#!/usr/bin/env python3
"""
TimeGAN Stage 1 - Minimal GAN
Master Thesis - Generative AI Workload Modeling
Fachhochschule Dortmund

WHAT THIS IS
============
Ablation Stage 1: the absolute minimum adversarial extension of LSTM v6.

Generator  = LSTM v6 unchanged (phase conditioning, regime latent, r_embed)
Discriminator = single BiLSTM -> scalar, no conditioning, standard BCE

Loss (generator):
    L_G = lambda_rec * MSE(recon, target)
        + lambda_adv * BCE(D(fake), real_label)

Loss (discriminator):
    L_D = BCE(D(real), 1) + BCE(D(fake.detach()), 0)

Nothing else added. Same LR for both. No spectral norm, no gradient penalty,
no feature matching. This is the baseline to compare every future stage against.

ABLATION PLAN
=============
Stage 1 (this file): minimal GAN
Stage 2: + spectral norm + two-timescale LR + n_disc_steps=2
Stage 3: + feature matching + moment matching
Stage 4: + WGAN-GP
Stage 5: + conditional discriminator (r + workload aware)
Stage 6: preprocessing switch (zero-mean vs full min-max)
Stage 7: epoch budget (200 vs 300 vs 500)

USAGE
-----
    python timegan_s1.py
    python timegan_s1.py --workloads bert --epochs 50
    python timegan_s1.py --lambda-adv 0.5
"""

import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

DATA_RAW_DIR = Path("data/processed/phase4/raw")
OUTPUT_DIR   = Path("outputs/phase4/timegan_s1")
MODEL_DIR    = Path("models/phase4/timegan_s1")

WORKLOADS = ["bert", "gpt2", "resnet152", "whisper", "yolo"]

ALL_METRICS = [
    "pod_cpu_usage", "pod_memory_bytes", "pod_psi_cpu",
    "pod_latency_avg", "pod_throughput",
    "gpu_utilization", "gpu_memory_used", "gpu_memory_total",
    "gpu_power_watts", "gpu_temperature",
]

DROP_PER_WORKLOAD = {
    "bert":      {"gpu_memory_total", "gpu_memory_used", "pod_memory_bytes",
                  "gpu_temperature",  "gpu_power_watts"},
    "gpt2":      {"gpu_memory_total", "pod_memory_bytes",
                  "gpu_temperature",  "gpu_power_watts"},
    "resnet152": {"gpu_memory_total", "gpu_memory_used", "pod_memory_bytes",
                  "gpu_temperature",  "gpu_power_watts"},
    "whisper":   {"gpu_memory_total", "gpu_memory_used", "pod_memory_bytes",
                  "pod_throughput",   "gpu_temperature",  "gpu_power_watts"},
    "yolo":      {"gpu_memory_total", "gpu_memory_used", "pod_memory_bytes",
                  "gpu_temperature",  "gpu_power_watts"},
}

DEFAULT_PHASE_BOUNDARIES = [0, 96, 180, 300, 420, 600]
N_PHASES = 6

# ------------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------------

GEN_CFG = {
    # Generator - identical to LSTM v6
    "hidden_dim":        128,
    "num_layers":        2,
    "latent_dim":        64,
    "regime_dim":        32,
    "pod_dim":           32,
    "replica_embed_dim": 8,
    "phase_embed_dim":   4,
    "dropout":           0.1,
    "grad_clip":         1.0,
    "seed":              42,
}

DISC_CFG = {
    # Discriminator - simple BiLSTM
    "hidden_dim":  128,
    "num_layers":  2,
    "dropout":     0.1,
}

TRAIN_CFG = {
    "batch_size":   16,
    "epochs":       200,
    "lr_gen":       1e-3,
    "lr_disc":      1e-3,    # same LR for Stage 1 (Stage 2 will split these)
    "weight_decay": 1e-5,
    "patience":     25,      # slightly more patience than LSTM (GAN loss is noisier)
    "min_delta":    1e-5,
    "lambda_rec":   1.0,     # reconstruction MSE weight
    "lambda_adv":   0.1,     # adversarial loss weight (start small, stable)
    "n_disc_steps": 1,       # discriminator updates per generator update (Stage 2 raises this)
    "clip_stds":    2.5,
    "n_gen":        5,
}


# ------------------------------------------------------------------
# Data helpers  (carried verbatim from LSTM v6)
# ------------------------------------------------------------------

def get_kept_metrics(workload):
    drop = DROP_PER_WORKLOAD[workload]
    kept_idx   = [i for i, m in enumerate(ALL_METRICS) if m not in drop]
    kept_names = [ALL_METRICS[i] for i in kept_idx]
    return kept_idx, kept_names


def load_raw_data(workload):
    path = DATA_RAW_DIR / f"{workload}_traces.npz"
    data = np.load(path, allow_pickle=True)
    norm_path = DATA_RAW_DIR / f"{workload}_normalization.json"
    with open(norm_path) as f:
        norm = json.load(f)
    return data, norm


def denormalize(traces_norm, metric_names, norm_params):
    lookup = norm_params["params"] if "params" in norm_params else norm_params
    out = traces_norm.copy().astype(np.float64)
    for j, m in enumerate(metric_names):
        if m not in lookup:
            continue
        mn = lookup[m]["min"]
        mx = lookup[m]["max"]
        out[:, :, j] = traces_norm[:, :, j] * (mx - mn) + mn
    return out


def apply_zeromean(traces, clip_stds=2.5):
    means = traces.mean(axis=1)
    zm    = traces - means[:, np.newaxis, :]
    if clip_stds is not None:
        metric_std = zm.std(axis=(0, 1), keepdims=True)
        clip_val   = clip_stds * metric_std
        zm         = np.clip(zm, -clip_val, clip_val)
    return zm.astype(np.float32), means.astype(np.float32)


def build_phase_sequence(seq_len, boundaries):
    phase_seq = np.zeros(seq_len, dtype=np.int64)
    for phase_idx, start in enumerate(boundaries):
        end = boundaries[phase_idx + 1] if phase_idx + 1 < len(boundaries) else seq_len
        phase_seq[start:end] = phase_idx
    return phase_seq


def append_phase_feature(traces, phase_seq):
    N, T, M = traces.shape
    phase_f = phase_seq.astype(np.float32) / (N_PHASES - 1)
    phase_f = np.broadcast_to(phase_f[np.newaxis, :, np.newaxis], (N, T, 1))
    return np.concatenate([traces, phase_f], axis=2).astype(np.float32)


def compute_metric_weights(zm_traces, train_idx, device):
    train_data = zm_traces[train_idx]
    var_per_m  = train_data.var(axis=(0, 1))
    var_per_m  = np.maximum(var_per_m, 1e-8)
    inv_var    = 1.0 / var_per_m
    weights    = inv_var / inv_var.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ------------------------------------------------------------------
# Post-hoc helpers  (carried from LSTM v6)
# ------------------------------------------------------------------

def fit_posthoc_power_regression(raw_traces, replica_counts, util_idx, power_idx):
    util  = raw_traces[:, :, util_idx].flatten()
    power = raw_traces[:, :, power_idx].flatten()
    mask  = util > 0.01
    if mask.sum() < 10:
        return 0.0, float(power.mean())
    X = util[mask]; y = power[mask]
    a = float(np.cov(X, y)[0, 1] / (np.var(X) + 1e-10))
    b = float(y.mean() - a * X.mean())
    return a, b


def reconstruct_posthoc(syn_orig, kept_names, n_pods, seq_len, posthoc_tables, workload):
    extra_cols  = []
    extra_names = []
    if "pod_memory_bytes" in posthoc_tables:
        val   = posthoc_tables["pod_memory_bytes"]
        col   = np.full((n_pods, seq_len, 1), val, dtype=np.float32)
        noise = np.random.normal(0, val * 0.02, (n_pods, seq_len, 1)).astype(np.float32)
        extra_cols.append(col + noise); extra_names.append("pod_memory_bytes")
    if "gpu_memory_used_lookup" in posthoc_tables:
        val   = posthoc_tables["gpu_memory_used_lookup"]
        col   = np.full((n_pods, seq_len, 1), val, dtype=np.float32)
        noise = np.random.normal(0, val * 0.01, (n_pods, seq_len, 1)).astype(np.float32)
        extra_cols.append(col + noise); extra_names.append("gpu_memory_used")
    if "power_regression" in posthoc_tables and "gpu_utilization" in kept_names:
        a, b     = posthoc_tables["power_regression"]
        util_idx = kept_names.index("gpu_utilization")
        util_col = syn_orig[:, :, util_idx]
        power    = np.maximum(a * util_col + b, 0.0)
        noise    = np.random.normal(0, abs(b) * 0.05, (n_pods, seq_len)).astype(np.float32)
        extra_cols.append((power + noise)[:, :, np.newaxis]); extra_names.append("gpu_power_watts")
    if extra_cols:
        return np.concatenate([syn_orig] + extra_cols, axis=2), kept_names + extra_names
    return syn_orig, kept_names


# ------------------------------------------------------------------
# Evaluation metrics  (carried from LSTM v6)
# ------------------------------------------------------------------

def compute_variance_ratio(real, synthetic, cap=5.0):
    vr = []
    for i in range(real.shape[2]):
        var_r = np.var(real[:, :, i])
        var_s = np.var(synthetic[:, :, i])
        vr.append(min(float(var_s / (var_r + 1e-10)), cap))
    return np.array(vr), float(np.mean(vr))


def compute_autocorr_similarity(real, synthetic, max_lag=20):
    diffs = []
    for i in range(real.shape[2]):
        r_flat = real[:, :, i].flatten()
        s_flat = synthetic[:, :, i].flatten()
        for lag in range(1, max_lag + 1):
            ac_r = float(np.corrcoef(r_flat[:-lag], r_flat[lag:])[0, 1])
            ac_s = float(np.corrcoef(s_flat[:-lag], s_flat[lag:])[0, 1])
            if not (np.isnan(ac_r) or np.isnan(ac_s)):
                diffs.append(abs(ac_r - ac_s))
    return float(np.mean(diffs)) if diffs else float("nan")


def detect_phase_jumps(traces, phase_seq):
    boundaries = [t for t in range(1, len(phase_seq))
                  if phase_seq[t] != phase_seq[t - 1]]
    if not boundaries:
        return 0.0
    boundary_deltas = []; within_deltas = []
    for i in range(traces.shape[0]):
        trace = traces[i]
        for t in range(1, trace.shape[0]):
            delta = float(np.mean(np.abs(trace[t] - trace[t - 1])))
            if t in boundaries:
                boundary_deltas.append(delta)
            else:
                within_deltas.append(delta)
    if not within_deltas or not boundary_deltas:
        return 0.0
    return float(np.mean(boundary_deltas)) / (float(np.mean(within_deltas)) + 1e-10)


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

class WorkloadDataset(Dataset):
    def __init__(self, zm_with_phase, zm_traces, means,
                 replica_counts, experiment_ids, phase_seq, idx):
        self.inp     = torch.tensor(zm_with_phase[idx], dtype=torch.float32)
        self.target  = torch.tensor(zm_traces[idx],     dtype=torch.float32)
        self.means   = torch.tensor(means[idx],         dtype=torch.float32)
        self.exp_ids = torch.tensor(experiment_ids[idx],dtype=torch.long)
        r = replica_counts[idx].astype(np.float32)
        self.r_norm  = torch.tensor((r - 1.0) / 9.0,   dtype=torch.float32)
        self.phase_t = torch.tensor(phase_seq,          dtype=torch.long)

    def __len__(self): return len(self.inp)

    def __getitem__(self, i):
        return (self.inp[i], self.target[i], self.means[i],
                self.r_norm[i], self.exp_ids[i], self.phase_t)


# ------------------------------------------------------------------
# Generator  (LSTM v6, copied exactly)
# ------------------------------------------------------------------

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, latent_dim, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])


class PhaseConditionedDecoder(nn.Module):
    def __init__(self, cond_dim, hidden_dim, num_layers,
                 output_dim, n_phases, phase_embed_dim, dropout):
        super().__init__()
        self.phase_embed = nn.Embedding(n_phases, phase_embed_dim)
        total_input      = cond_dim + phase_embed_dim
        self.h0          = nn.Linear(cond_dim, num_layers * hidden_dim)
        self.c0          = nn.Linear(cond_dim, num_layers * hidden_dim)
        self.lstm        = nn.LSTM(total_input, hidden_dim, num_layers,
                                   batch_first=True,
                                   dropout=dropout if num_layers > 1 else 0.0)
        self.fc_out      = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim  = hidden_dim
        self.num_layers  = num_layers

    def forward(self, z, phase_seq_batch):
        B, T = phase_seq_batch.shape
        h = self.h0(z).view(B, self.num_layers, self.hidden_dim).permute(1,0,2).contiguous()
        c = self.c0(z).view(B, self.num_layers, self.hidden_dim).permute(1,0,2).contiguous()
        z_exp  = z.unsqueeze(1).expand(B, T, -1)
        ph_emb = self.phase_embed(phase_seq_batch)
        inp    = torch.cat([z_exp, ph_emb], dim=2)
        out, _ = self.lstm(inp, (h, c))
        return self.fc_out(out)


class Generator(nn.Module):
    """LSTM v6 generator — unchanged from baseline."""
    def __init__(self, seq_len, n_metrics, gen_cfg):
        super().__init__()
        self.seq_len    = seq_len
        self.n_metrics  = n_metrics
        self.regime_dim = gen_cfg["regime_dim"]
        self.pod_dim    = gen_cfg["pod_dim"]

        self.pod_encoder = LSTMEncoder(
            n_metrics + 1, gen_cfg["hidden_dim"],
            gen_cfg["num_layers"], gen_cfg["pod_dim"], gen_cfg["dropout"])

        self.regime_proj = nn.Sequential(
            nn.Linear(gen_cfg["pod_dim"], gen_cfg["regime_dim"]), nn.ReLU())

        self.r_embed = nn.Sequential(
            nn.Linear(1, gen_cfg["replica_embed_dim"]), nn.ReLU())

        cond_dim = gen_cfg["regime_dim"] + gen_cfg["pod_dim"] + gen_cfg["replica_embed_dim"]

        self.decoder = PhaseConditionedDecoder(
            cond_dim, gen_cfg["hidden_dim"], gen_cfg["num_layers"],
            n_metrics, N_PHASES, gen_cfg["phase_embed_dim"], gen_cfg["dropout"])

    def forward(self, x_with_phase, r_norm, exp_ids, phase_seq_batch):
        B = x_with_phase.size(0)
        pod_lat    = self.pod_encoder(x_with_phase)
        regime_lat = torch.zeros(B, self.regime_dim, device=x_with_phase.device)
        for eid in exp_ids.unique():
            mask = (exp_ids == eid)
            if mask.sum() > 0:
                mean_pod = pod_lat[mask].mean(dim=0, keepdim=True)
                regime_lat[mask] = self.regime_proj(mean_pod).expand(mask.sum(), -1)
        r_e = self.r_embed(r_norm.unsqueeze(1))
        z   = torch.cat([regime_lat, pod_lat, r_e], dim=1)
        return torch.tanh(self.decoder(z, phase_seq_batch))

    def generate(self, r_norm_val, n_pods, phase_seq, device):
        self.eval()
        with torch.no_grad():
            regime_pod = torch.randn(1, self.pod_dim, device=device)
            regime_lat = self.regime_proj(regime_pod).expand(n_pods, -1)
            pod_lat    = torch.randn(n_pods, self.pod_dim, device=device)
            r_t        = torch.full((n_pods,), r_norm_val, device=device)
            r_e        = self.r_embed(r_t.unsqueeze(1))
            z          = torch.cat([regime_lat, pod_lat, r_e], dim=1)
            ph         = torch.tensor(phase_seq, dtype=torch.long, device=device)
            ph         = ph.unsqueeze(0).expand(n_pods, -1)
            return self.decoder(z, ph).cpu().numpy()


# ------------------------------------------------------------------
# Discriminator  (new for Stage 1)
# ------------------------------------------------------------------

class Discriminator(nn.Module):
    """
    Simple BiLSTM sequence discriminator.
    Input:  (B, T, M) - raw metric traces (zeromean normalized)
    Output: (B, 1)    - real/fake logit

    No conditioning on r or workload (Stage 5 will add this).
    No spectral normalization (Stage 2 will add this).
    """
    def __init__(self, n_metrics, disc_cfg):
        super().__init__()
        self.lstm = nn.LSTM(
            n_metrics,
            disc_cfg["hidden_dim"],
            disc_cfg["num_layers"],
            batch_first=True,
            bidirectional=True,
            dropout=disc_cfg["dropout"] if disc_cfg["num_layers"] > 1 else 0.0
        )
        # BiLSTM doubles hidden dim
        self.classifier = nn.Sequential(
            nn.Linear(disc_cfg["hidden_dim"] * 2, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
            # No sigmoid — using BCEWithLogitsLoss for numerical stability
        )

    def forward(self, x):
        """x: (B, T, M) -> logit: (B, 1)"""
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers*2, B, hidden_dim) for bidirectional
        # Take last layer forward + backward
        fwd = h_n[-2]   # (B, hidden_dim)
        bwd = h_n[-1]   # (B, hidden_dim)
        h   = torch.cat([fwd, bwd], dim=1)   # (B, hidden_dim*2)
        return self.classifier(h)            # (B, 1)


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train_timegan(generator, discriminator, train_dl, val_dl,
                  train_cfg, gen_cfg, device, metric_weights=None):
    """
    GAN training loop.

    Generator loss  = lambda_rec * MSE + lambda_adv * BCE(D(fake), real_label)
    Disc loss       = BCE(D(real), 1) + BCE(D(fake.detach()), 0)

    Early stopping monitors generator reconstruction loss (not adversarial),
    because adversarial loss oscillates by design and should not trigger stopping.
    """
    opt_G = torch.optim.Adam(generator.parameters(),
                             lr=train_cfg["lr_gen"],
                             weight_decay=train_cfg["weight_decay"])
    opt_D = torch.optim.Adam(discriminator.parameters(),
                             lr=train_cfg["lr_disc"],
                             weight_decay=train_cfg["weight_decay"])

    sched_G = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_G, factor=0.5, patience=7)

    bce  = nn.BCEWithLogitsLoss()
    mse  = nn.MSELoss(reduction="none")

    best_val_rec = float("inf")
    best_state_G = None
    best_state_D = None
    patience_c   = 0

    history = []   # for diagnostics

    for epoch in range(1, train_cfg["epochs"] + 1):
        generator.train()
        discriminator.train()

        epoch_loss_G = []; epoch_loss_D = []
        epoch_loss_rec = []; epoch_loss_adv = []

        for x_ph, target, _, r_norm, exp_ids, ph_batch in train_dl:
            x_ph    = x_ph.to(device)
            target  = target.to(device)
            r_norm  = r_norm.to(device)
            exp_ids = exp_ids.to(device)
            ph_batch = ph_batch.to(device)
            B = x_ph.size(0)

            real_label = torch.ones(B, 1, device=device)
            fake_label = torch.zeros(B, 1, device=device)

            # ---- Discriminator update(s) ----
            for _ in range(train_cfg["n_disc_steps"]):
                with torch.no_grad():
                    fake = generator(x_ph, r_norm, exp_ids, ph_batch)  # (B, T, M)

                logit_real = discriminator(target)
                logit_fake = discriminator(fake)

                loss_D = bce(logit_real, real_label) + bce(logit_fake, fake_label)

                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step()

            epoch_loss_D.append(loss_D.item())

            # ---- Generator update ----
            fake = generator(x_ph, r_norm, exp_ids, ph_batch)

            # Reconstruction loss
            raw_mse = mse(fake, target)   # (B, T, M)
            if metric_weights is not None:
                raw_mse = raw_mse * metric_weights.unsqueeze(0).unsqueeze(0)
            loss_rec = raw_mse.mean()

            # Adversarial loss: generator wants D to output real_label for fakes
            logit_fake = discriminator(fake)
            loss_adv   = bce(logit_fake, real_label)

            loss_G = (train_cfg["lambda_rec"] * loss_rec
                    + train_cfg["lambda_adv"] * loss_adv)

            opt_G.zero_grad()
            loss_G.backward()
            nn.utils.clip_grad_norm_(generator.parameters(), gen_cfg["grad_clip"])
            opt_G.step()

            epoch_loss_G.append(loss_G.item())
            epoch_loss_rec.append(loss_rec.item())
            epoch_loss_adv.append(loss_adv.item())

        # ---- Validation (reconstruction only for early stopping) ----
        generator.eval()
        discriminator.eval()
        val_recs = []
        with torch.no_grad():
            for x_ph, target, _, r_norm, exp_ids, ph_batch in val_dl:
                x_ph     = x_ph.to(device)
                target   = target.to(device)
                r_norm   = r_norm.to(device)
                exp_ids  = exp_ids.to(device)
                ph_batch = ph_batch.to(device)
                fake     = generator(x_ph, r_norm, exp_ids, ph_batch)
                raw_mse  = mse(fake, target)
                if metric_weights is not None:
                    raw_mse = raw_mse * metric_weights.unsqueeze(0).unsqueeze(0)
                val_recs.append(raw_mse.mean().item())

        val_rec = float(np.mean(val_recs))
        sched_G.step(val_rec)

        h = {
            "epoch":    epoch,
            "loss_G":   float(np.mean(epoch_loss_G)),
            "loss_D":   float(np.mean(epoch_loss_D)),
            "loss_rec": float(np.mean(epoch_loss_rec)),
            "loss_adv": float(np.mean(epoch_loss_adv)),
            "val_rec":  val_rec,
        }
        history.append(h)

        if epoch % 20 == 0 or epoch == 1:
            print(f"    ep {epoch:>4}  val_rec={val_rec:.5f}  "
                  f"G={h['loss_G']:.4f}  D={h['loss_D']:.4f}  "
                  f"rec={h['loss_rec']:.4f}  adv={h['loss_adv']:.4f}")

        if val_rec < best_val_rec - train_cfg["min_delta"]:
            best_val_rec = val_rec
            best_state_G = {k: v.cpu().clone() for k, v in generator.state_dict().items()}
            best_state_D = {k: v.cpu().clone() for k, v in discriminator.state_dict().items()}
            patience_c   = 0
        else:
            patience_c += 1
            if patience_c >= train_cfg["patience"]:
                print(f"    Early stop at epoch {epoch}")
                break

    generator.load_state_dict(best_state_G)
    discriminator.load_state_dict(best_state_D)
    return generator, discriminator, best_val_rec, epoch, history


# ------------------------------------------------------------------
# Evaluation  (same logic as LSTM v6)
# ------------------------------------------------------------------

def evaluate(generator, zm_with_phase, zm_traces, trace_means,
             raw_kept, replica_counts, phase_seq,
             train_idx, kept_names, norm_params, n_gen, device):

    real_norm    = raw_kept[train_idx]
    real_orig    = denormalize(real_norm, kept_names, norm_params)
    train_means  = trace_means[train_idx]
    train_r      = replica_counts[train_idx]

    unique_r   = np.unique(train_r)
    r_mean_map = {}
    for r_val in unique_r:
        mask = train_r == r_val
        r_mean_map[int(r_val)] = train_means[mask].mean(axis=0)

    syn_list   = []
    syn_r_list = []

    for r_val in unique_r:
        r_norm  = float((r_val - 1.0) / 9.0)
        r_mean  = r_mean_map[int(r_val)]
        for _ in range(n_gen):
            pkg = generator.generate(r_norm, int(r_val), phase_seq, device)
            for pod_trace in pkg:
                s_norm = np.clip(pod_trace + r_mean[np.newaxis, :], 0.0, 1.0)
                syn_list.append(s_norm)
                syn_r_list.append(int(r_val))

    syn_norm  = np.array(syn_list)
    syn_r_arr = np.array(syn_r_list, dtype=np.int32)
    syn_orig  = denormalize(syn_norm, kept_names, norm_params)
    real_r_arr = train_r

    vr, vr_mean = compute_variance_ratio(real_orig, syn_orig)
    ac_diff     = compute_autocorr_similarity(real_orig, syn_orig)
    real_jump   = detect_phase_jumps(real_orig, phase_seq)
    syn_jump    = detect_phase_jumps(syn_orig,  phase_seq)
    jump_ratio  = syn_jump / (real_jump + 1e-10)

    return (real_orig, syn_orig,
            real_r_arr, syn_r_arr,
            vr, vr_mean, ac_diff,
            real_jump, syn_jump, jump_ratio)


# ------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------

def plot_comparison(workload, real_orig, syn_orig,
                    real_r_arr, syn_r_arr,
                    kept_names, phase_seq, history, save_dir):
    N_SHOW_R = 4
    priority = ["pod_latency_avg", "pod_cpu_usage", "gpu_utilization",
                "pod_throughput", "gpu_power_watts", "pod_psi_cpu"]
    plot_idx = []
    for m in priority:
        if m in kept_names and len(plot_idx) < 4:
            plot_idx.append(kept_names.index(m))
    for i in range(len(kept_names)):
        if i not in plot_idx and len(plot_idx) < 4:
            plot_idx.append(i)

    boundaries = [t for t in range(1, len(phase_seq))
                  if phase_seq[t] != phase_seq[t - 1]]

    common_r = sorted(set(real_r_arr.tolist()) & set(syn_r_arr.tolist()))
    if len(common_r) > N_SHOW_R:
        idx_step = max(1, (len(common_r) - 1) // (N_SHOW_R - 1))
        selected = common_r[::idx_step][:N_SHOW_R]
        if common_r[-1] not in selected:
            selected[-1] = common_r[-1]
        common_r = sorted(set(selected))

    palette  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    r_colour = {r: palette[i % len(palette)] for i, r in enumerate(common_r)}

    t = np.arange(real_orig.shape[1])

    # 2x3 grid: 4 metric panels + 1 loss curve panel
    fig = plt.figure(figsize=(15, 10))
    gs  = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32)

    metric_axes = [fig.add_subplot(gs[r, c]) for r, c in [(0,0),(0,1),(1,0),(1,1)]]
    loss_ax     = fig.add_subplot(gs[:, 2])

    legend_handles = []

    for ax_i, m_i in enumerate(plot_idx):
        ax = metric_axes[ax_i]
        for r_val in common_r:
            col = r_colour[r_val]
            real_idxs = np.where(real_r_arr == r_val)[0]
            if len(real_idxs):
                ax.plot(t, real_orig[real_idxs[0], :, m_i],
                        color=col, alpha=0.85, linewidth=0.9, linestyle="-")
            syn_idxs = np.where(syn_r_arr == r_val)[0]
            if len(syn_idxs):
                ax.plot(t, syn_orig[syn_idxs[0], :, m_i],
                        color=col, alpha=0.75, linewidth=0.9, linestyle="--")
            if ax_i == 0 and len(real_idxs):
                import matplotlib.lines as mlines
                legend_handles.append(
                    mlines.Line2D([], [], color=col, linewidth=1.2,
                                  label=f"r={r_val}  solid=real  dashed=syn"))
        for b in boundaries:
            ax.axvline(x=b, color="silver", linewidth=0.7, linestyle=":")
        ax.set_title(kept_names[m_i].replace("pod_","").replace("gpu_",""), fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("timestep", fontsize=7)

    # Loss curve
    if history:
        eps      = [h["epoch"]    for h in history]
        rec_loss = [h["loss_rec"] for h in history]
        adv_loss = [h["loss_adv"] for h in history]
        disc_loss= [h["loss_D"]   for h in history]
        val_loss = [h["val_rec"]  for h in history]
        loss_ax.plot(eps, rec_loss,  label="train rec",  linewidth=1.0, color="#1f77b4")
        loss_ax.plot(eps, val_loss,  label="val rec",    linewidth=1.0, color="#1f77b4",
                     linestyle="--")
        loss_ax.plot(eps, adv_loss,  label="adv (gen)",  linewidth=1.0, color="#ff7f0e")
        loss_ax.plot(eps, disc_loss, label="disc loss",  linewidth=1.0, color="#2ca02c")
        loss_ax.set_title("Training losses", fontsize=9)
        loss_ax.set_xlabel("epoch", fontsize=7)
        loss_ax.legend(fontsize=7)
        loss_ax.tick_params(labelsize=7)

    fig.suptitle(
        f"{workload.upper()} - TimeGAN Stage 1 (minimal GAN)\n"
        f"solid=real  dashed=synthetic  grey=phase boundaries",
        fontsize=10, fontweight="bold")

    if legend_handles:
        fig.legend(handles=legend_handles, loc="lower center",
                   ncol=min(len(legend_handles), 4),
                   fontsize=7, framealpha=0.7,
                   bbox_to_anchor=(0.35, -0.01))

    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{workload}_timegan_s1.png"
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()
    return path


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workloads",    nargs="+", default=WORKLOADS)
    parser.add_argument("--epochs",       type=int,  default=TRAIN_CFG["epochs"])
    parser.add_argument("--lambda-rec",   type=float,default=TRAIN_CFG["lambda_rec"])
    parser.add_argument("--lambda-adv",   type=float,default=TRAIN_CFG["lambda_adv"])
    parser.add_argument("--n-gen",        type=int,  default=TRAIN_CFG["n_gen"])
    parser.add_argument("--device",       default="auto")
    parser.add_argument("--seed",         type=int,  default=GEN_CFG["seed"])
    parser.add_argument("--phase-boundaries", type=str,
                        default=",".join(map(str, DEFAULT_PHASE_BOUNDARIES)))
    args = parser.parse_args()

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    TRAIN_CFG["epochs"]     = args.epochs
    TRAIN_CFG["lambda_rec"] = args.lambda_rec
    TRAIN_CFG["lambda_adv"] = args.lambda_adv

    boundaries = list(map(int, args.phase_boundaries.split(",")))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    print("=" * 65)
    print("TimeGAN Stage 1 - Minimal GAN")
    print(f"Device      : {device}")
    print(f"Epochs      : {args.epochs}")
    print(f"lambda_rec  : {TRAIN_CFG['lambda_rec']}")
    print(f"lambda_adv  : {TRAIN_CFG['lambda_adv']}")
    print(f"n_disc_steps: {TRAIN_CFG['n_disc_steps']}")
    print(f"lr_gen      : {TRAIN_CFG['lr_gen']}  lr_disc: {TRAIN_CFG['lr_disc']}")
    print("=" * 65)

    for workload in args.workloads:
        print(f"\n{'='*65}")
        print(f"Workload: {workload.upper()}")
        print(f"{'='*65}")

        data, norm = load_raw_data(workload)
        raw_traces     = data["traces"]
        replica_counts = data["replica_counts"]
        train_idx      = data["train_idx"]
        val_idx        = data["val_idx"]
        metadata       = list(data["metadata"])
        seq_len        = raw_traces.shape[1]

        phase_seq = build_phase_sequence(seq_len, boundaries)

        # Experiment IDs
        exp_key_to_id = {}
        experiment_ids = np.zeros(len(metadata), dtype=np.int64)
        for i, meta in enumerate(metadata):
            key = (meta.get("experiment_id") if isinstance(meta, dict)
                   else f"{workload}_{replica_counts[i]}")
            if key not in exp_key_to_id:
                exp_key_to_id[key] = len(exp_key_to_id)
            experiment_ids[i] = exp_key_to_id[key]

        kept_idx, kept_names = get_kept_metrics(workload)
        n_metrics = len(kept_idx)
        raw_kept  = raw_traces[:, :, kept_idx]

        # Post-hoc tables
        norm_params = norm["params"] if "params" in norm else norm
        raw_phys    = denormalize(raw_kept, kept_names, norm_params)
        posthoc_tables = {}
        mem_raw = raw_traces[:, :, ALL_METRICS.index("pod_memory_bytes")]
        posthoc_tables["pod_memory_bytes"] = float(mem_raw.mean())
        if workload != "gpt2":
            gmem_raw = raw_traces[:, :, ALL_METRICS.index("gpu_memory_used")]
            posthoc_tables["gpu_memory_used_lookup"] = float(gmem_raw.mean())
        a, b = fit_posthoc_power_regression(
            denormalize(raw_traces, ALL_METRICS, norm_params),
            replica_counts,
            ALL_METRICS.index("gpu_utilization"),
            ALL_METRICS.index("gpu_power_watts"))
        posthoc_tables["power_regression"] = (a, b)

        # Preprocessing
        zm_traces, trace_means = apply_zeromean(raw_kept, TRAIN_CFG["clip_stds"])
        zm_with_phase = append_phase_feature(zm_traces, phase_seq)

        print(f"  Kept metrics ({n_metrics}): {kept_names}")
        print(f"  Train: {len(train_idx)}  Val: {len(val_idx)}")

        ds_train = WorkloadDataset(zm_with_phase, zm_traces, trace_means,
                                   replica_counts, experiment_ids, phase_seq, train_idx)
        ds_val   = WorkloadDataset(zm_with_phase, zm_traces, trace_means,
                                   replica_counts, experiment_ids, phase_seq, val_idx)
        dl_train = DataLoader(ds_train, batch_size=TRAIN_CFG["batch_size"], shuffle=True)
        dl_val   = DataLoader(ds_val,   batch_size=TRAIN_CFG["batch_size"])

        generator     = Generator(seq_len, n_metrics, GEN_CFG).to(device)
        discriminator = Discriminator(n_metrics, DISC_CFG).to(device)

        n_params_G = sum(p.numel() for p in generator.parameters())
        n_params_D = sum(p.numel() for p in discriminator.parameters())
        print(f"  Generator params    : {n_params_G:,}")
        print(f"  Discriminator params: {n_params_D:,}")

        metric_weights = compute_metric_weights(zm_traces, train_idx, device)

        t0 = time.time()
        generator, discriminator, best_val_rec, n_epochs, history = train_timegan(
            generator, discriminator, dl_train, dl_val,
            TRAIN_CFG, GEN_CFG, device, metric_weights)
        elapsed = time.time() - t0

        (real_orig, syn_orig,
         real_r_arr, syn_r_arr,
         vr, vr_mean, ac_diff,
         real_jump, syn_jump, jump_ratio) = evaluate(
            generator, zm_with_phase, zm_traces, trace_means,
            raw_kept, replica_counts, phase_seq,
            train_idx, kept_names, norm_params, args.n_gen, device)

        print(f"\n  Results:")
        print(f"    val_rec       = {best_val_rec:.5f}")
        print(f"    var_ratio     = {vr_mean:.4f}")
        print(f"    autocorr_diff = {ac_diff:.4f}")
        print(f"    jump_ratio    = {jump_ratio:.3f}x")
        print(f"    epochs        = {n_epochs}  time = {elapsed:.1f}s")
        print(f"  Per-metric VR:")
        for j, m in enumerate(kept_names):
            flag = " <-- low" if vr[j] < 0.5 else (" <-- high" if vr[j] > 3.0 else "")
            print(f"    {m:<22} {vr[j]:.4f}{flag}")

        plot_path = plot_comparison(
            workload, real_orig, syn_orig,
            real_r_arr, syn_r_arr,
            kept_names, phase_seq, history,
            OUTPUT_DIR / "plots")
        print(f"  Plot: {plot_path}")

        # Save
        mdir = MODEL_DIR / workload
        mdir.mkdir(parents=True, exist_ok=True)
        torch.save(generator.state_dict(),     mdir / "generator.pt")
        torch.save(discriminator.state_dict(), mdir / "discriminator.pt")

        cfg_out = {
            **GEN_CFG, **DISC_CFG,
            "workload": workload, "kept_metrics": kept_names,
            "n_metrics": n_metrics, "phase_boundaries": boundaries,
            "seq_len": seq_len, "lambda_rec": TRAIN_CFG["lambda_rec"],
            "lambda_adv": TRAIN_CFG["lambda_adv"],
        }
        with open(mdir / "config.json", "w") as f:
            json.dump(cfg_out, f, indent=2)
        with open(mdir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        all_results.append({
            "workload":             workload,
            "stage":                "s1_minimal_gan",
            "var_ratio_mean":       vr_mean,
            "var_ratio_per_metric": vr.tolist(),
            "autocorr_diff":        ac_diff,
            "phase_jump_ratio":     jump_ratio,
            "val_rec":              best_val_rec,
            "n_epochs":             n_epochs,
            "elapsed_s":            elapsed,
            "metrics_generated":    list(kept_names),
            "lambda_rec":           TRAIN_CFG["lambda_rec"],
            "lambda_adv":           TRAIN_CFG["lambda_adv"],
            "lstm_baseline_vr":     {
                "bert": 0.594, "gpt2": 0.676, "resnet152": 0.607,
                "whisper": 0.748, "yolo": 0.438
            }.get(workload),
        })

    # Summary
    print(f"\n{'='*65}")
    print("TIMEGAN STAGE 1 SUMMARY")
    print(f"{'='*65}")
    header = f"{'Workload':<12}  {'VR S1':>8}  {'VR LSTM':>8}  {'Delta':>8}  {'JumpRatio':>10}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        lstm_vr = r["lstm_baseline_vr"] or 0.0
        delta   = r["var_ratio_mean"] - lstm_vr
        print(f"  {r['workload']:<12}"
              f"  {r['var_ratio_mean']:>8.4f}"
              f"  {lstm_vr:>8.4f}"
              f"  {delta:>+8.4f}"
              f"  {r['phase_jump_ratio']:>10.3f}x")
    print("-" * len(header))
    vr_vals = [r["var_ratio_mean"] for r in all_results]
    print(f"  {'MEAN':<12}  {np.mean(vr_vals):>8.4f}")
    print()
    print("Compare against LSTM baseline mean VR = 0.612")
    print("If S1 > 0.612: adversarial loss helps even in minimal form")
    print("If S1 < 0.612: GAN instability hurting — Stage 2 stabilization needed")

    out_path = OUTPUT_DIR / "timegan_s1_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults : {out_path}")
    print(f"Plots   : {OUTPUT_DIR}/plots/")
    print(f"Models  : {MODEL_DIR}/")


if __name__ == "__main__":
    main()