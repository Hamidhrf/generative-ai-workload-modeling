#!/usr/bin/env python3
"""
TimeGAN S6 - Discriminator Warmup
==================================
Master Thesis - Generative AI Workload Modeling
Fachhochschule Dortmund

WHY S6 EXISTS
=============
S5a and S5b established that:
  - Lambda rebalancing (rec=0.1, adv=1.0) gave +28% VR over S4 for genA
  - Conditional discriminator (S5b) regressed genA by -20%
  - Both S5a and S5b show the same failure signature:
    disc loss starts at ~10, crashes to 0 within 5 epochs, never recovers
  - Generator adversarial loss has no useful signal after epoch 5
  - Within-phase variance remains collapsed, jump ratio 8-24x

ROOT CAUSE
==========
The BiLSTM discriminator (550k params) has far more capacity than it needs
to memorise a training set of ~49 samples. It reaches near-perfect
discrimination on the first pass through the data before the generator has
learned to produce anything realistic. Once the discriminator is dominant,
the adversarial gradient to the generator is essentially constant and
uninformative. The generator stops improving under adversarial pressure and
falls back on reconstruction loss alone.

S6 FIX: DISCRIMINATOR WARMUP
=============================
Train the generator alone for the first `disc_warmup` epochs with
reconstruction loss only (lambda_rec=1.0, lambda_adv=0.0, discriminator
frozen). The discriminator is not updated and does not receive any gradients
during warmup.

After warmup, switch to adversarial training with the S5a lambda schedule
(lambda_rec=0.1, lambda_adv=1.0). The generator now starts from a reasonable
reconstruction-quality baseline instead of random initialization, giving the
discriminator a harder task to solve and providing more informative gradients
in both directions.

Three warmup lengths are tested to bracket the right threshold:
  --disc-warmup 20   (default, conservative)
  --disc-warmup 30   (if 20 insufficient)
  --disc-warmup 50   (if 30 insufficient)

KEY DIAGNOSTIC
==============
After warmup ends, watch disc loss in the printed output and plots.
  - If disc loss crashes to 0 in first 5 post-warmup epochs: warmup too short
  - If disc loss stabilizes between 1 and 5: warmup worked, equilibrium found
  - If disc loss oscillates but does not collapse: adversarial game is live

ARCHITECTURE AND BASE CONFIG
=============================
Generator:     option A (reconstruction-based, LSTM v6)
               genB is dropped: consistently worse than genA and no fallback
               gradient when adversarial signal collapses.
Discriminator: same BiLSTM as S5a (no conditional — S5b showed it hurts)
Objective:     WGAN-GP (same as S5a)
Lambda warmup: rec=1.0, adv=0.0
Lambda post:   rec=0.1, adv=1.0
LR:            gen=1e-3, disc=2e-4 (two-timescale, same as S5a)
n_disc_steps:  2 during adversarial phase, 0 during warmup
GP lambda:     10.0
FM lambda:     0.01
Epochs:        200 (warmup epochs counted within budget)

USAGE
-----
    # Default warmup = 20 epochs
    python timegan_s6.py

    # Test different warmup lengths sequentially
    python timegan_s6.py --disc-warmup 20
    python timegan_s6.py --disc-warmup 30
    python timegan_s6.py --disc-warmup 50

    # Single workload for quick iteration
    python timegan_s6.py --disc-warmup 20 --workloads bert

    # All workloads, specific warmup
    python timegan_s6.py --disc-warmup 30 --workloads bert gpt2 resnet152 whisper yolo

OUTPUT STRUCTURE
----------------
    outputs/phase4/timegan_s6/warmup{N}/plots/{workload}_s6_w{N}.png
    models/phase4/timegan_s6/warmup{N}/{workload}/generator.pt
    models/phase4/timegan_s6/warmup{N}/{workload}/discriminator.pt
    models/phase4/timegan_s6/warmup{N}/{workload}/history.json
    outputs/phase4/timegan_s6/warmup{N}/results.json
"""

import argparse
import json
import math
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
# Paths and constants
# ------------------------------------------------------------------

DATA_RAW_DIR = Path("data/processed/phase4/raw")

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

# S5a was the best single-change result (VR=0.413 mean).
# S6 uses S5a as base with warmup added on top.
# Conditional discriminator is OFF (S5b showed it hurts).
BASE_CONFIG = {
    "adversarial":       "wgan",
    "spectral_norm":     False,
    "two_timescale_lr":  True,
    "n_disc_steps":      2,
    "lambda_rec_warmup": 1.0,   # reconstruction-only phase
    "lambda_adv_warmup": 0.0,
    "lambda_rec_post":   0.1,   # adversarial phase (S5a values)
    "lambda_adv_post":   1.0,
    "lambda_fm":         0.01,
    "lambda_gp":         10.0,
    "conditional_disc":  False,
    "preprocessing":     "zeromean",
    "epochs":            200,
}

GEN_CFG = {
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
    "hidden_dim":  128,
    "num_layers":  2,
    "dropout":     0.1,
}

# Reference VRs for comparison in summary table
REFERENCE_VR = {
    "bert":      {"lstm": 0.594, "s4": 0.117, "s5a_A": 0.141, "s5b_A": 0.107},
    "gpt2":      {"lstm": 0.676, "s4": 0.480, "s5a_A": 0.500, "s5b_A": 0.500},
    "resnet152": {"lstm": 0.607, "s4": 0.102, "s5a_A": 0.113, "s5b_A": 0.145},
    "whisper":   {"lstm": 0.748, "s4": 0.785, "s5a_A": 1.187, "s5b_A": 0.782},
    "yolo":      {"lstm": 0.438, "s4": 0.123, "s5a_A": 0.123, "s5b_A": 0.130},
}

# ------------------------------------------------------------------
# Data helpers
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


def compute_metric_weights(traces, train_idx, device):
    train_data = traces[train_idx]
    var_per_m  = train_data.var(axis=(0, 1))
    var_per_m  = np.maximum(var_per_m, 1e-8)
    inv_var    = 1.0 / var_per_m
    weights    = inv_var / inv_var.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)

# ------------------------------------------------------------------
# Evaluation metrics
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
    boundary_deltas = []
    within_deltas   = []
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
# Option A only: yields (zm_with_phase, zm_target, means, r_norm, exp_ids, phase)
# ------------------------------------------------------------------

class WorkloadDataset(Dataset):
    def __init__(self, zm_with_phase, zm_traces, means,
                 replica_counts, experiment_ids, phase_seq, idx):
        self.inp     = torch.tensor(zm_with_phase[idx], dtype=torch.float32)
        self.target  = torch.tensor(zm_traces[idx],     dtype=torch.float32)
        self.means   = torch.tensor(means[idx],          dtype=torch.float32)
        self.exp_ids = torch.tensor(experiment_ids[idx], dtype=torch.long)
        r = replica_counts[idx].astype(np.float32)
        self.r_norm  = torch.tensor((r - 1.0) / 9.0,    dtype=torch.float32)
        self.phase_t = torch.tensor(phase_seq,           dtype=torch.long)

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, i):
        return (self.inp[i], self.target[i], self.means[i],
                self.r_norm[i], self.exp_ids[i], self.phase_t)

# ------------------------------------------------------------------
# Generator (Option A: reconstruction-based, LSTM v6 unchanged)
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
        self.h0          = nn.Linear(cond_dim, num_layers * hidden_dim)
        self.c0          = nn.Linear(cond_dim, num_layers * hidden_dim)
        self.lstm        = nn.LSTM(cond_dim + phase_embed_dim, hidden_dim, num_layers,
                                   batch_first=True,
                                   dropout=dropout if num_layers > 1 else 0.0)
        self.fc_out      = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim  = hidden_dim
        self.num_layers  = num_layers

    def forward(self, z, phase_seq_batch):
        B, T = phase_seq_batch.shape
        h = self.h0(z).view(B, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        c = self.c0(z).view(B, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        z_exp  = z.unsqueeze(1).expand(B, T, -1)
        ph_emb = self.phase_embed(phase_seq_batch)
        inp    = torch.cat([z_exp, ph_emb], dim=2)
        out, _ = self.lstm(inp, (h, c))
        return self.fc_out(out)


class GeneratorA(nn.Module):
    """
    Reconstruction-based generator (LSTM v6).
    forward(): takes real trace as input, reconstructs it.
    generate(): substitutes random noise for the pod latent.
    """
    def __init__(self, seq_len, n_metrics, cfg):
        super().__init__()
        self.seq_len    = seq_len
        self.n_metrics  = n_metrics
        self.regime_dim = cfg["regime_dim"]
        self.pod_dim    = cfg["pod_dim"]

        self.pod_encoder = LSTMEncoder(
            n_metrics + 1, cfg["hidden_dim"],
            cfg["num_layers"], cfg["pod_dim"], cfg["dropout"])

        self.regime_proj = nn.Sequential(
            nn.Linear(cfg["pod_dim"], cfg["regime_dim"]), nn.ReLU())

        self.r_embed = nn.Sequential(
            nn.Linear(1, cfg["replica_embed_dim"]), nn.ReLU())

        cond_dim = cfg["regime_dim"] + cfg["pod_dim"] + cfg["replica_embed_dim"]
        self.decoder = PhaseConditionedDecoder(
            cond_dim, cfg["hidden_dim"], cfg["num_layers"],
            n_metrics, N_PHASES, cfg["phase_embed_dim"], cfg["dropout"])

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
# Discriminator (WGAN, unconditional — S5b showed conditional hurts)
# ------------------------------------------------------------------

class Discriminator(nn.Module):
    """
    BiLSTM critic. WGAN mode (no sigmoid). Unconditional.
    Same capacity as S5a (hidden_dim=128, num_layers=2).
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
        self.classifier = nn.Sequential(
            nn.Linear(disc_cfg["hidden_dim"] * 2, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            # No sigmoid: unbounded Wasserstein score
        )

    def forward(self, x, r_norm=None):
        _, (h_n, _) = self.lstm(x)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.classifier(h)

    def get_features(self, x, r_norm=None):
        _, (h_n, _) = self.lstm(x)
        return torch.cat([h_n[-2], h_n[-1]], dim=1)

# ------------------------------------------------------------------
# Gradient penalty (WGAN-GP)
# CuDNN disabled for double-backward through LSTM
# ------------------------------------------------------------------

def gradient_penalty(critic, real, fake, device, lam=10.0):
    B     = real.size(0)
    alpha = torch.rand(B, 1, 1, device=device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    with torch.backends.cudnn.flags(enabled=False):
        score_interp = critic(interp)
    grads = torch.autograd.grad(
        outputs=score_interp,
        inputs=interp,
        grad_outputs=torch.ones_like(score_interp),
        create_graph=True,
    )[0]
    grads_norm = grads.reshape(B, -1).norm(2, dim=1)
    return lam * ((grads_norm - 1) ** 2).mean()

# ------------------------------------------------------------------
# Training loop with discriminator warmup
#
# Warmup phase  (epoch <= disc_warmup):
#   - Generator trained with reconstruction loss only (lam_rec=1.0, lam_adv=0.0)
#   - Discriminator parameters are frozen (no update, no forward for disc)
#   - n_disc_steps = 0
#
# Adversarial phase  (epoch > disc_warmup):
#   - Switch to lam_rec=0.1, lam_adv=1.0
#   - Discriminator unfreezes and trains normally
#   - n_disc_steps = cfg["n_disc_steps"]
#
# The transition epoch is printed clearly so you can correlate it with
# the disc loss plot to judge whether the warmup length was sufficient.
# ------------------------------------------------------------------

def train(generator, discriminator, train_dl, val_dl,
          base_cfg, disc_warmup, gen_cfg, device, metric_weights=None):
    """
    Unified training loop with discriminator warmup.

    disc_warmup: int
        Number of epochs to train generator alone before introducing
        the discriminator. Set to 0 to reproduce S5a exactly (no warmup).
    """
    lam_gp   = base_cfg["lambda_gp"]
    lam_fm   = base_cfg["lambda_fm"]
    use_fm   = lam_fm > 0
    patience    = 25
    min_delta   = 1e-5

    lr_gen  = 1e-3
    lr_disc = 2e-4  # two-timescale

    betas = (0.0, 0.9)  # WGAN betas

    opt_G = torch.optim.Adam(generator.parameters(),
                             lr=lr_gen, betas=betas, weight_decay=1e-5)
    opt_D = torch.optim.Adam(discriminator.parameters(),
                             lr=lr_disc, betas=betas, weight_decay=1e-5)

    sched_G = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_G, factor=0.5, patience=7)

    mse_fn = nn.MSELoss(reduction="none")

    best_val_rec = float("inf")
    best_state_G = {k: v.cpu().clone() for k, v in generator.state_dict().items()}
    best_state_D = {k: v.cpu().clone() for k, v in discriminator.state_dict().items()}
    patience_c   = 0
    history      = []

    print(f"    Warmup: {disc_warmup} epochs (rec-only), "
          f"then adversarial from epoch {disc_warmup + 1}")

    for epoch in range(1, base_cfg["epochs"] + 1):

        # Determine phase and lambda values for this epoch
        in_warmup = (epoch <= disc_warmup)

        if in_warmup:
            lam_rec      = base_cfg["lambda_rec_warmup"]   # 1.0
            lam_adv      = base_cfg["lambda_adv_warmup"]   # 0.0
            n_disc_steps = 0                               # disc frozen
        else:
            lam_rec      = base_cfg["lambda_rec_post"]     # 0.1
            lam_adv      = base_cfg["lambda_adv_post"]     # 1.0
            n_disc_steps = base_cfg["n_disc_steps"]        # 2

        if epoch == disc_warmup + 1 and disc_warmup > 0:
            print(f"    [epoch {epoch}] Warmup complete. Switching to adversarial training.")

        generator.train()
        discriminator.train()

        ep_rec = ep_adv_g = ep_loss_d = ep_wdist = 0.0
        n_batches = 0

        for inp, target, _, r_norm, exp_ids, ph_batch in train_dl:
            inp      = inp.to(device)
            target   = target.to(device)
            r_norm   = r_norm.to(device)
            exp_ids  = exp_ids.to(device)
            ph_batch = ph_batch.to(device)
            B        = target.size(0)

            # ---- Discriminator update (adversarial phase only) ----
            if n_disc_steps > 0:
                with torch.no_grad():
                    fake_detach = generator(inp, r_norm, exp_ids, ph_batch)

                for _ in range(n_disc_steps):
                    opt_D.zero_grad()
                    score_real = discriminator(target)
                    score_fake = discriminator(fake_detach)
                    w_dist     = score_real.mean() - score_fake.mean()
                    loss_D     = -w_dist
                    gp         = gradient_penalty(
                        discriminator, target, fake_detach, device, lam_gp)
                    loss_D     = loss_D + gp
                    loss_D.backward()
                    opt_D.step()

                ep_loss_d += loss_D.item()
                ep_wdist  += w_dist.item()
            else:
                # Warmup: discriminator not updated, log zeros
                ep_loss_d += 0.0
                ep_wdist  += 0.0

            # ---- Generator update ----
            opt_G.zero_grad()
            fake = generator(inp, r_norm, exp_ids, ph_batch)

            raw_mse = mse_fn(fake, target)
            if metric_weights is not None:
                raw_mse = raw_mse * metric_weights.unsqueeze(0).unsqueeze(0)
            loss_rec = raw_mse.mean()

            if lam_adv > 0:
                score_fake_g = discriminator(fake)
                loss_adv     = -score_fake_g.mean()
            else:
                loss_adv = torch.tensor(0.0, device=device)

            loss_G = lam_rec * loss_rec + lam_adv * loss_adv

            if use_fm and not in_warmup:
                feat_real = discriminator.get_features(target).detach()
                feat_fake = discriminator.get_features(fake)
                loss_G    = loss_G + lam_fm * nn.functional.mse_loss(
                    feat_fake.mean(dim=0), feat_real.mean(dim=0))

            loss_G.backward()
            nn.utils.clip_grad_norm_(generator.parameters(), gen_cfg["grad_clip"])
            opt_G.step()

            ep_rec   += loss_rec.item()
            ep_adv_g += loss_adv.item() if hasattr(loss_adv, "item") else float(loss_adv)
            n_batches += 1

        # ---- Validation ----
        generator.eval()
        discriminator.eval()
        val_recs = []
        with torch.no_grad():
            for inp, target, _, r_norm, exp_ids, ph_batch in val_dl:
                inp      = inp.to(device)
                target   = target.to(device)
                r_norm   = r_norm.to(device)
                exp_ids  = exp_ids.to(device)
                ph_batch = ph_batch.to(device)
                B        = target.size(0)
                fake     = generator(inp, r_norm, exp_ids, ph_batch)
                raw_mse  = mse_fn(fake, target)
                if metric_weights is not None:
                    raw_mse = raw_mse * metric_weights.unsqueeze(0).unsqueeze(0)
                val_recs.append(raw_mse.mean().item())

        val_rec = float(np.mean(val_recs))
        sched_G.step(val_rec)

        phase_label = "warmup" if in_warmup else "adv"
        h = {
            "epoch":      epoch,
            "phase":      phase_label,
            "loss_G":     float(ep_adv_g / n_batches),
            "loss_D":     float(ep_loss_d / n_batches),
            "loss_rec":   float(ep_rec    / n_batches),
            "w_dist":     float(ep_wdist  / n_batches),
            "val_rec":    val_rec,
            "lam_rec":    lam_rec,
            "lam_adv":    lam_adv,
        }
        history.append(h)

        if epoch % 10 == 0 or epoch == 1 or epoch == disc_warmup + 1:
            print(f"    ep {epoch:>4} [{phase_label:>6}]  val_rec={val_rec:.5f}  "
                  f"G={h['loss_G']:.4f}  D={h['loss_D']:.4f}  "
                  f"rec={h['loss_rec']:.4f}  W={h['w_dist']:.4f}")

        if not math.isnan(val_rec) and val_rec < best_val_rec - min_delta:
            best_val_rec = val_rec
            best_state_G = {k: v.cpu().clone() for k, v in generator.state_dict().items()}
            best_state_D = {k: v.cpu().clone() for k, v in discriminator.state_dict().items()}
            patience_c   = 0
        else:
            patience_c += 1
            if patience_c >= patience:
                print(f"    Early stop at epoch {epoch}")
                break

    generator.load_state_dict(best_state_G)
    discriminator.load_state_dict(best_state_D)
    return generator, discriminator, best_val_rec, epoch, history

# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate(generator, gen_cfg,
             zm_traces, trace_means, raw_kept,
             replica_counts, phase_seq,
             train_idx, kept_names, norm_params, n_gen, device):

    real_norm   = raw_kept[train_idx]
    real_orig   = denormalize(real_norm, kept_names, norm_params)
    train_means = trace_means[train_idx]
    train_r     = replica_counts[train_idx]

    unique_r   = np.unique(train_r)
    r_mean_map = {int(rv): train_means[train_r == rv].mean(axis=0)
                  for rv in unique_r}

    syn_list   = []
    syn_r_list = []

    for r_val in unique_r:
        r_norm_val = float((r_val - 1.0) / 9.0)
        r_mean     = r_mean_map[int(r_val)]
        for _ in range(n_gen):
            pkg = generator.generate(r_norm_val, int(r_val), phase_seq, device)
            for pod_trace in pkg:
                s_norm = np.clip(pod_trace + r_mean[np.newaxis, :], 0.0, 1.0)
                syn_list.append(s_norm)
                syn_r_list.append(int(r_val))

    syn_norm  = np.array(syn_list)
    syn_r_arr = np.array(syn_r_list, dtype=np.int32)
    syn_orig  = denormalize(syn_norm, kept_names, norm_params)

    vr, vr_mean = compute_variance_ratio(real_orig, syn_orig)
    ac_diff     = compute_autocorr_similarity(real_orig, syn_orig)
    real_jump   = detect_phase_jumps(real_orig, phase_seq)
    syn_jump    = detect_phase_jumps(syn_orig,  phase_seq)
    jump_ratio  = syn_jump / (real_jump + 1e-10)

    return (real_orig, syn_orig, train_r, syn_r_arr,
            vr, vr_mean, ac_diff, jump_ratio)

# ------------------------------------------------------------------
# Visualization
# Adds a vertical line at the warmup/adversarial transition in the
# loss plot so you can immediately see whether disc loss crashes after it.
# ------------------------------------------------------------------

def plot_comparison(workload, real_orig, syn_orig,
                    real_r_arr, syn_r_arr, kept_names,
                    phase_seq, history, disc_warmup, save_dir, run_label):
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
    N_SHOW_R = 4
    if len(common_r) > N_SHOW_R:
        step = max(1, (len(common_r) - 1) // (N_SHOW_R - 1))
        selected = common_r[::step][:N_SHOW_R]
        if common_r[-1] not in selected:
            selected[-1] = common_r[-1]
        common_r = sorted(set(selected))

    palette  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    r_colour = {r: palette[i % len(palette)] for i, r in enumerate(common_r)}
    t        = np.arange(real_orig.shape[1])

    fig = plt.figure(figsize=(15, 10))
    gs  = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32)
    metric_axes    = [fig.add_subplot(gs[r, c]) for r, c in [(0,0),(0,1),(1,0),(1,1)]]
    loss_ax        = fig.add_subplot(gs[:, 2])
    legend_handles = []

    for ax_i, m_i in enumerate(plot_idx):
        ax = metric_axes[ax_i]
        for r_val in common_r:
            col       = r_colour[r_val]
            real_idxs = np.where(real_r_arr == r_val)[0]
            syn_idxs  = np.where(syn_r_arr  == r_val)[0]
            if len(real_idxs):
                ax.plot(t, real_orig[real_idxs[0], :, m_i],
                        color=col, alpha=0.85, linewidth=0.9)
            if len(syn_idxs):
                ax.plot(t, syn_orig[syn_idxs[0], :, m_i],
                        color=col, alpha=0.75, linewidth=0.9, linestyle="--")
            if ax_i == 0 and len(real_idxs):
                import matplotlib.lines as mlines
                legend_handles.append(
                    mlines.Line2D([], [], color=col, linewidth=1.2,
                                  label=f"r={r_val} solid=real dashed=syn"))
        for b in boundaries:
            ax.axvline(x=b, color="silver", linewidth=0.7, linestyle=":")
        ax.set_title(kept_names[m_i].replace("pod_", "").replace("gpu_", ""), fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("timestep", fontsize=7)

    if history:
        eps = [h["epoch"] for h in history]
        loss_ax.plot(eps, [h["loss_rec"] for h in history],
                     label="train rec", color="#1f77b4")
        loss_ax.plot(eps, [h["val_rec"]  for h in history],
                     label="val rec",   color="#1f77b4", linestyle="--")
        loss_ax.plot(eps, [h["loss_G"]   for h in history],
                     label="adv (gen)", color="#ff7f0e")
        loss_ax.plot(eps, [h["loss_D"]   for h in history],
                     label="disc loss", color="#2ca02c")
        loss_ax.plot(eps, [h["w_dist"]   for h in history],
                     label="W-dist",    color="#9467bd", linestyle=":")

        # Mark warmup/adversarial transition so the disc crash pattern is
        # easy to locate relative to the phase switch
        if disc_warmup > 0:
            loss_ax.axvline(x=disc_warmup + 0.5, color="red",
                            linewidth=1.2, linestyle="--", alpha=0.7,
                            label=f"warmup end (ep {disc_warmup})")

        loss_ax.legend(fontsize=7)
        loss_ax.set_title("Training losses", fontsize=9)
        loss_ax.set_xlabel("epoch", fontsize=7)
        loss_ax.tick_params(labelsize=7)

    fig.suptitle(
        f"{workload.upper()} - {run_label}\n"
        f"solid=real  dashed=synthetic  grey=phase boundaries",
        fontsize=10, fontweight="bold")
    if legend_handles:
        fig.legend(handles=legend_handles, loc="lower center",
                   ncol=min(len(legend_handles), 4), fontsize=7,
                   framealpha=0.7, bbox_to_anchor=(0.35, -0.01))

    save_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{workload}_{run_label.replace(' ', '_').replace('/', '_')}.png"
    path  = save_dir / fname
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()
    return path

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TimeGAN S6: discriminator warmup experiment")
    parser.add_argument(
        "--disc-warmup", type=int, default=20,
        help="Epochs to train generator alone before introducing discriminator. "
             "Recommended values: 20 (default), 30, 50. "
             "Set to 0 to reproduce S5a exactly (no warmup).")
    parser.add_argument(
        "--workloads", nargs="+", default=WORKLOADS,
        help="Subset of workloads to run (default: all five)")
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override total epoch budget (default: 200)")
    parser.add_argument(
        "--device", default="auto")
    parser.add_argument(
        "--seed", type=int, default=GEN_CFG["seed"])
    parser.add_argument(
        "--phase-boundaries", type=str,
        default=",".join(map(str, DEFAULT_PHASE_BOUNDARIES)))
    args = parser.parse_args()

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = BASE_CONFIG.copy()
    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    disc_warmup = args.disc_warmup
    boundaries  = list(map(int, args.phase_boundaries.split(",")))

    run_tag  = f"s6_w{disc_warmup}"
    out_dir  = Path(f"outputs/phase4/timegan_s6/warmup{disc_warmup}")
    model_dir = Path(f"models/phase4/timegan_s6/warmup{disc_warmup}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("TimeGAN S6 - Discriminator Warmup")
    print("=" * 65)
    print(f"Device          : {device}")
    print(f"Disc warmup     : {disc_warmup} epochs")
    print(f"Lambda (warmup) : rec={cfg['lambda_rec_warmup']}  adv={cfg['lambda_adv_warmup']}")
    print(f"Lambda (post)   : rec={cfg['lambda_rec_post']}  adv={cfg['lambda_adv_post']}")
    print(f"n_disc_steps    : {cfg['n_disc_steps']} (post-warmup)")
    print(f"GP lambda       : {cfg['lambda_gp']}")
    print(f"FM lambda       : {cfg['lambda_fm']}")
    print(f"Cond disc       : {cfg['conditional_disc']}")
    print(f"Epochs          : {cfg['epochs']}")
    print(f"Generator       : A (reconstruction-based, LSTM v6)")
    print(f"Output dir      : {out_dir}")
    print("=" * 65)

    if disc_warmup == 0:
        print("NOTE: disc_warmup=0 reproduces S5a exactly (no warmup).")
    elif disc_warmup < 20:
        print("NOTE: disc_warmup < 20 may be too short to see benefit. "
              "Recommend at least 20.")

    all_results = []

    for workload in args.workloads:
        print(f"\n  --- Workload: {workload.upper()} ---")

        data, norm    = load_raw_data(workload)
        raw_traces    = data["traces"]
        replica_counts = data["replica_counts"]
        train_idx     = data["train_idx"]
        val_idx       = data["val_idx"]
        metadata      = list(data["metadata"])
        seq_len       = raw_traces.shape[1]

        phase_seq = build_phase_sequence(seq_len, boundaries)

        exp_key_to_id  = {}
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

        norm_params = norm["params"] if "params" in norm else norm

        proc_traces, trace_means = apply_zeromean(raw_kept, clip_stds=2.5)
        zm_with_phase            = append_phase_feature(proc_traces, phase_seq)

        print(f"  Kept metrics ({n_metrics}): {kept_names}")
        print(f"  Train: {len(train_idx)}  Val: {len(val_idx)}")

        ds_train = WorkloadDataset(zm_with_phase, proc_traces, trace_means,
                                   replica_counts, experiment_ids,
                                   phase_seq, train_idx)
        ds_val   = WorkloadDataset(zm_with_phase, proc_traces, trace_means,
                                   replica_counts, experiment_ids,
                                   phase_seq, val_idx)
        dl_train = DataLoader(ds_train, batch_size=16, shuffle=True,  drop_last=True)
        dl_val   = DataLoader(ds_val,   batch_size=16, shuffle=False)

        generator     = GeneratorA(seq_len, n_metrics, GEN_CFG).to(device)
        discriminator = Discriminator(n_metrics, DISC_CFG).to(device)

        n_params_G = sum(p.numel() for p in generator.parameters())
        n_params_D = sum(p.numel() for p in discriminator.parameters())
        print(f"  Generator params    : {n_params_G:,}")
        print(f"  Discriminator params: {n_params_D:,}")

        metric_weights = compute_metric_weights(proc_traces, train_idx, device)

        t0 = time.time()
        generator, discriminator, best_val_rec, n_epochs, history = train(
            generator, discriminator, dl_train, dl_val,
            cfg, disc_warmup, GEN_CFG, device, metric_weights)
        elapsed = time.time() - t0

        (real_orig, syn_orig, real_r_arr, syn_r_arr,
         vr, vr_mean, ac_diff, jump_ratio) = evaluate(
            generator, GEN_CFG,
            proc_traces, trace_means, raw_kept,
            replica_counts, phase_seq,
            train_idx, kept_names, norm_params, 5, device)

        ref = REFERENCE_VR.get(workload, {})
        print(f"\n  Results (s6 warmup={disc_warmup}):")
        print(f"    val_rec       = {best_val_rec:.5f}")
        print(f"    var_ratio     = {vr_mean:.4f}")
        print(f"    autocorr_diff = {ac_diff:.4f}")
        print(f"    jump_ratio    = {jump_ratio:.3f}x")
        print(f"    epochs        = {n_epochs}  time = {elapsed:.1f}s")
        print(f"  Per-metric VR:")
        for j, m in enumerate(kept_names):
            flag = " <-- low" if vr[j] < 0.5 else (" <-- high" if vr[j] > 3.0 else "")
            print(f"    {m:<22} {vr[j]:.4f}{flag}")
        print(f"\n  Reference VRs for {workload}:")
        for k, v in ref.items():
            print(f"    {k:<12} {v:.4f}")

        run_label = f"s6 w{disc_warmup} genA"
        plot_path = plot_comparison(
            workload, real_orig, syn_orig, real_r_arr, syn_r_arr,
            kept_names, phase_seq, history, disc_warmup,
            out_dir / "plots", run_label)
        print(f"  Plot: {plot_path}")

        wl_model_dir = model_dir / workload
        wl_model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(generator.state_dict(),     wl_model_dir / "generator.pt")
        torch.save(discriminator.state_dict(), wl_model_dir / "discriminator.pt")
        with open(wl_model_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        with open(wl_model_dir / "config.json", "w") as f:
            json.dump({
                **{k: v for k, v in cfg.items()},
                **GEN_CFG,
                **DISC_CFG,
                "gen_option":   "a",
                "workload":     workload,
                "disc_warmup":  disc_warmup,
                "kept_metrics": kept_names,
            }, f, indent=2)

        all_results.append({
            "workload":             workload,
            "stage":                "s6",
            "gen_option":           "a",
            "disc_warmup":          disc_warmup,
            "var_ratio_mean":       vr_mean,
            "var_ratio_per_metric": vr.tolist(),
            "autocorr_diff":        ac_diff,
            "phase_jump_ratio":     jump_ratio,
            "val_rec":              best_val_rec,
            "n_epochs":             n_epochs,
            "elapsed_s":            elapsed,
            "metrics_generated":    list(kept_names),
            **{f"ref_{k}": v for k, v in ref.items()},
        })

    # Summary table
    print(f"\n{'='*65}")
    print(f"S6 SUMMARY  (disc_warmup={disc_warmup})")
    print(f"{'='*65}")
    header = f"  {'Workload':<12}  {'VR S6':>8}  {'VR S5a-A':>9}  {'LSTM':>8}  {'jump':>8}"
    print(header)
    print("  " + "-" * 52)
    vr_vals = []
    for r in all_results:
        vr_vals.append(r["var_ratio_mean"])
        s5a = r.get("ref_s5a_A", float("nan"))
        lstm = r.get("ref_lstm",  float("nan"))
        jr   = r["phase_jump_ratio"]
        print(f"  {r['workload']:<12}  {r['var_ratio_mean']:>8.4f}  "
              f"{s5a:>9.4f}  {lstm:>8.4f}  {jr:>7.3f}x")
    print("  " + "-" * 52)
    print(f"  {'MEAN':<12}  {float(np.mean(vr_vals)):>8.4f}")

    print(f"\nReference means:")
    print(f"  LSTM   mean VR = 0.612")
    print(f"  S5a-A  mean VR = 0.413  (best genA run)")
    print(f"  S4     mean VR = 0.322")

    print(f"\nDiagnostic guide:")
    print(f"  Good: disc loss does not crash to 0 within 5 epochs after warmup")
    print(f"  Good: VR S6 > 0.413 (beats S5a)")
    print(f"  Good: jump_ratio closer to 1.0 (within-phase variance recovered)")
    print(f"  Bad:  disc loss crashes same as before -> try larger --disc-warmup")

    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults: {results_path}")


if __name__ == "__main__":
    main()