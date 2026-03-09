#!/usr/bin/env python3
"""
TimeGAN S8 - Noise Injection + Data Augmentation + Fixed Epoch Budget
=======================================================================
Master Thesis - Generative AI Workload Modeling
Fachhochschule Dortmund

THREE ROOT CAUSES ADDRESSED IN S8
===================================

1. GENERATOR LOCKED IN RECONSTRUCTION MODE (FIX: encoder input dropout)
   The pod_encoder after 20 warmup epochs has memorised the phase-level
   mean of each real trace. The decoder reproduces that smooth structure.
   Adversarial signal cannot break it because reconstruction gradient
   always pulls toward smooth output.
   FIX: During adversarial training, randomly zero out 30% of encoder
   input timesteps per sample. Encoder can no longer extract a clean
   mean. Decoder must generate missing variance from latent noise.
   Applied only during forward() training. generate() bypasses encoder
   entirely (random pod_lat), so this has no effect on generation.

2. SMALL DATASET: 49 training samples (FIX: two augmentations)
   A) Phase boundary jitter: on each epoch, shift each interior phase
      boundary by Uniform(-8, +8) timesteps independently per sample.
      Same 49 traces appear with different boundary positions each epoch.
   B) Replica interpolation: linearly blend traces at consecutive r
      values (alpha=0.5) to produce free samples at intermediate r.
      Example: r=1 and r=3 -> new r=2 samples. Applied once at load.

3. EARLY STOP KILLS ADVERSARIAL TRAINING (FIX: fixed adv budget)
   Patience=25 on val_rec triggers at epoch 40-45 because val_rec
   converges during warmup and does not improve after. This gives only
   ~20 adversarial epochs.
   FIX: Warmup uses patience=15 on val_rec. Adversarial phase runs for
   a FIXED 150 epochs regardless of val_rec. Best W-dist checkpoint
   is saved during adversarial phase and used as final model.

USAGE
-----
    python timegan_s8.py                        # default: all fixes
    python timegan_s8.py --workloads bert       # single workload
    python timegan_s8.py --enc-dropout 0.0      # ablate noise injection
    python timegan_s8.py --jitter 0 --no-aug-interp  # ablate augmentation
    python timegan_s8.py --adv-epochs 100       # shorter budget
"""

import argparse
import json
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_RAW_DIR = Path("data/processed/phase4/raw")
WORKLOADS    = ["bert", "gpt2", "resnet152", "whisper", "yolo"]

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

BASE_CONFIG = {
    "adversarial":       "wgan",
    "n_disc_steps":      2,
    "lambda_rec_warmup": 1.0,
    "lambda_adv_warmup": 0.0,
    "lambda_rec_post":   0.1,
    "lambda_adv_post":   1.0,
    "lambda_fm":         0.01,
    "lambda_gp":         10.0,
    "preprocessing":     "zeromean",
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
    "hidden_dim": 32,
    "num_layers": 1,
    "dropout":    0.0,
}

REFERENCE_VR = {
    "bert":      {"lstm": 0.594, "s5a_A": 0.141, "s6_w20": 0.270, "s7": 0.240},
    "gpt2":      {"lstm": 0.676, "s5a_A": 0.500, "s6_w20": 0.568, "s7": 0.573},
    "resnet152": {"lstm": 0.607, "s5a_A": 0.113, "s6_w20": 0.239, "s7": 0.246},
    "whisper":   {"lstm": 0.748, "s5a_A": 1.187, "s6_w20": 0.751, "s7": 0.745},
    "yolo":      {"lstm": 0.438, "s5a_A": 0.123, "s6_w20": 0.262, "s7": 0.270},
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
    with open(DATA_RAW_DIR / f"{workload}_normalization.json") as f:
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


def replica_interpolation(zm_traces, zm_with_phase, trace_means,
                           replica_counts, experiment_ids, train_idx):
    """
    For each consecutive pair of replica counts in the training set,
    create interpolated samples at the midpoint r value (alpha=0.5).
    Returns extended arrays and a new train_idx including interpolated samples.
    """
    train_r  = replica_counts[train_idx]
    unique_r = np.unique(train_r)

    new_zm    = []
    new_zmph  = []
    new_means = []
    new_r     = []
    new_eids  = []
    max_eid   = experiment_ids.max() + 10000

    for i in range(len(unique_r) - 1):
        r_lo  = unique_r[i]
        r_hi  = unique_r[i + 1]
        r_mid = int(round((r_lo + r_hi) / 2.0))
        if r_mid == r_lo or r_mid == r_hi:
            continue

        idx_lo = train_idx[train_r == r_lo]
        idx_hi = train_idx[train_r == r_hi]
        n      = min(len(idx_lo), len(idx_hi))
        if n == 0:
            continue

        idx_lo = idx_lo[:n]
        idx_hi = idx_hi[:n]

        a = 0.5
        new_zm.append(a * zm_traces[idx_lo]    + (1 - a) * zm_traces[idx_hi])
        new_zmph.append(a * zm_with_phase[idx_lo] + (1 - a) * zm_with_phase[idx_hi])
        new_means.append(a * trace_means[idx_lo]  + (1 - a) * trace_means[idx_hi])
        new_r.append(np.full(n, r_mid, dtype=replica_counts.dtype))
        new_eids.append(np.arange(max_eid, max_eid + n, dtype=np.int64))
        max_eid += n

    if not new_zm:
        return zm_traces, zm_with_phase, trace_means, replica_counts, experiment_ids, train_idx

    n_orig = len(zm_traces)
    n_new  = sum(len(x) for x in new_zm)

    zm_ext   = np.concatenate([zm_traces]     + new_zm,    axis=0)
    zmph_ext = np.concatenate([zm_with_phase]  + new_zmph,  axis=0)
    m_ext    = np.concatenate([trace_means]   + new_means,  axis=0)
    r_ext    = np.concatenate([replica_counts] + new_r,     axis=0)
    e_ext    = np.concatenate([experiment_ids] + new_eids,   axis=0)
    ti_ext   = np.concatenate([train_idx,
                               np.arange(n_orig, n_orig + n_new, dtype=np.int64)])

    return zm_ext, zmph_ext, m_ext, r_ext, e_ext, ti_ext

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
# Dataset with phase boundary jitter
# ------------------------------------------------------------------

class WorkloadDataset(Dataset):
    def __init__(self, zm_with_phase, zm_traces, means,
                 replica_counts, experiment_ids, phase_seq, idx,
                 jitter=0, base_boundaries=None):
        self.inp      = torch.tensor(zm_with_phase[idx], dtype=torch.float32)
        self.target   = torch.tensor(zm_traces[idx],     dtype=torch.float32)
        self.means    = torch.tensor(means[idx],          dtype=torch.float32)
        self.exp_ids  = torch.tensor(experiment_ids[idx], dtype=torch.long)
        r = replica_counts[idx].astype(np.float32)
        self.r_norm   = torch.tensor((r - 1.0) / 9.0,    dtype=torch.float32)
        self.phase_t  = torch.tensor(phase_seq,           dtype=torch.long)
        self.jitter   = jitter
        self.seq_len  = zm_traces.shape[1]
        self.base_bnd = list(base_boundaries or DEFAULT_PHASE_BOUNDARIES)

    def __len__(self):
        return len(self.inp)

    def _jittered_phase(self):
        if self.jitter == 0:
            return self.phase_t
        bnd = [self.base_bnd[0]]
        for b in self.base_bnd[1:]:
            shift = int(np.random.randint(-self.jitter, self.jitter + 1))
            b_new = int(np.clip(b + shift, bnd[-1] + 1, self.seq_len - 1))
            bnd.append(b_new)
        return torch.tensor(
            build_phase_sequence(self.seq_len, bnd), dtype=torch.long)

    def __getitem__(self, i):
        return (self.inp[i], self.target[i], self.means[i],
                self.r_norm[i], self.exp_ids[i], self._jittered_phase())

# ------------------------------------------------------------------
# Generator with encoder input dropout
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
    Reconstruction-based generator with encoder input dropout.

    enc_dropout_p: fraction of encoder input timesteps zeroed during
                   adversarial training (enc_dropout_active=True).
    During generate(): encoder bypassed (random pod_lat), dropout irrelevant.
    """
    def __init__(self, seq_len, n_metrics, cfg, enc_dropout_p=0.3):
        super().__init__()
        self.seq_len            = seq_len
        self.n_metrics          = n_metrics
        self.regime_dim         = cfg["regime_dim"]
        self.pod_dim            = cfg["pod_dim"]
        self.enc_dropout_p      = enc_dropout_p
        self.enc_dropout_active = False  # toggled by training loop

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
        B      = x_with_phase.size(0)
        enc_in = x_with_phase

        # Timestep-level input dropout during adversarial training only
        if self.enc_dropout_active and self.enc_dropout_p > 0 and self.training:
            keep = 1.0 - self.enc_dropout_p
            mask = torch.bernoulli(
                torch.full((B, enc_in.size(1), 1), keep, device=enc_in.device))
            enc_in = enc_in * mask

        pod_lat    = self.pod_encoder(enc_in)
        regime_lat = torch.zeros(B, self.regime_dim, device=x_with_phase.device)
        for eid in exp_ids.unique():
            m = (exp_ids == eid)
            if m.sum() > 0:
                regime_lat[m] = self.regime_proj(
                    pod_lat[m].mean(dim=0, keepdim=True)).expand(m.sum(), -1)
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
# Discriminator (same as S7: h=32, l=1)
# ------------------------------------------------------------------

class Discriminator(nn.Module):
    def __init__(self, n_metrics, disc_cfg):
        super().__init__()
        hidden   = disc_cfg["hidden_dim"]
        n_layers = disc_cfg["num_layers"]
        dropout  = disc_cfg["dropout"] if n_layers > 1 else 0.0
        self.lstm = nn.LSTM(n_metrics, hidden, n_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, max(hidden, 16)),
            nn.LeakyReLU(0.2),
            nn.Linear(max(hidden, 16), 1))

    def forward(self, x, r_norm=None):
        _, (h_n, _) = self.lstm(x)
        return self.classifier(torch.cat([h_n[-2], h_n[-1]], dim=1))

    def get_features(self, x, r_norm=None):
        _, (h_n, _) = self.lstm(x)
        return torch.cat([h_n[-2], h_n[-1]], dim=1)


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
        create_graph=True)[0]
    grads_norm = grads.reshape(B, -1).norm(2, dim=1)
    return lam * ((grads_norm - 1) ** 2).mean()

# ------------------------------------------------------------------
# Training loop
# Key changes vs S7:
#   1. Two-phase stopping: warmup=patience-based, adv=fixed budget.
#   2. enc_dropout_active toggled at warmup->adv transition.
#   3. Best W-dist checkpoint saved during adv phase.
# ------------------------------------------------------------------

def train(generator, discriminator, train_dl, val_dl,
          base_cfg, disc_warmup, adv_epochs, gen_cfg,
          device, metric_weights=None):

    lam_gp  = base_cfg["lambda_gp"]
    lam_fm  = base_cfg["lambda_fm"]
    use_fm  = lam_fm > 0

    warmup_patience = 15
    min_delta       = 1e-5
    betas           = (0.0, 0.9)

    opt_G = torch.optim.Adam(generator.parameters(),
                             lr=1e-3, betas=betas, weight_decay=1e-5)
    opt_D = torch.optim.Adam(discriminator.parameters(),
                             lr=2e-4, betas=betas, weight_decay=1e-5)
    sched_G = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_G, factor=0.5, patience=7)
    mse_fn  = nn.MSELoss(reduction="none")

    best_val_rec    = float("inf")
    best_wdist      = float("-inf")
    best_G_rec      = {k: v.cpu().clone() for k, v in generator.state_dict().items()}
    best_G_wdist    = {k: v.cpu().clone() for k, v in generator.state_dict().items()}
    best_D          = {k: v.cpu().clone() for k, v in discriminator.state_dict().items()}
    patience_c      = 0
    history         = []
    warmup_done     = False
    actual_warmup   = disc_warmup  # may be shortened by early patience

    total_epochs = disc_warmup + adv_epochs

    print(f"    Warmup: {disc_warmup} epochs (patience={warmup_patience})")
    print(f"    Adv:    {adv_epochs} fixed epochs, total budget={total_epochs}")

    epoch = 0
    for epoch in range(1, total_epochs + 1):

        in_warmup    = (epoch <= actual_warmup)
        lam_rec      = base_cfg["lambda_rec_warmup"] if in_warmup else base_cfg["lambda_rec_post"]
        lam_adv      = base_cfg["lambda_adv_warmup"] if in_warmup else base_cfg["lambda_adv_post"]
        n_disc_steps = 0 if in_warmup else base_cfg["n_disc_steps"]

        if not in_warmup and not warmup_done:
            warmup_done = True
            generator.enc_dropout_active = True
            print(f"    [epoch {epoch}] Warmup complete. "
                  f"Adversarial training + enc dropout (p={generator.enc_dropout_p}).")

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

            if n_disc_steps > 0:
                with torch.no_grad():
                    fake_d = generator(inp, r_norm, exp_ids, ph_batch)
                for _ in range(n_disc_steps):
                    opt_D.zero_grad()
                    sr  = discriminator(target)
                    sf  = discriminator(fake_d)
                    wd  = sr.mean() - sf.mean()
                    ld  = -wd + gradient_penalty(discriminator, target, fake_d, device, lam_gp)
                    ld.backward()
                    opt_D.step()
                ep_loss_d += ld.item()
                ep_wdist  += wd.item()

            opt_G.zero_grad()
            fake    = generator(inp, r_norm, exp_ids, ph_batch)
            raw_mse = mse_fn(fake, target)
            if metric_weights is not None:
                raw_mse = raw_mse * metric_weights.unsqueeze(0).unsqueeze(0)
            loss_rec = raw_mse.mean()

            loss_adv = (-discriminator(fake).mean()
                        if lam_adv > 0
                        else torch.tensor(0.0, device=device))
            loss_G   = lam_rec * loss_rec + lam_adv * loss_adv

            if use_fm and not in_warmup:
                fr    = discriminator.get_features(target).detach()
                ff    = discriminator.get_features(fake)
                loss_G = loss_G + lam_fm * F.mse_loss(ff.mean(0), fr.mean(0))

            loss_G.backward()
            nn.utils.clip_grad_norm_(generator.parameters(), gen_cfg["grad_clip"])
            opt_G.step()

            ep_rec    += loss_rec.item()
            ep_adv_g  += loss_adv.item() if hasattr(loss_adv, "item") else 0.0
            n_batches += 1

        # Validation
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
                fake     = generator(inp, r_norm, exp_ids, ph_batch)
                raw_mse  = mse_fn(fake, target)
                if metric_weights is not None:
                    raw_mse = raw_mse * metric_weights.unsqueeze(0).unsqueeze(0)
                val_recs.append(raw_mse.mean().item())

        val_rec   = float(np.mean(val_recs))
        avg_wdist = ep_wdist / n_batches if n_batches else 0.0
        sched_G.step(val_rec)

        phase_label = "warmup" if in_warmup else "adv"
        h = {
            "epoch": epoch, "phase": phase_label,
            "loss_G": float(ep_adv_g / n_batches),
            "loss_D": float(ep_loss_d / n_batches),
            "loss_rec": float(ep_rec / n_batches),
            "w_dist": float(avg_wdist),
            "val_rec": val_rec,
        }
        history.append(h)

        if epoch % 10 == 0 or epoch == 1 or epoch == actual_warmup + 1:
            print(f"    ep {epoch:>4} [{phase_label:>6}]  val_rec={val_rec:.5f}  "
                  f"G={h['loss_G']:.4f}  D={h['loss_D']:.4f}  "
                  f"rec={h['loss_rec']:.4f}  W={avg_wdist:.4f}")

        # Warmup: patience-based checkpointing on val_rec
        if in_warmup:
            if val_rec < best_val_rec - min_delta:
                best_val_rec = val_rec
                best_G_rec   = {k: v.cpu().clone() for k, v in generator.state_dict().items()}
                best_D       = {k: v.cpu().clone() for k, v in discriminator.state_dict().items()}
                patience_c   = 0
            else:
                patience_c += 1
                if patience_c >= warmup_patience:
                    print(f"    [warmup] Patience at epoch {epoch}. "
                          f"Starting adversarial phase early.")
                    actual_warmup = epoch
                    warmup_done   = True
                    generator.enc_dropout_active = True
                    print(f"    Enc dropout activated (p={generator.enc_dropout_p}).")
                    patience_c = 0
                    # Adjust total budget to keep adv_epochs fixed
                    total_epochs = epoch + adv_epochs

        # Adversarial: save best W-dist
        else:
            if avg_wdist > best_wdist:
                best_wdist   = avg_wdist
                best_G_wdist = {k: v.cpu().clone() for k, v in generator.state_dict().items()}

    # Final model selection
    if warmup_done and best_wdist > float("-inf"):
        generator.load_state_dict(best_G_wdist)
        print(f"    Final: best W-dist checkpoint (W={best_wdist:.4f})")
    else:
        generator.load_state_dict(best_G_rec)
        print(f"    Final: best val_rec checkpoint (rec={best_val_rec:.5f})")
    discriminator.load_state_dict(best_D)
    return generator, discriminator, best_val_rec, epoch, history

# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate(generator, gen_cfg, zm_traces, trace_means, raw_kept,
             replica_counts, phase_seq, train_idx_orig,
             kept_names, norm_params, n_gen, device):
    real_norm   = raw_kept[train_idx_orig]
    real_orig   = denormalize(real_norm, kept_names, norm_params)
    train_means = trace_means[train_idx_orig]
    train_r     = replica_counts[train_idx_orig]
    unique_r    = np.unique(train_r)
    r_mean_map  = {int(rv): train_means[train_r == rv].mean(axis=0) for rv in unique_r}

    syn_list = []
    syn_r    = []
    for r_val in unique_r:
        r_norm_val = float((r_val - 1.0) / 9.0)
        r_mean     = r_mean_map[int(r_val)]
        for _ in range(n_gen):
            pkg = generator.generate(r_norm_val, int(r_val), phase_seq, device)
            for pod_trace in pkg:
                syn_list.append(np.clip(pod_trace + r_mean[np.newaxis, :], 0.0, 1.0))
                syn_r.append(int(r_val))

    syn_norm  = np.array(syn_list)
    syn_r_arr = np.array(syn_r, dtype=np.int32)
    syn_orig  = denormalize(syn_norm, kept_names, norm_params)

    vr, vr_mean = compute_variance_ratio(real_orig, syn_orig)
    ac_diff     = compute_autocorr_similarity(real_orig, syn_orig)
    rj          = detect_phase_jumps(real_orig, phase_seq)
    sj          = detect_phase_jumps(syn_orig,  phase_seq)
    jump_ratio  = sj / (rj + 1e-10)

    return (real_orig, syn_orig, replica_counts[train_idx_orig],
            syn_r_arr, vr, vr_mean, ac_diff, jump_ratio)

# ------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------

def plot_comparison(workload, real_orig, syn_orig,
                    real_r_arr, syn_r_arr, kept_names,
                    phase_seq, history, disc_warmup, adv_epochs,
                    save_dir, run_label):
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
    if len(common_r) > 4:
        step     = max(1, (len(common_r) - 1) // 3)
        selected = common_r[::step][:4]
        if common_r[-1] not in selected:
            selected[-1] = common_r[-1]
        common_r = sorted(set(selected))

    palette  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    r_colour = {r: palette[i % 4] for i, r in enumerate(common_r)}
    t        = np.arange(real_orig.shape[1])

    fig = plt.figure(figsize=(15, 10))
    gs  = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32)
    max_axes   = [fig.add_subplot(gs[r, c]) for r, c in [(0,0),(0,1),(1,0),(1,1)]]
    loss_ax    = fig.add_subplot(gs[:, 2])
    leg_handles = []

    for ax_i, m_i in enumerate(plot_idx):
        ax = max_axes[ax_i]
        for r_val in common_r:
            col  = r_colour[r_val]
            ri   = np.where(real_r_arr == r_val)[0]
            si   = np.where(syn_r_arr  == r_val)[0]
            if len(ri):
                ax.plot(t, real_orig[ri[0], :, m_i], color=col, alpha=0.85, linewidth=0.9)
            if len(si):
                ax.plot(t, syn_orig[si[0], :, m_i],
                        color=col, alpha=0.75, linewidth=0.9, linestyle="--")
            if ax_i == 0 and len(ri):
                import matplotlib.lines as mlines
                leg_handles.append(mlines.Line2D(
                    [], [], color=col, linewidth=1.2,
                    label=f"r={r_val} solid=real dashed=syn"))
        for b in boundaries:
            ax.axvline(x=b, color="silver", linewidth=0.7, linestyle=":")
        ax.set_title(kept_names[m_i].replace("pod_","").replace("gpu_",""), fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("timestep", fontsize=7)

    if history:
        eps = [h["epoch"] for h in history]
        loss_ax.plot(eps, [h["loss_rec"] for h in history], label="train rec", color="#1f77b4")
        loss_ax.plot(eps, [h["val_rec"]  for h in history], label="val rec",   color="#1f77b4", linestyle="--")
        loss_ax.plot(eps, [h["loss_G"]   for h in history], label="adv (gen)", color="#ff7f0e")
        loss_ax.plot(eps, [h["loss_D"]   for h in history], label="disc loss", color="#2ca02c")
        loss_ax.plot(eps, [h["w_dist"]   for h in history], label="W-dist",    color="#9467bd", linestyle=":")
        if disc_warmup > 0:
            loss_ax.axvline(x=disc_warmup + 0.5, color="red", linewidth=1.2,
                            linestyle="--", alpha=0.7, label=f"warmup end (ep {disc_warmup})")
        loss_ax.legend(fontsize=7)
        loss_ax.set_title(f"Training losses\nwarmup={disc_warmup} adv={adv_epochs}", fontsize=9)
        loss_ax.set_xlabel("epoch", fontsize=7)
        loss_ax.tick_params(labelsize=7)

    fig.suptitle(f"{workload.upper()} - {run_label}\n"
                 f"solid=real  dashed=synthetic  grey=phase boundaries",
                 fontsize=10, fontweight="bold")
    if leg_handles:
        fig.legend(handles=leg_handles, loc="lower center",
                   ncol=min(len(leg_handles), 4), fontsize=7,
                   framealpha=0.7, bbox_to_anchor=(0.35, -0.01))

    save_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{workload}_{run_label.replace(' ','_').replace('/','_')}.png"
    path  = save_dir / fname
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()
    return path

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TimeGAN S8")
    parser.add_argument("--enc-dropout",  type=float, default=0.3)
    parser.add_argument("--adv-epochs",   type=int,   default=150)
    parser.add_argument("--disc-warmup",  type=int,   default=20)
    parser.add_argument("--jitter",       type=int,   default=8)
    parser.add_argument("--aug-interp",   action="store_true",  default=True)
    parser.add_argument("--no-aug-interp",dest="aug_interp", action="store_false")
    parser.add_argument("--workloads",    nargs="+",  default=WORKLOADS)
    parser.add_argument("--device",       default="auto")
    parser.add_argument("--seed",         type=int,   default=GEN_CFG["seed"])
    parser.add_argument("--phase-boundaries", type=str,
                        default=",".join(map(str, DEFAULT_PHASE_BOUNDARIES)))
    args = parser.parse_args()

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg        = BASE_CONFIG.copy()
    boundaries = list(map(int, args.phase_boundaries.split(",")))
    interp_tag = "" if args.aug_interp else "_noint"
    run_tag    = (f"ed{int(args.enc_dropout*10):02d}"
                  f"_j{args.jitter}"
                  f"_ae{args.adv_epochs}{interp_tag}")
    out_dir   = Path(f"outputs/phase4/timegan_s8/{run_tag}")
    model_dir = Path(f"models/phase4/timegan_s8/{run_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("TimeGAN S8 - Noise Injection + Augmentation + Fixed Budget")
    print("=" * 65)
    print(f"Device          : {device}")
    print(f"Enc dropout     : {args.enc_dropout}  (adv phase only, 0=disabled)")
    print(f"Adv epochs      : {args.adv_epochs}  (fixed budget, no early stop)")
    print(f"Disc warmup     : {args.disc_warmup}")
    print(f"Phase jitter    : {args.jitter} timesteps (0=disabled)")
    print(f"Replica interp  : {args.aug_interp}")
    print(f"Disc capacity   : h=32 l=1 (from S7)")
    print(f"Lambda (warmup) : rec=1.0  adv=0.0")
    print(f"Lambda (post)   : rec=0.1  adv=1.0")
    print(f"Total epochs    : {args.disc_warmup + args.adv_epochs}")
    print(f"Output dir      : {out_dir}")
    print("=" * 65)

    all_results = []

    for workload in args.workloads:
        print(f"\n  --- Workload: {workload.upper()} ---")

        data, norm      = load_raw_data(workload)
        raw_traces      = data["traces"]
        replica_counts  = data["replica_counts"]
        train_idx_orig  = data["train_idx"]
        val_idx         = data["val_idx"]
        metadata        = list(data["metadata"])
        seq_len         = raw_traces.shape[1]

        phase_seq = build_phase_sequence(seq_len, boundaries)

        exp_key_map    = {}
        experiment_ids = np.zeros(len(metadata), dtype=np.int64)
        for i, meta in enumerate(metadata):
            key = (meta.get("experiment_id") if isinstance(meta, dict)
                   else f"{workload}_{replica_counts[i]}")
            if key not in exp_key_map:
                exp_key_map[key] = len(exp_key_map)
            experiment_ids[i] = exp_key_map[key]

        kept_idx, kept_names = get_kept_metrics(workload)
        n_metrics   = len(kept_idx)
        raw_kept    = raw_traces[:, :, kept_idx]
        norm_params = norm["params"] if "params" in norm else norm

        proc_traces, trace_means = apply_zeromean(raw_kept, clip_stds=2.5)
        zm_with_phase            = append_phase_feature(proc_traces, phase_seq)

        # Replica interpolation
        if args.aug_interp:
            (proc_aug, zmph_aug, means_aug,
             r_aug, eids_aug, ti_aug) = replica_interpolation(
                proc_traces, zm_with_phase, trace_means,
                replica_counts, experiment_ids, train_idx_orig)
            n_interp = len(ti_aug) - len(train_idx_orig)
        else:
            proc_aug = proc_traces
            zmph_aug = zm_with_phase
            means_aug = trace_means
            r_aug     = replica_counts
            eids_aug  = experiment_ids
            ti_aug    = train_idx_orig
            n_interp  = 0

        print(f"  Kept metrics ({n_metrics}): {kept_names}")
        print(f"  Train: {len(train_idx_orig)} orig + {n_interp} interp = {len(ti_aug)} total")
        print(f"  Val:   {len(val_idx)}")

        ds_train = WorkloadDataset(
            zmph_aug, proc_aug, means_aug, r_aug, eids_aug,
            phase_seq, ti_aug, jitter=args.jitter, base_boundaries=boundaries)
        ds_val = WorkloadDataset(
            zm_with_phase, proc_traces, trace_means,
            replica_counts, experiment_ids,
            phase_seq, val_idx, jitter=0)
        dl_train = DataLoader(ds_train, batch_size=16, shuffle=True,  drop_last=True)
        dl_val   = DataLoader(ds_val,   batch_size=16, shuffle=False)

        generator     = GeneratorA(seq_len, n_metrics, GEN_CFG,
                                   enc_dropout_p=args.enc_dropout).to(device)
        discriminator = Discriminator(n_metrics, DISC_CFG).to(device)

        n_params_G = sum(p.numel() for p in generator.parameters())
        n_params_D = sum(p.numel() for p in discriminator.parameters())
        print(f"  Generator params    : {n_params_G:,}")
        print(f"  Discriminator params: {n_params_D:,}")

        metric_weights = compute_metric_weights(proc_traces, train_idx_orig, device)

        t0 = time.time()
        generator, discriminator, best_val_rec, n_epochs, history = train(
            generator, discriminator, dl_train, dl_val,
            cfg, args.disc_warmup, args.adv_epochs, GEN_CFG,
            device, metric_weights)
        elapsed = time.time() - t0

        (real_orig, syn_orig, real_r_arr, syn_r_arr,
         vr, vr_mean, ac_diff, jump_ratio) = evaluate(
            generator, GEN_CFG, proc_traces, trace_means, raw_kept,
            replica_counts, phase_seq, train_idx_orig,
            kept_names, norm_params, 5, device)

        ref = REFERENCE_VR.get(workload, {})
        print(f"\n  Results (s8 {run_tag}):")
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

        run_label = f"s8 {run_tag} genA"
        plot_path = plot_comparison(
            workload, real_orig, syn_orig, real_r_arr, syn_r_arr,
            kept_names, phase_seq, history,
            args.disc_warmup, args.adv_epochs,
            out_dir / "plots", run_label)
        print(f"  Plot: {plot_path}")

        wl_dir = model_dir / workload
        wl_dir.mkdir(parents=True, exist_ok=True)
        torch.save(generator.state_dict(),     wl_dir / "generator.pt")
        torch.save(discriminator.state_dict(), wl_dir / "discriminator.pt")
        with open(wl_dir / "history.json", "w") as fh:
            json.dump(history, fh, indent=2)
        with open(wl_dir / "config.json", "w") as fh:
            json.dump({
                **cfg, **GEN_CFG,
                "disc_hidden": DISC_CFG["hidden_dim"],
                "disc_layers": DISC_CFG["num_layers"],
                "disc_warmup": args.disc_warmup,
                "adv_epochs":  args.adv_epochs,
                "enc_dropout": args.enc_dropout,
                "jitter":      args.jitter,
                "aug_interp":  args.aug_interp,
                "workload":    workload,
                "kept_metrics": kept_names,
                "n_params_G":  n_params_G,
                "n_params_D":  n_params_D,
                "n_train_orig": len(train_idx_orig),
                "n_train_aug":  len(ti_aug),
            }, fh, indent=2)

        all_results.append({
            "workload": workload, "stage": "s8", "run_tag": run_tag,
            "enc_dropout": args.enc_dropout, "adv_epochs": args.adv_epochs,
            "disc_warmup": args.disc_warmup, "jitter": args.jitter,
            "aug_interp": args.aug_interp,
            "n_train_orig": len(train_idx_orig), "n_train_aug": len(ti_aug),
            "var_ratio_mean": vr_mean, "var_ratio_per_metric": vr.tolist(),
            "autocorr_diff": ac_diff, "phase_jump_ratio": jump_ratio,
            "val_rec": best_val_rec, "n_epochs": n_epochs, "elapsed_s": elapsed,
            "metrics_generated": list(kept_names),
            **{f"ref_{k}": v for k, v in ref.items()},
        })

    print(f"\n{'='*65}")
    print(f"S8 SUMMARY  ({run_tag})")
    print(f"{'='*65}")
    print(f"  {'Workload':<12}  {'VR S8':>7}  {'VR S7':>7}  {'VR S6':>7}  {'LSTM':>7}  {'jump':>8}")
    print("  " + "-" * 58)
    vr_vals = []
    for r in all_results:
        vr_vals.append(r["var_ratio_mean"])
        print(f"  {r['workload']:<12}  {r['var_ratio_mean']:>7.4f}  "
              f"{r.get('ref_s7', float('nan')):>7.4f}  "
              f"{r.get('ref_s6_w20', float('nan')):>7.4f}  "
              f"{r.get('ref_lstm', float('nan')):>7.4f}  "
              f"{r['phase_jump_ratio']:>7.3f}x")
    print("  " + "-" * 58)
    print(f"  {'MEAN':<12}  {float(np.mean(vr_vals)):>7.4f}")

    print(f"\nReference means:  LSTM=0.612  S5a=0.413  S6=0.418  S7=0.415")
    print(f"\nDiagnostic guide:")
    print(f"  Target: mean VR > 0.50, jump_ratio < 5x on 3+ workloads")
    print(f"  If VR collapses vs S7: enc_dropout too high, try --enc-dropout 0.2")
    print(f"  If jump_ratio still 10x+: enc_dropout too low, try --enc-dropout 0.5")
    print(f"  If augmentation hurts: try --jitter 0 --no-aug-interp")

    with open(out_dir / "results.json", "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\nResults: {out_dir}/results.json")


if __name__ == "__main__":
    main()