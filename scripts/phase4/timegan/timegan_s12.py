#!/usr/bin/env python3
"""
TimeGAN S12 - Per-Replica Feature Matching + Stronger Reconstruction Anchor
============================================================================
Master Thesis - Generative AI Workload Modeling
Fachhochschule Dortmund

STAGE HISTORY (honest mean VR, excluding capped metrics)
==========================================================
LSTM baseline  0.612  reference
S7             0.415  under-generated
S8             1.678  best honest VR, but jump ratios bad (Whisper 42x)
S9 (vc3.5)     0.878  most balanced, ResNet152 collapsed (0.324)
S10            1.937  inflated - psi_cpu hits 5.0 cap on 4/5 workloads
S11            1.978  inflated + flat synthetics - honest quality regressed

S11 FAILURE ANALYSIS
======================
Three root causes identified from loss curves and plots:

1. DISCRIMINATOR DOMINANCE (flat synthetic traces)
   W-dist climbs monotonically (BERT: 0 to 22.5, ResNet152: 0 to 24.4).
   Disc keeps winning all 150 epochs. Generator's only escape is flat means.
   BERT val_rec jumped from 0.011 (warmup) to 0.38 (adversarial) = 34x
   reconstruction degradation. lambda_rec=0.1 is too weak to anchor the
   generator when the discriminator is this strong.
   FIX: Increase lambda_rec_post from 0.1 to 0.3.

2. psi_cpu STILL CAPPED AT 5.0 (evaluation inflated, loss too weak)
   Huber delta=0.5 behaves like L2 for errors below 0.5. For metrics where
   real std ~ 0.0001 in normalized space, the Huber loss gradient is tiny
   compared to the adversarial term. The var_reg loss value stays flat at
   0.025 throughout training (BERT, ResNet152, YOLO) -- it is not moving.
   FIX: Reduce Huber delta from 0.5 to 0.1 (L1 regime kicks in earlier).
        Increase lambda_var_reg from 0.5 to 1.0 (stronger pull).

3. GENERATOR OUTPUTS WRONG REPLICA LEVELS (throughput saturating at ceiling)
   Synthetic throughput for BERT r=3 is flat at 4.1 (real is 0.3). The
   evaluation adds r_mean back to zero-mean synthetic and clips to [0,1].
   When the generator outputs positive values in zero-mean space for a
   replica count where the real mean is already high, the sum saturates.
   The discriminator conditions on r_norm but the generator never received
   a direct gradient signal tying its zero-mean output to the correct
   per-replica distribution. The conditional discriminator helps in
   principle but in practice the discriminator collapses before it can
   provide this signal.
   FIX: Per-replica statistical feature matching. For each unique r_val
   in the batch, directly match fake mean and std to real mean and std.
   This is a direct, discriminator-independent gradient signal.

S12 CHANGES (three targeted fixes from S11)
=============================================

CHANGE 1: lambda_rec_post 0.1 -> 0.3
  Prevents reconstruction collapse during adversarial training.
  Generator stays anchored to real dynamics even when discriminator wins.

CHANGE 2: Huber delta 0.5 -> 0.1, lambda_var_reg 0.5 -> 1.0
  L1 regime kicks in at error > 0.1 instead of > 0.5.
  For near-zero real std metrics (psi_cpu, gpu_util in low-contention
  workloads), the Huber gradient is now meaningful and pulls the
  generator output down. Higher lambda ensures it competes with adv term.

CHANGE 3: Per-replica statistical feature matching (lambda_fm_stat=0.5)
  For each unique replica count in the batch:
      fm_mean = MSE(fake.mean([0,1]), real.mean([0,1]).detach())
      fm_std  = MSE(fake.std([0,1]),  real.std([0,1]).detach())
      loss_G += lambda_fm_stat * (fm_mean + fm_std)
  This provides a direct, discriminator-independent gradient signal that:
  - Fixes wrong replica level outputs (throughput, latency)
  - Provides correct per-replica std targets (complements var_reg)
  - Works even when the discriminator dominates

ARCHITECTURE UNCHANGED FROM S11
==================================
Generator:     GeneratorA, LSTM h=128 l=2, enc-dropout per workload
Discriminator: Conditional, biLSTM h=32 l=1 + r_norm at classifier
Warmup:        20 epochs reconstruction only (patience 15)
Adversarial:   150 fixed epochs, best W-dist checkpoint
Optimizer:     Adam lr=1e-3 (G), 2e-4 (D), betas=(0.0, 0.9), wd=1e-5

PER-WORKLOAD SETTINGS (carried from S11)
==========================================
bert:      enc_dropout=0.3,  lambda_smooth=0.1, n_disc_steps=2
gpt2:      enc_dropout=0.3,  lambda_smooth=0.1, n_disc_steps=2
resnet152: enc_dropout=0.3,  lambda_smooth=0.1, n_disc_steps=2
whisper:   enc_dropout=0.15, lambda_smooth=0.3, n_disc_steps=3
yolo:      enc_dropout=0.3,  lambda_smooth=0.1, n_disc_steps=2

HYPERPARAMETER CHANGES FROM S11
==================================
lambda_rec_post:  0.1  -> 0.3
lambda_var_reg:   0.5  -> 1.0
huber_delta:      0.5  -> 0.1
lambda_fm_stat:   -    -> 0.5  (new)

REFERENCES
===========
S7   mean VR = 0.415
S8   mean VR = 1.678  (honest, best so far, jump ratios bad)
S9   mean VR = 0.878  (honest, most balanced)
S10  mean VR = 1.937  (inflated)
S11  mean VR = 1.978  (inflated + flat synthetics)
LSTM mean VR = 0.612

USAGE
-----
    python timegan_s12.py                          # all workloads, defaults
    python timegan_s12.py --workloads bert gpt2    # subset
    python timegan_s12.py --var-reg 1.0            # variance regression strength
    python timegan_s12.py --fm-stat 0.5            # per-replica feature match strength
    python timegan_s12.py --lambda-rec 0.3         # reconstruction anchor strength

OUTPUT
------
    outputs/phase4/timegan_s12/{run_tag}/plots/
    models/phase4/timegan_s12/{run_tag}/{workload}/
    outputs/phase4/timegan_s12/{run_tag}/results.json

    run_tag = s12_vr{var_reg*10}_fm{fm_stat*10}_lr{lambda_rec*10}_ae{adv_epochs}
    e.g.      s12_vr10_fm05_lr03_ae150
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

# ------------------------------------------------------------------
# Paths and constants
# ------------------------------------------------------------------

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
    "lambda_rec_post":   0.3,    # S12: increased from 0.1 to prevent reconstruction collapse
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

# Per-workload overrides carried from S11
WORKLOAD_OVERRIDES = {
    "bert":      {"enc_dropout": 0.3,  "lambda_smooth": 0.1, "n_disc_steps": 2},
    "gpt2":      {"enc_dropout": 0.3,  "lambda_smooth": 0.1, "n_disc_steps": 2},
    "resnet152": {"enc_dropout": 0.3,  "lambda_smooth": 0.1, "n_disc_steps": 2},
    "whisper":   {"enc_dropout": 0.15, "lambda_smooth": 0.3, "n_disc_steps": 3},
    "yolo":      {"enc_dropout": 0.3,  "lambda_smooth": 0.1, "n_disc_steps": 2},
}

REFERENCE_VR = {
    "bert":      {"lstm": 0.594, "s8": 2.645, "s9": 1.213, "s10": 2.569, "s11": 1.841},
    "gpt2":      {"lstm": 0.676, "s8": 2.315, "s9": 0.664, "s10": 2.252, "s11": 2.708},
    "resnet152": {"lstm": 0.607, "s8": 1.055, "s9": 0.324, "s10": 2.017, "s11": 2.461},
    "whisper":   {"lstm": 0.748, "s8": 1.042, "s9": 0.743, "s10": 1.401, "s11": 1.592},
    "yolo":      {"lstm": 0.438, "s8": 1.334, "s9": 1.446, "s10": 1.446, "s11": 1.289},
}

REFERENCE_JUMP = {
    "bert":      {"s8":  3.9, "s9":  8.7, "s10": 0.33, "s11": 0.28},
    "gpt2":      {"s8":  3.0, "s9":  7.2, "s10": 1.98, "s11": 3.20},
    "resnet152": {"s8":  6.5, "s9": 17.1, "s10": 0.22, "s11": 0.18},
    "whisper":   {"s8": 42.1, "s9": 26.2, "s10": 0.89, "s11": 1.17},
    "yolo":      {"s8": 10.6, "s9": 11.2, "s10": 0.47, "s11": 0.77},
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


def validate_replica_counts(replica_counts, n_pods, metadata, workload):
    if len(replica_counts) == n_pods:
        return replica_counts.flatten().astype(np.int32)

    print(f"  [WARN] replica_counts has {len(replica_counts)} entries "
          f"but n_pods={n_pods}. Reconstructing from metadata.")
    rc = np.zeros(n_pods, dtype=np.int32)
    for i, meta in enumerate(metadata):
        if isinstance(meta, dict):
            rc[i] = int(meta.get("replica_count", meta.get("replicas", 1)))
        else:
            rc[i] = 1
    unique_rc = np.unique(rc)
    print(f"  [INFO] Reconstructed replica_counts. Unique values: {unique_rc}")
    return rc


def denormalize(traces_norm, metric_names, norm_params):
    lookup = norm_params["params"] if "params" in norm_params else norm_params
    out = traces_norm.copy().astype(np.float64)
    for j, m in enumerate(metric_names):
        if m not in lookup:
            continue
        lo = lookup[m].get("min", 0.0)
        hi = lookup[m].get("max", 1.0)
        out[:, :, j] = out[:, :, j] * (hi - lo) + lo
    return out


def apply_zeromean(raw_kept, clip_stds=2.5):
    means = raw_kept.mean(axis=1, keepdims=True)  # (N, 1, M)
    zm    = raw_kept - means
    if clip_stds > 0:
        std_global = zm.reshape(-1, zm.shape[-1]).std(axis=0, keepdims=True)
        zm = np.clip(zm, -clip_stds * std_global, clip_stds * std_global)
    return zm, means.squeeze(1)  # (N, T, M), (N, M)


def build_phase_sequence(seq_len, boundaries):
    phase_seq = np.zeros(seq_len, dtype=np.int64)
    for p_idx in range(len(boundaries) - 1):
        phase_seq[boundaries[p_idx]:boundaries[p_idx + 1]] = p_idx
    if boundaries[-1] < seq_len:
        phase_seq[boundaries[-1]:] = len(boundaries) - 1
    return phase_seq


def append_phase_feature(traces, phase_seq):
    N, T, M = traces.shape
    ph = phase_seq[np.newaxis, :, np.newaxis].repeat(N, axis=0).astype(np.float32)
    ph_norm = ph / max(phase_seq.max(), 1)
    return np.concatenate([traces, ph_norm], axis=2).astype(np.float32)


def compute_metric_weights(proc_traces, train_idx, device):
    train_data     = proc_traces[train_idx]
    var_per_metric = train_data.reshape(-1, train_data.shape[-1]).var(axis=0)
    var_per_metric = np.clip(var_per_metric, 1e-8, None)
    weights        = 1.0 / var_per_metric
    weights        = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ------------------------------------------------------------------
# Replica interpolation (carried unchanged from S9+)
# ------------------------------------------------------------------

def replica_interpolation(proc_traces, zmph, trace_means,
                          replica_counts, experiment_ids, train_idx):
    train_r         = replica_counts[train_idx]
    unique_r        = set(int(v) for v in np.unique(train_r))
    unique_r_sorted = sorted(unique_r)

    new_zm    = []
    new_proc  = []
    new_means = []
    new_r     = []
    new_eids  = []

    for i in range(len(unique_r_sorted) - 1):
        r_lo  = unique_r_sorted[i]
        r_hi  = unique_r_sorted[i + 1]
        r_mid = int(round((r_lo + r_hi) / 2.0))

        if r_mid in unique_r:
            continue
        if r_mid <= r_lo or r_mid >= r_hi:
            continue

        idx_lo = train_idx[train_r == r_lo]
        idx_hi = train_idx[train_r == r_hi]
        n      = min(len(idx_lo), len(idx_hi))
        if n == 0:
            continue

        alpha = 0.5
        for k in range(n):
            new_zm.append(alpha * zmph[idx_lo[k]]         + (1 - alpha) * zmph[idx_hi[k]])
            new_proc.append(alpha * proc_traces[idx_lo[k]] + (1 - alpha) * proc_traces[idx_hi[k]])
            new_means.append(alpha * trace_means[idx_lo[k]] + (1 - alpha) * trace_means[idx_hi[k]])
            new_r.append(r_mid)
            new_eids.append(-1)
        print(f"    [interp] r={r_lo}+r={r_hi} -> r={r_mid}: {n} new samples")

    if not new_zm:
        print(f"    [interp] 0 samples created. "
              f"unique_r={unique_r_sorted} -- all midpoints already covered.")
        return (proc_traces, zmph, trace_means,
                replica_counts, experiment_ids, train_idx)

    N_orig  = len(proc_traces)
    n_new   = len(new_zm)
    new_ids = np.arange(N_orig, N_orig + n_new)

    proc_aug  = np.concatenate([proc_traces, np.array(new_proc)],  axis=0)
    zmph_aug  = np.concatenate([zmph,        np.array(new_zm)],    axis=0)
    means_aug = np.concatenate([trace_means, np.array(new_means)], axis=0)
    r_aug     = np.concatenate([replica_counts, np.array(new_r)],  axis=0)
    eids_aug  = np.concatenate([experiment_ids, np.array(new_eids)], axis=0)
    ti_aug    = np.concatenate([train_idx, new_ids],                axis=0)

    return proc_aug, zmph_aug, means_aug, r_aug, eids_aug, ti_aug.astype(np.int64)


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

class WorkloadDataset(Dataset):
    def __init__(self, zmph, proc_traces, trace_means,
                 replica_counts, experiment_ids,
                 phase_seq, indices,
                 jitter=0, base_boundaries=None):
        self.zmph        = zmph
        self.proc        = proc_traces
        self.means       = trace_means
        self.rc          = replica_counts
        self.eids        = experiment_ids
        self.phase_seq   = phase_seq
        self.indices     = indices
        self.jitter      = jitter
        self.base_bounds = base_boundaries or DEFAULT_PHASE_BOUNDARIES
        self.r_max       = max(float(replica_counts.max()), 1.0)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx   = self.indices[i]
        inp   = torch.tensor(self.zmph[idx], dtype=torch.float32)
        tgt   = torch.tensor(self.proc[idx], dtype=torch.float32)
        r_val = int(self.rc[idx])   if idx < len(self.rc)   else 1
        eid   = int(self.eids[idx]) if idx < len(self.eids) else 0

        if self.jitter > 0:
            shift = np.random.randint(-self.jitter, self.jitter + 1)
            inp   = torch.roll(inp, shift, dims=0)
            tgt   = torch.roll(tgt, shift, dims=0)

        r_norm = torch.tensor((r_val - 1.0) / 9.0, dtype=torch.float32)
        exp_id = torch.tensor(eid, dtype=torch.long)
        T      = inp.shape[0]
        ph     = torch.tensor(self.phase_seq[:T], dtype=torch.long)

        return inp, tgt, torch.tensor(r_val), r_norm, exp_id, ph


# ------------------------------------------------------------------
# Evaluation metrics
# ------------------------------------------------------------------

def compute_variance_ratio(real, synthetic, cap=5.0):
    var_r = real.reshape(-1, real.shape[-1]).var(axis=0)
    var_s = synthetic.reshape(-1, synthetic.shape[-1]).var(axis=0)
    ratio = np.where(var_r > 1e-10,
                     np.clip(var_s / var_r, 0, cap),
                     np.ones_like(var_r))
    return ratio, float(ratio.mean())


def compute_autocorr_similarity(real, synthetic, max_lag=20):
    def acf(x):
        x   = x - x.mean(axis=(0, 1), keepdims=True)
        out = []
        for lag in range(1, max_lag + 1):
            c  = (x[:, lag:, :] * x[:, :-lag, :]).mean()
            v  = (x ** 2).mean()
            out.append(float(c / (v + 1e-10)))
        return np.array(out)
    return float(np.abs(acf(real) - acf(synthetic)).mean())


def detect_phase_jumps(traces, phase_seq):
    boundaries = []
    for t in range(1, len(phase_seq)):
        if phase_seq[t] != phase_seq[t - 1]:
            boundaries.append(t)
    if not boundaries:
        return 1.0
    jumps = []
    for t in boundaries:
        if t == 0 or t >= traces.shape[1]:
            continue
        delta = np.abs(traces[:, t, :] - traces[:, t - 1, :]).mean()
        base  = traces[:, t, :].std() + 1e-10
        jumps.append(float(delta / base))
    return float(np.mean(jumps)) if jumps else 1.0


# ------------------------------------------------------------------
# Generator (unchanged from S11)
# ------------------------------------------------------------------

class GeneratorA(nn.Module):
    def __init__(self, seq_len, n_metrics, cfg, enc_dropout_p=0.3):
        super().__init__()
        self.seq_len             = seq_len
        self.n_metrics           = n_metrics
        self.hidden_dim          = cfg["hidden_dim"]
        self.num_layers          = cfg["num_layers"]
        self.latent_dim          = cfg["latent_dim"]
        self.enc_dropout_p       = enc_dropout_p
        self.enc_dropout_active  = False

        input_dim = n_metrics + 1  # +1 for phase feature

        self.enc_rnn = nn.LSTM(input_dim, self.hidden_dim, self.num_layers,
                               batch_first=True,
                               dropout=cfg["dropout"] if self.num_layers > 1 else 0.0)
        self.enc_fc  = nn.Linear(self.hidden_dim, self.latent_dim)

        self.r_embed = nn.Sequential(
            nn.Linear(1, cfg["replica_embed_dim"]),
            nn.ReLU(),
        )
        self.ph_embed = nn.Embedding(N_PHASES + 1, cfg["phase_embed_dim"])

        dec_input = self.latent_dim + cfg["replica_embed_dim"]
        self.dec_rnn = nn.LSTM(dec_input, self.hidden_dim, self.num_layers,
                               batch_first=True,
                               dropout=cfg["dropout"] if self.num_layers > 1 else 0.0)
        self.dec_fc  = nn.Linear(self.hidden_dim + cfg["phase_embed_dim"], n_metrics)
        self.out_act = nn.Sigmoid()

    def forward(self, inp, r_norm, exp_ids, phase_ids):
        B, T, _ = inp.shape

        if self.enc_dropout_active and self.enc_dropout_p > 0 and self.training:
            mask    = (torch.rand(B, T, 1, device=inp.device) > self.enc_dropout_p).float()
            enc_inp = inp * mask
        else:
            enc_inp = inp

        _, (h, c)   = self.enc_rnn(enc_inp)
        z           = self.enc_fc(h[-1])
        r_emb       = self.r_embed(r_norm.unsqueeze(-1))
        dec_in_step = torch.cat([z, r_emb], dim=-1).unsqueeze(1).expand(-1, T, -1)
        dec_out, _  = self.dec_rnn(dec_in_step)
        ph_emb      = self.ph_embed(phase_ids)
        out_in      = torch.cat([dec_out, ph_emb], dim=-1)
        return self.out_act(self.dec_fc(out_in))

    @torch.no_grad()
    def generate(self, r_norm_val, r_int_val, phase_seq, device):
        self.eval()
        T  = self.seq_len
        B  = 1
        z  = torch.randn(B, self.latent_dim, device=device)

        r_norm = torch.tensor([[r_norm_val]], dtype=torch.float32, device=device)
        r_emb  = self.r_embed(r_norm)
        dec_in = torch.cat([z, r_emb], dim=-1).unsqueeze(1).expand(-1, T, -1)

        dummy_h = (torch.zeros(self.num_layers, B, self.hidden_dim, device=device),
                   torch.zeros(self.num_layers, B, self.hidden_dim, device=device))
        dec_out, _ = self.dec_rnn(dec_in, dummy_h)

        ph_ids = torch.tensor(phase_seq[:T], dtype=torch.long, device=device).unsqueeze(0)
        ph_emb = self.ph_embed(ph_ids)
        out    = self.out_act(self.dec_fc(torch.cat([dec_out, ph_emb], dim=-1)))
        return out.squeeze(0).cpu().numpy()[np.newaxis]  # (1, T, M)


# ------------------------------------------------------------------
# Discriminator (conditional on r_norm, unchanged from S11)
# ------------------------------------------------------------------

class Discriminator(nn.Module):
    """
    Bidirectional LSTM + conditional classifier.
    r_norm injected at classifier input (hidden*2+1).
    get_features returns LSTM-only output for feature matching.
    """
    def __init__(self, n_metrics, cfg):
        super().__init__()
        hidden   = cfg["hidden_dim"]
        n_layers = cfg["num_layers"]
        dropout  = cfg["dropout"] if n_layers > 1 else 0.0
        self.lstm = nn.LSTM(n_metrics, hidden, n_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2 + 1, max(hidden, 16)),
            nn.LeakyReLU(0.2),
            nn.Linear(max(hidden, 16), 1))

    def forward(self, x, r_norm=None):
        _, (h_n, _) = self.lstm(x)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        if r_norm is not None:
            h = torch.cat([h, r_norm.unsqueeze(1)], dim=1)
        else:
            h = torch.cat([h, torch.zeros(h.shape[0], 1, device=h.device)], dim=1)
        return self.classifier(h)

    def get_features(self, x, r_norm=None):
        _, (h_n, _) = self.lstm(x)
        return torch.cat([h_n[-2], h_n[-1]], dim=1)


def gradient_penalty(disc, real, fake, r_norm, device, lam=10.0):
    B   = real.shape[0]
    eps = torch.rand(B, 1, 1, device=device)
    mid = (eps * real + (1 - eps) * fake).requires_grad_(True)
    with torch.backends.cudnn.flags(enabled=False):
        d = disc(mid, r_norm)
    grad = torch.autograd.grad(d.sum(), mid, create_graph=True)[0]
    return lam * ((grad.norm(2, dim=(1, 2)) - 1) ** 2).mean()


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train(generator, discriminator, train_dl, val_dl,
          base_cfg, disc_warmup, adv_epochs, gen_cfg,
          lambda_var_reg, huber_delta,
          lambda_fm_stat,
          lambda_smooth, n_disc_steps_wl,
          device, metric_weights=None,
          warmup_patience=15, min_delta=1e-5):
    """
    S12 training loop.

    Changes from S11:
    1. lambda_rec_post increased (passed in via base_cfg["lambda_rec_post"]).
    2. Variance regression uses smaller Huber delta and higher lambda.
    3. Per-replica statistical feature matching added to generator loss.
       For each unique r_val in the batch:
           fm_mean = MSE(fake.mean([0,1]), real.mean([0,1]).detach())
           fm_std  = MSE(fake.std([0,1]),  real.std([0,1]).detach())
           loss_G += lambda_fm_stat * (fm_mean + fm_std)
       This is discriminator-independent and provides direct gradient signal
       for both the level error and the variance calibration per replica count.
    """
    use_fm  = base_cfg.get("lambda_fm", 0.0) > 0
    lam_fm  = base_cfg.get("lambda_fm", 0.0)
    lam_gp  = base_cfg.get("lambda_gp", 10.0)
    betas   = (0.0, 0.9)

    opt_G   = torch.optim.Adam(generator.parameters(),
                               lr=1e-3, betas=betas, weight_decay=1e-5)
    opt_D   = torch.optim.Adam(discriminator.parameters(),
                               lr=2e-4, betas=betas, weight_decay=1e-5)
    sched_G = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_G, factor=0.5, patience=7)
    mse_fn  = nn.MSELoss(reduction="none")

    best_val_rec  = float("inf")
    best_wdist    = float("-inf")
    best_G_rec    = {k: v.cpu().clone() for k, v in generator.state_dict().items()}
    best_G_wdist  = {k: v.cpu().clone() for k, v in generator.state_dict().items()}
    best_D        = {k: v.cpu().clone() for k, v in discriminator.state_dict().items()}
    patience_c    = 0
    history       = []
    warmup_done   = False
    actual_warmup = disc_warmup
    total_epochs  = disc_warmup + adv_epochs

    lam_rec_post = base_cfg["lambda_rec_post"]
    print(f"    Warmup: {disc_warmup} epochs (patience={warmup_patience})")
    print(f"    Adv:    {adv_epochs} fixed epochs  "
          f"lambda_rec={lam_rec_post}  var_reg={lambda_var_reg}  "
          f"huber_delta={huber_delta}  fm_stat={lambda_fm_stat}  "
          f"n_disc={n_disc_steps_wl}")

    for epoch in range(1, total_epochs + 1):

        in_warmup    = (epoch <= actual_warmup)
        lam_rec      = base_cfg["lambda_rec_warmup"] if in_warmup else lam_rec_post
        lam_adv      = base_cfg["lambda_adv_warmup"] if in_warmup else base_cfg["lambda_adv_post"]
        n_disc_steps = 0 if in_warmup else n_disc_steps_wl

        if not in_warmup and not warmup_done:
            warmup_done = True
            generator.enc_dropout_active = True
            print(f"    [epoch {epoch}] Adversarial start. "
                  f"enc_dropout p={generator.enc_dropout_p}  "
                  f"n_disc_steps={n_disc_steps_wl}")

        generator.train()
        discriminator.train()
        ep_rec = ep_adv_g = ep_loss_d = ep_wdist = ep_vreg = ep_sm = ep_fmstat = 0.0
        n_batches = 0

        for inp, target, r_int, r_norm, exp_ids, ph_batch in train_dl:
            inp      = inp.to(device)
            target   = target.to(device)
            r_norm   = r_norm.to(device)
            r_int    = r_int.to(device)
            exp_ids  = exp_ids.to(device)
            ph_batch = ph_batch.to(device)

            # Discriminator update
            if n_disc_steps > 0:
                with torch.no_grad():
                    fake_d = generator(inp, r_norm, exp_ids, ph_batch)
                for _ in range(n_disc_steps):
                    opt_D.zero_grad()
                    sr = discriminator(target, r_norm)
                    sf = discriminator(fake_d, r_norm)
                    wd = sr.mean() - sf.mean()
                    ld = -wd + gradient_penalty(
                             discriminator, target, fake_d, r_norm, device, lam_gp)
                    ld.backward()
                    opt_D.step()
                ep_loss_d += ld.item()
                ep_wdist  += wd.item()

            # ----------------------------------------------------------
            # Generator update - single backward pass.
            # All loss terms accumulated before backward().
            # ----------------------------------------------------------
            opt_G.zero_grad()
            fake = generator(inp, r_norm, exp_ids, ph_batch)

            # Reconstruction loss
            raw_mse  = mse_fn(fake, target)
            if metric_weights is not None:
                raw_mse = raw_mse * metric_weights.unsqueeze(0).unsqueeze(0)
            loss_rec = raw_mse.mean()

            # Adversarial loss (conditional discriminator)
            loss_adv = (-discriminator(fake, r_norm).mean()
                        if lam_adv > 0 else torch.tensor(0.0, device=device))
            loss_G   = lam_rec * loss_rec + lam_adv * loss_adv

            # LSTM feature matching (replica-agnostic, from S11)
            if use_fm and not in_warmup:
                fr     = discriminator.get_features(target).detach()
                ff     = discriminator.get_features(fake)
                loss_G = loss_G + lam_fm * F.mse_loss(ff.mean(0), fr.mean(0))

            # Variance regression: Huber loss on per-metric std.
            # delta=0.1 so L1 regime kicks in at error > 0.1 (was 0.5 in S11).
            # Provides stronger gradient for near-zero std metrics (psi_cpu).
            if lambda_var_reg > 0 and not in_warmup:
                std_real = target.std(dim=[0, 1]).detach()  # (M,)
                std_fake = fake.std(dim=[0, 1])              # (M,) differentiable
                var_reg  = F.huber_loss(std_fake, std_real, delta=huber_delta)
                loss_G   = loss_G + lambda_var_reg * var_reg
                ep_vreg += var_reg.item()

            # Per-replica statistical feature matching.
            # For each unique replica count in the batch, match fake mean and
            # std to real mean and std directly. This is discriminator-
            # independent and corrects both level errors and variance per r.
            if lambda_fm_stat > 0 and not in_warmup:
                unique_r_vals = r_int.unique()
                fm_stat_loss  = torch.tensor(0.0, device=device)
                n_r_matched   = 0
                for r_val in unique_r_vals:
                    mask = (r_int == r_val)
                    if mask.sum() < 2:
                        continue
                    real_r   = target[mask]
                    fake_r   = fake[mask]
                    fm_mean  = F.mse_loss(fake_r.mean(dim=[0, 1]),
                                          real_r.mean(dim=[0, 1]).detach())
                    fm_std   = F.mse_loss(fake_r.std(dim=[0, 1]),
                                          real_r.std(dim=[0, 1]).detach())
                    fm_stat_loss = fm_stat_loss + fm_mean + fm_std
                    n_r_matched += 1
                if n_r_matched > 0:
                    fm_stat_loss = fm_stat_loss / n_r_matched
                    loss_G       = loss_G + lambda_fm_stat * fm_stat_loss
                    ep_fmstat   += fm_stat_loss.item()

            # Temporal smoothing penalty
            if lambda_smooth > 0 and not in_warmup:
                diff   = fake[:, 1:, :] - fake[:, :-1, :]
                sm     = (diff ** 2).mean()
                loss_G = loss_G + lambda_smooth * sm
                ep_sm += sm.item()

            loss_G.backward()
            nn.utils.clip_grad_norm_(generator.parameters(),
                                     max_norm=gen_cfg["grad_clip"])
            opt_G.step()

            ep_rec   += loss_rec.item()
            ep_adv_g += loss_adv.item() if hasattr(loss_adv, "item") else 0.0
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

        phase_label = "warmup" if in_warmup else "   adv"
        h = {
            "epoch":    epoch,
            "phase":    phase_label.strip(),
            "loss_G":   float(ep_adv_g   / n_batches),
            "loss_D":   float(ep_loss_d  / n_batches),
            "loss_rec": float(ep_rec     / n_batches),
            "w_dist":   float(avg_wdist),
            "val_rec":  val_rec,
            "var_reg":  float(ep_vreg    / n_batches),
            "fm_stat":  float(ep_fmstat  / n_batches),
            "smooth":   float(ep_sm      / n_batches),
        }
        history.append(h)

        if epoch % 10 == 0 or epoch == 1 or epoch == actual_warmup + 1:
            print(f"    ep {epoch:>4} [{phase_label}]  val_rec={val_rec:.5f}  "
                  f"G={h['loss_G']:.4f}  D={h['loss_D']:.4f}  "
                  f"W={avg_wdist:.4f}  vr={h['var_reg']:.4f}  "
                  f"fm={h['fm_stat']:.4f}  sm={h['smooth']:.4f}")

        # Warmup bookkeeping
        if in_warmup:
            if val_rec < best_val_rec - min_delta:
                best_val_rec = val_rec
                best_G_rec   = {k: v.cpu().clone() for k, v in generator.state_dict().items()}
                best_D       = {k: v.cpu().clone() for k, v in discriminator.state_dict().items()}
                patience_c   = 0
            else:
                patience_c += 1
                if patience_c >= warmup_patience:
                    print(f"    [warmup] Patience at epoch {epoch}. Starting adv early.")
                    actual_warmup = epoch
                    warmup_done   = True
                    generator.enc_dropout_active = True
                    print(f"    enc_dropout activated (p={generator.enc_dropout_p})")
                    patience_c   = 0
                    total_epochs = epoch + adv_epochs
        else:
            if avg_wdist > best_wdist:
                best_wdist   = avg_wdist
                best_G_wdist = {k: v.cpu().clone() for k, v in generator.state_dict().items()}

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
# Visualisation
# ------------------------------------------------------------------

def plot_comparison(workload, real_orig, syn_orig, real_r_arr, syn_r_arr,
                    kept_names, phase_seq, history,
                    disc_warmup, adv_epochs, save_dir, run_label):
    boundary_ts = []
    for t in range(1, len(phase_seq)):
        if phase_seq[t] != phase_seq[t - 1]:
            boundary_ts.append(t)

    r_vals_plot = sorted(set(int(v) for v in np.unique(real_r_arr)))
    r_sample    = r_vals_plot[::max(1, len(r_vals_plot) // 4)][:4]
    colors      = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]

    plot_metrics = kept_names[:4]
    n_plots = len(plot_metrics) + 1
    n_cols  = 3
    n_rows  = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    leg_handles = []
    leg_labels  = []

    for pi, mname in enumerate(plot_metrics):
        ax  = axes[pi]
        col = (mname.replace("pod_", "").replace("gpu_", "")
                    .replace("_avg", "").replace("_usage", ""))
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("timestep", fontsize=8)

        for i, r_val in enumerate(r_sample):
            clr  = colors[i % len(colors)]
            ridx = np.where(real_r_arr == r_val)[0]
            sidx = np.where(syn_r_arr  == r_val)[0]
            m_idx = kept_names.index(mname)

            if len(ridx) > 0:
                h1, = ax.plot(real_orig[ridx[0], :, m_idx], color=clr,
                              alpha=0.85, linewidth=0.9)
                if pi == 0:
                    leg_handles.append(h1)
                    leg_labels.append(f"r={r_val} solid=real dashed=syn")
            if len(sidx) > 0:
                ax.plot(syn_orig[sidx[0], :, m_idx], color=clr,
                        alpha=0.85, linewidth=0.9, linestyle="--")

        for bt in boundary_ts:
            ax.axvline(bt, color="grey", alpha=0.3, linewidth=0.7)

    loss_ax = axes[len(plot_metrics)]
    eps     = [h["epoch"]   for h in history]
    loss_ax.plot(eps, [h["loss_rec"] for h in history], label="train rec",  color="#1f77b4")
    loss_ax.plot(eps, [h["val_rec"]  for h in history], label="val rec",    color="#1f77b4", linestyle="--")
    loss_ax.plot(eps, [h["loss_G"]   for h in history], label="adv (gen)",  color="#ff7f0e")
    loss_ax.plot(eps, [h["loss_D"]   for h in history], label="disc loss",  color="#2ca02c")
    loss_ax.plot(eps, [h["w_dist"]   for h in history], label="W-dist",     color="#9467bd", linestyle=":")
    loss_ax.plot(eps, [h["var_reg"]  for h in history], label="var_reg",    color="#d62728", linestyle="-.")
    loss_ax.plot(eps, [h["fm_stat"]  for h in history], label="fm_stat",    color="#e377c2", linestyle="-.")
    loss_ax.plot(eps, [h["smooth"]   for h in history], label="smooth",     color="#8c564b", linestyle=":")
    loss_ax.axvline(disc_warmup, color="red", linestyle="--", alpha=0.5,
                    label=f"warmup end (ep {disc_warmup})")
    loss_ax.set_title(f"Training losses\nwarmup={disc_warmup} adv={adv_epochs}", fontsize=9)
    loss_ax.set_xlabel("epoch", fontsize=8)
    loss_ax.legend(fontsize=6, loc="upper left")

    for ax in axes[n_plots:]:
        ax.set_visible(False)

    title = (f"{workload.upper()} - {run_label}\n"
             "solid=real  dashed=synthetic  grey=phase boundaries")
    fig.suptitle(title, fontsize=11, fontweight="bold")
    if leg_handles:
        fig.legend(handles=leg_handles, labels=leg_labels,
                   loc="lower center", ncol=min(len(leg_handles), 4),
                   fontsize=7, framealpha=0.7, bbox_to_anchor=(0.35, -0.01))

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
    parser = argparse.ArgumentParser(description="TimeGAN S12")
    parser.add_argument("--var-reg",     type=float, default=1.0,
                        help="Lambda for variance regression Huber loss (default 1.0)")
    parser.add_argument("--huber-delta", type=float, default=0.1,
                        help="Huber loss delta for variance regression (default 0.1)")
    parser.add_argument("--fm-stat",     type=float, default=0.5,
                        help="Lambda for per-replica statistical feature matching (default 0.5)")
    parser.add_argument("--lambda-rec",  type=float, default=0.3,
                        help="Lambda for reconstruction in adversarial phase (default 0.3)")
    parser.add_argument("--smooth",      type=float, default=None,
                        help="Lambda for smoothing (None = per-workload defaults)")
    parser.add_argument("--enc-dropout", type=float, default=None,
                        help="Override enc_dropout for all workloads")
    parser.add_argument("--adv-epochs",  type=int,   default=150)
    parser.add_argument("--disc-warmup", type=int,   default=20)
    parser.add_argument("--jitter",      type=int,   default=8)
    parser.add_argument("--aug-interp",  action="store_true",  default=True)
    parser.add_argument("--no-aug-interp", dest="aug_interp",  action="store_false")
    parser.add_argument("--workloads",   nargs="+",  default=WORKLOADS)
    parser.add_argument("--device",      default="auto")
    parser.add_argument("--seed",        type=int,   default=GEN_CFG["seed"])
    parser.add_argument("--phase-boundaries", type=str,
                        default=",".join(map(str, DEFAULT_PHASE_BOUNDARIES)))
    args = parser.parse_args()

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = BASE_CONFIG.copy()
    cfg["lambda_rec_post"] = args.lambda_rec  # override from CLI

    boundaries = list(map(int, args.phase_boundaries.split(",")))
    interp_tag = "" if args.aug_interp else "_ni"
    run_tag    = (f"s12_vr{int(args.var_reg * 10):02d}"
                  f"_fm{int(args.fm_stat   * 10):02d}"
                  f"_lr{int(args.lambda_rec * 10):02d}"
                  f"_ae{args.adv_epochs}{interp_tag}")

    out_dir   = Path(f"outputs/phase4/timegan_s12/{run_tag}")
    model_dir = Path(f"models/phase4/timegan_s12/{run_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TimeGAN S12 - Per-Replica FM + Stronger Reconstruction Anchor")
    print("=" * 70)
    print(f"Device          : {device}")
    print(f"lambda_rec_post : {args.lambda_rec}  (was 0.1 in S11)")
    print(f"Var regression  : lambda={args.var_reg}  Huber delta={args.huber_delta}  "
          f"(was 0.5/0.5 in S11)")
    print(f"Per-replica FM  : lambda_fm_stat={args.fm_stat}  (new in S12)")
    print(f"Smooth          : {'per-workload defaults' if args.smooth is None else args.smooth}")
    print(f"Enc dropout     : {'per-workload defaults' if args.enc_dropout is None else args.enc_dropout}")
    print(f"Adv epochs      : {args.adv_epochs}")
    print(f"Disc warmup     : {args.disc_warmup}")
    print(f"Phase jitter    : {args.jitter}")
    print(f"Replica interp  : {args.aug_interp}")
    print(f"Output dir      : {out_dir}")
    print("=" * 70)

    all_results = []

    for workload in args.workloads:
        wl_overrides    = WORKLOAD_OVERRIDES[workload]
        enc_dropout_p   = (args.enc_dropout if args.enc_dropout is not None
                           else wl_overrides["enc_dropout"])
        lambda_smooth   = (args.smooth      if args.smooth      is not None
                           else wl_overrides["lambda_smooth"])
        n_disc_steps_wl = wl_overrides["n_disc_steps"]

        print(f"\n  --- Workload: {workload.upper()} "
              f"(enc_dropout={enc_dropout_p}  smooth={lambda_smooth}"
              f"  n_disc={n_disc_steps_wl}) ---")

        data, norm     = load_raw_data(workload)
        raw_traces     = data["traces"]
        replica_counts = data["replica_counts"]
        train_idx_orig = data["train_idx"]
        val_idx        = data["val_idx"]
        metadata       = list(data["metadata"])
        seq_len        = raw_traces.shape[1]
        n_pods         = len(raw_traces)

        replica_counts = validate_replica_counts(
            replica_counts, n_pods, metadata, workload)

        phase_seq = build_phase_sequence(seq_len, boundaries)

        exp_key_map    = {}
        experiment_ids = np.zeros(n_pods, dtype=np.int64)
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

        if args.aug_interp:
            (proc_aug, zmph_aug, means_aug,
             r_aug, eids_aug, ti_aug) = replica_interpolation(
                proc_traces, zm_with_phase, trace_means,
                replica_counts, experiment_ids, train_idx_orig)
            n_interp = len(ti_aug) - len(train_idx_orig)
        else:
            proc_aug  = proc_traces
            zmph_aug  = zm_with_phase
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
                                   enc_dropout_p=enc_dropout_p).to(device)
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
            args.var_reg, args.huber_delta,
            args.fm_stat,
            lambda_smooth, n_disc_steps_wl,
            device, metric_weights)
        elapsed = time.time() - t0

        (real_orig, syn_orig, real_r_arr, syn_r_arr,
         vr, vr_mean, ac_diff, jump_ratio) = evaluate(
            generator, GEN_CFG, proc_traces, trace_means, raw_kept,
            replica_counts, phase_seq, train_idx_orig,
            kept_names, norm_params, 5, device)

        ref_vr   = REFERENCE_VR.get(workload, {})
        ref_jump = REFERENCE_JUMP.get(workload, {})
        print(f"\n  Results (s12 {run_tag}):")
        print(f"    val_rec       = {best_val_rec:.5f}")
        print(f"    var_ratio     = {vr_mean:.4f}  "
              f"(s11={ref_vr.get('s11','?'):.4f}  s9={ref_vr.get('s9','?'):.4f}  "
              f"lstm={ref_vr.get('lstm','?'):.4f})")
        print(f"    autocorr_diff = {ac_diff:.4f}")
        print(f"    jump_ratio    = {jump_ratio:.3f}x  "
              f"(s11={ref_jump.get('s11','?'):.3f}x  s9={ref_jump.get('s9','?'):.1f}x)")
        print(f"    epochs        = {n_epochs}  time = {elapsed:.1f}s")
        print(f"  Per-metric VR:")
        for j, m in enumerate(kept_names):
            flag = " <-- low"  if vr[j] < 0.5 else (
                   " <-- HIGH" if vr[j] > 3.0 else "")
            print(f"    {m:<22} {vr[j]:.4f}{flag}")

        run_label = f"s12 {run_tag} genA"
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
                "disc_hidden":     DISC_CFG["hidden_dim"],
                "disc_layers":     DISC_CFG["num_layers"],
                "disc_warmup":     args.disc_warmup,
                "adv_epochs":      args.adv_epochs,
                "enc_dropout":     enc_dropout_p,
                "lambda_smooth":   lambda_smooth,
                "lambda_var_reg":  args.var_reg,
                "huber_delta":     args.huber_delta,
                "lambda_fm_stat":  args.fm_stat,
                "lambda_rec_post": args.lambda_rec,
                "n_disc_steps_wl": n_disc_steps_wl,
                "jitter":          args.jitter,
                "aug_interp":      args.aug_interp,
                "workload":        workload,
                "kept_metrics":    kept_names,
                "n_params_G":      n_params_G,
                "n_params_D":      n_params_D,
                "n_train_orig":    len(train_idx_orig),
                "n_train_aug":     len(ti_aug),
            }, fh, indent=2)

        all_results.append({
            "workload":        workload,
            "stage":           "s12",
            "run_tag":         run_tag,
            "enc_dropout":     enc_dropout_p,
            "adv_epochs":      args.adv_epochs,
            "disc_warmup":     args.disc_warmup,
            "jitter":          args.jitter,
            "aug_interp":      args.aug_interp,
            "lambda_var_reg":  args.var_reg,
            "huber_delta":     args.huber_delta,
            "lambda_fm_stat":  args.fm_stat,
            "lambda_rec_post": args.lambda_rec,
            "lambda_smooth":   lambda_smooth,
            "n_disc_steps_wl": n_disc_steps_wl,
            "n_train_orig":    len(train_idx_orig),
            "n_train_aug":     len(ti_aug),
            "var_ratio_mean":       vr_mean,
            "var_ratio_per_metric": vr.tolist(),
            "autocorr_diff":        ac_diff,
            "phase_jump_ratio":     jump_ratio,
            "val_rec":              best_val_rec,
            "n_epochs":             n_epochs,
            "elapsed_s":            elapsed,
            "metrics_generated":    list(kept_names),
            **{f"ref_vr_{k}":   v for k, v in ref_vr.items()},
            **{f"ref_jump_{k}": v for k, v in ref_jump.items()},
        })

    # Summary table
    print(f"\n{'='*72}")
    print(f"S12 SUMMARY  ({run_tag})")
    print(f"{'='*72}")
    print(f"  {'Workload':<12}  {'VR S12':>7}  {'VR S11':>7}  {'VR S9':>7}  {'LSTM':>7}  "
          f"{'jump S12':>9}  {'jump S11':>9}")
    print("  " + "-" * 72)
    vr_vals = []
    for r in all_results:
        vr_vals.append(r["var_ratio_mean"])
        print(f"  {r['workload']:<12}  {r['var_ratio_mean']:>7.4f}  "
              f"{r.get('ref_vr_s11',  float('nan')):>7.4f}  "
              f"{r.get('ref_vr_s9',   float('nan')):>7.4f}  "
              f"{r.get('ref_vr_lstm', float('nan')):>7.4f}  "
              f"{r['phase_jump_ratio']:>8.3f}x  "
              f"{r.get('ref_jump_s11', float('nan')):>8.3f}x")
    print("  " + "-" * 72)
    print(f"  {'MEAN':<12}  {float(np.mean(vr_vals)):>7.4f}")
    print(f"\nReference means:  LSTM=0.612  S8=1.678 (honest best)  "
          f"S9=0.878 (honest balanced)  S11=1.978 (inflated)")

    with open(out_dir / "results.json", "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\nResults: {out_dir}/results.json")


if __name__ == "__main__":
    main()