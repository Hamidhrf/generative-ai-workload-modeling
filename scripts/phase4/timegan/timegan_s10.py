#!/usr/bin/env python3
"""
TimeGAN S10 - Adaptive Per-Metric Variance Cap + Gradient Clipping
===================================================================
Master Thesis - Generative AI Workload Modeling
Fachhochschule Dortmund

S9 RESULTS RECAP (best run: vc3.5)
====================================
Mean VR 0.878 beats LSTM 0.612. Four of five workloads beat LSTM.
But two structural problems remain:

1. RESNET152 VARIANCE COLLAPSE (VR 0.324 vs LSTM 0.607)
   Root cause: ResNet152 has near-zero real variance on GPU and latency
   (GPU ~3-36%, latency stable ~17ms). The var_pen ratio fires at epoch 21
   with value 26.2 even at cap=3.5, because:
       var_fake / var_real_tiny >> cap
   immediately when the generator produces any noise. The generator learns
   "produce zero variance" in the first adversarial epoch and never recovers.

   The gradient clipping in S9 (max_norm=1.0) runs AFTER loss_G.backward(),
   which clips the total gradient norm, but the var_pen term of 26.2 already
   dominates the gradient direction before clipping normalises the magnitude.
   The direction is already wrong.

   FIX A: Per-metric adaptive cap.
   Instead of a single scalar cap for all metrics, compute a cap per metric
   from the training data variance distribution. Metrics with near-zero real
   variance get a higher cap (the generator needs room to produce any signal).
   Metrics with high real variance get a tighter cap.

       var_real_global = traces_train.var(axis=(0,1))   # shape (M,)
       scale = var_mean / var_real_global.clamp(1e-8)   # inverse of relative var
       cap_per_metric = (scale * base_cap).clamp(base_cap, base_cap * 6.0)

   ResNet152 gpu_utilization has ~10x smaller variance than the mean, so it
   gets cap ~10*base_cap. The penalty does not fire on normal generator noise
   for that metric.

   FIX B: Scale down the effective lambda for var_pen in the combined loss.
   Use (lambda_var_pen * 0.1) so the penalty contributes to the direction
   but cannot dominate it even if the ratio is large at epoch 21.
   With the adaptive cap the penalty should not be large for low-variance
   metrics anyway, but the scaling provides a second layer of protection.
   Single backward pass -- CuDNN RNNs do not support double backward.

2. WHISPER JUMP RATIO 23-26x (still above 20x threshold)
   The smoothing penalty is not addressing the root cause. Looking at the
   Whisper loss plot: W-dist is noisy after epoch 130 (oscillating 7-8 instead
   of monotonically rising). The discriminator and generator are not in stable
   equilibrium. The jump ratio comes from this instability.

   The discriminator needs more updates per generator step during the period
   when Whisper's W-dist becomes unstable. Whisper gets n_disc_steps=3 instead
   of 2 from epoch 100 onward (when W-dist stabilises in other workloads but
   not in Whisper).

   FIX: Per-workload n_disc_steps override in WORKLOAD_OVERRIDES.
   Whisper: n_disc_steps=3 (vs 2 for others). More discriminator updates
   per generator step stabilises the equilibrium and reduces the jump ratio
   by forcing the discriminator to remain calibrated.

ARCHITECTURE UNCHANGED FROM S9
================================
Generator:     GeneratorA with enc-dropout (per-workload p)
Discriminator: h=32 l=1 (S7 capacity)
Warmup:        20 epochs rec-only
Adversarial:   150 fixed epochs, best W-dist checkpoint
Lambda:        rec=0.1, adv=1.0

PER-WORKLOAD DEFAULTS (updated)
=================================
bert:      enc_dropout=0.3,  lambda_smooth=0.1, n_disc_steps=2
gpt2:      enc_dropout=0.3,  lambda_smooth=0.1, n_disc_steps=2
resnet152: enc_dropout=0.3,  lambda_smooth=0.1, n_disc_steps=2
whisper:   enc_dropout=0.15, lambda_smooth=0.3, n_disc_steps=3
yolo:      enc_dropout=0.3,  lambda_smooth=0.1, n_disc_steps=2

NEW IN S10
===========
- cap_per_metric: per-metric adaptive variance cap (computed from train data)
- var_pen effective lambda scaled to (lambda_var_pen * 0.1) in combined loss
  so penalty cannot dominate gradient direction even at epoch-21 spike
- Single backward pass (CuDNN RNNs do not support double backward)
- Per-workload n_disc_steps in WORKLOAD_OVERRIDES (Whisper gets 3)
- run_tag uses "s10" prefix to separate from S9 output directories

REFERENCES (carry from S9)
===========================
S7  mean VR = 0.415
S8  mean VR = 1.678
S9  mean VR = 0.878  (vc3.5 run, best S9)
LSTM mean VR = 0.612

USAGE
-----
    python timegan_s10.py                          # all workloads, defaults
    python timegan_s10.py --workloads bert gpt2    # subset
    python timegan_s10.py --var-pen 0.5            # variance penalty strength
    python timegan_s10.py --base-cap 3.5           # base cap (scaled per metric)
    python timegan_s10.py --smooth 0.0             # disable smoothing
    python timegan_s10.py --enc-dropout 0.3        # override all workloads

OUTPUT
------
    outputs/phase4/timegan_s10/{run_tag}/plots/
    models/phase4/timegan_s10/{run_tag}/{workload}/
    outputs/phase4/timegan_s10/{run_tag}/results.json

    run_tag = s10_vp{var_pen*10}_bc{base_cap*10}_ae{adv_epochs}
    e.g.      s10_vp05_bc35_ae150
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
    "n_disc_steps":      2,        # default; overridden per-workload below
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

# Per-workload overrides
# S10 adds n_disc_steps: Whisper gets 3 to stabilise equilibrium
WORKLOAD_OVERRIDES = {
    "bert":      {"enc_dropout": 0.3,  "lambda_smooth": 0.1, "n_disc_steps": 2},
    "gpt2":      {"enc_dropout": 0.3,  "lambda_smooth": 0.1, "n_disc_steps": 2},
    "resnet152": {"enc_dropout": 0.3,  "lambda_smooth": 0.1, "n_disc_steps": 2},
    "whisper":   {"enc_dropout": 0.15, "lambda_smooth": 0.3, "n_disc_steps": 3},
    "yolo":      {"enc_dropout": 0.3,  "lambda_smooth": 0.1, "n_disc_steps": 2},
}

REFERENCE_VR = {
    "bert":      {"lstm": 0.594, "s7": 0.240, "s8": 2.645, "s9": 1.213},
    "gpt2":      {"lstm": 0.676, "s7": 0.573, "s8": 2.315, "s9": 0.664},
    "resnet152": {"lstm": 0.607, "s7": 0.246, "s8": 1.055, "s9": 0.324},
    "whisper":   {"lstm": 0.748, "s7": 0.745, "s8": 1.042, "s9": 0.743},
    "yolo":      {"lstm": 0.438, "s7": 0.270, "s8": 1.334, "s9": 1.446},
}

REFERENCE_JUMP = {
    "bert":      {"s7": 12.7, "s8":  3.9, "s9":  8.7},
    "gpt2":      {"s7": 10.5, "s8":  3.0, "s9":  7.2},
    "resnet152": {"s7": 12.6, "s8":  6.5, "s9": 17.1},
    "whisper":   {"s7": 16.2, "s8": 42.1, "s9": 26.2},
    "yolo":      {"s7": 10.2, "s8": 10.6, "s9": 11.2},
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
    """
    Ensure replica_counts has one value per pod, not per experiment.
    If length mismatch, reconstruct from metadata.
    """
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
    means = raw_kept.mean(axis=1, keepdims=True)  # (N,1,M)
    zm    = raw_kept - means
    if clip_stds > 0:
        std_global = zm.reshape(-1, zm.shape[-1]).std(axis=0, keepdims=True)
        zm = np.clip(zm, -clip_stds * std_global, clip_stds * std_global)
    return zm, means.squeeze(1)  # (N,T,M), (N,M)


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
    """
    Inverse-variance weights so low-variance metrics are not ignored by MSE.
    """
    train_data = proc_traces[train_idx]
    var_per_metric = train_data.reshape(-1, train_data.shape[-1]).var(axis=0)
    var_per_metric = np.clip(var_per_metric, 1e-8, None)
    weights = 1.0 / var_per_metric
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def compute_adaptive_cap(proc_traces, train_idx, base_cap, max_scale=6.0):
    """
    Per-metric variance cap computed from training data.

    Metrics with near-zero real variance get a higher cap (the generator needs
    room to produce any signal at all without the penalty immediately firing).
    Metrics with high real variance get a tighter cap.

    cap_j = (var_mean / var_j) * base_cap, clamped to [base_cap, base_cap*max_scale]

    Example with base_cap=3.5:
      - ResNet152 gpu_utilization (var ~10x smaller than mean): cap ~21.0
      - BERT psi_cpu (var ~2x smaller than mean): cap ~7.0
      - GPT2 latency (var ~equal to mean): cap ~3.5
    """
    train_data    = proc_traces[train_idx]
    var_per_metric = train_data.reshape(-1, train_data.shape[-1]).var(axis=0)
    var_per_metric = np.clip(var_per_metric, 1e-8, None)
    var_mean       = var_per_metric.mean()
    scale          = var_mean / var_per_metric
    cap            = np.clip(scale * base_cap, base_cap, base_cap * max_scale)
    return cap.astype(np.float32)


# ------------------------------------------------------------------
# Replica interpolation (fixed skip condition from S9)
# ------------------------------------------------------------------

def replica_interpolation(proc_traces, zmph, trace_means,
                          replica_counts, experiment_ids, train_idx):
    """
    Augment training set by linearly interpolating between adjacent
    replica counts. Skips pairs where the midpoint already exists in
    the dataset (real measurement present). Produces new training
    samples for gaps only.
    """
    train_r   = replica_counts[train_idx]
    unique_r  = set(int(v) for v in np.unique(train_r))
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

        # Skip if real data already exists at the midpoint
        if r_mid in unique_r:
            continue
        # Skip if midpoint is not strictly between endpoints (handles gap=1)
        if r_mid <= r_lo or r_mid >= r_hi:
            continue

        idx_lo = train_idx[train_r == r_lo]
        idx_hi = train_idx[train_r == r_hi]
        n      = min(len(idx_lo), len(idx_hi))
        if n == 0:
            continue

        alpha = 0.5
        for k in range(n):
            new_zm.append(alpha * zmph[idx_lo[k]]   + (1 - alpha) * zmph[idx_hi[k]])
            new_proc.append(alpha * proc_traces[idx_lo[k]] + (1 - alpha) * proc_traces[idx_hi[k]])
            new_means.append(alpha * trace_means[idx_lo[k]] + (1 - alpha) * trace_means[idx_hi[k]])
            new_r.append(r_mid)
            new_eids.append(-1)
        print(f"    [interp] r={r_lo}+r={r_hi} -> r={r_mid}: {n} new samples")

    if not new_zm:
        print(f"    [interp] 0 samples created. "
              f"unique_r={unique_r_sorted} -- all midpoints already covered.")
        n_orig = len(train_idx)
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
        self.zmph           = zmph
        self.proc           = proc_traces
        self.means          = trace_means
        self.rc             = replica_counts
        self.eids           = experiment_ids
        self.phase_seq      = phase_seq
        self.indices        = indices
        self.jitter         = jitter
        self.base_bounds    = base_boundaries or DEFAULT_PHASE_BOUNDARIES
        self.r_max          = max(float(replica_counts.max()), 1.0)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx   = self.indices[i]
        inp   = torch.tensor(self.zmph[idx],  dtype=torch.float32)
        tgt   = torch.tensor(self.proc[idx],  dtype=torch.float32)
        r_val = int(self.rc[idx]) if idx < len(self.rc) else 1
        eid   = int(self.eids[idx]) if idx < len(self.eids) else 0

        if self.jitter > 0:
            T     = inp.shape[0]
            shift = np.random.randint(-self.jitter, self.jitter + 1)
            inp   = torch.roll(inp, shift, dims=0)
            tgt   = torch.roll(tgt, shift, dims=0)

        r_norm = torch.tensor((r_val - 1.0) / 9.0, dtype=torch.float32)
        exp_id = torch.tensor(eid, dtype=torch.long)

        T = inp.shape[0]
        ph = torch.tensor(self.phase_seq[:T], dtype=torch.long)

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
# Generator (GeneratorA, unchanged from S8/S9)
# ------------------------------------------------------------------

class GeneratorA(nn.Module):
    """
    LSTM encoder-decoder with:
    - Replica embedding (continuous r_norm + learned embedding)
    - Phase embedding for phase-aware generation
    - Enc-dropout: activated only during adversarial phase
    """
    def __init__(self, seq_len, n_metrics, cfg, enc_dropout_p=0.3):
        super().__init__()
        self.seq_len          = seq_len
        self.n_metrics        = n_metrics
        self.hidden_dim       = cfg["hidden_dim"]
        self.num_layers       = cfg["num_layers"]
        self.latent_dim       = cfg["latent_dim"]
        self.enc_dropout_p    = enc_dropout_p
        self.enc_dropout_active = False  # toggled on at adv start

        input_dim = n_metrics + 1  # +1 for phase feature

        self.enc_rnn = nn.LSTM(input_dim, self.hidden_dim, self.num_layers,
                               batch_first=True, dropout=cfg["dropout"] if self.num_layers > 1 else 0.0)
        self.enc_fc  = nn.Linear(self.hidden_dim, self.latent_dim)

        self.r_embed = nn.Sequential(
            nn.Linear(1, cfg["replica_embed_dim"]),
            nn.ReLU(),
        )
        self.ph_embed = nn.Embedding(N_PHASES + 1, cfg["phase_embed_dim"])

        dec_input = self.latent_dim + cfg["replica_embed_dim"]
        self.dec_rnn = nn.LSTM(dec_input, self.hidden_dim, self.num_layers,
                               batch_first=True, dropout=cfg["dropout"] if self.num_layers > 1 else 0.0)
        self.dec_fc  = nn.Linear(self.hidden_dim + cfg["phase_embed_dim"], n_metrics)
        self.out_act  = nn.Sigmoid()

    def forward(self, inp, r_norm, exp_ids, phase_ids):
        B, T, _ = inp.shape

        if self.enc_dropout_active and self.enc_dropout_p > 0 and self.training:
            mask = (torch.rand(B, T, 1, device=inp.device) > self.enc_dropout_p).float()
            enc_inp = inp * mask
        else:
            enc_inp = inp

        _, (h, c) = self.enc_rnn(enc_inp)
        z = self.enc_fc(h[-1])  # (B, latent_dim)

        r_emb = self.r_embed(r_norm.unsqueeze(-1))  # (B, replica_embed_dim)
        dec_in_step = torch.cat([z, r_emb], dim=-1).unsqueeze(1).expand(-1, T, -1)

        dec_out, _ = self.dec_rnn(dec_in_step)  # (B, T, hidden_dim)

        ph_emb = self.ph_embed(phase_ids)  # (B, T, phase_embed_dim)
        out_in = torch.cat([dec_out, ph_emb], dim=-1)
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
# Discriminator (unchanged from S9)
# ------------------------------------------------------------------

class Discriminator(nn.Module):
    def __init__(self, n_metrics, cfg):
        super().__init__()
        hidden   = cfg["hidden_dim"]
        n_layers = cfg["num_layers"]
        dropout  = cfg["dropout"] if n_layers > 1 else 0.0
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


def gradient_penalty(disc, real, fake, device, lam=10.0):
    B   = real.shape[0]
    eps = torch.rand(B, 1, 1, device=device)
    mid = (eps * real + (1 - eps) * fake).requires_grad_(True)
    # Disable CuDNN for the discriminator forward pass so that
    # create_graph=True works with LSTM. CuDNN RNNs do not support
    # double backward; the non-CuDNN path does.
    with torch.backends.cudnn.flags(enabled=False):
        d = disc(mid)
    grad = torch.autograd.grad(d.sum(), mid, create_graph=True)[0]
    return lam * ((grad.norm(2, dim=(1, 2)) - 1) ** 2).mean()


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train(generator, discriminator, train_dl, val_dl,
          base_cfg, disc_warmup, adv_epochs, gen_cfg,
          lambda_var_pen, cap_per_metric_tensor,
          lambda_smooth, n_disc_steps_wl,
          device, metric_weights=None,
          warmup_patience=15, min_delta=1e-5):
    """
    Training loop with S10 changes:

    1. Per-metric adaptive variance cap (cap_per_metric_tensor computed from
       training data before this call). Prevents the epoch-21 vpen spike that
       collapsed ResNet152 in S9 -- near-zero-variance metrics get a higher cap
       so the penalty does not fire on normal generator noise.

    2. Single backward pass (CuDNN RNNs do not support double backward).
       All loss terms accumulated into loss_G, then backward once.
       var_pen uses a reduced effective lambda (lambda_var_pen * 0.1) so that
       even if vpen is large at epoch 21, it cannot dominate the gradient
       direction. With the adaptive cap the penalty should not be large anyway,
       but the scaling provides a second layer of protection.

    3. Per-workload n_disc_steps passed in as n_disc_steps_wl.
       Whisper uses 3 discriminator steps per generator step.
    """
    use_fm  = base_cfg.get("lambda_fm", 0.0) > 0
    lam_fm  = base_cfg.get("lambda_fm", 0.0)
    lam_gp  = base_cfg.get("lambda_gp", 10.0)
    betas   = (0.0, 0.9)

    opt_G = torch.optim.Adam(generator.parameters(),
                             lr=1e-3, betas=betas, weight_decay=1e-5)
    opt_D = torch.optim.Adam(discriminator.parameters(),
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

    print(f"    Warmup: {disc_warmup} epochs (patience={warmup_patience})")
    print(f"    Adv:    {adv_epochs} fixed epochs  "
          f"var_pen={lambda_var_pen}  n_disc={n_disc_steps_wl}")

    for epoch in range(1, total_epochs + 1):

        in_warmup    = (epoch <= actual_warmup)
        lam_rec      = base_cfg["lambda_rec_warmup"] if in_warmup else base_cfg["lambda_rec_post"]
        lam_adv      = base_cfg["lambda_adv_warmup"] if in_warmup else base_cfg["lambda_adv_post"]
        # Use per-workload disc steps during adversarial phase
        n_disc_steps = 0 if in_warmup else n_disc_steps_wl

        if not in_warmup and not warmup_done:
            warmup_done = True
            generator.enc_dropout_active = True
            print(f"    [epoch {epoch}] Adversarial start. "
                  f"enc_dropout p={generator.enc_dropout_p}  "
                  f"n_disc_steps={n_disc_steps_wl}")

        generator.train()
        discriminator.train()
        ep_rec = ep_adv_g = ep_loss_d = ep_wdist = ep_vpen = ep_sm = 0.0
        n_batches = 0

        for inp, target, _, r_norm, exp_ids, ph_batch in train_dl:
            inp      = inp.to(device)
            target   = target.to(device)
            r_norm   = r_norm.to(device)
            exp_ids  = exp_ids.to(device)
            ph_batch = ph_batch.to(device)

            # Discriminator update
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

            # ----------------------------------------------------------
            # Generator update - single backward pass
            # CuDNN LSTMs do not support double backward, so all loss
            # terms are accumulated into loss_G and backward is called once.
            # ----------------------------------------------------------
            opt_G.zero_grad()
            fake = generator(inp, r_norm, exp_ids, ph_batch)

            raw_mse  = mse_fn(fake, target)
            if metric_weights is not None:
                raw_mse = raw_mse * metric_weights.unsqueeze(0).unsqueeze(0)
            loss_rec = raw_mse.mean()

            loss_adv = (-discriminator(fake).mean()
                        if lam_adv > 0 else torch.tensor(0.0, device=device))
            loss_G   = lam_rec * loss_rec + lam_adv * loss_adv

            if use_fm and not in_warmup:
                fr     = discriminator.get_features(target).detach()
                ff     = discriminator.get_features(fake)
                loss_G = loss_G + lam_fm * F.mse_loss(ff.mean(0), fr.mean(0))

            # Per-metric variance penalty with adaptive cap
            # lambda scaled by 0.1 so that even if vpen fires at epoch 21,
            # it cannot dominate the gradient direction. The adaptive cap
            # should prevent large vpen values for near-zero-variance metrics,
            # but the scaling is a second layer of protection.
            if lambda_var_pen > 0 and not in_warmup:
                var_real  = target.var(dim=[0, 1]).detach()   # (M,)
                var_fake  = fake.var(dim=[0, 1])               # (M,) differentiable
                var_ratio = var_fake / var_real.clamp(min=1e-8)
                excess    = F.relu(var_ratio - cap_per_metric_tensor)
                var_pen   = (excess ** 2).mean()
                loss_G    = loss_G + (lambda_var_pen * 0.1) * var_pen
                ep_vpen  += var_pen.item()

            # Temporal smoothing penalty
            if lambda_smooth > 0 and not in_warmup:
                diff   = fake[:, 1:, :] - fake[:, :-1, :]
                sm     = (diff ** 2).mean()
                loss_G = loss_G + lambda_smooth * sm
                ep_sm += sm.item()

            loss_G.backward()
            nn.utils.clip_grad_norm_(generator.parameters(), max_norm=gen_cfg["grad_clip"])
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
            "loss_G":   float(ep_adv_g  / n_batches),
            "loss_D":   float(ep_loss_d / n_batches),
            "loss_rec": float(ep_rec    / n_batches),
            "w_dist":   float(avg_wdist),
            "val_rec":  val_rec,
            "var_pen":  float(ep_vpen   / n_batches),
            "smooth":   float(ep_sm     / n_batches),
        }
        history.append(h)

        if epoch % 10 == 0 or epoch == 1 or epoch == actual_warmup + 1:
            print(f"    ep {epoch:>4} [{phase_label}]  val_rec={val_rec:.5f}  "
                  f"G={h['loss_G']:.4f}  D={h['loss_D']:.4f}  "
                  f"W={avg_wdist:.4f}  vpen={h['var_pen']:.4f}  sm={h['smooth']:.4f}")

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

    # Decide which metrics to plot (max 4)
    plot_metrics = kept_names[:4]
    n_plots = len(plot_metrics) + 1  # +1 for loss
    n_cols  = 3
    n_rows  = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    leg_handles = []
    leg_labels  = []

    for pi, mname in enumerate(plot_metrics):
        ax  = axes[pi]
        col = mname.replace("pod_", "").replace("gpu_", "").replace("_avg", "").replace("_usage", "")
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("timestep", fontsize=8)

        for i, r_val in enumerate(r_sample):
            clr   = colors[i % len(colors)]
            ridx  = np.where(real_r_arr == r_val)[0]
            sidx  = np.where(syn_r_arr  == r_val)[0]
            m_idx = kept_names.index(mname)

            if len(ridx) > 0:
                h1, = ax.plot(real_orig[ridx[0], :, m_idx], color=clr, alpha=0.85, linewidth=0.9)
                if pi == 0:
                    leg_handles.append(h1)
                    leg_labels.append(f"r={r_val} solid=real dashed=syn")
            if len(sidx) > 0:
                ax.plot(syn_orig[sidx[0], :, m_idx], color=clr, alpha=0.85,
                        linewidth=0.9, linestyle="--")

        for bt in boundary_ts:
            ax.axvline(bt, color="grey", alpha=0.3, linewidth=0.7)

    # Loss plot
    loss_ax = axes[len(plot_metrics)]
    eps     = [h["epoch"]   for h in history]
    loss_ax.plot(eps, [h["loss_rec"] for h in history], label="train rec",  color="#1f77b4")
    loss_ax.plot(eps, [h["val_rec"]  for h in history], label="val rec",    color="#1f77b4", linestyle="--")
    loss_ax.plot(eps, [h["loss_G"]   for h in history], label="adv (gen)",  color="#ff7f0e")
    loss_ax.plot(eps, [h["loss_D"]   for h in history], label="disc loss",  color="#2ca02c")
    loss_ax.plot(eps, [h["w_dist"]   for h in history], label="W-dist",     color="#9467bd", linestyle=":")
    loss_ax.plot(eps, [h["var_pen"]  for h in history], label="var_pen",    color="#d62728", linestyle="-.")
    loss_ax.plot(eps, [h["smooth"]   for h in history], label="smooth",     color="#8c564b", linestyle=":")
    loss_ax.axvline(disc_warmup, color="red", linestyle="--", alpha=0.5,
                    label=f"warmup end (ep {disc_warmup})")
    loss_ax.set_title(f"Training losses\nwarmup={disc_warmup} adv={adv_epochs}", fontsize=9)
    loss_ax.set_xlabel("epoch", fontsize=8)
    loss_ax.legend(fontsize=6, loc="upper left")

    # Hide unused axes
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
    parser = argparse.ArgumentParser(description="TimeGAN S10")
    parser.add_argument("--var-pen",     type=float, default=0.5,
                        help="Lambda for variance penalty (default 0.5)")
    parser.add_argument("--base-cap",    type=float, default=3.5,
                        help="Base cap for adaptive per-metric variance cap (default 3.5)")
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

    cfg        = BASE_CONFIG.copy()
    boundaries = list(map(int, args.phase_boundaries.split(",")))
    interp_tag = "" if args.aug_interp else "_ni"
    run_tag    = (f"s10_vp{int(args.var_pen * 10):02d}"
                  f"_bc{int(args.base_cap  * 10):02d}"
                  f"_ae{args.adv_epochs}{interp_tag}")

    out_dir   = Path(f"outputs/phase4/timegan_s10/{run_tag}")
    model_dir = Path(f"models/phase4/timegan_s10/{run_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TimeGAN S10 - Adaptive Per-Metric Cap + Separated Gradient Passes")
    print("=" * 70)
    print(f"Device          : {device}")
    print(f"Var penalty     : lambda={args.var_pen}  base_cap={args.base_cap} (adaptive per metric)")
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
        wl_overrides  = WORKLOAD_OVERRIDES[workload]
        enc_dropout_p = (args.enc_dropout if args.enc_dropout is not None
                         else wl_overrides["enc_dropout"])
        lambda_smooth = (args.smooth      if args.smooth      is not None
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

        # Compute per-metric adaptive variance cap from training data
        cap_per_metric = compute_adaptive_cap(
            proc_traces, train_idx_orig, args.base_cap, max_scale=6.0)
        cap_tensor = torch.tensor(cap_per_metric, dtype=torch.float32, device=device)

        print(f"  Adaptive variance caps per metric:")
        for j, m in enumerate(kept_names):
            print(f"    {m:<22} cap={cap_per_metric[j]:.2f}")

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
            args.var_pen, cap_tensor,
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
        print(f"\n  Results (s10 {run_tag}):")
        print(f"    val_rec       = {best_val_rec:.5f}")
        print(f"    var_ratio     = {vr_mean:.4f}  "
              f"(s9={ref_vr.get('s9','?'):.4f}  s8={ref_vr.get('s8','?'):.4f}  "
              f"lstm={ref_vr.get('lstm','?'):.4f})")
        print(f"    autocorr_diff = {ac_diff:.4f}")
        print(f"    jump_ratio    = {jump_ratio:.3f}x  "
              f"(s9={ref_jump.get('s9','?'):.1f}x  s8={ref_jump.get('s8','?'):.1f}x)")
        print(f"    epochs        = {n_epochs}  time = {elapsed:.1f}s")
        print(f"  Per-metric VR:")
        for j, m in enumerate(kept_names):
            flag = " <-- low"  if vr[j] < 0.5 else (
                   " <-- HIGH" if vr[j] > 3.0 else "")
            print(f"    {m:<22} {vr[j]:.4f}  (cap={cap_per_metric[j]:.1f}){flag}")

        run_label = f"s10 {run_tag} genA"
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
                "lambda_var_pen":  args.var_pen,
                "base_cap":        args.base_cap,
                "cap_per_metric":  cap_per_metric.tolist(),
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
            "stage":           "s10",
            "run_tag":         run_tag,
            "enc_dropout":     enc_dropout_p,
            "adv_epochs":      args.adv_epochs,
            "disc_warmup":     args.disc_warmup,
            "jitter":          args.jitter,
            "aug_interp":      args.aug_interp,
            "lambda_var_pen":  args.var_pen,
            "base_cap":        args.base_cap,
            "cap_per_metric":  cap_per_metric.tolist(),
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

    # Summary
    print(f"\n{'='*72}")
    print(f"S10 SUMMARY  ({run_tag})")
    print(f"{'='*72}")
    print(f"  {'Workload':<12}  {'VR S10':>7}  {'VR S9':>7}  {'VR S8':>7}  {'LSTM':>7}  "
          f"{'jump S10':>9}  {'jump S9':>9}")
    print("  " + "-" * 72)
    vr_vals = []
    for r in all_results:
        vr_vals.append(r["var_ratio_mean"])
        print(f"  {r['workload']:<12}  {r['var_ratio_mean']:>7.4f}  "
              f"{r.get('ref_vr_s9',  float('nan')):>7.4f}  "
              f"{r.get('ref_vr_s8',  float('nan')):>7.4f}  "
              f"{r.get('ref_vr_lstm',float('nan')):>7.4f}  "
              f"{r['phase_jump_ratio']:>8.3f}x  "
              f"{r.get('ref_jump_s9', float('nan')):>8.1f}x")
    print("  " + "-" * 72)
    print(f"  {'MEAN':<12}  {float(np.mean(vr_vals)):>7.4f}")
    print(f"\nReference means:  LSTM=0.612  S7=0.415  S8=1.678  S9=0.878")

    with open(out_dir / "results.json", "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\nResults: {out_dir}/results.json")


if __name__ == "__main__":
    main()