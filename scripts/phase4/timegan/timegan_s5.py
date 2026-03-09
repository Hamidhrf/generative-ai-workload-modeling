#!/usr/bin/env python3
"""
TimeGAN Ablation - Full Redo S1 through S7
===========================================
Master Thesis - Generative AI Workload Modeling
Fachhochschule Dortmund

WHY WE ARE REDOING S1-S4
=========================
The original S1-S4 used the LSTM v6 generator unchanged from the supervised
baseline. That generator's forward() takes the REAL trace as input:

    pod_lat = self.pod_encoder(x_with_phase)   # encodes the real trace

This means during GAN training the generator is a reconstruction model:
    input  -> real trace (B, T, M+1)
    output -> reconstruction of that real trace (B, T, M)

The MSE loss is trivially minimized by encoding then decoding the real input.
The adversarial signal never had a real job to do because the generator already
has the answer (the real trace) in its input. The generate() method substituted
random noise at inference time, but the model was never trained on noise inputs,
so generation was out-of-distribution from the start.

This explains why VR was consistently low across S1-S4 regardless of adversarial
objective changes: the problem was never the loss, it was that the generator was
not a generator at all.

TWO GENERATOR VARIANTS
=======================
We run all stages with both variants to isolate exactly what contributes:

Option A (reconstruction): keep LSTM v6 generator unchanged.
  - Training: generator(real_trace, r, phase) -> reconstructed_trace
  - Adversarial loss is a regularizer on top of reconstruction
  - Honest framing: conditional autoencoder with adversarial regularization
  - Generates by substituting noise at inference (same as original S1-S4)

Option B (generative): new noise-input generator.
  - Training: generator(noise, r, phase) -> synthetic_trace
  - MSE loss computed against real traces sampled from the dataset (not the
    specific trace the generator was conditioned on)
  - True GAN: generator never sees the real trace it is evaluated against
  - Generates by sampling fresh noise at inference (consistent with training)

ABLATION STAGES
===============
Stage 1 — Minimal GAN
  Generator:     option A or B
  Discriminator: simple BiLSTM -> scalar (BCE)
  Loss:          MSE + adversarial (BCE)
  LR:            same for both (1e-3)
  Epochs:        200

Stage 2 — Stabilization
  Add: spectral normalization on discriminator
  Add: two-timescale LR (gen=1e-3, disc=2e-4)
  Add: n_disc_steps=2
  Everything else same as Stage 1

Stage 3 — Richer loss
  Add: feature matching loss (lambda_fm=0.01)
  Add: moment matching loss (lambda_var=0.1, calibrated)
  Everything else same as Stage 2

Stage 4 — Gradient penalty
  Drop: BCE -> Wasserstein distance
  Drop: spectral norm (GP replaces it)
  Add: WGAN-GP gradient penalty (lambda_gp=10)
  Keep: feature matching at lambda_fm=0.01
  Drop: moment matching (S3 proved it destabilizes)
  Everything else same as Stage 2

Stage 5a — Lambda rebalancing
  Change: lambda_rec 1.0->0.1, lambda_adv 0.1->1.0
  Everything else same as Stage 4

Stage 5b — Conditional discriminator
  Add: discriminator receives r_norm broadcast at each timestep
  Everything else same as Stage 5a

Stage 6 — Preprocessing experiment
  Switch: zero-mean -> full min-max normalization
  Base: best stage from above
  Compare directly

Stage 7 — Epoch budget
  Run 300 then 500 epochs with best stage
  Compare VR vs 200 epochs

EACH STAGE ANSWERS ONE QUESTION
================================
Stage 1 vs LSTM:  Does any adversarial training help at all?
Stage 2 vs 1:     Does stabilization improve or was Stage 1 already stable?
Stage 3 vs 2:     Do feature/moment matching add on top of stability?
Stage 4 vs 3:     Does WGAN-GP improve over BCE?
Stage 5a vs 4:    Does lambda rebalancing fix reconstruction dominance?
Stage 5b vs 5a:   Does conditional discriminator matter additionally?
Stage 6 vs best:  Does preprocessing choice matter?
Stage 7 vs best:  Are we compute-limited at 200 epochs?
Option A vs B:    Does the generator architecture (reconstruction vs generative) matter?

USAGE
-----
    # Run a single stage with both generator options
    python timegan_ablation.py --stage s1
    python timegan_ablation.py --stage s2
    python timegan_ablation.py --stage s3
    python timegan_ablation.py --stage s4
    python timegan_ablation.py --stage s5a
    python timegan_ablation.py --stage s5b
    python timegan_ablation.py --stage s6
    python timegan_ablation.py --stage s7 --epochs 300
    python timegan_ablation.py --stage s7 --epochs 500

    # Run only one generator option
    python timegan_ablation.py --stage s1 --gen-option a
    python timegan_ablation.py --stage s1 --gen-option b

    # Run specific workloads only
    python timegan_ablation.py --stage s1 --workloads bert resnet152
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
# Paths
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

# ------------------------------------------------------------------
# Reference VRs (from original S1-S4 runs, option A baseline)
# ------------------------------------------------------------------

REFERENCE_VR = {
    "bert":      {"lstm": 0.594, "s1": 0.191, "s2": 0.252, "s3": 0.276, "s4": 0.117},
    "gpt2":      {"lstm": 0.676, "s1": 0.532, "s2": 0.552, "s3": 0.539, "s4": 0.480},
    "resnet152": {"lstm": 0.607, "s1": 0.108, "s2": 0.308, "s3": 0.288, "s4": 0.102},
    "whisper":   {"lstm": 0.748, "s1": 0.754, "s2": 0.761, "s3": 0.746, "s4": 0.785},
    "yolo":      {"lstm": 0.438, "s1": 0.271, "s2": 0.680, "s3": 0.190, "s4": 0.123},
}

# ------------------------------------------------------------------
# Stage configs: what each stage enables/changes
# ------------------------------------------------------------------

STAGE_CONFIGS = {
    "s1": {
        "adversarial":       "bce",
        "spectral_norm":     False,
        "two_timescale_lr":  False,
        "n_disc_steps":      1,
        "lambda_rec":        1.0,
        "lambda_adv":        0.1,
        "lambda_fm":         0.0,
        "lambda_var":        0.0,
        "lambda_gp":         0.0,
        "conditional_disc":  False,
        "preprocessing":     "zeromean",
        "epochs":            200,
    },
    "s2": {
        "adversarial":       "bce",
        "spectral_norm":     True,     # added
        "two_timescale_lr":  True,     # added
        "n_disc_steps":      2,        # added
        "lambda_rec":        1.0,
        "lambda_adv":        0.1,
        "lambda_fm":         0.0,
        "lambda_var":        0.0,
        "lambda_gp":         0.0,
        "conditional_disc":  False,
        "preprocessing":     "zeromean",
        "epochs":            200,
    },
    "s3": {
        "adversarial":       "bce",
        "spectral_norm":     True,
        "two_timescale_lr":  True,
        "n_disc_steps":      2,
        "lambda_rec":        1.0,
        "lambda_adv":        0.1,
        "lambda_fm":         0.01,    # added
        "lambda_var":        0.1,     # added (calibrated, not 0.5)
        "lambda_gp":         0.0,
        "conditional_disc":  False,
        "preprocessing":     "zeromean",
        "epochs":            200,
    },
    "s4": {
        "adversarial":       "wgan",   # changed
        "spectral_norm":     False,    # removed (GP replaces it)
        "two_timescale_lr":  True,
        "n_disc_steps":      2,
        "lambda_rec":        1.0,
        "lambda_adv":        0.1,
        "lambda_fm":         0.01,
        "lambda_var":        0.0,     # removed (proved destabilizing)
        "lambda_gp":         10.0,    # added
        "conditional_disc":  False,
        "preprocessing":     "zeromean",
        "epochs":            200,
    },
    "s5a": {
        "adversarial":       "wgan",
        "spectral_norm":     False,
        "two_timescale_lr":  True,
        "n_disc_steps":      2,
        "lambda_rec":        0.1,     # changed
        "lambda_adv":        1.0,     # changed
        "lambda_fm":         0.01,
        "lambda_var":        0.0,
        "lambda_gp":         10.0,
        "conditional_disc":  False,
        "preprocessing":     "zeromean",
        "epochs":            200,
    },
    "s5b": {
        "adversarial":       "wgan",
        "spectral_norm":     False,
        "two_timescale_lr":  True,
        "n_disc_steps":      2,
        "lambda_rec":        0.1,
        "lambda_adv":        1.0,
        "lambda_fm":         0.01,
        "lambda_var":        0.0,
        "lambda_gp":         10.0,
        "conditional_disc":  True,    # added
        "preprocessing":     "zeromean",
        "epochs":            200,
    },
    "s6": {
        # base = best stage from above, only preprocessing changes
        # set via --base-stage argument; defaults to s5b
        "adversarial":       "wgan",
        "spectral_norm":     False,
        "two_timescale_lr":  True,
        "n_disc_steps":      2,
        "lambda_rec":        0.1,
        "lambda_adv":        1.0,
        "lambda_fm":         0.01,
        "lambda_var":        0.0,
        "lambda_gp":         10.0,
        "conditional_disc":  True,
        "preprocessing":     "minmax",  # changed
        "epochs":            200,
    },
    "s7": {
        # base = best stage, epochs change only
        "adversarial":       "wgan",
        "spectral_norm":     False,
        "two_timescale_lr":  True,
        "n_disc_steps":      2,
        "lambda_rec":        0.1,
        "lambda_adv":        1.0,
        "lambda_fm":         0.01,
        "lambda_var":        0.0,
        "lambda_gp":         10.0,
        "conditional_disc":  True,
        "preprocessing":     "zeromean",
        "epochs":            300,      # overridden by --epochs flag
    },
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


def apply_minmax(traces):
    """
    Full min-max normalization to [0, 1] per metric across all samples.
    Returns normalized traces and (min, max) per metric for inversion.
    """
    mn = traces.min(axis=(0, 1), keepdims=True)  # (1, 1, M)
    mx = traces.max(axis=(0, 1), keepdims=True)
    rng = np.maximum(mx - mn, 1e-8)
    normalized = ((traces - mn) / rng).astype(np.float32)
    return normalized, mn.squeeze(), mx.squeeze()


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
# Post-hoc helpers
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
# Option A: yields (input_trace, target_trace, r_norm, ...)
#           input_trace = zm_with_phase (real trace + phase feature)
#           generator receives real trace -> reconstruction model
# Option B: yields (target_trace, r_norm, ...)
#           generator receives noise -> true generative model
# Both yield the same target_trace for MSE loss computation
# ------------------------------------------------------------------

class WorkloadDataset(Dataset):
    def __init__(self, processed_traces, target_traces, means,
                 replica_counts, experiment_ids, phase_seq, idx, gen_option):
        self.gen_option = gen_option
        # processed_traces: zm_with_phase for option A, zm_traces for option B
        self.inp     = torch.tensor(processed_traces[idx], dtype=torch.float32)
        self.target  = torch.tensor(target_traces[idx],    dtype=torch.float32)
        self.means   = torch.tensor(means[idx],            dtype=torch.float32)
        self.exp_ids = torch.tensor(experiment_ids[idx],   dtype=torch.long)
        r = replica_counts[idx].astype(np.float32)
        self.r_norm  = torch.tensor((r - 1.0) / 9.0,      dtype=torch.float32)
        self.phase_t = torch.tensor(phase_seq,             dtype=torch.long)

    def __len__(self): return len(self.inp)

    def __getitem__(self, i):
        return (self.inp[i], self.target[i], self.means[i],
                self.r_norm[i], self.exp_ids[i], self.phase_t)


# ------------------------------------------------------------------
# Generator Option A: reconstruction-based (original LSTM v6)
# Training: generator(real_trace, r, phase) -> reconstructed_trace
# Inference: generator(noise, r, phase) -> synthetic_trace
# Adversarial training is a regularizer on top of reconstruction.
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
        h = self.h0(z).view(B, self.num_layers, self.hidden_dim).permute(1,0,2).contiguous()
        c = self.c0(z).view(B, self.num_layers, self.hidden_dim).permute(1,0,2).contiguous()
        z_exp  = z.unsqueeze(1).expand(B, T, -1)
        ph_emb = self.phase_embed(phase_seq_batch)
        inp    = torch.cat([z_exp, ph_emb], dim=2)
        out, _ = self.lstm(inp, (h, c))
        return self.fc_out(out)


class GeneratorA(nn.Module):
    """
    Option A: reconstruction-based generator (LSTM v6 unchanged).
    forward() takes the real trace as input and reconstructs it.
    Adversarial loss is a regularizer, not a true generative objective.
    Inference uses random noise in place of the pod latent.
    """
    def __init__(self, seq_len, n_metrics, cfg):
        super().__init__()
        self.seq_len    = seq_len
        self.n_metrics  = n_metrics
        self.regime_dim = cfg["regime_dim"]
        self.pod_dim    = cfg["pod_dim"]

        # Encodes the real input trace to a pod-level latent
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
        """Training forward: reads real trace x_with_phase."""
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
        """Inference: substitutes random noise for the pod latent."""
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
# Generator Option B: noise-input (true generative model)
# Training: generator(noise, r, phase) -> synthetic_trace
# Inference: same — consistent with training distribution
# MSE loss is computed against randomly sampled real traces,
# not the specific trace the generator is conditioned on.
# ------------------------------------------------------------------

class GeneratorB(nn.Module):
    """
    Option B: noise-input generator. True GAN generator.
    Never sees the real trace during training or inference.

    Architecture:
    - noise: (B, latent_dim) sampled from N(0,1)
    - r_embed: maps normalized replica count to hidden_dim
    - Initial hidden/cell states of decoder LSTM are learned from noise + r
    - PhaseConditionedDecoder produces (B, T, n_metrics)
    - tanh output keeps values in [-1, 1] (compatible with zero-mean targets)
    """
    def __init__(self, seq_len, n_metrics, cfg):
        super().__init__()
        self.seq_len   = seq_len
        self.n_metrics = n_metrics
        self.latent_dim = cfg["latent_dim"]

        self.r_embed = nn.Sequential(
            nn.Linear(1, cfg["replica_embed_dim"]),
            nn.ReLU(),
        )

        # Project noise + r_embed to condition vector for decoder
        cond_dim = cfg["latent_dim"] + cfg["replica_embed_dim"]
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, cfg["hidden_dim"]),
            nn.ReLU(),
        )

        self.decoder = PhaseConditionedDecoder(
            cfg["hidden_dim"], cfg["hidden_dim"], cfg["num_layers"],
            n_metrics, N_PHASES, cfg["phase_embed_dim"], cfg["dropout"])

    def forward(self, noise, r_norm, phase_seq_batch):
        """
        noise: (B, latent_dim)
        r_norm: (B,)
        phase_seq_batch: (B, T)
        """
        r_e  = self.r_embed(r_norm.unsqueeze(1))          # (B, replica_embed_dim)
        cond = torch.cat([noise, r_e], dim=1)              # (B, latent_dim + replica_embed_dim)
        z    = self.cond_proj(cond)                        # (B, hidden_dim)
        return torch.tanh(self.decoder(z, phase_seq_batch))

    def generate(self, r_norm_val, n_pods, phase_seq, device):
        """Inference: identical to training — sample noise, decode."""
        self.eval()
        with torch.no_grad():
            noise = torch.randn(n_pods, self.latent_dim, device=device)
            r_t   = torch.full((n_pods,), r_norm_val, device=device)
            ph    = torch.tensor(phase_seq, dtype=torch.long, device=device)
            ph    = ph.unsqueeze(0).expand(n_pods, -1)
            return self.forward(noise, r_t, ph).cpu().numpy()


# ------------------------------------------------------------------
# Discriminator
# Supports: BCE (S1-S3) and WGAN (S4-S7)
# Supports: unconditional (S1-S5a) and conditional on r_norm (S5b-S7)
# Spectral norm applied per-layer when enabled (S2-S3 only)
# ------------------------------------------------------------------

def _apply_spectral_norm(module):
    for name, layer in module.named_children():
        if isinstance(layer, (nn.Linear, nn.LSTM)):
            if isinstance(layer, nn.LSTM):
                # Apply spectral norm to all weight matrices in LSTM
                for param_name in [n for n, _ in layer.named_parameters()
                                    if 'weight' in n]:
                    nn.utils.parametrize.register_parametrization(
                        layer, param_name,
                        nn.utils.parametrizations.spectral_norm(
                            nn.Linear(1, 1)  # placeholder
                        )
                    )
            else:
                nn.utils.spectral_norm(layer)


class Discriminator(nn.Module):
    """
    BiLSTM critic/discriminator.

    S1-S3:  adversarial_mode='bce'  -> sigmoid output, BCEWithLogitsLoss
            spectral_norm: optional (S2-S3 only)
    S4-S7:  adversarial_mode='wgan' -> no sigmoid, unbounded Wasserstein score
            spectral_norm: False (GP replaces it)

    S5b-S7: conditional=True -> r_norm broadcast concatenated to input at each timestep
    """
    def __init__(self, n_metrics, disc_cfg, adversarial_mode="bce",
                 spectral_norm=False, conditional=False):
        super().__init__()
        self.adversarial_mode = adversarial_mode
        self.conditional      = conditional

        in_dim = n_metrics + (1 if conditional else 0)

        self.lstm = nn.LSTM(
            in_dim,
            disc_cfg["hidden_dim"],
            disc_cfg["num_layers"],
            batch_first=True,
            bidirectional=True,
            dropout=disc_cfg["dropout"] if disc_cfg["num_layers"] > 1 else 0.0
        )

        head_layers = [
            nn.Linear(disc_cfg["hidden_dim"] * 2, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        ]
        # BCE mode: add sigmoid. WGAN mode: no sigmoid (unbounded score).
        if adversarial_mode == "bce":
            head_layers.append(nn.Sigmoid())

        self.classifier = nn.Sequential(*head_layers)

        # Spectral norm on linear layers (S2-S3 only)
        if spectral_norm:
            for i, layer in enumerate(self.classifier):
                if isinstance(layer, nn.Linear):
                    nn.utils.spectral_norm(layer)

    def _prepare_input(self, x, r_norm=None):
        if self.conditional and r_norm is not None:
            B, T, _ = x.shape
            r_col = r_norm.view(B, 1, 1).expand(B, T, 1)
            return torch.cat([x, r_col], dim=2)
        return x

    def forward(self, x, r_norm=None):
        inp = self._prepare_input(x, r_norm)
        _, (h_n, _) = self.lstm(inp)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.classifier(h)

    def get_features(self, x, r_norm=None):
        inp = self._prepare_input(x, r_norm)
        _, (h_n, _) = self.lstm(inp)
        return torch.cat([h_n[-2], h_n[-1]], dim=1)


# ------------------------------------------------------------------
# Gradient penalty (WGAN-GP, S4+)
# CuDNN disabled for double-backward through LSTM
# ------------------------------------------------------------------

def gradient_penalty(critic, real, fake, r_norm, device, lam=10.0):
    B = real.size(0)
    alpha  = torch.rand(B, 1, 1, device=device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    with torch.backends.cudnn.flags(enabled=False):
        score_interp = critic(interp, r_norm)
    grads = torch.autograd.grad(
        outputs=score_interp,
        inputs=interp,
        grad_outputs=torch.ones_like(score_interp),
        create_graph=True,
    )[0]
    grads_norm = grads.reshape(B, -1).norm(2, dim=1)
    return lam * ((grads_norm - 1) ** 2).mean()


# ------------------------------------------------------------------
# Training loop
# Handles both generator options and all stage configurations
# ------------------------------------------------------------------

def train(generator, discriminator, train_dl, val_dl,
          stage_cfg, gen_cfg, gen_option, device, metric_weights=None):
    """
    Unified training loop for all stages and both generator options.

    Generator option A: batch yields (real_trace, target, means, r_norm, exp_ids, phase)
                        generator.forward(real_trace, r_norm, exp_ids, phase) -> fake
    Generator option B: batch yields (_, target, means, r_norm, exp_ids, phase)
                        generator.forward(noise, r_norm, phase) -> fake
                        MSE computed against a randomly drawn real target from the batch

    For option B, the MSE is computed against a randomly permuted real target
    (not the specific target paired with the noise sample). This prevents the
    generator from trivially copying input.
    """
    adv_mode    = stage_cfg["adversarial"]
    use_wgan    = (adv_mode == "wgan")
    use_gp      = stage_cfg["lambda_gp"] > 0
    use_fm      = stage_cfg["lambda_fm"] > 0
    use_var     = stage_cfg["lambda_var"] > 0
    cond_disc   = stage_cfg["conditional_disc"]
    lam_rec     = stage_cfg["lambda_rec"]
    lam_adv     = stage_cfg["lambda_adv"]
    lam_fm      = stage_cfg["lambda_fm"]
    lam_var     = stage_cfg["lambda_var"]
    lam_gp      = stage_cfg["lambda_gp"]
    n_disc_steps = stage_cfg["n_disc_steps"]
    patience    = 25
    min_delta   = 1e-5

    lr_gen  = 1e-3
    lr_disc = 2e-4 if stage_cfg["two_timescale_lr"] else 1e-3

    betas = (0.0, 0.9) if use_wgan else (0.9, 0.999)

    opt_G = torch.optim.Adam(generator.parameters(),
                             lr=lr_gen, betas=betas, weight_decay=1e-5)
    opt_D = torch.optim.Adam(discriminator.parameters(),
                             lr=lr_disc, betas=betas, weight_decay=1e-5)

    sched_G = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_G, factor=0.5, patience=7)

    mse_fn  = nn.MSELoss(reduction="none")
    bce_fn  = nn.BCELoss()

    best_val_rec = float("inf")
    best_state_G = {k: v.cpu().clone() for k, v in generator.state_dict().items()}
    best_state_D = {k: v.cpu().clone() for k, v in discriminator.state_dict().items()}
    patience_c   = 0
    history      = []

    for epoch in range(1, stage_cfg["epochs"] + 1):
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

            # Generate fake traces
            if gen_option == "a":
                with torch.no_grad():
                    fake_detach = generator(inp, r_norm, exp_ids, ph_batch)
            else:
                noise = torch.randn(B, gen_cfg["latent_dim"], device=device)
                with torch.no_grad():
                    fake_detach = generator(noise, r_norm, ph_batch)

            # ---- Discriminator / Critic update ----
            for _ in range(n_disc_steps):
                opt_D.zero_grad()

                if use_wgan:
                    score_real = discriminator(target,        r_norm if cond_disc else None)
                    score_fake = discriminator(fake_detach,   r_norm if cond_disc else None)
                    w_dist     = score_real.mean() - score_fake.mean()
                    loss_D     = -w_dist
                    if use_gp:
                        gp     = gradient_penalty(discriminator, target, fake_detach,
                                                   r_norm if cond_disc else None,
                                                   device, lam_gp)
                        loss_D = loss_D + gp
                else:
                    real_labels = torch.ones(B,  1, device=device)
                    fake_labels = torch.zeros(B, 1, device=device)
                    score_real  = discriminator(target,      r_norm if cond_disc else None)
                    score_fake  = discriminator(fake_detach, r_norm if cond_disc else None)
                    loss_D      = bce_fn(score_real, real_labels) + \
                                  bce_fn(score_fake, fake_labels)
                    w_dist      = torch.tensor(0.0)

                loss_D.backward()
                opt_D.step()

            ep_loss_d += loss_D.item()
            ep_wdist  += w_dist.item() if hasattr(w_dist, 'item') else float(w_dist)

            # ---- Generator update ----
            opt_G.zero_grad()

            if gen_option == "a":
                fake = generator(inp, r_norm, exp_ids, ph_batch)
                # MSE against the paired real target
                mse_target = target
            else:
                noise = torch.randn(B, gen_cfg["latent_dim"], device=device)
                fake  = generator(noise, r_norm, ph_batch)
                # MSE against a randomly drawn real target (not input-paired)
                # Permute within the batch to decouple noise from target
                perm       = torch.randperm(B, device=device)
                mse_target = target[perm]

            raw_mse = mse_fn(fake, mse_target)
            if metric_weights is not None:
                raw_mse = raw_mse * metric_weights.unsqueeze(0).unsqueeze(0)
            loss_rec = raw_mse.mean()

            if use_wgan:
                score_fake_g = discriminator(fake, r_norm if cond_disc else None)
                loss_adv     = -score_fake_g.mean()
            else:
                real_labels  = torch.ones(B, 1, device=device)
                score_fake_g = discriminator(fake, r_norm if cond_disc else None)
                loss_adv     = bce_fn(score_fake_g, real_labels)

            loss_G = lam_rec * loss_rec + lam_adv * loss_adv

            if use_fm:
                feat_real = discriminator.get_features(
                    target, r_norm if cond_disc else None).detach()
                feat_fake = discriminator.get_features(
                    fake,   r_norm if cond_disc else None)
                loss_G = loss_G + lam_fm * nn.functional.mse_loss(
                    feat_fake.mean(dim=0), feat_real.mean(dim=0))

            if use_var:
                # Moment matching: penalize difference in per-metric variance
                # Calibrated: normalize by real variance to avoid scale dominance
                var_real = target.var(dim=1).mean(dim=0).detach()  # (M,)
                var_fake = fake.var(dim=1).mean(dim=0)
                var_real_safe = torch.clamp(var_real, min=1e-6)
                loss_var = ((var_fake - var_real) ** 2 / var_real_safe).mean()
                loss_G   = loss_G + lam_var * loss_var

            loss_G.backward()
            nn.utils.clip_grad_norm_(generator.parameters(), gen_cfg["grad_clip"])
            opt_G.step()

            ep_rec   += loss_rec.item()
            ep_adv_g += loss_adv.item()
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
                if gen_option == "a":
                    fake = generator(inp, r_norm, exp_ids, ph_batch)
                    mse_target = target
                else:
                    noise = torch.randn(B, gen_cfg["latent_dim"], device=device)
                    fake  = generator(noise, r_norm, ph_batch)
                    mse_target = target
                raw_mse = mse_fn(fake, mse_target)
                if metric_weights is not None:
                    raw_mse = raw_mse * metric_weights.unsqueeze(0).unsqueeze(0)
                val_recs.append(raw_mse.mean().item())

        val_rec = float(np.mean(val_recs))
        sched_G.step(val_rec)

        h = {
            "epoch":   epoch,
            "loss_G":  float(ep_adv_g / n_batches),
            "loss_D":  float(ep_loss_d / n_batches),
            "loss_rec":float(ep_rec    / n_batches),
            "w_dist":  float(ep_wdist  / n_batches),
            "val_rec": val_rec,
        }
        history.append(h)

        if epoch % 20 == 0 or epoch == 1:
            print(f"    ep {epoch:>4}  val_rec={val_rec:.5f}  "
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

def evaluate(generator, gen_option, gen_cfg,
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
# ------------------------------------------------------------------

def plot_comparison(workload, real_orig, syn_orig,
                    real_r_arr, syn_r_arr, kept_names,
                    phase_seq, history, save_dir, run_label):
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
    metric_axes = [fig.add_subplot(gs[r, c]) for r, c in [(0,0),(0,1),(1,0),(1,1)]]
    loss_ax     = fig.add_subplot(gs[:, 2])
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
        ax.set_title(kept_names[m_i].replace("pod_","").replace("gpu_",""), fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("timestep", fontsize=7)

    if history:
        eps  = [h["epoch"]   for h in history]
        loss_ax.plot(eps, [h["loss_rec"] for h in history], label="train rec",
                     color="#1f77b4")
        loss_ax.plot(eps, [h["val_rec"]  for h in history], label="val rec",
                     color="#1f77b4", linestyle="--")
        loss_ax.plot(eps, [h["loss_G"]   for h in history], label="adv (gen)",
                     color="#ff7f0e")
        loss_ax.plot(eps, [h["loss_D"]   for h in history], label="disc loss",
                     color="#2ca02c")
        loss_ax.plot(eps, [h["w_dist"]   for h in history], label="W-dist",
                     color="#9467bd", linestyle=":")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=list(STAGE_CONFIGS.keys()), required=True)
    parser.add_argument("--gen-option", choices=["a", "b", "both"], default="both",
                        help="a=reconstruction  b=noise-input  both=run both")
    parser.add_argument("--workloads",  nargs="+", default=WORKLOADS)
    parser.add_argument("--epochs",     type=int,  default=None,
                        help="Override epoch count (used for s7)")
    parser.add_argument("--device",     default="auto")
    parser.add_argument("--seed",       type=int,  default=GEN_CFG["seed"])
    parser.add_argument("--phase-boundaries", type=str,
                        default=",".join(map(str, DEFAULT_PHASE_BOUNDARIES)))
    args = parser.parse_args()

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    stage_cfg = STAGE_CONFIGS[args.stage].copy()
    if args.epochs is not None:
        stage_cfg["epochs"] = args.epochs

    boundaries = list(map(int, args.phase_boundaries.split(",")))

    gen_options = ["a", "b"] if args.gen_option == "both" else [args.gen_option]

    print("=" * 65)
    print(f"TimeGAN Ablation: {args.stage.upper()}")
    print(f"Device     : {device}")
    print(f"Gen option : {args.gen_option}")
    print(f"Epochs     : {stage_cfg['epochs']}")
    print(f"Adversarial: {stage_cfg['adversarial']}")
    print(f"Spec norm  : {stage_cfg['spectral_norm']}")
    print(f"2-ts LR    : {stage_cfg['two_timescale_lr']}")
    print(f"n_disc     : {stage_cfg['n_disc_steps']}")
    print(f"lam_rec    : {stage_cfg['lambda_rec']}")
    print(f"lam_adv    : {stage_cfg['lambda_adv']}")
    print(f"lam_fm     : {stage_cfg['lambda_fm']}")
    print(f"lam_var    : {stage_cfg['lambda_var']}")
    print(f"lam_gp     : {stage_cfg['lambda_gp']}")
    print(f"cond_disc  : {stage_cfg['conditional_disc']}")
    print(f"preproc    : {stage_cfg['preprocessing']}")
    print("=" * 65)

    all_results = []

    for gen_opt in gen_options:
        gen_label = f"{args.stage}_gen{gen_opt.upper()}"
        out_dir   = Path(f"outputs/phase4/timegan_ablation/{args.stage}/gen{gen_opt.upper()}")
        model_dir = Path(f"models/phase4/timegan_ablation/{args.stage}/gen{gen_opt.upper()}")
        out_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*65}")
        print(f"Generator option {gen_opt.upper()}: "
              f"{'reconstruction-based (LSTM v6)' if gen_opt == 'a' else 'noise-input (true GAN)'}")
        print(f"{'='*65}")

        for workload in args.workloads:
            print(f"\n  --- Workload: {workload.upper()} ---")

            data, norm = load_raw_data(workload)
            raw_traces     = data["traces"]
            replica_counts = data["replica_counts"]
            train_idx      = data["train_idx"]
            val_idx        = data["val_idx"]
            metadata       = list(data["metadata"])
            seq_len        = raw_traces.shape[1]

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

            # Preprocessing
            if stage_cfg["preprocessing"] == "zeromean":
                proc_traces, trace_means = apply_zeromean(raw_kept, clip_stds=2.5)
                zm_traces = proc_traces
            else:
                proc_traces, mn_mm, mx_mm = apply_minmax(raw_kept)
                trace_means = np.zeros((len(proc_traces), n_metrics), dtype=np.float32)
                zm_traces   = proc_traces  # no zero-mean for min-max option

            # Generator option A needs phase feature appended to input
            # Generator option B does not use the trace as input at all,
            # but we still build zm_with_phase for the dataset (unused by B)
            zm_with_phase = append_phase_feature(zm_traces, phase_seq)

            print(f"  Kept metrics ({n_metrics}): {kept_names}")
            print(f"  Train: {len(train_idx)}  Val: {len(val_idx)}")

            # Dataset: option A uses zm_with_phase as inp, option B's inp is ignored
            inp_traces = zm_with_phase if gen_opt == "a" else zm_traces

            ds_train = WorkloadDataset(inp_traces, zm_traces, trace_means,
                                       replica_counts, experiment_ids,
                                       phase_seq, train_idx, gen_opt)
            ds_val   = WorkloadDataset(inp_traces, zm_traces, trace_means,
                                       replica_counts, experiment_ids,
                                       phase_seq, val_idx, gen_opt)
            dl_train = DataLoader(ds_train, batch_size=16, shuffle=True,  drop_last=True)
            dl_val   = DataLoader(ds_val,   batch_size=16, shuffle=False)

            # Build generator
            if gen_opt == "a":
                generator = GeneratorA(seq_len, n_metrics, GEN_CFG).to(device)
            else:
                generator = GeneratorB(seq_len, n_metrics, GEN_CFG).to(device)

            discriminator = Discriminator(
                n_metrics, DISC_CFG,
                adversarial_mode=stage_cfg["adversarial"],
                spectral_norm=stage_cfg["spectral_norm"],
                conditional=stage_cfg["conditional_disc"]
            ).to(device)

            n_params_G = sum(p.numel() for p in generator.parameters())
            n_params_D = sum(p.numel() for p in discriminator.parameters())
            print(f"  Generator params    : {n_params_G:,}")
            print(f"  Discriminator params: {n_params_D:,}")

            metric_weights = compute_metric_weights(zm_traces, train_idx, device)

            t0 = time.time()
            generator, discriminator, best_val_rec, n_epochs, history = train(
                generator, discriminator, dl_train, dl_val,
                stage_cfg, GEN_CFG, gen_opt, device, metric_weights)
            elapsed = time.time() - t0

            (real_orig, syn_orig, real_r_arr, syn_r_arr,
             vr, vr_mean, ac_diff, jump_ratio) = evaluate(
                generator, gen_opt, GEN_CFG,
                zm_traces, trace_means, raw_kept,
                replica_counts, phase_seq,
                train_idx, kept_names, norm_params, 5, device)

            ref = REFERENCE_VR.get(workload, {})
            print(f"\n  Results (gen={gen_opt.upper()}):")
            print(f"    val_rec       = {best_val_rec:.5f}")
            print(f"    var_ratio     = {vr_mean:.4f}")
            print(f"    autocorr_diff = {ac_diff:.4f}")
            print(f"    jump_ratio    = {jump_ratio:.3f}x")
            print(f"    epochs        = {n_epochs}  time = {elapsed:.1f}s")
            print(f"  Per-metric VR:")
            for j, m in enumerate(kept_names):
                flag = " <-- low" if vr[j] < 0.5 else (" <-- high" if vr[j] > 3.0 else "")
                print(f"    {m:<22} {vr[j]:.4f}{flag}")

            run_label = f"{args.stage} gen{gen_opt.upper()}"
            plot_path = plot_comparison(
                workload, real_orig, syn_orig, real_r_arr, syn_r_arr,
                kept_names, phase_seq, history,
                out_dir / "plots", run_label)
            print(f"  Plot: {plot_path}")

            wl_model_dir = model_dir / workload
            wl_model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(generator.state_dict(),     wl_model_dir / "generator.pt")
            torch.save(discriminator.state_dict(), wl_model_dir / "discriminator.pt")
            with open(wl_model_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)
            with open(wl_model_dir / "config.json", "w") as f:
                json.dump({**stage_cfg, **GEN_CFG, **DISC_CFG,
                           "gen_option": gen_opt, "workload": workload,
                           "kept_metrics": kept_names}, f, indent=2)

            all_results.append({
                "workload":             workload,
                "stage":                args.stage,
                "gen_option":           gen_opt,
                "var_ratio_mean":       vr_mean,
                "var_ratio_per_metric": vr.tolist(),
                "autocorr_diff":        ac_diff,
                "phase_jump_ratio":     jump_ratio,
                "val_rec":              best_val_rec,
                "n_epochs":             n_epochs,
                "elapsed_s":            elapsed,
                "metrics_generated":    list(kept_names),
                **{f"ref_{k}": v for k, v in ref.items()},
                **stage_cfg,
            })

    # Summary table
    print(f"\n{'='*65}")
    print(f"TIMEGAN ABLATION SUMMARY: {args.stage.upper()}")
    print(f"{'='*65}")

    for gen_opt in gen_options:
        opt_results = [r for r in all_results if r["gen_option"] == gen_opt]
        if not opt_results:
            continue
        print(f"\nGenerator option {gen_opt.upper()}: "
              f"{'reconstruction-based' if gen_opt == 'a' else 'noise-input (true GAN)'}")
        header = f"  {'Workload':<12}  {'VR':>8}  {'LSTM':>8}  {'jump_ratio':>11}"
        print(header)
        print("  " + "-" * 45)
        vr_vals = []
        for r in opt_results:
            vr   = r["var_ratio_mean"]
            vr_vals.append(vr)
            lstm = r.get("ref_lstm", 0.0)
            jr   = r["phase_jump_ratio"]
            print(f"  {r['workload']:<12}  {vr:>8.4f}  {lstm:>8.4f}  {jr:>11.3f}x")
        print("  " + "-" * 45)
        print(f"  {'MEAN':<12}  {float(np.mean(vr_vals)):>8.4f}")

    print(f"\nReference:")
    print(f"  LSTM mean VR = 0.612")
    print(f"  S2 mean VR   = 0.511  (best original GAN run)")
    print(f"  S4 mean VR   = 0.322  (original WGAN-GP)")

    results_path = Path(f"outputs/phase4/timegan_ablation/{args.stage}/results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults: {results_path}")


if __name__ == "__main__":
    main()