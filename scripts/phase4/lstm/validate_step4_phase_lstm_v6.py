#!/usr/bin/env python3
"""
Phase 4 - Step 4: Phase-Conditioned LSTM (v6 - matched-r plots) with Revised Metric Set
Master Thesis - Generative AI Workload Modeling
Fachhochschule Dortmund

WHAT CHANGED FROM STEPS 1-3
============================
1. Revised metric drops (flatness analysis results):
   - gpu_temperature dropped from ALL workloads (flat_ratio 0.028-0.053)
   - pod_throughput dropped from Whisper only (flat_ratio 0.048)
   - gpu_memory_used dropped from GPT2 (was borderline 0.136, now consistent)

   Final metric counts:
     BERT      6: cpu_usage, psi_cpu, latency_avg, throughput, gpu_util, gpu_power
     GPT2      7: cpu_usage, psi_cpu, latency_avg, throughput, gpu_util, gpu_memory_used, gpu_power
     ResNet152 6: same as BERT
     Whisper   5: cpu_usage, psi_cpu, latency_avg, gpu_util, gpu_power
     YOLO      6: same as BERT

2. Phase conditioning added:
   The experiment runner drives six load phases during each 60-minute experiment.
   Timestep boundaries are deterministic. Each timestep gets a phase index 0-5
   which is appended to the model input as an additional feature.

   Phase schedule from run_experiment_v3.py (Business Day, 5s intervals):
     Phase 0 NIGHT:   t=0..95    (0-8 min,   0.3 req/s)
     Phase 1 RAMP:    t=96..179  (8-15 min,  2.0 req/s)
     Phase 2 MORNING: t=180..299 (15-25 min, 4.0 req/s)
     Phase 3 LUNCH:   t=300..419 (25-35 min, 1.5 req/s)
     Phase 4 PEAK:    t=420..599 (35-50 min, 5.0 req/s)
     Phase 5 EVENING: t=600..714 (50-60 min, 1.0 req/s)

   If your experiment runner uses different boundaries, set --phase-boundaries
   to a comma-separated list of 6 start timesteps, e.g.: 0,96,180,300,420,600

   At generation time, the phase sequence is deterministic (not generated) -
   the model receives the real phase schedule and generates metric values
   conditioned on knowing which phase each timestep belongs to.

ARCHITECTURE
============
  Input per timestep: (n_metrics + 1) where +1 is the phase index (0-5)
  Encoder:   LSTM -> pod latent
  Regime:    mean of pod latents from same experiment -> regime latent (Step 3)
  Decoder:   (regime_latent + pod_latent + r_embed + phase_embed) -> LSTM -> output
             The phase embedding at decode time is provided timestep-by-timestep
             via a learnable embedding table (6 phases x phase_embed_dim).

  The decoder receives phase information at every timestep so it can produce
  discrete jumps between phases rather than smooth interpolation.

WHY THIS SHOULD WORK
====================
  The staircase problem: real traces show discrete level changes at phase
  boundaries. The LSTM decoder given only a Gaussian latent has no signal
  for WHEN to jump. With phase conditioning, the decoder knows the current
  phase at every timestep. It can learn: "in phase 2, pod_throughput = X;
  in phase 4, pod_throughput = Y." The jump becomes a learned conditional
  rather than an implicit temporal structure.

  This is architecturally clean and directly defensible in the thesis:
  "We condition the generative model on the experiment's load phase schedule,
  which is a known external signal from the experiment runner. This allows
  the model to reproduce discrete contention transitions corresponding to
  load phase changes while the random latent captures between-pod variability."

CHANGES IN v2
=============
After observing BERT Step 4 results (VR=0.50, gpu_power_watts=0.02, JumpRatio=12x):

  1. Std-based zeromean clipping (clip_stds=2.5):
     Deviations beyond 2.5 std per metric are clipped. This preserves real
     latency spikes in the training signal while stopping extreme outliers
     from dominating the MSE loss and causing the decoder to smooth them away.

  2. Per-metric inverse-variance loss weighting:
     Metrics with near-zero within-trace variance (e.g. gpu_power_watts after
     zeromean for BERT) were being ignored by plain MSE. Inverse-variance
     weighting forces the model to attend to flat metrics proportionally.
     Weights are normalized so the overall loss scale is unchanged.

  3. Reduced phase_embed_dim: 8 -> 4
     JumpRatio=12x showed the phase embedding was over-dominating the decoder.
     The decoder learned to fire at every phase boundary regardless of latent
     content. Smaller embedding dampens this while keeping phase awareness.

  4. Default epochs raised: 150 -> 200

Usage
-----
    python scripts/phase4/validate_step4_phase_lstm.py

    # Quick test
    python scripts/phase4/validate_step4_phase_lstm.py --workloads bert --epochs 50

    # Custom phase boundaries (6 start timesteps, must sum to cover 715)
    python scripts/phase4/validate_step4_phase_lstm.py --phase-boundaries 0,120,240,360,480,595

    # Compare with step3 best model (loads saved step3 results for reference)
    python scripts/phase4/validate_step4_phase_lstm.py --compare-step3
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
OUTPUT_DIR   = Path("outputs/phase4/step4_phase")
MODEL_DIR    = Path("models/phase4/step4_phase")
STEP3_RESULTS = Path("outputs/phase4/step_validation/step_validation_results.json")

WORKLOADS = ["bert", "gpt2", "resnet152", "whisper", "yolo"]

ALL_METRICS = [
    "pod_cpu_usage", "pod_memory_bytes", "pod_psi_cpu",
    "pod_latency_avg", "pod_throughput",
    "gpu_utilization", "gpu_memory_used", "gpu_memory_total",
    "gpu_power_watts", "gpu_temperature",
]

# Final metric classification (v3) based on thesis requirements and flatness analysis
#
# GENERATE (model trains on these - have genuine temporal dynamics):
#   pod_cpu_usage, pod_psi_cpu, pod_latency_avg, pod_throughput, gpu_utilization
#   + gpu_memory_used for GPT2 only (dynamic KV cache, flat_ratio=0.136)
#
# POST-HOC (added after generation via lookup/regression, not generated):
#   pod_memory_bytes    -> lookup table per workload (constant model weights)
#   gpu_memory_used     -> lookup table per (workload, r) for non-GPT2
#   gpu_power_watts     -> linear regression on gpu_utilization
#
# DROP (no thesis value):
#   gpu_memory_total    -> hardware constant, zero temporal dynamics
#   gpu_temperature     -> not mentioned in thesis proposal, flat within-trace
#
# Thesis justification:
#   "Metrics with deterministic behavior are reconstructed analytically.
#    The generative model focuses on metrics carrying genuine temporal uncertainty."
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

# Post-hoc metrics: added to synthetic traces after generation
# These are NOT trained by the model - they are reconstructed from real data
POSTHOC_METRICS = {
    "pod_memory_bytes": "lookup_per_workload",
    "gpu_memory_used":  "lookup_per_workload_r",   # non-GPT2 only
    "gpu_power_watts":  "regression_on_utilization",
}

# Phase boundaries derived from experiment runner (run_experiment_v3.py)
# Business Day load profile, 5-second scrape interval:
#   Phase 0 NIGHT:   t=0..95    (0-8 min,   0.3 req/s)
#   Phase 1 RAMP:    t=96..179  (8-15 min,  2.0 req/s)
#   Phase 2 MORNING: t=180..299 (15-25 min, 4.0 req/s)
#   Phase 3 LUNCH:   t=300..419 (25-35 min, 1.5 req/s)
#   Phase 4 PEAK:    t=420..599 (35-50 min, 5.0 req/s)
#   Phase 5 EVENING: t=600..714 (50-60 min, 1.0 req/s)
# Total: 720 timesteps at 5s intervals = 60 min (dataset trimmed to 715)
DEFAULT_PHASE_BOUNDARIES = [0, 96, 180, 300, 420, 600]
N_PHASES = 6

TRAIN_CFG = {
    "hidden_dim":        128,
    "num_layers":        2,
    "latent_dim":        64,
    "regime_dim":        32,
    "pod_dim":           32,
    "replica_embed_dim": 8,
    "phase_embed_dim":   4,
    "dropout":           0.1,
    "batch_size":        16,
    "epochs":            200,
    "learning_rate":     1e-3,
    "weight_decay":      1e-5,
    "patience":          20,
    "min_delta":         1e-5,
    "grad_clip":         1.0,
    "seed":              42,
}


# ------------------------------------------------------------------
# Phase sequence builder
# ------------------------------------------------------------------

def build_phase_sequence(seq_len, boundaries):
    """
    Build integer phase index array of shape (seq_len,).
    boundaries: list of 6 start timesteps, e.g. [0, 120, 240, 360, 480, 595]
    """
    phase_seq = np.zeros(seq_len, dtype=np.int64)
    for phase_idx, start in enumerate(boundaries):
        end = boundaries[phase_idx + 1] if phase_idx + 1 < len(boundaries) else seq_len
        phase_seq[start:end] = phase_idx
    return phase_seq


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
    """
    Convert [0,1] normalized traces back to physical units.
    metric_names: list of metric names corresponding to axis-2 of traces_norm.
    Handles both norm file formats:
      Format A (raw files):    norm[metric]["min"] / norm[metric]["max"]
      Format B (phase4 files): norm["params"][metric]["min"] / ...
    """
    lookup = norm_params["params"] if "params" in norm_params else norm_params
    out = traces_norm.copy().astype(np.float64)
    for j, m in enumerate(metric_names):
        if m not in lookup:
            continue   # metric not in norm file (e.g. full ALL_METRICS list)
        mn = lookup[m]["min"]
        mx = lookup[m]["max"]
        out[:, :, j] = traces_norm[:, :, j] * (mx - mn) + mn
    return out


def apply_zeromean(traces, clip_stds=2.5):
    """
    Per-trace zero-mean normalization with std-based clipping.
    clip_stds: clip deviations beyond this many standard deviations per metric.
               2.5 preserves real variation (latency spikes) while preventing
               extreme outliers from dominating the MSE loss.
               Set to None to disable clipping.
    """
    means = traces.mean(axis=1)                        # (N, M)
    zm    = traces - means[:, np.newaxis, :]           # (N, T, M)

    if clip_stds is not None:
        # Per-metric std across all pods and timesteps
        metric_std = zm.std(axis=(0, 1), keepdims=True)   # (1, 1, M)
        clip_val   = clip_stds * metric_std
        zm         = np.clip(zm, -clip_val, clip_val)

    return zm.astype(np.float32), means.astype(np.float32)


def append_phase_feature(traces, phase_seq):
    """
    Append phase index as an additional feature to traces.
    traces:    (N, T, M)
    phase_seq: (T,) integer array 0..N_PHASES-1
    Returns:   (N, T, M+1) float32
    """
    N, T, M = traces.shape
    phase_f = phase_seq.astype(np.float32) / (N_PHASES - 1)   # normalize to [0,1]
    phase_f = np.broadcast_to(phase_f[np.newaxis, :, np.newaxis], (N, T, 1))
    return np.concatenate([traces, phase_f], axis=2).astype(np.float32)


# ------------------------------------------------------------------
# Post-hoc metric reconstruction
# ------------------------------------------------------------------

def fit_posthoc_lookup(raw_traces, replica_counts, metric_idx):
    """
    Fit a lookup table: mean value per replica count.
    Returns dict {r: mean_value} from real (unnormalized) traces.
    """
    table = {}
    for r in np.unique(replica_counts):
        mask = replica_counts == r
        vals = raw_traces[mask, :, metric_idx]   # (n_pods, T)
        table[int(r)] = float(vals.mean())
    return table


def fit_posthoc_power_regression(raw_traces, replica_counts,
                                  util_idx, power_idx):
    """
    Fit: power = a * utilization + b
    Simple OLS using all timesteps across all pods.
    Returns (a, b) coefficients in original (unnormalized) scale.
    """
    util  = raw_traces[:, :, util_idx].flatten()
    power = raw_traces[:, :, power_idx].flatten()
    # Remove near-zero utilization points (model not loaded yet)
    mask  = util > 0.01
    if mask.sum() < 10:
        return 0.0, float(power.mean())
    X = util[mask]
    y = power[mask]
    a = float(np.cov(X, y)[0, 1] / (np.var(X) + 1e-10))
    b = float(y.mean() - a * X.mean())
    return a, b


def reconstruct_posthoc(syn_orig, kept_names, n_pods, seq_len,
                         posthoc_tables, workload, norm_params):
    """
    Append post-hoc metrics to synthetic traces.
    syn_orig:      (n_pods, T, M) in physical units
    posthoc_tables: dict with fitted values
    Returns extended array (n_pods, T, M + n_posthoc) and updated metric names.
    """
    extra_cols   = []
    extra_names  = []

    # pod_memory_bytes: constant per workload
    if "pod_memory_bytes" in posthoc_tables:
        val     = posthoc_tables["pod_memory_bytes"]
        col     = np.full((n_pods, seq_len, 1), val, dtype=np.float32)
        noise   = np.random.normal(0, val * 0.02, (n_pods, seq_len, 1)).astype(np.float32)
        extra_cols.append(col + noise)
        extra_names.append("pod_memory_bytes")

    # gpu_memory_used: lookup per (workload, r) -- non-GPT2 only
    if "gpu_memory_used_lookup" in posthoc_tables:
        # Use mean across all r since we generate for a specific r
        val  = posthoc_tables["gpu_memory_used_lookup"]
        col  = np.full((n_pods, seq_len, 1), val, dtype=np.float32)
        noise = np.random.normal(0, val * 0.01, (n_pods, seq_len, 1)).astype(np.float32)
        extra_cols.append(col + noise)
        extra_names.append("gpu_memory_used")

    # gpu_power_watts: regression on utilization
    if "power_regression" in posthoc_tables and "gpu_utilization" in kept_names:
        a, b     = posthoc_tables["power_regression"]
        util_idx = kept_names.index("gpu_utilization")
        util_col = syn_orig[:, :, util_idx]              # (n_pods, T)
        power    = a * util_col + b                      # (n_pods, T)
        power    = np.maximum(power, 0.0)
        noise    = np.random.normal(0, abs(b) * 0.05,
                                    (n_pods, seq_len)).astype(np.float32)
        extra_cols.append((power + noise)[:, :, np.newaxis])
        extra_names.append("gpu_power_watts")

    if extra_cols:
        extended = np.concatenate([syn_orig] + extra_cols, axis=2)
        return extended, kept_names + extra_names
    return syn_orig, kept_names


def compute_variance_ratio(real, synthetic, cap=5.0):
    vr = []
    for i in range(real.shape[2]):
        var_r = np.var(real[:, :, i])
        var_s = np.var(synthetic[:, :, i])
        ratio = var_s / (var_r + 1e-10)
        vr.append(min(float(ratio), cap))
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


def detect_phase_jumps(traces, phase_seq, threshold=0.05):
    """
    Measure how much traces change at phase boundaries vs within phases.
    Returns ratio: mean(|delta at boundary|) / mean(|delta within phase|)
    Higher ratio = more pronounced phase transitions (good for us to reproduce).
    """
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
# ------------------------------------------------------------------

class Step4Dataset(Dataset):
    """
    Zeromean traces + phase feature appended + package experiment IDs.
    The phase feature is the last column in the input (M+1 features total).
    The model outputs only M metrics (no phase prediction needed).
    """
    def __init__(self, zm_with_phase, zm_traces, means,
                 replica_counts, experiment_ids, idx):
        # zm_with_phase: (N, T, M+1)
        # zm_traces:     (N, T, M)  for target reconstruction
        self.inp     = torch.tensor(zm_with_phase[idx], dtype=torch.float32)
        self.target  = torch.tensor(zm_traces[idx],     dtype=torch.float32)
        self.means   = torch.tensor(means[idx],         dtype=torch.float32)
        self.exp_ids = torch.tensor(experiment_ids[idx],dtype=torch.long)
        r = replica_counts[idx].astype(np.float32)
        self.r_norm  = torch.tensor((r - 1.0) / 9.0,   dtype=torch.float32)

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return (self.inp[idx], self.target[idx],
                self.means[idx], self.r_norm[idx], self.exp_ids[idx])


# ------------------------------------------------------------------
# Model
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
    """
    LSTM decoder that receives the phase embedding at every timestep.

    At each timestep t:
      input = concat(regime_latent, pod_latent, r_embed, phase_embed[phase_t])
    This is provided as a (B, T, cond_dim) sequence to the LSTM.

    The phase embedding is a learnable table (N_PHASES, phase_embed_dim).
    """
    def __init__(self, cond_dim, hidden_dim, num_layers,
                 output_dim, n_phases, phase_embed_dim, dropout):
        super().__init__()
        self.phase_embed = nn.Embedding(n_phases, phase_embed_dim)
        total_input = cond_dim + phase_embed_dim
        self.h0   = nn.Linear(cond_dim, num_layers * hidden_dim)
        self.c0   = nn.Linear(cond_dim, num_layers * hidden_dim)
        self.lstm = nn.LSTM(total_input, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, z, phase_seq_batch):
        """
        z:              (B, cond_dim)  - static conditioning vector
        phase_seq_batch:(B, T)         - integer phase indices per timestep
        Returns:        (B, T, output_dim)
        """
        B, T = phase_seq_batch.shape
        h = self.h0(z).view(B, self.num_layers,
                            self.hidden_dim).permute(1, 0, 2).contiguous()
        c = self.c0(z).view(B, self.num_layers,
                            self.hidden_dim).permute(1, 0, 2).contiguous()

        # Expand z across time and append phase embeddings
        z_exp    = z.unsqueeze(1).expand(B, T, -1)      # (B, T, cond_dim)
        ph_emb   = self.phase_embed(phase_seq_batch)     # (B, T, phase_embed_dim)
        inp      = torch.cat([z_exp, ph_emb], dim=2)     # (B, T, cond_dim+phase_embed_dim)

        out, _ = self.lstm(inp, (h, c))
        return self.fc_out(out)    # (B, T, output_dim)


class Step4Model(nn.Module):
    """
    Phase-conditioned package-aware LSTM autoencoder.

    Encoder input:  (n_metrics + 1) features per timestep
                    (last feature = normalized phase index)
    Decoder output: n_metrics (no phase column in target)

    Architecture:
      - Pod encoder: LSTM(M+1) -> pod_latent (pod_dim)
      - Regime:      mean of pod_latents in same experiment -> regime_proj -> regime_latent
      - r_embed:     replica_count -> r_embed_dim
      - Conditioning: concat(regime_latent, pod_latent, r_embed) -> z
      - Decoder:     z + phase_embedding[t] at each timestep -> LSTM -> M outputs
    """
    def __init__(self, seq_len, n_metrics, cfg):
        super().__init__()
        self.seq_len    = seq_len
        self.n_metrics  = n_metrics
        self.regime_dim = cfg["regime_dim"]
        self.pod_dim    = cfg["pod_dim"]

        # Encoder takes M+1 features (metrics + phase)
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
        """
        x_with_phase:   (B, T, M+1)
        r_norm:         (B,)
        exp_ids:        (B,) integer
        phase_seq_batch:(B, T) integer phase indices
        Returns:        (B, T, M) - reconstruction of the M metrics only
        """
        B = x_with_phase.size(0)

        pod_lat = self.pod_encoder(x_with_phase)   # (B, pod_dim)

        # Regime latent: mean of pod latents within same experiment
        regime_lat = torch.zeros(B, self.regime_dim, device=x_with_phase.device)
        for eid in exp_ids.unique():
            mask = (exp_ids == eid)
            if mask.sum() > 0:
                mean_pod = pod_lat[mask].mean(dim=0, keepdim=True)
                regime_lat[mask] = self.regime_proj(mean_pod).expand(mask.sum(), -1)

        r_e  = self.r_embed(r_norm.unsqueeze(1))       # (B, r_embed_dim)
        z    = torch.cat([regime_lat, pod_lat, r_e], dim=1)   # (B, cond_dim)

        out  = self.decoder(z, phase_seq_batch)        # (B, T, M)
        return torch.tanh(out)

    def generate(self, r_norm_val, n_pods, phase_seq, device):
        """
        Generate n_pods traces for a single experiment (shared regime).
        phase_seq: (T,) integer numpy array - the phase schedule to use
        """
        self.eval()
        with torch.no_grad():
            # One shared regime for all pods
            regime_pod = torch.randn(1, self.pod_dim, device=device)
            regime_lat = self.regime_proj(regime_pod).expand(n_pods, -1)

            # Individual pod latents
            pod_lat = torch.randn(n_pods, self.pod_dim, device=device)

            r_t = torch.full((n_pods,), r_norm_val, device=device)
            r_e = self.r_embed(r_t.unsqueeze(1))

            z = torch.cat([regime_lat, pod_lat, r_e], dim=1)

            # Broadcast phase_seq to batch
            ph = torch.tensor(phase_seq, dtype=torch.long, device=device)
            ph = ph.unsqueeze(0).expand(n_pods, -1)    # (n_pods, T)

            out = self.decoder(z, ph)
            return out.cpu().numpy()    # (n_pods, T, M)


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def compute_metric_weights(zm_traces, train_idx, device):
    """
    Per-metric loss weights: inverse variance weighting.
    Metrics with very small variance (e.g. flat gpu_power_watts after zeromean)
    get a HIGHER weight so the model cannot ignore them.
    Normalized so weights sum to n_metrics (preserves overall loss scale).
    """
    train_data  = zm_traces[train_idx]                     # (N_train, T, M)
    var_per_m   = train_data.var(axis=(0, 1))              # (M,)
    var_per_m   = np.maximum(var_per_m, 1e-8)              # avoid div/0
    inv_var     = 1.0 / var_per_m
    weights     = inv_var / inv_var.mean()                 # normalize: mean=1
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train(model, train_dl, val_dl, cfg, device, metric_weights=None):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg["learning_rate"],
                                 weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=7)
    criterion = nn.MSELoss(reduction="none")

    best_val   = float("inf")
    best_state = None
    patience_c = 0

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        for x_ph, target, _, r_norm, exp_ids, ph_batch in train_dl:
            x_ph, target = x_ph.to(device), target.to(device)
            r_norm       = r_norm.to(device)
            exp_ids      = exp_ids.to(device)
            ph_batch     = ph_batch.to(device)

            pred = model(x_ph, r_norm, exp_ids, ph_batch)
            raw_loss = criterion(pred, target)       # (B, T, M)
            if metric_weights is not None:
                raw_loss = raw_loss * metric_weights.unsqueeze(0).unsqueeze(0)
            loss = raw_loss.mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_ph, target, _, r_norm, exp_ids, ph_batch in val_dl:
                x_ph, target = x_ph.to(device), target.to(device)
                r_norm       = r_norm.to(device)
                exp_ids      = exp_ids.to(device)
                ph_batch     = ph_batch.to(device)
                pred = model(x_ph, r_norm, exp_ids, ph_batch)
                raw_loss = criterion(pred, target)
                if metric_weights is not None:
                    raw_loss = raw_loss * metric_weights.unsqueeze(0).unsqueeze(0)
                val_losses.append(raw_loss.mean().item())

        val_mse = float(np.mean(val_losses))
        scheduler.step(val_mse)

        if val_mse < best_val - cfg["min_delta"]:
            best_val   = val_mse
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            patience_c = 0
        else:
            patience_c += 1
            if patience_c >= cfg["patience"]:
                break

    model.load_state_dict(best_state)
    return model, best_val, epoch


# ------------------------------------------------------------------
# Dataset with phase batch column
# ------------------------------------------------------------------

class Step4DatasetFull(Dataset):
    """
    Dataset that also provides the phase sequence as a per-sample tensor.
    The phase sequence is the same for every sample (fixed experiment schedule)
    so we store it once and return it for every item.
    """
    def __init__(self, zm_with_phase, zm_traces, means,
                 replica_counts, experiment_ids, phase_seq, idx):
        self.inp     = torch.tensor(zm_with_phase[idx], dtype=torch.float32)
        self.target  = torch.tensor(zm_traces[idx],     dtype=torch.float32)
        self.means   = torch.tensor(means[idx],         dtype=torch.float32)
        self.exp_ids = torch.tensor(experiment_ids[idx],dtype=torch.long)
        r = replica_counts[idx].astype(np.float32)
        self.r_norm  = torch.tensor((r - 1.0) / 9.0,   dtype=torch.float32)
        # Same phase sequence for all samples
        self.phase_t = torch.tensor(phase_seq, dtype=torch.long)

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return (self.inp[idx], self.target[idx],
                self.means[idx], self.r_norm[idx],
                self.exp_ids[idx], self.phase_t)


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate(model, zm_with_phase, zm_traces, trace_means,
             raw_kept, replica_counts, experiment_ids,
             phase_seq, train_idx, kept_names, norm_params,
             n_gen, device):
    """
    Generate synthetic traces and evaluate in physical scale.
    """
    real_norm = raw_kept[train_idx]
    real_orig = denormalize(real_norm, kept_names, norm_params)

    train_means   = trace_means[train_idx]     # (N_train, M)
    train_r       = replica_counts[train_idx]  # (N_train,)

    # Per-r mean: mean of trace_means for pods with that replica count
    # This is the correct baseline to add back to zeromean decoder output.
    # Using grand mean (mean across all r) collapses all r-level differences
    # and causes near-zero VR for r-dependent metrics like gpu_utilization.
    unique_r   = np.unique(train_r)
    r_mean_map = {}
    for r_val in unique_r:
        mask = train_r == r_val
        r_mean_map[int(r_val)] = train_means[mask].mean(axis=0)  # (M,)

    syn_list   = []
    syn_r_list = []   # parallel list: which r_val each synthetic pod was generated at

    for r_val in unique_r:
        r_norm    = float((r_val - 1.0) / 9.0)
        r_mean    = r_mean_map[int(r_val)]   # (M,) - mean for THIS r value
        for _ in range(n_gen):
            pkg = model.generate(r_norm, int(r_val), phase_seq, device)
            for pod_trace in pkg:
                # Add back the per-r mean (not grand mean)
                s_norm = pod_trace + r_mean[np.newaxis, :]
                s_norm = np.clip(s_norm, 0.0, 1.0)
                syn_list.append(s_norm)
                syn_r_list.append(int(r_val))

    syn_norm  = np.array(syn_list)
    syn_r_arr = np.array(syn_r_list, dtype=np.int32)   # (N_syn,)
    syn_orig  = denormalize(syn_norm, kept_names, norm_params)

    # real_r_arr: replica counts for the training pods in the same order as real_orig
    real_r_arr = train_r   # (N_train,) already aligned with real_orig rows

    vr, vr_mean = compute_variance_ratio(real_orig, syn_orig)
    ac_diff     = compute_autocorr_similarity(real_orig, syn_orig)

    # Phase jump fidelity
    real_jump = detect_phase_jumps(real_orig, phase_seq)
    syn_jump  = detect_phase_jumps(syn_orig,  phase_seq)
    jump_ratio = syn_jump / (real_jump + 1e-10)

    return (real_orig, syn_orig,
            real_r_arr, syn_r_arr,
            vr, vr_mean, ac_diff, real_jump, syn_jump, jump_ratio)


# ------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------

def plot_comparison(workload, real_orig, syn_orig,
                    real_r_arr, syn_r_arr,
                    kept_names, phase_seq, save_dir):
    """
    4-panel matched-r plot: for each replica count present in both real and
    synthetic sets, draw one real trace (blue solid) and one synthetic trace
    (red dashed) at the SAME r value so the comparison is fair.

    Up to N_SHOW_R distinct r values are shown per panel, cycling through
    a perceptually distinct colour palette so each r-pair is easy to follow.
    Phase boundaries are marked with vertical grey lines.
    """
    N_SHOW_R = 4   # max r values to show per panel (keeps plot readable)

    priority = ["pod_latency_avg", "pod_cpu_usage", "gpu_utilization",
                "pod_throughput", "gpu_power_watts", "pod_psi_cpu",
                "pod_memory_bytes", "gpu_memory_used"]
    plot_idx = []
    for m in priority:
        if m in kept_names and len(plot_idx) < 4:
            plot_idx.append(kept_names.index(m))
    for i in range(len(kept_names)):
        if i not in plot_idx and len(plot_idx) < 4:
            plot_idx.append(i)

    boundaries = [t for t in range(1, len(phase_seq))
                  if phase_seq[t] != phase_seq[t - 1]]

    # Pick r values that appear in both sets; spread across the range
    common_r = sorted(set(real_r_arr.tolist()) & set(syn_r_arr.tolist()))
    if len(common_r) > N_SHOW_R:
        # Sample: lowest, highest, and evenly spaced in between
        idx_step = max(1, (len(common_r) - 1) // (N_SHOW_R - 1))
        selected = common_r[::idx_step][:N_SHOW_R]
        if common_r[-1] not in selected:
            selected[-1] = common_r[-1]
        common_r = sorted(set(selected))

    # Colour palette: one distinct colour per r value
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd",
               "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]
    r_colour = {r: palette[i % len(palette)] for i, r in enumerate(common_r)}

    t = np.arange(real_orig.shape[1])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    legend_handles = []   # build once, attach to figure legend

    for ax_i, m_i in enumerate(plot_idx):
        ax = axes.flatten()[ax_i]

        for r_val in common_r:
            col = r_colour[r_val]

            # One real pod at this r (first match)
            real_idxs = np.where(real_r_arr == r_val)[0]
            if len(real_idxs):
                k = real_idxs[0]
                line_real, = ax.plot(t, real_orig[k, :, m_i],
                                     color=col, alpha=0.85, linewidth=0.9,
                                     linestyle="-")

            # One synthetic pod at this r (first match)
            syn_idxs = np.where(syn_r_arr == r_val)[0]
            if len(syn_idxs):
                k = syn_idxs[0]
                ax.plot(t, syn_orig[k, :, m_i],
                        color=col, alpha=0.75, linewidth=0.9,
                        linestyle="--")

            # Build legend entry once (first panel only)
            if ax_i == 0 and len(real_idxs):
                import matplotlib.lines as mlines
                h = mlines.Line2D([], [], color=col, linewidth=1.2,
                                  label=f"r={r_val}  solid=real  dashed=syn")
                legend_handles.append(h)

        for b in boundaries:
            ax.axvline(x=b, color="silver", linewidth=0.8, linestyle=":")

        ax.set_title(kept_names[m_i].replace("pod_", "").replace("gpu_", ""),
                     fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("timestep", fontsize=7)

    fig.suptitle(
        f"{workload.upper()} - Step 4: Phase Conditioning  "
        f"(matched-r: solid=real, dashed=synthetic)\n"
        f"grey lines=phase boundaries",
        fontsize=10, fontweight="bold")

    if legend_handles:
        fig.legend(handles=legend_handles, loc="lower center",
                   ncol=min(len(legend_handles), 4),
                   fontsize=7, framealpha=0.7,
                   bbox_to_anchor=(0.5, -0.01))

    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{workload}_step4_phase.png"
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()
    return path


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workloads",        nargs="+", default=WORKLOADS)
    parser.add_argument("--epochs",           type=int,  default=TRAIN_CFG["epochs"])
    parser.add_argument("--n-gen",            type=int,  default=5)
    parser.add_argument("--device",           default="auto")
    parser.add_argument("--seed",             type=int,  default=42)
    parser.add_argument("--phase-boundaries", type=str,
                        default=",".join(map(str, DEFAULT_PHASE_BOUNDARIES)),
                        help="6 comma-separated start timesteps for phases")
    parser.add_argument("--compare-step3",    action="store_true",
                        help="Print step3 variance ratios alongside step4 for comparison")
    args = parser.parse_args()

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    TRAIN_CFG["epochs"] = args.epochs

    boundaries = list(map(int, args.phase_boundaries.split(",")))
    assert len(boundaries) == N_PHASES, \
        f"Need exactly {N_PHASES} phase start timesteps, got {len(boundaries)}"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load step3 results if comparison requested
    step3_vr = {}
    if args.compare_step3 and STEP3_RESULTS.exists():
        with open(STEP3_RESULTS) as f:
            s3 = json.load(f)
        for r in s3:
            if r["step"] == 3:
                step3_vr[r["workload"]] = r["var_ratio_mean"]

    all_results = []

    print("=" * 65)
    print("Phase 4 - Step 4: Phase-Conditioned LSTM (v6 - matched-r plots)")
    print(f"Device           : {device}")
    print(f"Epochs           : {args.epochs}")
    print(f"n_gen            : {args.n_gen}")
    print(f"Phase boundaries : {boundaries}")
    print("  Phase names: NIGHT(0) RAMP(1) MORNING(2) LUNCH(3) PEAK(4) EVENING(5)")
    print(f"N phases         : {N_PHASES}")
    print("=" * 65)
    print()
    print("Final metric strategy (v6):")
    print("  GENERATE:  cpu_usage, psi_cpu, latency_avg, throughput, gpu_utilization")
    print("             + gpu_memory_used for GPT2 only (dynamic KV cache)")
    print("  POST-HOC:  pod_memory_bytes (lookup), gpu_memory_used non-GPT2 (lookup),")
    print("             gpu_power_watts (regression on utilization)")
    print("  DROP:      gpu_memory_total (hardware constant), gpu_temperature (not in thesis)")
    print()

    for workload in args.workloads:
        print(f"\n{'='*65}")
        print(f"Workload: {workload.upper()}")
        print(f"{'='*65}")

        data, norm = load_raw_data(workload)
        raw_traces     = data["traces"]           # (N, 715, 10)
        replica_counts = data["replica_counts"]   # (N,)
        train_idx      = data["train_idx"]
        val_idx        = data["val_idx"]
        metadata       = list(data["metadata"])

        seq_len = raw_traces.shape[1]   # 715

        # Build phase sequence
        phase_seq = build_phase_sequence(seq_len, boundaries)
        phase_counts = [(phase_seq == p).sum() for p in range(N_PHASES)]
        print(f"  Phase timestep counts: {phase_counts}")

        # Build experiment IDs
        exp_key_to_id = {}
        experiment_ids = np.zeros(len(metadata), dtype=np.int64)
        for i, meta in enumerate(metadata):
            key = (meta.get("experiment_id")
                   if isinstance(meta, dict)
                   else f"{workload}_{replica_counts[i]}")
            if key not in exp_key_to_id:
                exp_key_to_id[key] = len(exp_key_to_id)
            experiment_ids[i] = exp_key_to_id[key]

        # Select kept metrics
        kept_idx, kept_names = get_kept_metrics(workload)
        n_metrics = len(kept_idx)
        raw_kept  = raw_traces[:, :, kept_idx]    # (N, 715, M)

        # Fit post-hoc tables from real (unnormalized) data
        # These are fitted once here and applied after generation
        raw_phys = denormalize(raw_kept, kept_names,
                               norm["params"] if "params" in norm else norm)
        posthoc_tables = {}

        # pod_memory_bytes: constant per workload
        mem_idx_raw = ALL_METRICS.index("pod_memory_bytes")
        mem_raw     = raw_traces[:, :, mem_idx_raw]
        posthoc_tables["pod_memory_bytes"] = float(mem_raw.mean())

        # gpu_memory_used: lookup per (workload, r) -- fitted on all data
        # For non-GPT2 workloads only (GPT2 keeps it in the model)
        if workload != "gpt2":
            gmem_idx_raw = ALL_METRICS.index("gpu_memory_used")
            gmem_raw     = raw_traces[:, :, gmem_idx_raw]
            # Use mean across ALL r for lookup (we query at generation time by r)
            gmem_by_r    = {}
            for r_val in np.unique(replica_counts):
                mask = replica_counts == r_val
                gmem_by_r[int(r_val)] = float(gmem_raw[mask].mean())
            posthoc_tables["gpu_memory_used_by_r"] = gmem_by_r
            # Default (mean across all r) used in reconstruct_posthoc
            posthoc_tables["gpu_memory_used_lookup"] = float(gmem_raw.mean())

        # gpu_power_watts: regression on gpu_utilization
        util_idx_raw  = ALL_METRICS.index("gpu_utilization")
        power_idx_raw = ALL_METRICS.index("gpu_power_watts")
        a, b = fit_posthoc_power_regression(
            denormalize(raw_traces, ALL_METRICS,
                        norm["params"] if "params" in norm else norm),
            replica_counts, util_idx_raw, power_idx_raw)
        posthoc_tables["power_regression"] = (a, b)

        print(f"  Post-hoc tables fitted:")
        print(f"    pod_memory_bytes  = {posthoc_tables['pod_memory_bytes']:.2f} bytes (constant)")
        if workload != "gpt2":
            r_vals = sorted(posthoc_tables["gpu_memory_used_by_r"].keys())
            r_sample = {r: f"{posthoc_tables['gpu_memory_used_by_r'][r]:.0f}" for r in r_vals[:3]}
            print(f"    gpu_memory_used   = {r_sample} ... (lookup by r)")
        print(f"    gpu_power_watts   = {a:.4f} * utilization + {b:.4f} (regression)")

        # Zeromean
        zm_traces, trace_means = apply_zeromean(raw_kept)

        # Append phase feature to input
        zm_with_phase = append_phase_feature(zm_traces, phase_seq)  # (N, 715, M+1)

        print(f"  Kept metrics ({n_metrics}): {kept_names}")
        print(f"  Input features per timestep: {n_metrics + 1} (metrics + phase)")
        print(f"  Train: {len(train_idx)}  Val: {len(val_idx)}")
        print(f"  Experiments: {len(exp_key_to_id)}")

        # Phase jump fidelity in real data (tells us how much the model
        # needs to learn to reproduce)
        real_kept_orig = denormalize(raw_kept[train_idx], kept_names,
                                     norm["params"] if "params" in norm else norm)
        real_jump = detect_phase_jumps(real_kept_orig, phase_seq)
        print(f"  Real phase jump ratio: {real_jump:.3f}  "
              f"(ratio of boundary delta to within-phase delta)")

        # Build datasets
        ds_train = Step4DatasetFull(zm_with_phase, zm_traces, trace_means,
                                    replica_counts, experiment_ids,
                                    phase_seq, train_idx)
        ds_val   = Step4DatasetFull(zm_with_phase, zm_traces, trace_means,
                                    replica_counts, experiment_ids,
                                    phase_seq, val_idx)
        dl_train = DataLoader(ds_train, batch_size=TRAIN_CFG["batch_size"],
                              shuffle=True)
        dl_val   = DataLoader(ds_val,   batch_size=TRAIN_CFG["batch_size"])

        # Build and train model
        t0    = time.time()
        model = Step4Model(seq_len, n_metrics, TRAIN_CFG).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {n_params:,}")

        metric_weights = compute_metric_weights(zm_traces, train_idx, device)
        print(f"  Metric weights (inv-var): "
              + "  ".join(f"{kept_names[i].replace('pod_','').replace('gpu_','')}"
                          f"={metric_weights[i].item():.2f}"
                          for i in range(n_metrics)))
        model, val_mse, n_epochs = train(model, dl_train, dl_val, TRAIN_CFG, device,
                                         metric_weights=metric_weights)
        elapsed = time.time() - t0

        # Evaluate
        norm_params = norm["params"] if "params" in norm else norm
        (real_orig, syn_orig,
         real_r_arr, syn_r_arr,
         vr, vr_mean,
         ac_diff, real_jump, syn_jump, jump_ratio) = evaluate(
            model, zm_with_phase, zm_traces, trace_means,
            raw_kept, replica_counts, experiment_ids,
            phase_seq, train_idx, kept_names, norm_params,
            args.n_gen, device)

        print(f"\n  Results:")
        print(f"    val_mse       = {val_mse:.5f}")
        print(f"    var_ratio     = {vr_mean:.4f}  (target: close to 1.0)")
        print(f"    autocorr_diff = {ac_diff:.4f}  (lower = better)")
        print(f"    phase_jump_ratio (real/syn): {real_jump:.3f} / {syn_jump:.3f}"
              f"  -> {jump_ratio:.2f}x  (>0.5 = reasonable jump reproduction)")
        print(f"    epochs        = {n_epochs}  time = {elapsed:.1f}s")
        print(f"  Per-metric VR:")
        for j, m in enumerate(kept_names):
            flag = " <-- low" if vr[j] < 0.3 else ("" if vr[j] < 3.0 else " <-- high")
            print(f"    {m:<22} {vr[j]:.4f}{flag}")

        if workload in step3_vr:
            print(f"  Step3 var_ratio was: {step3_vr[workload]:.4f}  "
                  f"(delta: {vr_mean - step3_vr[workload]:+.4f})")

        # Post-hoc reconstruction: append memory and power to synthetic traces
        n_syn_pods = syn_orig.shape[0]
        syn_extended, extended_names = reconstruct_posthoc(
            syn_orig, list(kept_names), n_syn_pods, seq_len,
            posthoc_tables, workload,
            norm["params"] if "params" in norm else norm)

        posthoc_added = [m for m in extended_names if m not in kept_names]
        if posthoc_added:
            print(f"  Post-hoc appended: {posthoc_added}")
            print(f"  Full output shape: {syn_extended.shape}  "
                  f"({len(extended_names)} metrics total)")

        # Plot (generated metrics only - what the model produced)
        plot_path = plot_comparison(workload, real_orig, syn_orig,
                                    real_r_arr, syn_r_arr,
                                    kept_names, phase_seq, OUTPUT_DIR / "plots")
        print(f"  Plot: {plot_path}")

        # Save model
        mdir = MODEL_DIR / workload
        mdir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), mdir / "model.pt")
        cfg_out = {**TRAIN_CFG,
                   "workload": workload,
                   "kept_metrics": kept_names,
                   "n_metrics": n_metrics,
                   "phase_boundaries": boundaries,
                   "seq_len": seq_len}
        with open(mdir / "config.json", "w") as f:
            json.dump(cfg_out, f, indent=2)

        # Serialize posthoc tables for saving
        posthoc_serializable = {}
        if "pod_memory_bytes" in posthoc_tables:
            posthoc_serializable["pod_memory_bytes_mean"] = posthoc_tables["pod_memory_bytes"]
        if "gpu_memory_used_by_r" in posthoc_tables:
            posthoc_serializable["gpu_memory_used_by_r"] = posthoc_tables["gpu_memory_used_by_r"]
        if "power_regression" in posthoc_tables:
            posthoc_serializable["power_regression_a"] = posthoc_tables["power_regression"][0]
            posthoc_serializable["power_regression_b"] = posthoc_tables["power_regression"][1]

        all_results.append({
            "workload":               workload,
            "step":                   4,
            "step_label":             "phase_conditioning_v6",
            "val_mse":                val_mse,
            "var_ratio_mean":         vr_mean,
            "var_ratio_per_metric":   vr.tolist(),
            "autocorr_diff":          ac_diff,
            "phase_jump_ratio":       jump_ratio,
            "real_jump_ratio":        real_jump,
            "syn_jump_ratio":         syn_jump,
            "metrics_generated":      list(kept_names),
            "metrics_posthoc":        posthoc_added,
            "metrics_full_output":    extended_names,
            "n_metrics_generated":    n_metrics,
            "n_metrics_total":        len(extended_names),
            "n_epochs":               n_epochs,
            "elapsed_s":              elapsed,
            "step3_var_ratio":        step3_vr.get(workload),
            "posthoc_tables":         posthoc_serializable,
        })

        # Save posthoc tables alongside model config
        with open(mdir / "posthoc_tables.json", "w") as f:
            json.dump(posthoc_serializable, f, indent=2)

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print(f"\n{'='*65}")
    print("STEP 4 SUMMARY")
    print(f"{'='*65}")
    col = 12
    header = (f"{'Workload':<12}  {'VarRatio':>{col}}  {'AC_diff':>{col}}"
              f"  {'JumpRatio':>{col}}  {'Step3 VR':>{col}}")
    print(header)
    print("-" * len(header))
    for r in all_results:
        s3 = f"{r['step3_var_ratio']:.4f}" if r["step3_var_ratio"] else "N/A"
        jr = r["phase_jump_ratio"]
        jr_flag = " *" if jr >= 0.5 else "  "
        print(f"  {r['workload']:<12}"
              f"  {r['var_ratio_mean']:>{col}.4f}"
              f"  {r['autocorr_diff']:>{col}.4f}"
              f"  {jr:>{col}.4f}{jr_flag}"
              f"  {s3:>{col}}")
    print("-" * len(header))
    vr_vals = [r["var_ratio_mean"] for r in all_results]
    ac_vals = [r["autocorr_diff"]  for r in all_results]
    jr_vals = [r["phase_jump_ratio"] for r in all_results]
    print(f"  {'MEAN':<12}"
          f"  {np.mean(vr_vals):>{col}.4f}"
          f"  {np.mean(ac_vals):>{col}.4f}"
          f"  {np.mean(jr_vals):>{col}.4f}")
    print()
    print("  VarRatio: target 0.8-1.5")
    print("  AC_diff:  lower is better (temporal autocorrelation fidelity)")
    print("  JumpRatio: syn_jump / real_jump at phase boundaries")
    print("             >= 0.5 means synthetic traces show meaningful phase transitions")
    print("  * = jump ratio >= 0.5 (phase transitions reproduced)")

    # Save results
    out_path = OUTPUT_DIR / "step4_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults: {out_path}")
    print(f"Plots:   {OUTPUT_DIR}/plots/")
    print(f"Models:  {MODEL_DIR}/")


if __name__ == "__main__":
    main()