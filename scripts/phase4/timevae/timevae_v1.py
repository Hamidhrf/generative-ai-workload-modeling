#!/usr/bin/env python3
"""
Phase 4 - TimeVAE: Conditional Variational Autoencoder for Workload Trace Generation
Master Thesis - Generative AI Workload Modeling
Fachhochschule Dortmund

WHAT THIS IS
============
A conditional VAE that trains on FULL min-max normalized traces (not zeromean).
The encoder maps a pod trace to a latent distribution (mu, logvar).
The decoder reconstructs the full trace conditioned on:
  - a sampled latent z
  - replica count embedding (r)
  - phase embedding at every timestep

WHY VAE OVER LSTM BASELINE
===========================
The LSTM baseline (VR=0.612) trains on zeromean deviations and adds a
per-r mean back at generation time. This splits the learning task:
  - model learns deviations
  - post-hoc logic handles mean levels

TimeVAE trains end-to-end on full traces. The decoder directly predicts
absolute [0,1] normalized values, so:
  - r-dependent mean levels are learned inside the model
  - spike amplitudes (gpu_utilization, psi_cpu) are in the training target
  - no post-hoc mean correction needed

The VAE latent space encodes both the level AND the temporal dynamics of
each pod's trace. At generation time we sample z ~ N(0,I) and decode
with the target r and phase schedule.

KEY DESIGN CHOICES
==================
1. Train on RAW (min-max normalized, not zeromean) traces from data/processed/phase4/raw/
2. Encoder: BiLSTM over full sequence -> (mu, logvar) of shape (latent_dim,)
3. Decoder: LSTM initialized from z, conditioned on r_embed at every step,
            phase_embed[t] appended at every step
4. KL annealing: beta starts at 0 and ramps to beta_max over warmup epochs.
   This prevents posterior collapse (latent z ignored) which is the main
   failure mode of VAEs on time series.
5. Sigmoid output: keeps generated values in [0,1] consistent with min-max normalization.
   (LSTM used tanh on zeromean, which was correct for deviations. Here we use sigmoid
   on the full trace.)
6. Reconstruction loss: MSE with per-metric inverse-variance weighting (same as LSTM).
   Impulsive metrics (gpu_utilization, psi_cpu) have higher variance, so their raw
   MSE weight is naturally lower -- but unlike LSTM, the model sees the actual spike
   amplitudes as training targets.

METRIC SELECTION
================
Identical to LSTM v6. Same DROP_PER_WORKLOAD, same post-hoc tables.
The evaluation script and VR/AC metrics are also carried forward unchanged.

ARCHITECTURE
============
  Encoder:
    BiLSTM(M, enc_hidden) -> last hidden state -> FC -> (mu, logvar) (latent_dim,)

  Decoder:
    z (latent_dim) + r_embed (r_embed_dim) -> init_h, init_c
    At each timestep t:
      input = concat(prev_output, r_embed, phase_embed[t])
    LSTM(input_dim, dec_hidden) -> FC -> sigmoid -> M values

  This is a "generative LSTM decoder" rather than a "recurrent decoder with
  teacher forcing." At generation time the decoder autoregressively conditions
  on its own previous output, allowing it to reproduce temporal dependencies.

GENERATION (no teacher forcing)
=================================
  z ~ N(0, I)
  h0, c0 = decode_init(z, r_embed)
  x_t = zeros (M)      # start token
  for t in 0..T-1:
      inp_t = concat(x_t, r_embed, phase_embed[t])
      x_t, (h, c) = lstm_step(inp_t, (h, c))
      x_t = sigmoid(fc_out(x_t))
      output[t] = x_t

Usage
-----
    python scripts/phase4/timevae_v1.py

    # Single workload quick test
    python scripts/phase4/timevae_v1.py --workloads bert --epochs 50

    # Full run all workloads
    python scripts/phase4/timevae_v1.py --epochs 300
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
OUTPUT_DIR   = Path("outputs/phase4/timevae")
MODEL_DIR    = Path("models/phase4/timevae")
LSTM_RESULTS = Path("outputs/phase4/lstm/step4_phase/step4_results.json")

WORKLOADS = ["bert", "gpt2", "resnet152", "whisper", "yolo"]

ALL_METRICS = [
    "pod_cpu_usage", "pod_memory_bytes", "pod_psi_cpu",
    "pod_latency_avg", "pod_throughput",
    "gpu_utilization", "gpu_memory_used", "gpu_memory_total",
    "gpu_power_watts", "gpu_temperature",
]

# Identical to LSTM v6 - carried forward unchanged
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

# Phase boundaries from experiment runner (run_experiment_v3.py)
DEFAULT_PHASE_BOUNDARIES = [0, 96, 180, 300, 420, 600]
N_PHASES = 6

# ------------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------------

TRAIN_CFG = {
    # Architecture
    "enc_hidden_dim":    128,
    "enc_num_layers":    2,
    "dec_hidden_dim":    128,
    "dec_num_layers":    2,
    "latent_dim":        64,
    "r_embed_dim":       16,
    "phase_embed_dim":   8,
    "dropout":           0.1,
    # Training
    "batch_size":        16,
    "epochs":            300,
    "learning_rate":     1e-3,
    "weight_decay":      1e-5,
    "patience":          25,
    "min_delta":         1e-5,
    "grad_clip":         1.0,
    "seed":              42,
    # KL annealing: beta ramps from 0 -> beta_max over warmup_epochs
    # This prevents posterior collapse where the encoder ignores the input
    # and z ~ N(0,I) regardless of the trace (a common VAE failure mode).
    "beta_max":          0.5,
    "warmup_epochs":     50,
}


# ------------------------------------------------------------------
# Phase helpers
# ------------------------------------------------------------------

def build_phase_sequence(seq_len, boundaries):
    phase_seq = np.zeros(seq_len, dtype=np.int64)
    for phase_idx, start in enumerate(boundaries):
        end = boundaries[phase_idx + 1] if phase_idx + 1 < len(boundaries) else seq_len
        phase_seq[start:end] = phase_idx
    return phase_seq


# ------------------------------------------------------------------
# Data helpers  (denormalize, posthoc carried from LSTM v6)
# ------------------------------------------------------------------

def get_kept_metrics(workload):
    drop       = DROP_PER_WORKLOAD[workload]
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


# ------------------------------------------------------------------
# Post-hoc metric reconstruction  (identical to LSTM v6)
# ------------------------------------------------------------------

def fit_posthoc_tables(raw_traces, replica_counts, kept_names, workload, norm_params):
    lookup = norm_params["params"] if "params" in norm_params else norm_params
    tables = {}
    all_m  = ALL_METRICS

    # pod_memory_bytes: constant per workload (lookup mean across all pods)
    if "pod_memory_bytes" in all_m:
        midx = all_m.index("pod_memory_bytes")
        vals = raw_traces[:, :, midx]   # still in [0,1]
        # denormalize to physical
        mn = lookup["pod_memory_bytes"]["min"]
        mx = lookup["pod_memory_bytes"]["max"]
        phys = vals * (mx - mn) + mn
        tables["pod_memory_bytes"] = float(phys.mean())

    # gpu_memory_used (non-GPT2): mean per r
    if workload != "gpt2" and "gpu_memory_used" in all_m:
        midx = all_m.index("gpu_memory_used")
        mn   = lookup["gpu_memory_used"]["min"]
        mx   = lookup["gpu_memory_used"]["max"]
        val_phys = raw_traces[:, :, midx] * (mx - mn) + mn
        tables["gpu_memory_used_lookup"] = float(val_phys.mean())

    # gpu_power_watts: linear regression on gpu_utilization
    if "gpu_power_watts" in all_m and "gpu_utilization" in all_m:
        uidx  = all_m.index("gpu_utilization")
        pidx  = all_m.index("gpu_power_watts")
        umn, umx = lookup["gpu_utilization"]["min"], lookup["gpu_utilization"]["max"]
        pmn, pmx = lookup["gpu_power_watts"]["min"],  lookup["gpu_power_watts"]["max"]
        util  = (raw_traces[:, :, uidx] * (umx - umn) + umn).flatten()
        power = (raw_traces[:, :, pidx] * (pmx - pmn) + pmn).flatten()
        mask  = util > 0.01
        if mask.sum() >= 10:
            X = util[mask]; y = power[mask]
            a = float(np.cov(X, y)[0, 1] / (np.var(X) + 1e-10))
            b = float(y.mean() - a * X.mean())
        else:
            a, b = 0.0, float(power.mean())
        tables["power_regression"] = (a, b)

    return tables


def reconstruct_posthoc(syn_orig, kept_names, n_pods, seq_len,
                         posthoc_tables, workload):
    extra_cols  = []
    extra_names = []

    if "pod_memory_bytes" in posthoc_tables:
        val   = posthoc_tables["pod_memory_bytes"]
        col   = np.full((n_pods, seq_len, 1), val, dtype=np.float32)
        noise = np.random.normal(0, val * 0.02, (n_pods, seq_len, 1)).astype(np.float32)
        extra_cols.append(col + noise)
        extra_names.append("pod_memory_bytes")

    if "gpu_memory_used_lookup" in posthoc_tables:
        val   = posthoc_tables["gpu_memory_used_lookup"]
        col   = np.full((n_pods, seq_len, 1), val, dtype=np.float32)
        noise = np.random.normal(0, val * 0.01, (n_pods, seq_len, 1)).astype(np.float32)
        extra_cols.append(col + noise)
        extra_names.append("gpu_memory_used")

    if "power_regression" in posthoc_tables and "gpu_utilization" in kept_names:
        a, b     = posthoc_tables["power_regression"]
        util_idx = kept_names.index("gpu_utilization")
        util_col = syn_orig[:, :, util_idx]
        power    = a * util_col + b
        power    = np.maximum(power, 0.0)
        noise    = np.random.normal(0, abs(b) * 0.05,
                                    (n_pods, seq_len)).astype(np.float32)
        extra_cols.append((power + noise)[:, :, np.newaxis])
        extra_names.append("gpu_power_watts")

    if extra_cols:
        extended = np.concatenate([syn_orig] + extra_cols, axis=2)
        return extended, kept_names + extra_names
    return syn_orig, kept_names


# ------------------------------------------------------------------
# Evaluation metrics  (identical to LSTM v6)
# ------------------------------------------------------------------

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
# ------------------------------------------------------------------

class TimeVAEDataset(Dataset):
    """
    Returns full min-max normalized traces (not zeromean).
    Target = input (VAE reconstructs the same trace).
    Conditioning: r_norm scalar, phase sequence (T,) int tensor.
    """
    def __init__(self, traces, replica_counts, phase_seq, idx):
        # traces: (N, T, M) float64 in [0,1]
        self.x       = torch.tensor(traces[idx], dtype=torch.float32)
        r            = replica_counts[idx].astype(np.float32)
        self.r_norm  = torch.tensor((r - 1.0) / 9.0, dtype=torch.float32)
        self.phase_t = torch.tensor(phase_seq, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.r_norm[i], self.phase_t


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

class TimeVAEEncoder(nn.Module):
    """
    BiLSTM encoder: (N, T, M) -> (N, latent_dim) mu, (N, latent_dim) logvar.
    Uses last hidden states from both directions.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, latent_dim, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0)
        # bidirectional -> 2 * hidden_dim
        self.fc_mu     = nn.Linear(2 * hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(2 * hidden_dim, latent_dim)

    def forward(self, x):
        # x: (B, T, M)
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers * 2, B, hidden_dim)
        # Take last layer, both directions: h_n[-2] (fwd) and h_n[-1] (bwd)
        h_fwd = h_n[-2]   # (B, hidden_dim)
        h_bwd = h_n[-1]   # (B, hidden_dim)
        h     = torch.cat([h_fwd, h_bwd], dim=1)  # (B, 2*hidden_dim)
        return self.fc_mu(h), self.fc_logvar(h)


class TimeVAEDecoder(nn.Module):
    """
    Autoregressive LSTM decoder conditioned on z, r, and phase at every step.

    Init hidden state from z + r_embed.
    At each timestep: input = concat(prev_output, r_embed, phase_embed[t])
    Output: sigmoid(FC(hidden)) -> M values in [0,1]

    Autoregressive generation (no teacher forcing at generation time) allows
    the decoder to capture temporal dependencies in the output sequence.
    During TRAINING we use teacher forcing (real x_{t-1} as prev_output)
    for stable gradients. This is standard practice for VAE decoders.
    """
    def __init__(self, latent_dim, r_embed_dim, phase_embed_dim,
                 hidden_dim, num_layers, output_dim, n_phases, dropout):
        super().__init__()
        self.phase_embed = nn.Embedding(n_phases, phase_embed_dim)
        self.r_proj      = nn.Sequential(
            nn.Linear(1, r_embed_dim), nn.Tanh())

        # Init hidden/cell from z + r_embed
        cond_dim = latent_dim + r_embed_dim
        self.h0  = nn.Linear(cond_dim, num_layers * hidden_dim)
        self.c0  = nn.Linear(cond_dim, num_layers * hidden_dim)

        # LSTM input: prev_output + r_embed + phase_embed
        lstm_input = output_dim + r_embed_dim + phase_embed_dim
        self.lstm  = nn.LSTM(
            lstm_input, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0)
        self.fc_out    = nn.Linear(hidden_dim, output_dim)

        self.output_dim   = output_dim
        self.hidden_dim   = hidden_dim
        self.num_layers   = num_layers
        self.r_embed_dim  = r_embed_dim

    def _init_hidden(self, z, r_embed, device):
        cond  = torch.cat([z, r_embed], dim=1)  # (B, latent+r_embed)
        B     = z.size(0)
        h0    = self.h0(cond).view(B, self.num_layers, self.hidden_dim)
        h0    = h0.permute(1, 0, 2).contiguous()
        c0    = self.c0(cond).view(B, self.num_layers, self.hidden_dim)
        c0    = c0.permute(1, 0, 2).contiguous()
        return h0, c0

    def forward_teacher(self, z, r_norm, phase_seq_batch, x_real):
        """
        Teacher-forced forward pass used during training.
        x_real: (B, T, M) - real trace used as prev_output at each step.
        The input at step t is x_real[:, t-1, :] (shifted right).
        """
        B, T, M = x_real.shape
        r_embed = self.r_proj(r_norm.unsqueeze(1))   # (B, r_embed_dim)
        h, c    = self._init_hidden(z, r_embed, x_real.device)

        ph_emb  = self.phase_embed(phase_seq_batch)  # (B, T, phase_embed_dim)
        r_exp   = r_embed.unsqueeze(1).expand(B, T, -1)  # (B, T, r_embed_dim)

        # Shift x right: prepend zeros as start token
        x_shifted = torch.cat([
            torch.zeros(B, 1, M, device=x_real.device),
            x_real[:, :-1, :]
        ], dim=1)   # (B, T, M)

        lstm_in = torch.cat([x_shifted, r_exp, ph_emb], dim=2)  # (B, T, M+r+ph)
        out, _  = self.lstm(lstm_in, (h, c))
        return torch.sigmoid(self.fc_out(out))   # (B, T, M) in [0,1]

    def forward_generate(self, z, r_norm, phase_seq, device):
        """
        Autoregressive generation (no teacher forcing).
        phase_seq: (T,) int64 numpy array.
        Returns (T, M) numpy array in [0,1].
        """
        T       = len(phase_seq)
        r_embed = self.r_proj(r_norm.unsqueeze(1))   # (1, r_embed_dim)
        h, c    = self._init_hidden(z, r_embed, device)

        ph_t    = torch.tensor(phase_seq, dtype=torch.long, device=device)
        ph_emb  = self.phase_embed(ph_t)             # (T, phase_embed_dim)

        x_t   = torch.zeros(1, self.output_dim, device=device)  # start token
        outs  = []
        for t in range(T):
            inp  = torch.cat([x_t, r_embed, ph_emb[t:t+1]], dim=1)  # (1, M+r+ph)
            inp  = inp.unsqueeze(1)                                   # (1, 1, M+r+ph)
            out_h, (h, c) = self.lstm(inp, (h, c))
            x_t  = torch.sigmoid(self.fc_out(out_h.squeeze(1)))     # (1, M)
            outs.append(x_t)

        return torch.cat(outs, dim=0).cpu().numpy()   # (T, M)


class TimeVAE(nn.Module):
    def __init__(self, seq_len, n_metrics, cfg):
        super().__init__()
        self.encoder = TimeVAEEncoder(
            n_metrics, cfg["enc_hidden_dim"],
            cfg["enc_num_layers"], cfg["latent_dim"], cfg["dropout"])
        self.decoder = TimeVAEDecoder(
            cfg["latent_dim"], cfg["r_embed_dim"], cfg["phase_embed_dim"],
            cfg["dec_hidden_dim"], cfg["dec_num_layers"],
            n_metrics, N_PHASES, cfg["dropout"])
        self.latent_dim = cfg["latent_dim"]

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x, r_norm, phase_seq_batch):
        mu, logvar = self.encoder(x)
        z          = self.reparameterize(mu, logvar)
        recon      = self.decoder.forward_teacher(z, r_norm, phase_seq_batch, x)
        return recon, mu, logvar

    def generate(self, r_norm_val, n_pods, phase_seq, device):
        """
        Generate n_pods traces.
        r_norm_val: scalar float in [0,1]  (= (r-1)/9)
        phase_seq:  (T,) int64 numpy array
        Returns:    (n_pods, T, M) numpy array in [0,1]
        """
        self.eval()
        with torch.no_grad():
            traces = []
            for _ in range(n_pods):
                z      = torch.randn(1, self.latent_dim, device=device)
                r_norm = torch.tensor([r_norm_val], dtype=torch.float32, device=device)
                trace  = self.decoder.forward_generate(z, r_norm, phase_seq, device)
                traces.append(trace)  # (T, M)
        return np.stack(traces, axis=0)   # (n_pods, T, M)


# ------------------------------------------------------------------
# Loss
# ------------------------------------------------------------------

def vae_loss(recon, target, mu, logvar, beta, metric_weights=None):
    """
    ELBO loss = reconstruction loss + beta * KL divergence.

    Reconstruction: weighted MSE per metric, averaged over (B, T).
    KL: -0.5 * sum(1 + logvar - mu^2 - exp(logvar)), averaged over batch.

    beta is annealed from 0 -> beta_max over warmup_epochs.
    Low beta early on lets the decoder learn reconstruction before
    the KL term forces the latent to be Gaussian.
    """
    # MSE reconstruction
    raw_mse = (recon - target) ** 2   # (B, T, M)
    if metric_weights is not None:
        raw_mse = raw_mse * metric_weights.unsqueeze(0).unsqueeze(0)
    recon_loss = raw_mse.mean()

    # KL divergence: closed form for Gaussian prior N(0,I)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * kl, recon_loss.item(), kl.item()


def compute_metric_weights(traces, train_idx, device):
    train_data = traces[train_idx]           # (N_train, T, M)
    var_per_m  = train_data.var(axis=(0,1))  # (M,)
    var_per_m  = np.maximum(var_per_m, 1e-8)
    inv_var    = 1.0 / var_per_m
    weights    = inv_var / inv_var.mean()    # normalize: mean=1
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train(model, train_dl, val_dl, cfg, device, metric_weights=None):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg["learning_rate"],
                                 weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=8)

    best_val   = float("inf")
    best_state = None
    patience_c = 0

    for epoch in range(1, cfg["epochs"] + 1):
        # KL annealing: linearly ramp beta from 0 to beta_max
        beta = cfg["beta_max"] * min(1.0, epoch / max(cfg["warmup_epochs"], 1))

        model.train()
        for x, r_norm, phase_t in train_dl:
            x, r_norm = x.to(device), r_norm.to(device)
            # phase_t is (B, T) - same phase sequence for each sample
            phase_t   = phase_t.to(device)

            recon, mu, logvar = model(x, r_norm, phase_t)
            loss, _, _ = vae_loss(recon, x, mu, logvar, beta, metric_weights)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, r_norm, phase_t in val_dl:
                x, r_norm = x.to(device), r_norm.to(device)
                phase_t   = phase_t.to(device)
                recon, mu, logvar = model(x, r_norm, phase_t)
                loss, _, _ = vae_loss(recon, x, mu, logvar, beta, metric_weights)
                val_losses.append(loss.item())

        val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)

        # Log every 25 epochs
        if epoch % 25 == 0 or epoch == 1:
            print(f"    epoch {epoch:4d}  val_loss={val_loss:.5f}  beta={beta:.3f}")

        if val_loss < best_val - cfg["min_delta"]:
            best_val   = val_loss
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            patience_c = 0
        else:
            patience_c += 1
            if patience_c >= cfg["patience"]:
                print(f"    early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model, best_val, epoch


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate(model, raw_kept, replica_counts, train_idx,
             kept_names, norm_params, phase_seq, n_gen, device):
    """
    Generate synthetic traces and evaluate in physical units.
    No post-hoc mean correction needed: VAE generates full-scale traces.
    """
    real_norm = raw_kept[train_idx]
    real_orig = denormalize(real_norm, kept_names, norm_params)
    train_r   = replica_counts[train_idx]

    unique_r   = np.unique(train_r)
    syn_list   = []
    syn_r_list = []

    model.eval()
    for r_val in unique_r:
        r_norm_val = float((r_val - 1.0) / 9.0)
        for _ in range(n_gen):
            # Generate r_val pods in one call (one experiment worth)
            pkg = model.generate(r_norm_val, int(r_val), phase_seq, device)
            # pkg: (r_val, T, M) in [0,1]
            for pod_trace in pkg:
                syn_list.append(pod_trace)
                syn_r_list.append(int(r_val))

    syn_norm  = np.stack(syn_list, axis=0)          # (N_syn, T, M)
    syn_r_arr = np.array(syn_r_list, dtype=np.int32)
    real_r_arr = train_r

    # Denormalize synthetic traces to physical units for evaluation
    syn_orig = denormalize(syn_norm, kept_names, norm_params)

    vr, vr_mean = compute_variance_ratio(real_orig, syn_orig)
    ac_diff     = compute_autocorr_similarity(real_orig, syn_orig)
    real_jump   = detect_phase_jumps(real_orig, phase_seq)
    syn_jump    = detect_phase_jumps(syn_orig,  phase_seq)
    jump_ratio  = syn_jump / (real_jump + 1e-10)

    return (real_orig, syn_orig,
            real_r_arr, syn_r_arr,
            vr, vr_mean, ac_diff, real_jump, syn_jump, jump_ratio)


# ------------------------------------------------------------------
# Visualization  (matched-r plots, identical logic to LSTM v6)
# ------------------------------------------------------------------

def plot_comparison(workload, real_orig, syn_orig,
                    real_r_arr, syn_r_arr,
                    kept_names, phase_seq, save_dir):
    N_SHOW_R = 4
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

    common_r = sorted(set(real_r_arr.tolist()) & set(syn_r_arr.tolist()))
    if len(common_r) > N_SHOW_R:
        idx_step = max(1, (len(common_r) - 1) // (N_SHOW_R - 1))
        selected = common_r[::idx_step][:N_SHOW_R]
        if common_r[-1] not in selected:
            selected[-1] = common_r[-1]
        common_r = sorted(set(selected))

    palette  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]
    r_colour = {r: palette[i % len(palette)] for i, r in enumerate(common_r)}
    t = np.arange(real_orig.shape[1])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    legend_handles = []

    for ax_i, m_i in enumerate(plot_idx):
        ax = axes.flatten()[ax_i]
        for r_val in common_r:
            col = r_colour[r_val]
            real_idxs = np.where(real_r_arr == r_val)[0]
            if len(real_idxs):
                k = real_idxs[0]
                ax.plot(t, real_orig[k, :, m_i],
                        color=col, alpha=0.85, linewidth=0.9, linestyle="-")
            syn_idxs = np.where(syn_r_arr == r_val)[0]
            if len(syn_idxs):
                k = syn_idxs[0]
                ax.plot(t, syn_orig[k, :, m_i],
                        color=col, alpha=0.75, linewidth=0.9, linestyle="--")
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
        f"{workload.upper()} - TimeVAE  "
        f"(matched-r: solid=real, dashed=synthetic)\n"
        f"grey lines=phase boundaries",
        fontsize=10, fontweight="bold")
    if legend_handles:
        fig.legend(handles=legend_handles, loc="lower center",
                   ncol=min(len(legend_handles), 4),
                   fontsize=7, framealpha=0.7,
                   bbox_to_anchor=(0.5, -0.01))

    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{workload}_timevae.png"
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()
    return path


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workloads", nargs="+", default=WORKLOADS)
    parser.add_argument("--epochs",   type=int,  default=TRAIN_CFG["epochs"])
    parser.add_argument("--n-gen",    type=int,  default=5,
                        help="Number of synthetic experiments per r value")
    parser.add_argument("--device",   default="auto")
    parser.add_argument("--phase-boundaries", default=None,
                        help="Comma-separated phase start timesteps, e.g. 0,96,180,300,420,600")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    TRAIN_CFG["epochs"] = args.epochs

    boundaries = DEFAULT_PHASE_BOUNDARIES
    if args.phase_boundaries:
        boundaries = [int(x) for x in args.phase_boundaries.split(",")]

    # Load LSTM baseline results for comparison
    lstm_vr = {}
    if LSTM_RESULTS.exists():
        with open(LSTM_RESULTS) as f:
            lstm_data = json.load(f)
        for r in lstm_data:
            lstm_vr[r["workload"]] = r["var_ratio_mean"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "plots").mkdir(parents=True, exist_ok=True)

    phase_seq = build_phase_sequence(715, boundaries)

    np.random.seed(TRAIN_CFG["seed"])
    torch.manual_seed(TRAIN_CFG["seed"])

    all_results = []

    for workload in args.workloads:
        print(f"\n{'='*60}")
        print(f"  {workload.upper()}")
        print(f"{'='*60}")

        data, norm = load_raw_data(workload)
        traces         = data["traces"].astype(np.float64)   # (55, 715, 10)
        replica_counts = data["replica_counts"]
        train_idx      = data["train_idx"]
        val_idx        = data["val_idx"]

        kept_idx, kept_names = get_kept_metrics(workload)
        raw_kept  = traces[:, :, kept_idx].astype(np.float32)  # (55, 715, M)
        seq_len   = raw_kept.shape[1]
        n_metrics = len(kept_names)

        norm_params = norm["params"] if "params" in norm else norm

        print(f"  Metrics to generate ({n_metrics}): {kept_names}")
        print(f"  Train pods: {len(train_idx)}  Val pods: {len(val_idx)}")
        print(f"  Replica counts in train: {sorted(np.unique(replica_counts[train_idx]).tolist())}")

        # Fit post-hoc tables from training data
        posthoc_tables = fit_posthoc_tables(
            traces[train_idx], replica_counts[train_idx],
            kept_names, workload, norm_params)

        # Per-metric inverse-variance weights from full (not zeromean) training traces
        metric_weights = compute_metric_weights(raw_kept, train_idx, device)
        print(f"  Metric weights (inv-var): "
              + "  ".join(f"{kept_names[i].replace('pod_','').replace('gpu_','')}"
                          f"={metric_weights[i].item():.2f}"
                          for i in range(n_metrics)))

        # Datasets and dataloaders
        ds_train = TimeVAEDataset(raw_kept, replica_counts, phase_seq, train_idx)
        ds_val   = TimeVAEDataset(raw_kept, replica_counts, phase_seq, val_idx)
        dl_train = DataLoader(ds_train, batch_size=TRAIN_CFG["batch_size"], shuffle=True)
        dl_val   = DataLoader(ds_val,   batch_size=TRAIN_CFG["batch_size"])

        # Build model
        t0    = time.time()
        model = TimeVAE(seq_len, n_metrics, TRAIN_CFG).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {n_params:,}")

        model, best_val, n_epochs = train(
            model, dl_train, dl_val, TRAIN_CFG, device, metric_weights)
        elapsed = time.time() - t0

        # Evaluate
        (real_orig, syn_orig,
         real_r_arr, syn_r_arr,
         vr, vr_mean,
         ac_diff, real_jump, syn_jump, jump_ratio) = evaluate(
            model, raw_kept, replica_counts, train_idx,
            kept_names, norm_params, phase_seq, args.n_gen, device)

        print(f"\n  Results:")
        print(f"    best_val_loss = {best_val:.5f}")
        print(f"    var_ratio     = {vr_mean:.4f}  (target: close to 1.0)")
        print(f"    autocorr_diff = {ac_diff:.4f}  (lower = better)")
        print(f"    phase_jump_ratio (real/syn): {real_jump:.3f} / {syn_jump:.3f}"
              f"  -> {jump_ratio:.2f}x")
        print(f"    epochs        = {n_epochs}  time = {elapsed:.1f}s")
        print(f"  Per-metric VR:")
        for j, m in enumerate(kept_names):
            flag = " <-- low" if vr[j] < 0.3 else ("" if vr[j] < 3.0 else " <-- high")
            lstm_m = ""
            print(f"    {m:<22} {vr[j]:.4f}{flag}{lstm_m}")

        if workload in lstm_vr:
            delta = vr_mean - lstm_vr[workload]
            print(f"  LSTM baseline VR: {lstm_vr[workload]:.4f}  "
                  f"(delta: {delta:+.4f}  {'improved' if delta > 0 else 'regressed'})")

        # Post-hoc reconstruction
        n_syn_pods = syn_orig.shape[0]
        syn_extended, extended_names = reconstruct_posthoc(
            syn_orig, list(kept_names), n_syn_pods, seq_len,
            posthoc_tables, workload)

        posthoc_added = [m for m in extended_names if m not in kept_names]
        if posthoc_added:
            print(f"  Post-hoc appended: {posthoc_added}")
            print(f"  Full output shape: {syn_extended.shape}")

        # Plot
        plot_path = plot_comparison(
            workload, real_orig, syn_orig,
            real_r_arr, syn_r_arr,
            kept_names, phase_seq, OUTPUT_DIR / "plots")
        print(f"  Plot: {plot_path}")

        # Save model
        mdir = MODEL_DIR / workload
        mdir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), mdir / "model.pt")
        cfg_out = {**TRAIN_CFG,
                   "workload":         workload,
                   "kept_metrics":     kept_names,
                   "n_metrics":        n_metrics,
                   "phase_boundaries": boundaries,
                   "seq_len":          seq_len}
        with open(mdir / "config.json", "w") as f:
            json.dump(cfg_out, f, indent=2)

        # Serialize posthoc tables
        posthoc_serializable = {}
        if "pod_memory_bytes" in posthoc_tables:
            posthoc_serializable["pod_memory_bytes_mean"] = posthoc_tables["pod_memory_bytes"]
        if "gpu_memory_used_lookup" in posthoc_tables:
            posthoc_serializable["gpu_memory_used_lookup"] = posthoc_tables["gpu_memory_used_lookup"]
        if "power_regression" in posthoc_tables:
            posthoc_serializable["power_regression_a"] = posthoc_tables["power_regression"][0]
            posthoc_serializable["power_regression_b"] = posthoc_tables["power_regression"][1]
        with open(mdir / "posthoc_tables.json", "w") as f:
            json.dump(posthoc_serializable, f, indent=2)

        all_results.append({
            "workload":             workload,
            "model":                "timevae_v1",
            "best_val_loss":        best_val,
            "var_ratio_mean":       vr_mean,
            "var_ratio_per_metric": vr.tolist(),
            "autocorr_diff":        ac_diff,
            "phase_jump_ratio":     jump_ratio,
            "real_jump_ratio":      real_jump,
            "syn_jump_ratio":       syn_jump,
            "metrics_generated":    list(kept_names),
            "metrics_posthoc":      posthoc_added,
            "metrics_full_output":  extended_names,
            "n_metrics_generated":  n_metrics,
            "n_metrics_total":      len(extended_names),
            "n_epochs":             n_epochs,
            "elapsed_s":            elapsed,
            "lstm_var_ratio":       lstm_vr.get(workload),
            "posthoc_tables":       posthoc_serializable,
        })

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*65}")
    print("TIMEVAE SUMMARY")
    print(f"{'='*65}")
    col = 12
    header = (f"{'Workload':<12}  {'VarRatio':>{col}}  {'AC_diff':>{col}}"
              f"  {'JumpRatio':>{col}}  {'LSTM VR':>{col}}  {'Delta':>{col}}")
    print(header)
    print("-" * len(header))
    for r in all_results:
        lstm_r = r["lstm_var_ratio"]
        delta  = (r["var_ratio_mean"] - lstm_r) if lstm_r else float("nan")
        lstm_s = f"{lstm_r:.4f}" if lstm_r else "N/A"
        delta_s = f"{delta:+.4f}" if not np.isnan(delta) else "N/A"
        print(f"  {r['workload']:<12}"
              f"  {r['var_ratio_mean']:>{col}.4f}"
              f"  {r['autocorr_diff']:>{col}.4f}"
              f"  {r['phase_jump_ratio']:>{col}.4f}"
              f"  {lstm_s:>{col}}"
              f"  {delta_s:>{col}}")
    print("-" * len(header))
    vr_vals = [r["var_ratio_mean"] for r in all_results]
    ac_vals = [r["autocorr_diff"]  for r in all_results]
    print(f"  {'MEAN':<12}"
          f"  {np.mean(vr_vals):>{col}.4f}"
          f"  {np.mean(ac_vals):>{col}.4f}")

    print()
    print("  VarRatio: target 0.8-1.5  (1.0 = perfect variance match)")
    print("  AC_diff:  lower is better")
    print("  JumpRatio: syn_jump / real_jump  (>=0.5 = phase transitions reproduced)")
    if lstm_vr:
        print(f"  LSTM baseline mean VR: {np.mean(list(lstm_vr.values())):.4f}")

    out_path = OUTPUT_DIR / "timevae_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults: {out_path}")
    print(f"Plots:   {OUTPUT_DIR}/plots/")
    print(f"Models:  {MODEL_DIR}/")


if __name__ == "__main__":
    main()