"""
Phase 4 - TimeVAE v3: Fully-Connected Decoder (Approach 1)
Master Thesis - Generative AI Workload Modeling
Fachhochschule Dortmund

CHANGES FROM v1/v2
==================
v1 mean VR = 0.6128  (autoregressive LSTM decoder, tf=1.0)
v2 mean VR = 0.5165  (same + scheduled sampling + phase_embed=4, undertrained)

Root cause of VR ceiling in v1/v2:
  Autoregressive generation feeds the decoder's own smooth output as input
  at each timestep. Because MSE loss discourages high predictions, the decoder
  outputs near-mean values, and the next step is conditioned on that smooth
  value. Spikes cannot propagate: the input that would trigger t+1 spike
  never appears.

Fix (Approach 1 - FC decoder):
  Replace the autoregressive LSTM decoder with a fully-connected network
  that generates the entire sequence (T, M) in a single forward pass.
  Input: z (latent) + r_embed + phase_embed for every timestep (pre-computed).
  No sequential dependency means no smoothing cascade.
  The decoder sees all timesteps simultaneously and can place a spike at t=50
  without needing to have output a spike at t=49 first.

  Architecture:
    1. Compute r_embed from r_norm: Linear(1, r_embed_dim) -> Tanh
    2. Compute phase_embed for every timestep: Embedding(N_PHASES, phase_embed_dim)
       result: (T, phase_embed_dim)
    3. Expand z + r_embed to (T, latent+r_embed)
    4. Concatenate with phase_embed: (T, latent + r_embed + phase_embed)
    5. Pass through shared MLP (same weights for each timestep):
       FC(latent+r+ph -> fc_dim) -> LayerNorm -> GELU
       FC(fc_dim -> fc_dim)      -> LayerNorm -> GELU
       FC(fc_dim -> M)           -> Sigmoid
    6. Output: (T, M) in [0,1]

  The shared MLP (same weights per timestep) is equivalent to a 1x1 conv
  over time. This allows spike generation because each output timestep is
  independently computed from the latent code, not from the previous output.

Also carries forward:
  phase_embed_dim = 4  (v2 fix: prevents phase boundary explosion)
  All other infrastructure identical to v1 (encoder, loss, evaluation, plots)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

DATA_RAW_DIR  = Path("data/processed/phase4/raw")
OUTPUT_DIR    = Path("outputs/phase4/timevae_v3")
MODEL_DIR     = Path("models/phase4/timevae_v3")
LSTM_RESULTS  = Path("outputs/phase4/lstm/step4_phase/step4_results.json")
VAE_V1_RESULTS = Path("outputs/phase4/timevae/timevae_results.json")

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

TRAIN_CFG = {
    # Encoder (identical to v1: BiLSTM)
    "enc_hidden_dim":    128,
    "enc_num_layers":    2,
    "latent_dim":        64,
    "dropout":           0.1,
    # Conditioning
    "r_embed_dim":       16,
    # Fix from v2: phase_embed_dim=4 prevents phase boundary explosion
    "phase_embed_dim":   4,
    # FC decoder hidden dim
    "fc_hidden_dim":     256,
    # Training
    "batch_size":        16,
    "epochs":            300,
    "learning_rate":     1e-3,
    "weight_decay":      1e-5,
    "patience":          25,
    "min_delta":         1e-5,
    "grad_clip":         1.0,
    "seed":              42,
    # KL annealing
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
# Data helpers
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
# Post-hoc metric reconstruction (identical to v1)
# ------------------------------------------------------------------

def fit_posthoc_tables(raw_traces, replica_counts, kept_names, workload, norm_params):
    lookup = norm_params["params"] if "params" in norm_params else norm_params
    tables = {}
    all_m  = ALL_METRICS

    if "pod_memory_bytes" in all_m:
        midx = all_m.index("pod_memory_bytes")
        vals = raw_traces[:, :, midx]
        mn = lookup["pod_memory_bytes"]["min"]
        mx = lookup["pod_memory_bytes"]["max"]
        phys = vals * (mx - mn) + mn
        tables["pod_memory_bytes"] = float(phys.mean())

    if workload != "gpt2" and "gpu_memory_used" in all_m:
        midx = all_m.index("gpu_memory_used")
        mn   = lookup["gpu_memory_used"]["min"]
        mx   = lookup["gpu_memory_used"]["max"]
        val_phys = raw_traces[:, :, midx] * (mx - mn) + mn
        tables["gpu_memory_used_lookup"] = float(val_phys.mean())

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
# Evaluation metrics (identical to v1)
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
# Dataset (identical to v1)
# ------------------------------------------------------------------

class TimeVAEDataset(Dataset):
    """
    Returns (trace, r_norm, phase_seq_batch).
    trace:          (T, M) float32 in [0,1]
    r_norm:         scalar float32 in [0,1]  = (r-1)/9
    phase_seq_batch: (T,) int64
    """
    def __init__(self, traces, replica_counts, phase_seq, idx):
        self.traces   = traces[idx]                           # (N, T, M)
        self.r_norm   = ((replica_counts[idx] - 1.0) / 9.0).astype(np.float32)
        self.phase    = phase_seq

    def __len__(self):
        return len(self.traces)

    def __getitem__(self, i):
        x     = torch.tensor(self.traces[i],  dtype=torch.float32)  # (T, M)
        r     = torch.tensor(self.r_norm[i],  dtype=torch.float32)  # scalar
        phase = torch.tensor(self.phase,       dtype=torch.long)     # (T,)
        return x, r, phase

# ------------------------------------------------------------------
# Model: Encoder (identical to v1)
# ------------------------------------------------------------------

class TimeVAEEncoder(nn.Module):
    """
    BiLSTM encoder: (N, T, M) -> (N, latent_dim) mu, (N, latent_dim) logvar.
    Identical to v1.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, latent_dim, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0)
        self.fc_mu     = nn.Linear(2 * hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(2 * hidden_dim, latent_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_fwd = h_n[-2]
        h_bwd = h_n[-1]
        h     = torch.cat([h_fwd, h_bwd], dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

# ------------------------------------------------------------------
# Model: FC Decoder (Approach 1 - replaces autoregressive LSTM)
# ------------------------------------------------------------------

class FCDecoder(nn.Module):
    """
    Fully-connected decoder: generates the entire (T, M) sequence in one pass.

    No autoregression: no sequential dependency between timesteps.
    This prevents the smoothing cascade where t+1 is conditioned on t's
    smooth output and can never recover spikes.

    Forward pass per sample:
      1. r_embed  = Tanh(Linear(r_norm, r_embed_dim))          shape: (r_embed_dim,)
      2. ph_embed = Embedding(phase_seq)                        shape: (T, phase_embed_dim)
      3. z_exp    = z repeated T times                          shape: (T, latent_dim)
      4. r_exp    = r_embed repeated T times                    shape: (T, r_embed_dim)
      5. inp      = concat(z_exp, r_exp, ph_embed)              shape: (T, latent+r+ph)
      6. MLP(inp) = FC->LN->GELU -> FC->LN->GELU -> FC->Sigmoid shape: (T, M)

    The MLP weights are SHARED across timesteps (same linear layer applied to
    each row of inp independently). This is equivalent to a 1x1 convolution
    and means:
      - No temporal smoothing from recurrence
      - Each timestep independently decoded from (z, r, phase_id)
      - Spike at t=50 does not depend on output at t=49

    Spike generation mechanism:
      The latent z encodes the "style" of the trace including spike events.
      Phase embedding tells the decoder which phase we are in.
      Because the MLP sees (z, phase_id) independently at each t,
      it can output a spike at any t where z encodes high activity,
      without needing the previous timestep to have been a spike.
    """
    def __init__(self, seq_len, latent_dim, r_embed_dim, phase_embed_dim,
                 fc_hidden_dim, output_dim, n_phases):
        super().__init__()
        self.seq_len      = seq_len
        self.phase_embed  = nn.Embedding(n_phases, phase_embed_dim)
        self.r_proj       = nn.Sequential(
            nn.Linear(1, r_embed_dim), nn.Tanh())

        inp_dim = latent_dim + r_embed_dim + phase_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(inp_dim,     fc_hidden_dim),
            nn.LayerNorm(fc_hidden_dim),
            nn.GELU(),
            nn.Linear(fc_hidden_dim, fc_hidden_dim),
            nn.LayerNorm(fc_hidden_dim),
            nn.GELU(),
            nn.Linear(fc_hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, z, r_norm, phase_seq_batch):
        """
        z:              (B, latent_dim)
        r_norm:         (B,) float in [0,1]
        phase_seq_batch:(B, T) int64

        Returns: (B, T, M) in [0,1]
        """
        B, T = phase_seq_batch.shape

        r_embed = self.r_proj(r_norm.unsqueeze(1))           # (B, r_embed_dim)
        ph_emb  = self.phase_embed(phase_seq_batch)           # (B, T, phase_embed_dim)

        z_exp   = z.unsqueeze(1).expand(B, T, -1)            # (B, T, latent_dim)
        r_exp   = r_embed.unsqueeze(1).expand(B, T, -1)      # (B, T, r_embed_dim)

        inp     = torch.cat([z_exp, r_exp, ph_emb], dim=2)   # (B, T, latent+r+ph)

        # Apply shared MLP to each timestep independently
        # Reshape to (B*T, inp_dim), apply MLP, reshape back
        B_T = B * T
        inp_flat = inp.reshape(B_T, -1)                      # (B*T, inp_dim)
        out_flat = self.mlp(inp_flat)                         # (B*T, M)
        return out_flat.reshape(B, T, -1)                     # (B, T, M)

    def generate(self, z, r_norm_val, phase_seq, device):
        """
        Generate a single trace.
        z:           (1, latent_dim)
        r_norm_val:  scalar float
        phase_seq:   (T,) int64 numpy array
        Returns:     (T, M) numpy array in [0,1]
        """
        T       = len(phase_seq)
        r_t     = torch.tensor([r_norm_val], dtype=torch.float32, device=device)
        ph_t    = torch.tensor(phase_seq, dtype=torch.long, device=device)
        ph_t    = ph_t.unsqueeze(0)    # (1, T)

        with torch.no_grad():
            out = self.forward(z, r_t, ph_t)  # (1, T, M)
        return out.squeeze(0).cpu().numpy()    # (T, M)

# ------------------------------------------------------------------
# Model: TimeVAE wrapper
# ------------------------------------------------------------------

class TimeVAE(nn.Module):
    def __init__(self, seq_len, n_metrics, cfg):
        super().__init__()
        self.encoder = TimeVAEEncoder(
            n_metrics, cfg["enc_hidden_dim"],
            cfg["enc_num_layers"], cfg["latent_dim"], cfg["dropout"])
        self.decoder = FCDecoder(
            seq_len,
            cfg["latent_dim"], cfg["r_embed_dim"], cfg["phase_embed_dim"],
            cfg["fc_hidden_dim"], n_metrics, N_PHASES)
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
        recon      = self.decoder(z, r_norm, phase_seq_batch)
        return recon, mu, logvar

    def generate(self, r_norm_val, n_pods, phase_seq, device):
        """
        Generate n_pods traces.
        r_norm_val: scalar float in [0,1]  = (r-1)/9
        phase_seq:  (T,) int64 numpy array
        Returns:    (n_pods, T, M) numpy array in [0,1]
        """
        self.eval()
        traces = []
        for _ in range(n_pods):
            z     = torch.randn(1, self.latent_dim, device=device)
            r_t   = torch.tensor([r_norm_val], dtype=torch.float32, device=device)
            trace = self.decoder.generate(z, r_t.item(), phase_seq, device)
            traces.append(trace)
        return np.stack(traces, axis=0)   # (n_pods, T, M)

# ------------------------------------------------------------------
# Loss (identical to v1)
# ------------------------------------------------------------------

def vae_loss(recon, target, mu, logvar, beta, metric_weights=None):
    raw_mse = (recon - target) ** 2
    if metric_weights is not None:
        raw_mse = raw_mse * metric_weights.unsqueeze(0).unsqueeze(0)
    recon_loss = raw_mse.mean()
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss.item(), kl.item()


def compute_metric_weights(traces, train_idx, device):
    train_data = traces[train_idx]
    var_per_m  = train_data.var(axis=(0, 1))
    var_per_m  = np.maximum(var_per_m, 1e-8)
    inv_var    = 1.0 / var_per_m
    weights    = inv_var / inv_var.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)

# ------------------------------------------------------------------
# Training (identical to v1 - no scheduled sampling)
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
        beta = cfg["beta_max"] * min(1.0, epoch / max(cfg["warmup_epochs"], 1))

        model.train()
        for x, r_norm, phase_t in train_dl:
            x, r_norm = x.to(device), r_norm.to(device)
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
# Evaluation (identical to v1)
# ------------------------------------------------------------------

def evaluate(model, raw_kept, replica_counts, train_idx,
             kept_names, norm_params, phase_seq, n_gen, device):
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
            pkg = model.generate(r_norm_val, int(r_val), phase_seq, device)
            for pod_trace in pkg:
                syn_list.append(pod_trace)
                syn_r_list.append(int(r_val))

    syn_norm  = np.stack(syn_list, axis=0)
    syn_r_arr = np.array(syn_r_list, dtype=np.int32)

    syn_orig = denormalize(syn_norm, kept_names, norm_params)

    vr, vr_mean = compute_variance_ratio(real_orig, syn_orig)
    ac_diff     = compute_autocorr_similarity(real_orig, syn_orig)
    real_jump   = detect_phase_jumps(real_orig, phase_seq)
    syn_jump    = detect_phase_jumps(syn_orig,  phase_seq)
    jump_ratio  = syn_jump / (real_jump + 1e-10)

    return (real_orig, syn_orig,
            train_r, syn_r_arr,
            vr, vr_mean, ac_diff, real_jump, syn_jump, jump_ratio)

# ------------------------------------------------------------------
# Visualization (identical to v1 except title label)
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
        f"{workload.upper()} - TimeVAE v3 FC-Decoder"
        f"  (matched-r: solid=real, dashed=synthetic)\n"
        f"grey lines=phase boundaries",
        fontsize=10, fontweight="bold")
    if legend_handles:
        fig.legend(handles=legend_handles, loc="lower center",
                   ncol=min(len(legend_handles), 4),
                   fontsize=7, framealpha=0.7,
                   bbox_to_anchor=(0.5, -0.01))

    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{workload}_timevae_v3.png"
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
    parser.add_argument("--n-gen",    type=int,  default=5)
    parser.add_argument("--device",   default="auto")
    parser.add_argument("--phase-boundaries", default=None)
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

    lstm_vr = {}
    if LSTM_RESULTS.exists():
        with open(LSTM_RESULTS) as f:
            for r in json.load(f):
                lstm_vr[r["workload"]] = r["var_ratio_mean"]

    vae_v1_vr = {}
    if VAE_V1_RESULTS.exists():
        with open(VAE_V1_RESULTS) as f:
            for r in json.load(f):
                vae_v1_vr[r["workload"]] = r["var_ratio_mean"]

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
        traces         = data["traces"].astype(np.float64)
        replica_counts = data["replica_counts"]
        train_idx      = data["train_idx"]
        val_idx        = data["val_idx"]

        kept_idx, kept_names = get_kept_metrics(workload)
        raw_kept  = traces[:, :, kept_idx].astype(np.float32)
        seq_len   = raw_kept.shape[1]
        n_metrics = len(kept_names)

        norm_params = norm["params"] if "params" in norm else norm

        print(f"  Metrics to generate ({n_metrics}): {kept_names}")
        print(f"  Train pods: {len(train_idx)}  Val pods: {len(val_idx)}")
        print(f"  Replica counts in train: {sorted(np.unique(replica_counts[train_idx]).tolist())}")

        posthoc_tables = fit_posthoc_tables(
            traces[train_idx], replica_counts[train_idx],
            kept_names, workload, norm_params)

        metric_weights = compute_metric_weights(raw_kept, train_idx, device)
        print(f"  Metric weights (inv-var): "
              + "  ".join(f"{kept_names[i].replace('pod_','').replace('gpu_','')}"
                          f"={metric_weights[i].item():.2f}"
                          for i in range(n_metrics)))

        ds_train = TimeVAEDataset(raw_kept, replica_counts, phase_seq, train_idx)
        ds_val   = TimeVAEDataset(raw_kept, replica_counts, phase_seq, val_idx)
        dl_train = DataLoader(ds_train, batch_size=TRAIN_CFG["batch_size"], shuffle=True)
        dl_val   = DataLoader(ds_val,   batch_size=TRAIN_CFG["batch_size"])

        t0    = time.time()
        model = TimeVAE(seq_len, n_metrics, TRAIN_CFG).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {n_params:,}")

        model, best_val, n_epochs = train(
            model, dl_train, dl_val, TRAIN_CFG, device, metric_weights)
        elapsed = time.time() - t0

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
            print(f"    {m:<22} {vr[j]:.4f}{flag}")

        if workload in lstm_vr:
            delta_lstm = vr_mean - lstm_vr[workload]
            delta_v1   = vr_mean - vae_v1_vr.get(workload, float("nan"))
            print(f"  LSTM baseline VR:  {lstm_vr[workload]:.4f}"
                  f"  (delta vs LSTM: {delta_lstm:+.4f})")
            if workload in vae_v1_vr:
                print(f"  TimeVAE v1 VR:     {vae_v1_vr[workload]:.4f}"
                      f"  (delta vs v1:  {delta_v1:+.4f})")

        n_syn_pods = syn_orig.shape[0]
        syn_extended, extended_names = reconstruct_posthoc(
            syn_orig, list(kept_names), n_syn_pods, seq_len,
            posthoc_tables, workload)

        posthoc_added = [m for m in extended_names if m not in kept_names]
        if posthoc_added:
            print(f"  Post-hoc appended: {posthoc_added}")
            print(f"  Full output shape: {syn_extended.shape}")

        plot_path = plot_comparison(
            workload, real_orig, syn_orig,
            real_r_arr, syn_r_arr,
            kept_names, phase_seq, OUTPUT_DIR / "plots")
        print(f"  Plot: {plot_path}")

        mdir = MODEL_DIR / workload
        mdir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), mdir / "model.pt")
        cfg_out = {**TRAIN_CFG,
                   "workload":         workload,
                   "kept_metrics":     kept_names,
                   "n_metrics":        n_metrics,
                   "phase_boundaries": boundaries,
                   "seq_len":          seq_len,
                   "decoder_type":     "fc_shared_mlp"}
        with open(mdir / "config.json", "w") as f:
            json.dump(cfg_out, f, indent=2)

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
            "model":                "timevae_v3",
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
            "vae_v1_var_ratio":     vae_v1_vr.get(workload),
            "posthoc_tables":       posthoc_serializable,
        })

    # Summary
    print(f"\n{'='*72}")
    print("TIMEVAE v3 SUMMARY  (FC decoder, phase_embed_dim=4)")
    print(f"{'='*72}")
    print(f"  {'Workload':<12}  {'VAE-v3':>8}  {'VAE-v1':>8}  {'LSTM':>8}"
          f"  {'vs v1':>8}  {'vs LSTM':>8}  {'JumpRatio':>10}")
    print("  " + "-" * 68)
    for r in all_results:
        v1  = vae_v1_vr.get(r["workload"], float("nan"))
        lv  = lstm_vr.get(r["workload"],   float("nan"))
        dv1 = r["var_ratio_mean"] - v1
        dlv = r["var_ratio_mean"] - lv
        print(f"  {r['workload']:<12}"
              f"  {r['var_ratio_mean']:>8.4f}"
              f"  {v1:>8.4f}"
              f"  {lv:>8.4f}"
              f"  {dv1:>+8.4f}"
              f"  {dlv:>+8.4f}"
              f"  {r['phase_jump_ratio']:>10.2f}x")
    print("  " + "-" * 68)
    vr_vals = [r["var_ratio_mean"] for r in all_results]
    print(f"  {'MEAN':<12}  {np.mean(vr_vals):>8.4f}"
          f"  {np.mean(list(vae_v1_vr.values())):>8.4f}"
          f"  {np.mean(list(lstm_vr.values())):>8.4f}")

    out_path = OUTPUT_DIR / "timevae_v3_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults: {out_path}")
    print(f"Plots:   {OUTPUT_DIR}/plots/")
    print(f"Models:  {MODEL_DIR}/")


if __name__ == "__main__":
    main()