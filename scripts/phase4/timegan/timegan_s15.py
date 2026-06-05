#!/usr/bin/env python3
"""
TimeGAN S15 - Remove Zeromean, Raw Normalized Targets
=======================================================
Master Thesis - Generative AI Workload Modeling
Fachhochschule Dortmund

S14 DIAGNOSIS
==============
S14 (mean VR=1.2036) fixed the flat-trace problem by removing the encoder.
Synthetic traces now show temporal dynamics and phase structure.

Two remaining problems:

PROBLEM 1: WRONG VALUE LEVELS (throughput on BERT, ResNet152, YOLO)
  S14 used zeromean preprocessing:
      proc_trace = raw_trace - per_pod_mean     # can be negative
  Then trains with sigmoid generator (output bounded to (0, 1)).
  Zeromean targets can be negative. Sigmoid can never reproduce them.
  fm_stat pushes generator mean toward the real zeromean mean (~0),
  but then evaluation adds back the per-r average mean.
  For throughput: within-trace variation spans a wide range, and the
  zeromean target for high-phase timesteps is large positive (>1 even).
  Sigmoid clips this. The generator ends up outputting wrong levels.

  Evidence: BERT throughput synthetic r=1 at ~3.4, real at ~0.4.

PROBLEM 2: LOW JUMP RATIOS (0.006 - 0.155, target > 0.3)
  LSTM decoder naturally produces smooth outputs. Real data has abrupt
  step changes at phase boundaries (e.g. BERT CPU: 0.02 -> 0.20 in 1 step).
  Smooth LSTM = low jump ratio. Not addressed in this stage, will revisit
  after the value level issue is fixed.

S15 FIX: REMOVE ZEROMEAN, TRAIN ON RAW NORMALIZED [0, 1] DATA
================================================================
Remove apply_zeromean entirely. Training targets are the raw normalized
traces which are already in [0, 1] by construction (MinMax normalization).
Generator sigmoid output is also in (0, 1).

Now training targets and generator output live in the SAME [0, 1] space:
  - fm_stat loss: matches absolute means, which are correct replica-level values
  - var_reg loss: matches absolute stds in [0, 1] space
  - Discriminator: compares real [0,1] vs fake [0,1] directly
  - No r_mean offset needed in evaluation

The r_emb + phase_emb conditioning in GeneratorB directly learns:
  "For r=1, output throughput ~0.4. For r=5, output ~0.8."
This is exactly what these embeddings are for.

Evaluation is also simplified: generator output IS the normalized trace.
Denormalize directly for plotting and VR computation. No offset addition.

CHANGES FROM S14
==================
Preprocessing:   apply_zeromean removed, raw_kept used directly as targets
Dataset:         no trace_means, no zmph - uses raw_kept[:, :, kept_idx]
Evaluation:      no r_mean addition - generator output = normalized trace
Interpolation:   simplified - no zeromean arrays needed

ALL ELSE UNCHANGED FROM S14
==============================
Architecture:    GeneratorB (encoder-free, sigmoid output)
Discriminator:   conditional biLSTM h=32 l=1 + r_norm
Losses:          log-ratio var_reg + fm_stat + smooth (post-warmup)
Warmup:          20 epochs, statistical only (fm_stat + var_reg)
lambda_var_reg:  0.3 (log-ratio)
lambda_fm_stat:  0.5
lambda_smooth:   per-workload (bert/gpt2/resnet152/yolo=0.1, whisper=0.3)
n_disc_steps:    per-workload (bert/gpt2/resnet152/yolo=2, whisper=3)
Adv epochs:      150 fixed, best W-dist checkpoint
Optimizer:       Adam lr=1e-3 (G), 2e-4 (D), betas=(0.0, 0.9), wd=1e-5

STAGE HISTORY
===============
LSTM baseline  0.612  reference
S7             0.415  under-generated (encoder mismatch)
S8             1.678  honest best, jump ratios bad (Whisper 42x)
S9             0.878  most balanced, ResNet152 0.324
S10            1.937  inflated - psi_cpu capped 4/5
S11            1.978  inflated + flat synthetics
S12            2.053  psi_cpu still capped 3/5
S14            1.204  encoder fixed, dynamics restored, level errors remain
S15            THIS   zeromean removed, raw [0,1] targets, consistent space

USAGE
-----
    python timegan_s15.py                          # all workloads
    python timegan_s15.py --workloads bert yolo    # subset
    python timegan_s15.py --var-reg 0.3
    python timegan_s15.py --fm-stat 0.5
    python timegan_s15.py --adv-epochs 150

OUTPUT
------
    outputs/phase4/timegan_s15/{run_tag}/plots/
    models/phase4/timegan_s15/{run_tag}/{workload}/
    outputs/phase4/timegan_s15/{run_tag}/results.json

    run_tag = s15_vr{var_reg*10}_fm{fm_stat*10}_ae{adv_epochs}
    e.g.      s15_vr03_fm05_ae150
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
    "lambda_adv_warmup": 0.0,
    "lambda_adv_post":   1.0,
    "lambda_fm":         0.01,
    "lambda_gp":         10.0,
    "preprocessing":     "zeromean",
}

GEN_CFG = {
    "hidden_dim":        128,
    "num_layers":        2,
    "latent_dim":        64,
    "replica_embed_dim": 16,
    "phase_embed_dim":   8,
    "dropout":           0.1,
    "grad_clip":         1.0,
    "seed":              42,
}

DISC_CFG = {
    "hidden_dim": 32,
    "num_layers": 1,
    "dropout":    0.0,
}

WORKLOAD_OVERRIDES = {
    "bert":      {"lambda_smooth": 0.1, "n_disc_steps": 2},
    "gpt2":      {"lambda_smooth": 0.1, "n_disc_steps": 2},
    "resnet152": {"lambda_smooth": 0.1, "n_disc_steps": 2},
    "whisper":   {"lambda_smooth": 0.3, "n_disc_steps": 3},
    "yolo":      {"lambda_smooth": 0.1, "n_disc_steps": 2},
}

REFERENCE_VR = {
    "bert":      {"lstm": 0.594, "s9": 1.213, "s14": 0.718},
    "gpt2":      {"lstm": 0.676, "s9": 0.664, "s14": 1.232},
    "resnet152": {"lstm": 0.607, "s9": 0.324, "s14": 0.992},
    "whisper":   {"lstm": 0.748, "s9": 0.743, "s14": 1.654},
    "yolo":      {"lstm": 0.438, "s9": 1.446, "s14": 1.422},
}

REFERENCE_JUMP = {
    "bert":      {"s8":  3.9, "s9":  8.7, "s14": 0.058},
    "gpt2":      {"s8":  3.0, "s9":  7.2, "s14": 0.155},
    "resnet152": {"s8":  6.5, "s9": 17.1, "s14": 0.024},
    "whisper":   {"s8": 42.1, "s9": 26.2, "s14": 0.019},
    "yolo":      {"s8": 10.6, "s9": 11.2, "s14": 0.006},
}

# ------------------------------------------------------------------
# Data helpers (unchanged from S12)
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
    print(f"  [INFO] Unique replica_counts: {np.unique(rc)}")
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


def build_phase_sequence(seq_len, boundaries):
    phase_seq = np.zeros(seq_len, dtype=np.int64)
    for p_idx in range(len(boundaries) - 1):
        phase_seq[boundaries[p_idx]:boundaries[p_idx + 1]] = p_idx
    if boundaries[-1] < seq_len:
        phase_seq[boundaries[-1]:] = len(boundaries) - 1
    return phase_seq


def compute_metric_weights(raw_traces, train_idx, device):
    train_data     = raw_traces[train_idx]
    var_per_metric = train_data.reshape(-1, train_data.shape[-1]).var(axis=0)
    var_per_metric = np.clip(var_per_metric, 1e-8, None)
    weights        = 1.0 / var_per_metric
    weights        = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def replica_interpolation(raw_traces, replica_counts, experiment_ids, train_idx):
    """
    Interpolate between adjacent replica counts to augment training data.
    Operates on raw_traces directly (no zeromean involved).
    """
    train_r         = replica_counts[train_idx]
    unique_r_sorted = sorted(set(int(v) for v in np.unique(train_r)))

    new_traces = []
    new_r      = []
    new_eids   = []

    for i in range(len(unique_r_sorted) - 1):
        r_lo  = unique_r_sorted[i]
        r_hi  = unique_r_sorted[i + 1]
        r_mid = int(round((r_lo + r_hi) / 2.0))
        if r_mid in set(unique_r_sorted) or r_mid <= r_lo or r_mid >= r_hi:
            continue
        idx_lo = train_idx[train_r == r_lo]
        idx_hi = train_idx[train_r == r_hi]
        n      = min(len(idx_lo), len(idx_hi))
        if n == 0:
            continue
        for k in range(n):
            new_traces.append(0.5 * raw_traces[idx_lo[k]] + 0.5 * raw_traces[idx_hi[k]])
            new_r.append(r_mid)
            new_eids.append(-1)
        print(f"    [interp] r={r_lo}+r={r_hi} -> r={r_mid}: {n} samples")

    if not new_traces:
        print(f"    [interp] 0 samples created - all midpoints already covered.")
        return raw_traces, replica_counts, experiment_ids, train_idx

    N_orig  = len(raw_traces)
    n_new   = len(new_traces)
    new_ids = np.arange(N_orig, N_orig + n_new)

    raw_aug  = np.concatenate([raw_traces,     np.array(new_traces)], axis=0)
    r_aug    = np.concatenate([replica_counts, np.array(new_r)],      axis=0)
    eids_aug = np.concatenate([experiment_ids, np.array(new_eids)],   axis=0)
    ti_aug   = np.concatenate([train_idx,      new_ids],              axis=0)
    return raw_aug, r_aug, eids_aug, ti_aug.astype(np.int64)


# ------------------------------------------------------------------
# Dataset
# S15: target is raw_kept (normalized [0,1]), no zeromean.
# Generator sigmoid output lives in the same [0,1] space.
# ------------------------------------------------------------------

class WorkloadDataset(Dataset):
    def __init__(self, raw_traces, replica_counts, experiment_ids,
                 phase_seq, indices, jitter=0):
        self.raw   = raw_traces
        self.rc    = replica_counts
        self.eids  = experiment_ids
        self.phase_seq = phase_seq
        self.indices   = indices
        self.jitter    = jitter
        self.r_max     = max(float(replica_counts.max()), 1.0)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx   = self.indices[i]
        tgt   = torch.tensor(self.raw[idx], dtype=torch.float32)
        r_val = int(self.rc[idx])   if idx < len(self.rc)   else 1
        eid   = int(self.eids[idx]) if idx < len(self.eids) else 0

        if self.jitter > 0:
            shift = np.random.randint(-self.jitter, self.jitter + 1)
            tgt   = torch.roll(tgt, shift, dims=0)

        r_norm = torch.tensor((r_val - 1.0) / 9.0, dtype=torch.float32)
        T      = tgt.shape[0]
        ph     = torch.tensor(self.phase_seq[:T], dtype=torch.long)

        return tgt, torch.tensor(r_val), r_norm, ph


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
            c = (x[:, lag:, :] * x[:, :-lag, :]).mean()
            v = (x ** 2).mean()
            out.append(float(c / (v + 1e-10)))
        return np.array(out)
    return float(np.abs(acf(real) - acf(synthetic)).mean())


def detect_phase_jumps(traces, phase_seq):
    boundaries = [t for t in range(1, len(phase_seq))
                  if phase_seq[t] != phase_seq[t - 1]]
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
# GeneratorB - encoder-free
# ------------------------------------------------------------------

class GeneratorB(nn.Module):
    """
    Encoder-free conditional generator.

    z ~ N(0, latent_dim) is sampled identically at train and generate time.
    There is no encoder. No train/test mismatch.

    z + r_emb -> MLP -> initial hidden state (h0, c0)
    per-timestep input: cat(z_expanded, r_emb_expanded, phase_emb[t])
    LSTM unrolls T steps from (h0, c0)
    output: sigmoid( fc( lstm_hidden ) )
    """

    def __init__(self, seq_len, n_metrics, cfg):
        super().__init__()
        self.seq_len   = seq_len
        self.n_metrics = n_metrics
        hidden         = cfg["hidden_dim"]
        n_layers       = cfg["num_layers"]
        latent_dim     = cfg["latent_dim"]
        r_emb_dim      = cfg["replica_embed_dim"]
        ph_emb_dim     = cfg["phase_embed_dim"]
        dropout        = cfg["dropout"] if n_layers > 1 else 0.0

        # Replica count embedding
        self.r_embed = nn.Sequential(
            nn.Linear(1, r_emb_dim),
            nn.Tanh(),
        )

        # Phase embedding (one per timestep)
        self.ph_embed = nn.Embedding(N_PHASES + 1, ph_emb_dim)

        # Initial hidden state from z and r
        init_in = latent_dim + r_emb_dim
        self.h_init = nn.Sequential(
            nn.Linear(init_in, hidden * n_layers),
            nn.Tanh(),
        )
        self.c_init = nn.Sequential(
            nn.Linear(init_in, hidden * n_layers),
            nn.Tanh(),
        )

        # Decoder LSTM: input = cat(z, r_emb, phase_emb) per timestep
        dec_in_dim = latent_dim + r_emb_dim + ph_emb_dim
        self.dec_rnn = nn.LSTM(dec_in_dim, hidden, n_layers,
                               batch_first=True, dropout=dropout)

        # Output projection
        self.out_fc  = nn.Linear(hidden, n_metrics)
        self.out_act = nn.Sigmoid()

        self.latent_dim = latent_dim
        self.n_layers   = n_layers
        self.hidden_dim = hidden

    def forward(self, r_norm, phase_ids, z=None):
        """
        Args:
            r_norm:    (B,) normalized replica count in [0, 1]
            phase_ids: (B, T) long tensor of phase indices
            z:         (B, latent_dim) optional; sampled internally if None

        Returns:
            out: (B, T, n_metrics) in [0, 1]
        """
        B  = r_norm.shape[0]
        T  = phase_ids.shape[1]
        device = r_norm.device

        if z is None:
            z = torch.randn(B, self.latent_dim, device=device)

        r_emb = self.r_embed(r_norm.unsqueeze(-1))  # (B, r_emb_dim)

        zr = torch.cat([z, r_emb], dim=-1)  # (B, latent+r_emb)

        # Build initial hidden and cell states
        h0 = self.h_init(zr)  # (B, hidden * n_layers)
        c0 = self.c_init(zr)
        h0 = h0.view(B, self.n_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        c0 = c0.view(B, self.n_layers, self.hidden_dim).permute(1, 0, 2).contiguous()

        # Per-timestep input: cat(z, r_emb, phase_emb)
        ph_emb  = self.ph_embed(phase_ids)       # (B, T, ph_emb_dim)
        z_exp   = z.unsqueeze(1).expand(-1, T, -1)
        r_exp   = r_emb.unsqueeze(1).expand(-1, T, -1)
        dec_in  = torch.cat([z_exp, r_exp, ph_emb], dim=-1)  # (B, T, dec_in_dim)

        dec_out, _ = self.dec_rnn(dec_in, (h0, c0))
        return self.out_act(self.out_fc(dec_out))  # (B, T, n_metrics)

    @torch.no_grad()
    def generate(self, r_norm_val, phase_seq, device, n_samples=1):
        """
        Generate n_samples traces for a given replica count.
        Returns array of shape (n_samples, T, n_metrics).
        """
        self.eval()
        T      = self.seq_len
        r_norm = torch.full((n_samples,), r_norm_val,
                            dtype=torch.float32, device=device)
        ph_ids = torch.tensor(phase_seq[:T], dtype=torch.long, device=device)
        ph_ids = ph_ids.unsqueeze(0).expand(n_samples, -1)
        out    = self(r_norm, ph_ids)  # (n_samples, T, n_metrics)
        return out.cpu().numpy()


# ------------------------------------------------------------------
# Discriminator (unchanged from S12 - conditional on r_norm)
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

    def get_features(self, x):
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
          base_cfg, stat_warmup, adv_epochs, gen_cfg,
          lambda_var_reg, lambda_fm_stat, lambda_smooth,
          n_disc_steps_wl, device, metric_weights=None,
          warmup_patience=15, min_delta=1e-4):
    """
    S14 training.

    Warmup (stat_warmup epochs):
        loss_G = lambda_fm_stat * fm_stat + lambda_var_reg * var_reg
        No discriminator training, no adversarial term.
        Forces generator to output correct per-replica statistics first.

    Adversarial (adv_epochs epochs):
        loss_G = adv + lambda_fm_stat * fm_stat + lambda_var_reg * var_reg
                     + lambda_smooth * smooth
        Discriminator trained n_disc_steps_wl times per generator step.
        Best W-dist checkpoint saved.
    """
    use_fm  = base_cfg.get("lambda_fm", 0.0) > 0
    lam_fm  = base_cfg.get("lambda_fm", 0.0)
    lam_gp  = base_cfg.get("lambda_gp", 10.0)
    lam_adv = base_cfg.get("lambda_adv_post", 1.0)
    betas   = (0.0, 0.9)

    opt_G   = torch.optim.Adam(generator.parameters(),
                               lr=1e-3, betas=betas, weight_decay=1e-5)
    opt_D   = torch.optim.Adam(discriminator.parameters(),
                               lr=2e-4, betas=betas, weight_decay=1e-5)
    sched_G = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_G, factor=0.5, patience=7)

    best_wdist   = float("-inf")
    best_G       = {k: v.cpu().clone() for k, v in generator.state_dict().items()}
    best_D       = {k: v.cpu().clone() for k, v in discriminator.state_dict().items()}
    best_stat    = float("inf")
    best_G_stat  = {k: v.cpu().clone() for k, v in generator.state_dict().items()}
    patience_c   = 0
    warmup_done  = False
    actual_warmup = stat_warmup
    total_epochs  = stat_warmup + adv_epochs
    history       = []

    print(f"    Statistical warmup: {stat_warmup} epochs (patience={warmup_patience})")
    print(f"    Adversarial:        {adv_epochs} fixed epochs")
    print(f"    lambda_var_reg={lambda_var_reg} (log-ratio)  "
          f"lambda_fm_stat={lambda_fm_stat}  "
          f"lambda_smooth={lambda_smooth}  n_disc={n_disc_steps_wl}")

    for epoch in range(1, total_epochs + 1):

        in_warmup    = (epoch <= actual_warmup)
        n_disc_steps = 0 if in_warmup else n_disc_steps_wl

        if not in_warmup and not warmup_done:
            warmup_done = True
            print(f"    [epoch {epoch}] Adversarial start.")

        generator.train()
        discriminator.train()
        ep_adv_g = ep_loss_d = ep_wdist = ep_vreg = ep_sm = ep_fmstat = 0.0
        ep_stat_loss = 0.0
        n_batches = 0

        for target, r_int, r_norm, ph_batch in train_dl:
            target   = target.to(device)
            r_norm   = r_norm.to(device)
            r_int    = r_int.to(device)
            ph_batch = ph_batch.to(device)

            # ----- Discriminator -----
            if n_disc_steps > 0:
                with torch.no_grad():
                    fake_d = generator(r_norm, ph_batch)
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

            # ----- Generator -----
            opt_G.zero_grad()
            fake = generator(r_norm, ph_batch)

            loss_G = torch.tensor(0.0, device=device)

            # Adversarial term (post-warmup only)
            if not in_warmup:
                loss_adv = -discriminator(fake, r_norm).mean()
                loss_G   = loss_G + lam_adv * loss_adv
                ep_adv_g += loss_adv.item()

                # LSTM feature matching
                if use_fm:
                    fr     = discriminator.get_features(target).detach()
                    ff     = discriminator.get_features(fake)
                    loss_G = loss_G + lam_fm * F.mse_loss(ff.mean(0), fr.mean(0))

            # Log-ratio variance loss (warmup + adversarial)
            if lambda_var_reg > 0:
                std_real  = target.std(dim=[0, 1]).detach()
                std_fake  = fake.std(dim=[0, 1])
                log_ratio = torch.log((std_fake + 1e-8) / (std_real + 1e-8))
                var_reg   = (log_ratio ** 2).mean()
                loss_G    = loss_G + lambda_var_reg * var_reg
                ep_vreg  += var_reg.item()

            # Per-replica statistical feature matching (warmup + adversarial)
            if lambda_fm_stat > 0:
                unique_r_vals = r_int.unique()
                fm_stat_loss  = torch.tensor(0.0, device=device)
                n_r_matched   = 0
                for r_val in unique_r_vals:
                    mask = (r_int == r_val)
                    if mask.sum() < 2:
                        continue
                    real_r  = target[mask]
                    fake_r  = fake[mask]
                    fm_mean = F.mse_loss(fake_r.mean(dim=[0, 1]),
                                         real_r.mean(dim=[0, 1]).detach())
                    fm_std  = F.mse_loss(fake_r.std(dim=[0, 1]),
                                         real_r.std(dim=[0, 1]).detach())
                    fm_stat_loss = fm_stat_loss + fm_mean + fm_std
                    n_r_matched += 1
                if n_r_matched > 0:
                    fm_stat_loss = fm_stat_loss / n_r_matched
                    loss_G       = loss_G + lambda_fm_stat * fm_stat_loss
                    ep_fmstat   += fm_stat_loss.item()

            # Temporal smoothing (adversarial only)
            if lambda_smooth > 0 and not in_warmup:
                diff   = fake[:, 1:, :] - fake[:, :-1, :]
                sm     = (diff ** 2).mean()
                loss_G = loss_G + lambda_smooth * sm
                ep_sm += sm.item()

            loss_G.backward()
            nn.utils.clip_grad_norm_(generator.parameters(),
                                     max_norm=gen_cfg["grad_clip"])
            opt_G.step()

            ep_stat_loss += (ep_fmstat + ep_vreg) / max(n_batches + 1, 1)
            n_batches    += 1

        # Validation: compute stat loss on val set
        generator.eval()
        discriminator.eval()
        val_stats = []
        with torch.no_grad():
            for target, r_int, r_norm, ph_batch in val_dl:
                target   = target.to(device)
                r_norm   = r_norm.to(device)
                r_int    = r_int.to(device)
                ph_batch = ph_batch.to(device)
                fake     = generator(r_norm, ph_batch)
                std_real  = target.std(dim=[0, 1])
                std_fake  = fake.std(dim=[0, 1])
                log_ratio = torch.log((std_fake + 1e-8) / (std_real + 1e-8))
                val_stats.append((log_ratio ** 2).mean().item())

        val_stat  = float(np.mean(val_stats))
        avg_wdist = ep_wdist / n_batches if n_batches else 0.0
        sched_G.step(val_stat)

        phase_label = "warmup" if in_warmup else "   adv"
        h = {
            "epoch":    epoch,
            "phase":    phase_label.strip(),
            "loss_G":   float(ep_adv_g   / n_batches),
            "loss_D":   float(ep_loss_d  / n_batches),
            "val_stat": val_stat,
            "w_dist":   float(avg_wdist),
            "var_reg":  float(ep_vreg    / n_batches),
            "fm_stat":  float(ep_fmstat  / n_batches),
            "smooth":   float(ep_sm      / n_batches),
        }
        history.append(h)

        if epoch % 10 == 0 or epoch == 1 or epoch == actual_warmup + 1:
            print(f"    ep {epoch:>4} [{phase_label}]  val_stat={val_stat:.5f}  "
                  f"G={h['loss_G']:.4f}  D={h['loss_D']:.4f}  "
                  f"W={avg_wdist:.4f}  vr={h['var_reg']:.4f}  "
                  f"fm={h['fm_stat']:.4f}  sm={h['smooth']:.4f}")

        if in_warmup:
            if val_stat < best_stat - min_delta:
                best_stat   = val_stat
                best_G_stat = {k: v.cpu().clone() for k, v in generator.state_dict().items()}
                best_D      = {k: v.cpu().clone() for k, v in discriminator.state_dict().items()}
                patience_c  = 0
            else:
                patience_c += 1
                if patience_c >= warmup_patience:
                    print(f"    [warmup] Patience at epoch {epoch}. Starting adv early.")
                    actual_warmup = epoch
                    warmup_done   = True
                    total_epochs  = epoch + adv_epochs
                    patience_c    = 0
        else:
            if avg_wdist > best_wdist:
                best_wdist = avg_wdist
                best_G     = {k: v.cpu().clone() for k, v in generator.state_dict().items()}

    if warmup_done and best_wdist > float("-inf"):
        generator.load_state_dict(best_G)
        print(f"    Final: best W-dist checkpoint (W={best_wdist:.4f})")
    else:
        generator.load_state_dict(best_G_stat)
        print(f"    Final: best stat checkpoint (stat={best_stat:.5f})")
    discriminator.load_state_dict(best_D)
    return generator, discriminator, best_stat, epoch, history


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate(generator, raw_kept, replica_counts, phase_seq,
             train_idx_orig, kept_names, norm_params, n_gen, device):
    """
    S15: generator output is in [0, 1] (sigmoid), same as raw_kept.
    No zeromean offset needed. Denormalize both directly for VR computation.
    """
    real_norm = raw_kept[train_idx_orig]
    real_orig = denormalize(real_norm, kept_names, norm_params)
    train_r   = replica_counts[train_idx_orig]
    unique_r  = np.unique(train_r)

    syn_list = []
    syn_r    = []
    for r_val in unique_r:
        r_norm_val = float((r_val - 1.0) / 9.0)
        pkg = generator.generate(r_norm_val, phase_seq, device, n_samples=n_gen)
        for pod_trace in pkg:
            syn_list.append(pod_trace)  # already in [0, 1]
            syn_r.append(int(r_val))

    syn_norm  = np.array(syn_list)
    syn_r_arr = np.array(syn_r, dtype=np.int32)
    syn_orig  = denormalize(syn_norm, kept_names, norm_params)

    vr, vr_mean = compute_variance_ratio(real_orig, syn_orig)
    ac_diff     = compute_autocorr_similarity(real_orig, syn_orig)
    rj          = detect_phase_jumps(real_orig, phase_seq)
    sj          = detect_phase_jumps(syn_orig,  phase_seq)
    jump_ratio  = sj / (rj + 1e-10)

    return (real_orig, syn_orig,
            replica_counts[train_idx_orig], syn_r_arr,
            vr, vr_mean, ac_diff, jump_ratio)


# ------------------------------------------------------------------
# Visualisation
# ------------------------------------------------------------------

def plot_comparison(workload, real_orig, syn_orig, real_r_arr, syn_r_arr,
                    kept_names, phase_seq, history,
                    stat_warmup, adv_epochs, save_dir, run_label):
    boundary_ts = [t for t in range(1, len(phase_seq))
                   if phase_seq[t] != phase_seq[t - 1]]

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
                    leg_labels.append(f"r={r_val}")
            if len(sidx) > 0:
                ax.plot(syn_orig[sidx[0], :, m_idx], color=clr,
                        alpha=0.85, linewidth=0.9, linestyle="--")

        for bt in boundary_ts:
            ax.axvline(bt, color="grey", alpha=0.3, linewidth=0.7)

    loss_ax = axes[len(plot_metrics)]
    eps     = [h["epoch"]   for h in history]
    loss_ax.plot(eps, [h["val_stat"] for h in history], label="val stat",  color="#1f77b4")
    loss_ax.plot(eps, [h["loss_G"]   for h in history], label="adv (gen)", color="#ff7f0e")
    loss_ax.plot(eps, [h["loss_D"]   for h in history], label="disc",      color="#2ca02c")
    loss_ax.plot(eps, [h["w_dist"]   for h in history], label="W-dist",    color="#9467bd", linestyle=":")
    loss_ax.plot(eps, [h["var_reg"]  for h in history], label="var_reg",   color="#d62728", linestyle="-.")
    loss_ax.plot(eps, [h["fm_stat"]  for h in history], label="fm_stat",   color="#e377c2", linestyle="-.")
    loss_ax.plot(eps, [h["smooth"]   for h in history], label="smooth",    color="#8c564b", linestyle=":")
    loss_ax.axvline(stat_warmup, color="red", linestyle="--", alpha=0.5,
                    label=f"warmup end (ep {stat_warmup})")
    loss_ax.set_title(f"Training losses\nwarmup={stat_warmup} adv={adv_epochs}", fontsize=9)
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
    parser = argparse.ArgumentParser(description="TimeGAN S14 - Encoder-Free Generator")
    parser.add_argument("--var-reg",     type=float, default=0.3,
                        help="Lambda for log-ratio variance loss (default 0.3)")
    parser.add_argument("--fm-stat",     type=float, default=0.5,
                        help="Lambda for per-replica statistical feature matching (default 0.5)")
    parser.add_argument("--smooth",      type=float, default=None,
                        help="Lambda for smoothing (None = per-workload defaults)")
    parser.add_argument("--adv-epochs",  type=int,   default=150)
    parser.add_argument("--stat-warmup", type=int,   default=20,
                        help="Statistical warmup epochs (fm_stat + var_reg only, no adversarial)")
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

    boundaries = list(map(int, args.phase_boundaries.split(",")))
    interp_tag = "" if args.aug_interp else "_ni"
    run_tag    = (f"s15_vr{int(args.var_reg * 10):02d}"
                  f"_fm{int(args.fm_stat   * 10):02d}"
                  f"_ae{args.adv_epochs}{interp_tag}")

    out_dir   = Path(f"outputs/phase4/timegan_s15/{run_tag}")
    model_dir = Path(f"models/phase4/timegan_s15/{run_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TimeGAN S15 - Encoder-Free Generator, Raw Normalized Targets")
    print("=" * 70)
    print(f"Device          : {device}")
    print(f"Architecture    : GeneratorB (z -> decoder, NO encoder)")
    print(f"Warmup          : statistical only (fm_stat + var_reg, {args.stat_warmup} epochs)")
    print(f"Var regression  : lambda={args.var_reg}  log-ratio")
    print(f"Per-replica FM  : lambda_fm_stat={args.fm_stat}")
    print(f"Adv epochs      : {args.adv_epochs}")
    print(f"Replica interp  : {args.aug_interp}")
    print(f"Output dir      : {out_dir}")
    print("=" * 70)

    all_results = []

    for workload in args.workloads:
        wl_overrides    = WORKLOAD_OVERRIDES[workload]
        lambda_smooth   = (args.smooth if args.smooth is not None
                           else wl_overrides["lambda_smooth"])
        n_disc_steps_wl = wl_overrides["n_disc_steps"]

        print(f"\n  --- Workload: {workload.upper()} "
              f"(smooth={lambda_smooth}  n_disc={n_disc_steps_wl}) ---")

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
        raw_kept    = raw_traces[:, :, kept_idx].astype(np.float32)
        norm_params = norm["params"] if "params" in norm else norm

        # S15: no zeromean. raw_kept is already in [0,1] from MinMax normalization.
        # Generator sigmoid output is in (0,1). Same space. No offset needed.
        if args.aug_interp:
            (raw_aug, r_aug, eids_aug, ti_aug) = replica_interpolation(
                raw_kept, replica_counts, experiment_ids, train_idx_orig)
            n_interp = len(ti_aug) - len(train_idx_orig)
        else:
            raw_aug  = raw_kept
            r_aug    = replica_counts
            eids_aug = experiment_ids
            ti_aug   = train_idx_orig
            n_interp = 0

        print(f"  Kept metrics ({n_metrics}): {kept_names}")
        print(f"  Train: {len(train_idx_orig)} orig + {n_interp} interp = {len(ti_aug)} total")
        print(f"  Val:   {len(val_idx)}")

        ds_train = WorkloadDataset(
            raw_aug, r_aug, eids_aug,
            phase_seq, ti_aug, jitter=args.jitter)
        ds_val = WorkloadDataset(
            raw_kept, replica_counts, experiment_ids,
            phase_seq, val_idx, jitter=0)
        dl_train = DataLoader(ds_train, batch_size=16, shuffle=True,  drop_last=True)
        dl_val   = DataLoader(ds_val,   batch_size=16, shuffle=False)

        generator     = GeneratorB(seq_len, n_metrics, GEN_CFG).to(device)
        discriminator = Discriminator(n_metrics, DISC_CFG).to(device)

        n_params_G = sum(p.numel() for p in generator.parameters())
        n_params_D = sum(p.numel() for p in discriminator.parameters())
        print(f"  GeneratorB params   : {n_params_G:,}")
        print(f"  Discriminator params: {n_params_D:,}")

        t0 = time.time()
        generator, discriminator, best_stat, n_epochs, history = train(
            generator, discriminator, dl_train, dl_val,
            BASE_CONFIG, args.stat_warmup, args.adv_epochs, GEN_CFG,
            args.var_reg, args.fm_stat,
            lambda_smooth, n_disc_steps_wl,
            device)
        elapsed = time.time() - t0

        (real_orig, syn_orig, real_r_arr, syn_r_arr,
         vr, vr_mean, ac_diff, jump_ratio) = evaluate(
            generator, raw_kept,
            replica_counts, phase_seq, train_idx_orig,
            kept_names, norm_params, 5, device)

        ref_vr   = REFERENCE_VR.get(workload, {})
        ref_jump = REFERENCE_JUMP.get(workload, {})
        print(f"\n  Results (s15 {run_tag}):")
        print(f"    best_stat     = {best_stat:.5f}")
        print(f"    var_ratio     = {vr_mean:.4f}  "
              f"(s14={ref_vr.get('s14','?'):.4f}  s9={ref_vr.get('s9','?'):.4f}  "
              f"lstm={ref_vr.get('lstm','?'):.4f})")
        print(f"    autocorr_diff = {ac_diff:.4f}")
        print(f"    jump_ratio    = {jump_ratio:.3f}x  "
              f"(s14={ref_jump.get('s14','?'):.3f}x  s9={ref_jump.get('s9','?'):.1f}x)")
        print(f"    epochs        = {n_epochs}  time = {elapsed:.1f}s")
        print(f"  Per-metric VR:")
        for j, m in enumerate(kept_names):
            flag = " <-- low"  if vr[j] < 0.5 else (
                   " <-- HIGH" if vr[j] > 3.0 else "")
            print(f"    {m:<22} {vr[j]:.4f}{flag}")

        run_label = f"s15 {run_tag} genB"
        plot_path = plot_comparison(
            workload, real_orig, syn_orig, real_r_arr, syn_r_arr,
            kept_names, phase_seq, history,
            args.stat_warmup, args.adv_epochs,
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
                "architecture":    "GeneratorB",
                "preprocessing":   "raw_normalized_no_zeromean",
                "warmup_type":     "statistical",
                **GEN_CFG,
                "disc_hidden":     DISC_CFG["hidden_dim"],
                "disc_layers":     DISC_CFG["num_layers"],
                "stat_warmup":     args.stat_warmup,
                "adv_epochs":      args.adv_epochs,
                "lambda_smooth":   lambda_smooth,
                "lambda_var_reg":  args.var_reg,
                "var_loss_type":   "log_ratio",
                "lambda_fm_stat":  args.fm_stat,
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
            "stage":           "s15",
            "architecture":    "GeneratorB",
            "preprocessing":   "raw_normalized",
            "run_tag":         run_tag,
            "adv_epochs":      args.adv_epochs,
            "stat_warmup":     args.stat_warmup,
            "jitter":          args.jitter,
            "aug_interp":      args.aug_interp,
            "lambda_var_reg":  args.var_reg,
            "var_loss_type":   "log_ratio",
            "lambda_fm_stat":  args.fm_stat,
            "lambda_smooth":   lambda_smooth,
            "n_disc_steps_wl": n_disc_steps_wl,
            "n_train_orig":    len(train_idx_orig),
            "n_train_aug":     len(ti_aug),
            "var_ratio_mean":       vr_mean,
            "var_ratio_per_metric": vr.tolist(),
            "autocorr_diff":        ac_diff,
            "phase_jump_ratio":     jump_ratio,
            "best_stat":            best_stat,
            "n_epochs":             n_epochs,
            "elapsed_s":            elapsed,
            "metrics_generated":    list(kept_names),
            **{f"ref_vr_{k}":   v for k, v in ref_vr.items()},
            **{f"ref_jump_{k}": v for k, v in ref_jump.items()},
        })

    # Summary
    print(f"\n{'='*72}")
    print(f"S15 SUMMARY  ({run_tag})")
    print(f"{'='*72}")
    print(f"  {'Workload':<12}  {'VR S15':>7}  {'VR S14':>7}  {'VR S9':>7}  {'LSTM':>7}  "
          f"{'jump S15':>9}  {'jump S14':>9}")
    print("  " + "-" * 72)
    vr_vals = []
    for r in all_results:
        vr_vals.append(r["var_ratio_mean"])
        print(f"  {r['workload']:<12}  {r['var_ratio_mean']:>7.4f}  "
              f"{r.get('ref_vr_s14',  float('nan')):>7.4f}  "
              f"{r.get('ref_vr_s9',   float('nan')):>7.4f}  "
              f"{r.get('ref_vr_lstm', float('nan')):>7.4f}  "
              f"{r['phase_jump_ratio']:>8.3f}x  "
              f"{r.get('ref_jump_s14', float('nan')):>8.3f}x")
    print("  " + "-" * 72)
    print(f"  {'MEAN':<12}  {float(np.mean(vr_vals)):>7.4f}")
    print(f"\nTarget: mean VR > 0.8, all metrics in [0.5, 3.0], jump ratio in [0.3, 5.0]")
    print(f"Reference: LSTM=0.612  S9=0.878 (honest balanced)  S14=1.204 (level errors)")

    with open(out_dir / "results.json", "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\nResults: {out_dir}/results.json")


if __name__ == "__main__":
    main()