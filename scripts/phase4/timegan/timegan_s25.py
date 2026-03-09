"""
TimeGAN S25 - Segment-Based Generation + Fixed Autocorrelation Loss
========================================
Master Thesis - Generative AI Workload Modeling
Fachhochschule Dortmund

S25 vs S24: THREE TARGETED FIXES
==================================

Fix 1 - autocorr_loss magnitude bug (critical):
  S24 computed raw covariance:  (x[:, :-lag] * x[:, lag:]).mean()
  For metrics with small magnitudes (PSI ~1e-4, latency ~0.009) this
  produces covariance values ~1e-8. The squared difference is ~1e-16.
  Even lambda_ac=0.1 contributes nothing -- the loss was dead in S24
  (confirmed: ac column showed 0.0001 throughout all workloads).

  S25 fix: normalize per metric by std before computing autocorrelation,
  so all values land in [-1, 1] and the loss has meaningful magnitude.

Fix 2 - YOLO n_disc_steps 2 -> 1:
  S24 YOLO showed continuous adv_gen growth to ~5-6 (generator explosion).
  Discriminator was too strong with 2 update steps per generator step.
  Matching Whisper's proven n_disc_steps=1 setting.

Fix 3 - ResNet152 lambda_smooth 0.1 -> 0.2:
  S24 ResNet152 throughput showed large intra-phase oscillations.
  Discriminator encouraging high-frequency variation the real data lacks.
  Stronger smoothing penalty to suppress this.

Fix 4 - Whisper lambda_fm_stat 0.5 -> 0.8:
  S24 Whisper CPU showed level error (r=1 synthetic ~5 vs real ~2).
  The fm_stat anchor was drifting during adversarial phase.
  Tighter statistical anchor to hold per-phase means.

S24 RESULTS (starting point for S25)
======================================
  bert      VR=0.733  (s21=0.672)  autocorr_diff=0.227  ac_loss=0.0001 (DEAD)
  gpt2      VR=1.082  (s21=0.996)  autocorr_diff=0.308  ac_loss=0.0005 (DEAD)
  resnet152 VR=0.747  (s21=0.960)  autocorr_diff=0.241  ac_loss=0.0004 (DEAD)
  whisper   VR=1.493  (s21=1.284)  autocorr_diff=0.146  ac_loss=0.0121
  yolo      VR=0.606  (s21=0.661)  autocorr_diff=0.265  ac_loss=0.0003 (DEAD)
  MEAN      0.9323

S25 TARGETS
============
1. ac_loss column should show values ~0.01-0.1 (not 0.0001) -- proof fix works
2. BERT/YOLO/ResNet152 autocorr_diff should decrease vs S24
3. ResNet152 VR should recover toward S21 level (0.96)
4. YOLO adv_gen should stay bounded (not grow to 5-6)
5. Mean VR >= S21 (0.9146)

USAGE
-----
    python timegan_s25.py
    python timegan_s25.py --workloads bert yolo
    python timegan_s25.py --lambda-ac 0.1
    python timegan_s25.py --var-reg 0.3 --fm-stat 1.0 --adv-epochs 150

OUTPUT
------
    outputs/phase4/timegan_s25/{run_tag}/plots/
    models/phase4/timegan_s25/{run_tag}/{workload}/
    outputs/phase4/timegan_s25/{run_tag}/results.json

    run_tag = s25_seg_vr{var_reg*10}_fm{fm_stat*10}_ae{adv_epochs}_ac{ac*10}
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
N_PHASES     = 6
SEGMENT_LEN  = 120
FULL_SEQ_LEN = 720

BASE_CONFIG = {
    "adversarial":       "wgan",
    "n_disc_steps":      2,
    "lambda_adv_warmup": 0.0,
    "lambda_adv_post":   1.0,
    "lambda_fm":         0.01,
    "lambda_gp":         10.0,
    "preprocessing":     "raw_normalized",
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

# S25: three overrides changed vs S24
#   resnet152: lambda_smooth 0.1 -> 0.2  (suppress intra-phase oscillation)
#   yolo:      n_disc_steps  2   -> 1    (prevent generator explosion)
#   whisper:   lambda_fm_stat 0.5 -> 0.8 (tighten anchor vs cpu level drift)
WORKLOAD_OVERRIDES = {
    "bert":      {"lambda_smooth": 0.05, "n_disc_steps": 2,
                  "lambda_fm_stat": 1.5,  "lambda_var_reg": 0.5,
                  "checkpoint_by": "best_W"},
    "gpt2":      {"lambda_smooth": 0.1,  "n_disc_steps": 2,
                  "lambda_fm_stat": 2.0,  "lambda_var_reg": 0.3,
                  "checkpoint_by": "best_W"},
    "resnet152": {"lambda_smooth": 0.2,  "n_disc_steps": 2,
                  "lambda_fm_stat": 1.0,  "lambda_var_reg": 0.4,
                  "checkpoint_by": "best_W"},
    "whisper":   {"lambda_smooth": 0.05, "n_disc_steps": 1,
                  "lambda_fm_stat": 0.8,  "lambda_var_reg": 0.5,
                  "checkpoint_by": "best_W"},
    "yolo":      {"lambda_smooth": 0.1,  "n_disc_steps": 1,
                  "lambda_fm_stat": 1.2,  "lambda_var_reg": 0.3,
                  "checkpoint_by": "best_W"},
}

REFERENCE_VR = {
    "bert":      {"lstm": 0.594, "s19": 1.226, "s21": 0.672, "s24": 0.733},
    "gpt2":      {"lstm": 0.676, "s19": 1.011, "s21": 0.996, "s24": 1.082},
    "resnet152": {"lstm": 0.607, "s19": 1.024, "s21": 0.960, "s24": 0.747},
    "whisper":   {"lstm": 0.748, "s19": 0.788, "s21": 1.284, "s24": 1.493},
    "yolo":      {"lstm": 0.438, "s19": 1.228, "s21": 0.661, "s24": 0.606},
}

REFERENCE_JUMP = {
    "bert":      {"s19": 0.083},
    "gpt2":      {"s19": 0.158},
    "resnet152": {"s19": 0.095},
    "whisper":   {"s19": 0.285},
    "yolo":      {"s19": 0.122},
}

# ------------------------------------------------------------------
# Data loading (identical to S21)
# ------------------------------------------------------------------

def load_raw_data(workload):
    path = DATA_RAW_DIR / f"{workload}_traces.npz"
    d    = np.load(path, allow_pickle=True)
    with open(DATA_RAW_DIR / f"{workload}_normalization.json") as f:
        norm = json.load(f)
    return dict(d), norm


def get_kept_metrics(workload):
    drop = DROP_PER_WORKLOAD.get(workload, set())
    kept_idx   = []
    kept_names = []
    for i, m in enumerate(ALL_METRICS):
        if m not in drop:
            kept_idx.append(i)
            kept_names.append(m)
    return kept_idx, kept_names


def denormalize(traces, kept_names, norm_params):
    lookup = norm_params["params"] if "params" in norm_params else norm_params
    out = traces.copy().astype(np.float64)
    for j, name in enumerate(kept_names):
        if name not in lookup:
            continue
        mn = lookup[name].get("min", 0.0)
        mx = lookup[name].get("max", 1.0)
        out[..., j] = out[..., j] * (mx - mn) + mn
    return out


def validate_replica_counts(replica_counts, n_pods, metadata, workload):
    if len(replica_counts) == n_pods:
        return replica_counts.flatten().astype(np.int32)
    print(f"  [WARN] replica_counts has {len(replica_counts)} entries "
          f"but n_pods={n_pods}; rebuilding from metadata")
    rc = np.ones(n_pods, dtype=np.int32)
    for i, meta in enumerate(metadata):
        if isinstance(meta, dict):
            rc[i] = int(meta.get("replica_count", meta.get("replicas", 1)))
    print(f"  [INFO] Unique replica_counts: {np.unique(rc)}")
    return rc


def replica_interpolation(raw_traces, replica_counts, experiment_ids, train_idx):
    train_r  = replica_counts[train_idx]
    unique_r = np.unique(train_r)
    new_traces, new_r, new_eids = [], [], []
    for i in range(len(unique_r) - 1):
        r_lo, r_hi = unique_r[i], unique_r[i + 1]
        r_mid = (r_lo + r_hi) // 2
        if r_mid in unique_r or r_mid == r_lo or r_mid == r_hi:
            continue
        lo_idx = train_idx[train_r == r_lo]
        hi_idx = train_idx[train_r == r_hi]
        n = min(len(lo_idx), len(hi_idx))
        for j in range(n):
            t = np.random.uniform(0.3, 0.7)
            new_traces.append((1 - t) * raw_traces[lo_idx[j]]
                              + t * raw_traces[hi_idx[j]])
            new_r.append(r_mid)
            new_eids.append(-1)
    if not new_traces:
        print(f"    [interp] 0 samples created - all midpoints already covered.")
        return raw_traces, replica_counts, experiment_ids, train_idx
    new_traces = np.array(new_traces, dtype=np.float32)
    n_orig = len(raw_traces)
    print(f"    [interp] {len(new_traces)} samples created.")
    raw_aug  = np.concatenate([raw_traces, new_traces], axis=0)
    r_aug    = np.concatenate([replica_counts, np.array(new_r)], axis=0)
    eids_aug = np.concatenate([experiment_ids, np.array(new_eids, dtype=np.int64)])
    ti_aug   = np.concatenate([train_idx,
                               np.arange(n_orig, n_orig + len(new_traces))])
    return raw_aug, r_aug, eids_aug, ti_aug


# ------------------------------------------------------------------
# Segment reshaping (identical to S21)
# ------------------------------------------------------------------

def segment_traces(traces, replica_counts, phase_boundaries, n_phases, seg_len):
    N, T, M  = traces.shape
    bounds   = list(phase_boundaries) + [T]
    assert len(bounds) == n_phases + 1

    segments      = []
    seg_rc        = []
    seg_phase_idx = []
    seg_trace_idx = []

    for trace_i in range(N):
        for ph in range(n_phases):
            t_start = bounds[ph]
            t_end   = bounds[ph + 1]
            raw_seg = traces[trace_i, t_start:t_end, :]
            span    = t_end - t_start
            if span == seg_len:
                resampled = raw_seg
            else:
                src_t = np.linspace(0, 1, span)
                dst_t = np.linspace(0, 1, seg_len)
                resampled = np.stack(
                    [np.interp(dst_t, src_t, raw_seg[:, m]) for m in range(M)],
                    axis=1
                ).astype(np.float32)
            segments.append(resampled)
            seg_rc.append(int(replica_counts[trace_i]))
            seg_phase_idx.append(ph)
            seg_trace_idx.append(trace_i)

    return (np.array(segments,      dtype=np.float32),
            np.array(seg_rc,        dtype=np.int32),
            np.array(seg_phase_idx, dtype=np.int32),
            np.array(seg_trace_idx, dtype=np.int32))


# ------------------------------------------------------------------
# SegmentDataset (identical to S21)
# ------------------------------------------------------------------

class SegmentDataset(Dataset):
    def __init__(self, segments, seg_rc, seg_phase_idx, indices, jitter=0):
        self.segs    = segments
        self.rc      = seg_rc
        self.ph      = seg_phase_idx
        self.indices = indices
        self.jitter  = jitter
        self.r_max   = max(float(seg_rc.max()), 1.0)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx   = self.indices[i]
        seg   = torch.tensor(self.segs[idx], dtype=torch.float32)
        r_val = int(self.rc[idx])
        ph    = int(self.ph[idx])

        if self.jitter > 0:
            shift = np.random.randint(-self.jitter, self.jitter + 1)
            seg   = torch.roll(seg, shift, dims=0)

        r_norm    = torch.tensor((r_val - 1.0) / 9.0, dtype=torch.float32)
        phase_idx = torch.tensor(ph, dtype=torch.long)

        return seg, torch.tensor(r_val, dtype=torch.long), r_norm, phase_idx


# ------------------------------------------------------------------
# Evaluation metrics (identical to S21)
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
# GeneratorSeg (identical to S21)
# ------------------------------------------------------------------

class GeneratorSeg(nn.Module):
    def __init__(self, seg_len, n_metrics, cfg):
        super().__init__()
        self.seg_len   = seg_len
        self.n_metrics = n_metrics
        hidden         = cfg["hidden_dim"]
        n_layers       = cfg["num_layers"]
        latent_dim     = cfg["latent_dim"]
        r_emb_dim      = cfg["replica_embed_dim"]
        ph_emb_dim     = cfg["phase_embed_dim"]
        dropout        = cfg["dropout"] if n_layers > 1 else 0.0

        self.r_embed = nn.Sequential(
            nn.Linear(1, r_emb_dim),
            nn.Tanh(),
        )
        self.ph_embed = nn.Embedding(N_PHASES + 1, ph_emb_dim)

        init_in = latent_dim + r_emb_dim + ph_emb_dim
        self.h_init = nn.Sequential(
            nn.Linear(init_in, hidden * n_layers),
            nn.Tanh(),
        )
        self.c_init = nn.Sequential(
            nn.Linear(init_in, hidden * n_layers),
            nn.Tanh(),
        )

        dec_in_dim = latent_dim + r_emb_dim + ph_emb_dim
        self.dec_rnn = nn.LSTM(dec_in_dim, hidden, n_layers,
                               batch_first=True, dropout=dropout)

        self.out_fc  = nn.Linear(hidden, n_metrics)
        self.out_act = nn.Sigmoid()

        self.latent_dim = latent_dim
        self.n_layers   = n_layers
        self.hidden_dim = hidden

    def forward(self, r_norm, phase_idx, z=None):
        B      = r_norm.shape[0]
        T      = self.seg_len
        device = r_norm.device

        if z is None:
            z = torch.randn(B, self.latent_dim, device=device)

        r_emb  = self.r_embed(r_norm.unsqueeze(-1))
        ph_emb = self.ph_embed(phase_idx)

        zrp = torch.cat([z, r_emb, ph_emb], dim=-1)
        h0  = self.h_init(zrp)
        c0  = self.c_init(zrp)
        h0  = h0.view(B, self.n_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        c0  = c0.view(B, self.n_layers, self.hidden_dim).permute(1, 0, 2).contiguous()

        z_exp  = z.unsqueeze(1).expand(-1, T, -1)
        r_exp  = r_emb.unsqueeze(1).expand(-1, T, -1)
        ph_exp = ph_emb.unsqueeze(1).expand(-1, T, -1)
        dec_in = torch.cat([z_exp, r_exp, ph_exp], dim=-1)

        dec_out, _ = self.dec_rnn(dec_in, (h0, c0))
        return self.out_act(self.out_fc(dec_out))

    @torch.no_grad()
    def generate_segment(self, r_norm_val, phase_idx_val, device, n_samples=1):
        self.eval()
        r_norm    = torch.full((n_samples,), r_norm_val,
                               dtype=torch.float32, device=device)
        ph_idx    = torch.full((n_samples,), phase_idx_val,
                               dtype=torch.long, device=device)
        return self(r_norm, ph_idx).cpu().numpy()

    @torch.no_grad()
    def generate_trace(self, r_norm_val, device, n_samples=1):
        self.eval()
        segments = []
        for ph in range(N_PHASES):
            seg = self.generate_segment(r_norm_val, ph, device, n_samples)
            segments.append(seg)
        return np.concatenate(segments, axis=1)


# ------------------------------------------------------------------
# Discriminator (identical to S21)
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
# FM stat loss (identical to S21)
# ------------------------------------------------------------------

def precompute_fm_targets(segments, seg_rc, seg_phase_idx, device):
    targets = {}
    for ph_val in set(seg_phase_idx.tolist()):
        for r_val in set(seg_rc.tolist()):
            mask = (seg_rc == r_val) & (seg_phase_idx == ph_val)
            if mask.sum() < 2:
                continue
            group = segments[mask]
            targets[(int(ph_val), int(r_val))] = {
                'mean': torch.tensor(
                    group.mean(axis=(0, 1)), dtype=torch.float32).to(device),
                'std':  torch.tensor(
                    group.std(axis=(0, 1)),  dtype=torch.float32).to(device) + 1e-8,
            }
    return targets


def compute_segment_fm_stat(fake, r_int, phase_idx_batch, fm_targets, device):
    total_loss = torch.tensor(0.0, device=device)
    count      = 0

    unique_r  = torch.unique(r_int)
    unique_ph = torch.unique(phase_idx_batch)

    for r_val in unique_r:
        for ph_val in unique_ph:
            key  = (int(ph_val.item()), int(r_val.item()))
            if key not in fm_targets:
                continue
            mask = (r_int == r_val) & (phase_idx_batch == ph_val)
            if mask.sum() < 1:
                continue

            fake_seg = fake[mask]
            t_mean   = fm_targets[key]['mean']
            t_std    = fm_targets[key]['std']

            fake_mean = fake_seg.mean(dim=(0, 1))
            scale     = (t_mean.abs() + t_std + 1e-6).detach()
            total_loss = total_loss + (
                (fake_mean - t_mean.detach()) / scale
            ).pow(2).mean()

            fake_std  = fake_seg.std(dim=(0, 1)) + 1e-8
            log_std   = torch.log(fake_std / t_std.detach())
            total_loss = total_loss + (log_std ** 2).mean()

            count += 1

    return total_loss / max(count, 1)


# ------------------------------------------------------------------
# S24 NEW: Autocorrelation loss
# ------------------------------------------------------------------

def autocorr_loss(real_seg, fake_seg, max_lag=20):
    """
    Penalises the difference in mean autocorrelation at each lag 1..max_lag.

    Both inputs: (B, T, M) -- 120-step segments.
    real_seg is detached so gradients only flow through fake_seg.

    S25 FIX vs S24: normalize each metric by its std before computing the
    autocorrelation product. This maps all metrics to comparable scale
    (autocorr in [-1, 1]) so metrics with small absolute magnitude (e.g.
    PSI_CPU ~1e-4, latency ~0.009) contribute equally to the loss instead
    of producing covariance values ~1e-16 that are invisible to the optimizer.

    Loss is averaged over lags so magnitude is independent of max_lag.
    """
    real_d = real_seg.detach()
    r_std  = real_d.std(dim=[0, 1], keepdim=True) + 1e-8
    f_std  = fake_seg.std(dim=[0, 1], keepdim=True).detach() + 1e-8
    real_n = real_d   / r_std
    fake_n = fake_seg / f_std
    loss   = torch.tensor(0.0, device=fake_seg.device)
    for lag in range(1, max_lag + 1):
        ac_real = (real_n[:, :-lag, :] * real_n[:, lag:, :]).mean()
        ac_fake = (fake_n[:, :-lag, :] * fake_n[:, lag:, :]).mean()
        loss    = loss + (ac_real - ac_fake) ** 2
    return loss / max_lag


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train(generator, discriminator, train_dl, val_dl,
          base_cfg, stat_warmup, adv_epochs, gen_cfg,
          lambda_var_reg, lambda_fm_stat, lambda_smooth,
          n_disc_steps_wl, device, fm_targets,
          lambda_ac, ac_max_lag,
          warmup_patience=15, min_delta=1e-4,
          checkpoint_by="best_W"):
    """
    S25 training loop.

    Identical to S24 except:
      - autocorr_loss uses variance-normalized signals (S25 Fix 1)
      - per-workload overrides changed for resnet152/yolo/whisper (S25 Fix 2-4)

    Set lambda_ac=0 to reproduce exact S21 behaviour.
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
    print(f"    lambda_var_reg={lambda_var_reg}  lambda_fm_stat={lambda_fm_stat}  "
          f"lambda_smooth={lambda_smooth}  lambda_ac={lambda_ac}  "
          f"ac_max_lag={ac_max_lag}  n_disc={n_disc_steps_wl}")

    for epoch in range(1, total_epochs + 1):

        in_warmup    = (epoch <= actual_warmup)
        n_disc_steps = 0 if in_warmup else n_disc_steps_wl

        if not in_warmup and not warmup_done:
            warmup_done = True
            print(f"    [epoch {epoch}] Adversarial start.")

        generator.train()
        discriminator.train()
        ep_adv_g = ep_loss_d = ep_wdist = ep_vreg = ep_sm = ep_fmstat = ep_ac = 0.0
        n_batches = 0

        for target, r_int, r_norm, phase_idx in train_dl:
            target    = target.to(device)
            r_norm    = r_norm.to(device)
            r_int     = r_int.to(device)
            phase_idx = phase_idx.to(device)

            # ----- Discriminator -----
            if n_disc_steps > 0:
                with torch.no_grad():
                    fake_d = generator(r_norm, phase_idx)
                for _ in range(n_disc_steps):
                    opt_D.zero_grad()
                    sr = discriminator(target, r_norm)
                    sf = discriminator(fake_d, r_norm)
                    wd = sr.mean() - sf.mean()
                    ld = -wd + gradient_penalty(
                             discriminator, target, fake_d, r_norm, device, lam_gp)
                    ld.backward()
                    nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=5.0)
                    opt_D.step()
                ep_loss_d += ld.item()
                ep_wdist  += wd.item()

            # ----- Generator -----
            opt_G.zero_grad()
            fake = generator(r_norm, phase_idx)

            loss_G = torch.tensor(0.0, device=device)

            # Adversarial term (post-warmup only)
            if not in_warmup:
                loss_adv = -discriminator(fake, r_norm).mean()
                loss_G   = loss_G + lam_adv * loss_adv
                ep_adv_g += loss_adv.item()

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

            # Per-segment fm_stat (warmup + adversarial)
            if lambda_fm_stat > 0:
                fm_stat_loss = compute_segment_fm_stat(
                    fake, r_int, phase_idx, fm_targets, device)
                loss_G    = loss_G + lambda_fm_stat * fm_stat_loss
                ep_fmstat += fm_stat_loss.item()

            # Temporal smoothing (adversarial only)
            if lambda_smooth > 0 and not in_warmup:
                diff   = fake[:, 1:, :] - fake[:, :-1, :]
                sm     = (diff ** 2).mean()
                loss_G = loss_G + lambda_smooth * sm
                ep_sm += sm.item()

            # S25 Fix 1: Autocorrelation loss with variance-normalised signals
            if lambda_ac > 0:
                ac_loss = autocorr_loss(target, fake, max_lag=ac_max_lag)
                loss_G  = loss_G + lambda_ac * ac_loss
                ep_ac  += ac_loss.item()

            loss_G.backward()
            nn.utils.clip_grad_norm_(generator.parameters(),
                                     max_norm=gen_cfg["grad_clip"])
            opt_G.step()
            n_batches += 1

        # ----- Validation -----
        generator.eval()
        val_stat_sum = 0.0
        n_val = 0
        with torch.no_grad():
            for target_v, r_int_v, r_norm_v, phase_idx_v in val_dl:
                target_v    = target_v.to(device)
                r_norm_v    = r_norm_v.to(device)
                r_int_v     = r_int_v.to(device)
                phase_idx_v = phase_idx_v.to(device)
                fake_v      = generator(r_norm_v, phase_idx_v)
                vs = compute_segment_fm_stat(
                    fake_v, r_int_v, phase_idx_v, fm_targets, device)
                val_stat_sum += vs.item()
                n_val += 1
        val_stat = val_stat_sum / max(n_val, 1)

        if n_batches > 0:
            ep_adv_g  /= n_batches
            ep_loss_d /= n_batches
            ep_wdist  /= n_batches
            ep_vreg   /= n_batches
            ep_sm     /= n_batches
            ep_fmstat /= n_batches
            ep_ac     /= n_batches

        sched_G.step(val_stat)

        if warmup_done and ep_wdist > best_wdist:
            best_wdist = ep_wdist
            best_G     = {k: v.cpu().clone() for k, v in generator.state_dict().items()}
            best_D     = {k: v.cpu().clone() for k, v in discriminator.state_dict().items()}

        if val_stat < best_stat - min_delta:
            best_stat   = val_stat
            best_G_stat = {k: v.cpu().clone() for k, v in generator.state_dict().items()}
            patience_c  = 0
        else:
            patience_c += 1

        if in_warmup and patience_c >= warmup_patience:
            actual_warmup = epoch
            print(f"    [warmup] Patience at epoch {epoch}. Starting adv early.")

        history.append({
            "epoch": epoch, "val_stat": val_stat,
            "adv_g": ep_adv_g, "disc": ep_loss_d, "wdist": ep_wdist,
            "var_reg": ep_vreg, "fm_stat": ep_fmstat, "smooth": ep_sm,
            "ac": ep_ac,
        })

        if epoch % 10 == 1 or epoch == total_epochs:
            phase_str = "warmup" if in_warmup else "   adv"
            print(f"    ep {epoch:4d} [{phase_str}]  "
                  f"val_stat={val_stat:.5f}  "
                  f"G={ep_adv_g:.4f}  D={ep_loss_d:.4f}  W={ep_wdist:.4f}  "
                  f"vr={ep_vreg:.4f}  fm={ep_fmstat:.4f}  "
                  f"sm={ep_sm:.4f}  ac={ep_ac:.4f}")

    if warmup_done and best_wdist > float("-inf") and checkpoint_by == "best_W":
        generator.load_state_dict(best_G)
        discriminator.load_state_dict(best_D)
        print(f"    Final: best W-dist checkpoint (W={best_wdist:.4f})")
    else:
        generator.load_state_dict(best_G_stat)
        if checkpoint_by == "best_stat":
            print(f"    Final: best stat checkpoint (stat={best_stat:.5f}) "
                  f"[forced by checkpoint_by]")
        else:
            print(f"    Final: best stat checkpoint (stat={best_stat:.5f}) "
                  f"[warmup never completed]")

    return generator, discriminator, best_stat, total_epochs, history


# ------------------------------------------------------------------
# Evaluate (identical to S21)
# ------------------------------------------------------------------

def evaluate(generator, raw_kept, replica_counts, phase_seq,
             train_idx_orig, kept_names, norm_params, n_gen, device):
    real_norm = raw_kept[train_idx_orig]
    real_orig = denormalize(real_norm, kept_names, norm_params)
    train_r   = replica_counts[train_idx_orig]
    unique_r  = np.unique(train_r)

    syn_list = []
    syn_r    = []
    for r_val in unique_r:
        r_norm_val = float((r_val - 1.0) / 9.0)
        traces = generator.generate_trace(r_norm_val, device, n_samples=n_gen)
        for trace in traces:
            syn_list.append(trace)
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
# Visualisation (identical to S21, ac curve added to loss plot)
# ------------------------------------------------------------------

def build_phase_sequence(full_seq_len, boundaries):
    phase_seq = np.zeros(full_seq_len, dtype=np.int64)
    for ph, (t0, t1) in enumerate(
            zip(boundaries, list(boundaries[1:]) + [full_seq_len])):
        phase_seq[t0:t1] = ph
    return phase_seq


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
                              linewidth=0.9, alpha=0.85)
                if pi == 0:
                    leg_handles.append(h1)
                    leg_labels.append(f"r={r_val}")
            if len(sidx) > 0:
                ax.plot(syn_orig[sidx[0], :, m_idx], color=clr,
                        linewidth=1.2, linestyle="--", alpha=0.7)

        for bt in boundary_ts:
            ax.axvline(bt, color="grey", linewidth=0.5, alpha=0.4)

    ax_loss = axes[len(plot_metrics)]
    if history:
        epochs  = [h["epoch"]    for h in history]
        vs_vals = [h["val_stat"] for h in history]
        g_vals  = [h["adv_g"]   for h in history]
        d_vals  = [h["disc"]    for h in history]
        w_vals  = [h["wdist"]   for h in history]
        vr_vals = [h["var_reg"] for h in history]
        fm_vals = [h["fm_stat"] for h in history]
        sm_vals = [h["smooth"]  for h in history]
        ac_vals = [h["ac"]      for h in history]

        adv_start = stat_warmup + 1
        ax_loss.plot(epochs, vs_vals, label="val stat",  color="blue",   lw=1.0)
        ax_loss.plot(epochs, g_vals,  label="adv (gen)", color="orange", lw=1.0)
        ax_loss.plot(epochs, d_vals,  label="disc",      color="green",  lw=1.0)
        ax_loss.plot(epochs, w_vals,  label="W-dist",    color="navy",
                     lw=0.8, linestyle=":")
        ax_loss.plot(epochs, vr_vals, label="var_reg",   color="red",    lw=0.8)
        ax_loss.plot(epochs, fm_vals, label="fm_stat",   color="pink",   lw=0.8)
        ax_loss.plot(epochs, sm_vals, label="smooth",    color="purple",
                     lw=0.8, linestyle=":")
        ax_loss.plot(epochs, ac_vals, label="autocorr",  color="brown",  lw=0.8)
        ax_loss.axvline(adv_start, color="pink", linewidth=1.0,
                        linestyle="--", label=f"warmup end (ep {stat_warmup})")
        ax_loss.set_title(f"Training losses\nwarmup={stat_warmup} adv={adv_epochs}",
                          fontsize=9)
        ax_loss.set_xlabel("epoch", fontsize=8)
        ax_loss.legend(fontsize=6, loc="upper left")
        ax_loss.set_ylim(-10, 10)

    for ax in axes[n_plots:]:
        ax.set_visible(False)

    if leg_handles:
        fig.legend(leg_handles, leg_labels,
                   loc="lower center", ncol=len(leg_handles),
                   fontsize=8, frameon=False)

    title = (f"{workload.upper()} - {run_label}\n"
             "solid=real  dashed=synthetic  grey=phase boundaries")
    fig.suptitle(title, fontsize=10, y=1.01)
    plt.tight_layout()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{workload}_{run_label.replace(' ', '_')}.png"
    path  = save_dir / fname
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TimeGAN S25 segment-based + fixed autocorr loss")
    parser.add_argument("--workloads",    nargs="+", default=WORKLOADS)
    parser.add_argument("--var-reg",      type=float, default=0.3)
    parser.add_argument("--fm-stat",      type=float, default=1.0)
    parser.add_argument("--adv-epochs",   type=int,   default=150)
    parser.add_argument("--stat-warmup",  type=int,   default=20)
    parser.add_argument("--jitter",       type=int,   default=4)
    parser.add_argument("--aug-interp",   action="store_true", default=True)
    parser.add_argument("--smooth",       type=float, default=None)
    # S24/S25: autocorrelation loss arguments
    parser.add_argument("--lambda-ac",    type=float, default=0.1,
                        help="Autocorrelation loss weight (0 = S21 behaviour)")
    parser.add_argument("--ac-max-lag",   type=int,   default=20,
                        help="Max lag for autocorrelation loss")
    args = parser.parse_args()

    torch.manual_seed(GEN_CFG["seed"])
    np.random.seed(GEN_CFG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_tag   = (f"s25_seg_vr{int(args.var_reg * 10):02d}"
                 f"_fm{int(args.fm_stat * 10):02d}"
                 f"_ae{args.adv_epochs}"
                 f"_ac{int(args.lambda_ac * 10):02d}")
    out_dir   = Path(f"outputs/phase4/timegan_s25/{run_tag}")
    model_dir = Path(f"models/phase4/timegan_s25/{run_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    boundaries = DEFAULT_PHASE_BOUNDARIES

    print("=" * 70)
    print("TimeGAN S25 - Segment-Based Generation + Fixed Autocorrelation Loss")
    print("=" * 70)
    print(f"Device         : {device}")
    print(f"Segment length : {SEGMENT_LEN} steps x {N_PHASES} phases = {FULL_SEQ_LEN}")
    print(f"Warmup         : statistical only ({args.stat_warmup} epochs)")
    print(f"Adv epochs     : {args.adv_epochs}")
    print(f"lambda_ac      : {args.lambda_ac}  (0 = S21 behaviour, no ac loss)")
    print(f"ac_max_lag     : {args.ac_max_lag}")
    print(f"Replica interp : {args.aug_interp}")
    print(f"Output dir     : {out_dir}")
    print("=" * 70)

    all_results = []

    for workload in args.workloads:
        wl_overrides      = WORKLOAD_OVERRIDES[workload]
        lambda_smooth     = (args.smooth if args.smooth is not None
                             else wl_overrides["lambda_smooth"])
        n_disc_steps_wl   = wl_overrides["n_disc_steps"]
        lambda_fm_stat_wl = wl_overrides.get("lambda_fm_stat", args.fm_stat)
        lambda_var_reg_wl = wl_overrides.get("lambda_var_reg", args.var_reg)
        checkpoint_by_wl  = wl_overrides.get("checkpoint_by",  "best_W")

        print(f"\n  --- Workload: {workload.upper()} "
              f"(smooth={lambda_smooth}  n_disc={n_disc_steps_wl}"
              f"  fm={lambda_fm_stat_wl}  vr={lambda_var_reg_wl}"
              f"  ckpt={checkpoint_by_wl}  ac={args.lambda_ac}) ---")

        data, norm     = load_raw_data(workload)
        raw_traces     = data["traces"]
        replica_counts = data["replica_counts"]
        train_idx_orig = data["train_idx"]
        val_idx        = data["val_idx"]
        metadata       = list(data["metadata"])
        n_pods         = len(raw_traces)

        replica_counts = validate_replica_counts(
            replica_counts, n_pods, metadata, workload)

        kept_idx, kept_names = get_kept_metrics(workload)
        n_metrics   = len(kept_idx)
        raw_kept    = raw_traces[:, :, kept_idx].astype(np.float32)
        norm_params = norm

        (all_segs, all_seg_rc,
         all_seg_ph, all_seg_ti) = segment_traces(
            raw_kept, replica_counts,
            boundaries, N_PHASES, SEGMENT_LEN)

        train_set     = set(train_idx_orig.tolist())
        val_set       = set(val_idx.tolist())
        seg_train_idx = np.where(np.isin(all_seg_ti, list(train_set)))[0]
        seg_val_idx   = np.where(np.isin(all_seg_ti, list(val_set)))[0]

        if args.aug_interp:
            (raw_aug, r_aug, _, ti_aug) = replica_interpolation(
                raw_kept, replica_counts,
                np.arange(n_pods, dtype=np.int64), train_idx_orig)
            n_interp = len(ti_aug) - len(train_idx_orig)
            if n_interp > 0:
                (aug_segs, aug_seg_rc,
                 aug_seg_ph, _) = segment_traces(
                    raw_aug[len(raw_kept):],
                    r_aug[len(replica_counts):],
                    boundaries, N_PHASES, SEGMENT_LEN)
                all_segs_tr          = np.concatenate([all_segs[seg_train_idx], aug_segs])
                all_rc_tr            = np.concatenate([all_seg_rc[seg_train_idx], aug_seg_rc])
                all_ph_tr            = np.concatenate([all_seg_ph[seg_train_idx], aug_seg_ph])
                seg_train_idx_merged = np.arange(len(all_segs_tr))
            else:
                n_interp             = 0
                all_segs_tr          = all_segs[seg_train_idx]
                all_rc_tr            = all_seg_rc[seg_train_idx]
                all_ph_tr            = all_seg_ph[seg_train_idx]
                seg_train_idx_merged = np.arange(len(all_segs_tr))
        else:
            n_interp             = 0
            all_segs_tr          = all_segs[seg_train_idx]
            all_rc_tr            = all_seg_rc[seg_train_idx]
            all_ph_tr            = all_seg_ph[seg_train_idx]
            seg_train_idx_merged = np.arange(len(all_segs_tr))

        n_train_segs      = len(seg_train_idx_merged)
        n_val_segs        = len(seg_val_idx)
        n_train_orig_segs = len(seg_train_idx)

        print(f"  Kept metrics ({n_metrics}): {kept_names}")
        print(f"  Trace splits: train={len(train_idx_orig)} orig, val={len(val_idx)}")
        print(f"  Segments: train={n_train_segs} "
              f"({n_train_orig_segs} orig + {n_interp * N_PHASES} interp), "
              f"val={n_val_segs}")

        ds_train = SegmentDataset(all_segs_tr, all_rc_tr, all_ph_tr,
                                  seg_train_idx_merged, jitter=args.jitter)
        ds_val   = SegmentDataset(all_segs[seg_val_idx],
                                  all_seg_rc[seg_val_idx],
                                  all_seg_ph[seg_val_idx],
                                  np.arange(n_val_segs), jitter=0)
        dl_train = DataLoader(ds_train, batch_size=32, shuffle=True,  drop_last=True)
        dl_val   = DataLoader(ds_val,   batch_size=32, shuffle=False)

        generator     = GeneratorSeg(SEGMENT_LEN, n_metrics, GEN_CFG).to(device)
        discriminator = Discriminator(n_metrics, DISC_CFG).to(device)

        n_params_G = sum(p.numel() for p in generator.parameters())
        n_params_D = sum(p.numel() for p in discriminator.parameters())
        print(f"  GeneratorSeg params : {n_params_G:,}")
        print(f"  Discriminator params: {n_params_D:,}")

        fm_targets = precompute_fm_targets(
            all_segs_tr, all_rc_tr, all_ph_tr, device)
        print(f"  FM targets: {len(fm_targets)} (phase,r) groups precomputed")

        t0 = time.time()
        generator, discriminator, best_stat, n_epochs, history = train(
            generator, discriminator, dl_train, dl_val,
            BASE_CONFIG, args.stat_warmup, args.adv_epochs, GEN_CFG,
            lambda_var_reg_wl, lambda_fm_stat_wl,
            lambda_smooth, n_disc_steps_wl, device, fm_targets,
            lambda_ac=args.lambda_ac,
            ac_max_lag=args.ac_max_lag,
            checkpoint_by=checkpoint_by_wl)
        elapsed = time.time() - t0

        phase_seq = build_phase_sequence(FULL_SEQ_LEN, boundaries)
        (real_orig, syn_orig, real_r_arr, syn_r_arr,
         vr, vr_mean, ac_diff, jump_ratio) = evaluate(
            generator, raw_kept,
            replica_counts, phase_seq, train_idx_orig,
            kept_names, norm_params, 5, device)

        ref_vr   = REFERENCE_VR.get(workload, {})
        ref_jump = REFERENCE_JUMP.get(workload, {})
        print(f"\n  Results ({run_tag}):")
        print(f"    best_stat     = {best_stat:.5f}")
        print(f"    var_ratio     = {vr_mean:.4f}  "
              f"(s21={ref_vr.get('s21', '?')}  s24={ref_vr.get('s24', '?')}  "
              f"lstm={ref_vr.get('lstm', '?')})")
        print(f"    autocorr_diff = {ac_diff:.4f}")
        print(f"    jump_ratio    = {jump_ratio:.3f}x  "
              f"(s19={ref_jump.get('s19', '?')}x)")
        print(f"    epochs        = {n_epochs}  time = {elapsed:.1f}s")
        print(f"  Per-metric VR:")
        for j, m in enumerate(kept_names):
            flag = " <-- low"  if vr[j] < 0.5 else (
                   " <-- HIGH" if vr[j] > 3.0 else "")
            print(f"    {m:<22} {vr[j]:.4f}{flag}")

        run_label = f"s25 {run_tag} segB"
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
                "architecture":  "GeneratorSeg",
                "preprocessing": "raw_normalized",
                "fm_stat_type":  "per_segment",
                "seg_len":       SEGMENT_LEN,
                "n_phases":      N_PHASES,
                "lambda_ac":     args.lambda_ac,
                "ac_max_lag":    args.ac_max_lag,
                **GEN_CFG,
                "disc_hidden":   DISC_CFG["hidden_dim"],
            }, fh, indent=2)

        result = {
            "workload":          workload,
            "stage":             "s25",
            "run_tag":           run_tag,
            "adv_epochs":        args.adv_epochs,
            "stat_warmup":       args.stat_warmup,
            "jitter":            args.jitter,
            "aug_interp":        args.aug_interp,
            "lambda_var_reg":    lambda_var_reg_wl,
            "lambda_fm_stat":    lambda_fm_stat_wl,
            "lambda_smooth":     lambda_smooth,
            "lambda_ac":         args.lambda_ac,
            "ac_max_lag":        args.ac_max_lag,
            "n_disc_steps_wl":   n_disc_steps_wl,
            "n_train_segs":      n_train_segs,
            "var_ratio_mean":    float(vr_mean),
            "var_ratio_per_metric": [float(v) for v in vr],
            "autocorr_diff":     float(ac_diff),
            "phase_jump_ratio":  float(jump_ratio),
            "best_stat":         float(best_stat),
            "n_epochs":          n_epochs,
            "elapsed_s":         elapsed,
            "metrics_generated": kept_names,
            "ref_vr_s21":        ref_vr.get("s21"),
            "ref_vr_s19":        ref_vr.get("s19"),
            "ref_vr_lstm":       ref_vr.get("lstm"),
        }
        all_results.append(result)

    print(f"\n{'=' * 72}")
    print(f"S25 SUMMARY  ({run_tag})")
    print(f"{'=' * 72}")
    print(f"  {'Workload':<12} {'VR S25':>8} {'VR S21':>8} {'VR S24':>8} "
          f"{'LSTM':>8} {'jump S25':>10}")
    print(f"  {'-' * 72}")
    vr_means = []
    for r in all_results:
        vr_means.append(r["var_ratio_mean"])
        ref_vr   = REFERENCE_VR.get(r["workload"], {})
        ref_jump = REFERENCE_JUMP.get(r["workload"], {})
        print(f"  {r['workload']:<12} "
              f"{r['var_ratio_mean']:>8.4f} "
              f"{ref_vr.get('s21', 0):>8.4f} "
              f"{ref_vr.get('s24', 0):>8.4f} "
              f"{ref_vr.get('lstm', 0):>8.4f} "
              f"{r['phase_jump_ratio']:>10.3f}x")
    print(f"  {'-' * 72}")
    print(f"  {'MEAN':<12} {sum(vr_means)/len(vr_means):>8.4f}")
    print(f"\nTarget: mean VR > 0.8, all metrics [0.5, 3.0], jump ratio [0.3, 5.0]")
    print(f"S21 baseline: mean VR=0.9146  S24 baseline: mean VR=0.9323")

    result_path = out_dir / "results.json"
    with open(result_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\nResults: {result_path}")


if __name__ == "__main__":
    main()