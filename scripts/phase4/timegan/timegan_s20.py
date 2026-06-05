"""
TimeGAN S20 - Segment-Based Generation
========================================
Master Thesis - Generative AI Workload Modeling
Fachhochschule Dortmund

S19 RESULTS (FINAL END-TO-END STAGE)
======================================
S19 achieved mean VR=1.056 -- best result across all stages.

  bert      VR=1.226   all per-metric [0.87, 1.73]
  gpt2      VR=1.011   all per-metric [0.57, 1.41]
  resnet152 VR=1.024   all per-metric [1.00, 1.04]
  whisper   VR=0.788   all per-metric [0.75, 0.84]
  yolo      VR=1.228   throughput=0.650 (persistent)
  MEAN      1.056      > 0.8 target achieved

ONE REMAINING LIMITATION: JUMP RATIO
=======================================
All five workloads show jump_ratio in [0.08, 0.29]x vs target [0.3, 5.0]x.
This is an architectural limitation of end-to-end LSTM generation, not a
hyperparameter issue.

Root cause: at a phase boundary t=120, the LSTM hidden state h_t is a
continuous function of h_{t-1}. There is no mechanism in the architecture
that says "produce a sharp transition here". The generator learns to drift
toward the new phase level over 20-40 steps rather than jump sharply.

No amount of lambda tuning fixes this because no loss term explicitly
rewards sharp transitions at phase boundaries.

S20: SEGMENT-BASED GENERATION
================================
Structural fix: instead of generating 720-step traces end-to-end, generate
each of the 6 phases as an independent 120-step segment, then concatenate.

DATA RESHAPING:
  Before: (N, 720, M)    -- 49 full traces per workload
  After:  (N*6, 120, M)  -- 294 segments per workload (6x more samples)
  Conditioning per segment: (replica_count, phase_index, workload)

WHY IT FIXES JUMP RATIO:
  Phase boundaries become explicit concatenation points. Segment 2 (phase 2)
  and segment 3 (phase 3) are generated independently from different noise
  vectors. The level difference at the boundary is structural -- it is the
  difference between two independently-sampled distributions:

    P(x | r, phase=2)  vs  P(x | r, phase=3)

  The generator never needs to produce a jump because the jump IS the
  concatenation. Jump ratio becomes a function of how different the per-phase
  conditional distributions are, which is directly controlled by phase_index
  conditioning and learned from real data.

WHAT IT DOES NOT FIX:
  Inter-segment continuity. The final timestep of segment k and the first
  timestep of segment k+1 are independently sampled -- there may be a small
  discontinuity at each concatenation point. For Kwok simulation this is
  acceptable since Kwok cares about per-phase resource levels. For thesis
  evaluation this should be acknowledged: "segment-based generation produces
  structurally correct phase-level statistics and sharp transitions but
  introduces discontinuities at concatenation points due to independent
  segment sampling."

ARCHITECTURE CHANGES FROM S19:
================================
1. SEGMENT_LEN = 120  (was 720)
2. New segment_traces() reshapes (N, 720, M) -> (N*6, 120, M)
3. SegmentDataset: returns (segment_120, r_norm, r_int, phase_idx_scalar)
4. GeneratorSeg: identical to GeneratorB but seq_len=120, phase conditioning
   uses a single scalar phase_idx (same value broadcast to all 120 steps)
5. Discriminator: unchanged, just sees (B, 120, M) instead of (B, 720, M)
6. compute_segment_fm_stat(): simpler than per-phase version -- each training
   sample is already one phase, so fm_stat = per-(phase_idx, r_int) matching
   of mean and std. No phase masking needed.
7. evaluate(): generate 6 segments per r_val, concatenate -> 720-step traces,
   then use existing VR / autocorr / jump metrics on full traces
8. Training loop: same WGAN structure, same disc clipping

HYPERPARAMETERS:
  Same WORKLOAD_OVERRIDES as S19 (fm_stat, var_reg, smooth per workload).
  checkpoint_by="best_W" for all (segment training engages adversarially).
  No "best_stat" override needed since Whisper's adversarial training
  engages with segment-level sequences (shorter, easier to discriminate).

EXPECTED OUTCOME:
  VR:        similar to S19 (~1.0 mean) -- segment fm_stat maintains level
  Jump ratio: target [0.3, 5.0]x        -- structural fix should deliver this
  Autocorr:  similar or slightly worse  -- independent segment sampling may
             reduce long-range autocorrelation

STAGE HISTORY
===============
LSTM baseline  0.612  reference
S7             0.415  under-generated (encoder mismatch)
S8             1.678  best VR but temporal incoherence (Whisper 42x jump)
S9             0.878  balanced, ResNet152 0.324
S14            1.204  encoder fixed, level errors remain
S15            1.144  raw [0,1] targets, throughput still flat
S16            1.009  per-phase fm_stat; YOLO disc explosion; Whisper W~0
S17            1.003  disc gradient clipping; fm_stat 1.0; Whisper n_disc=1
S18            0.957  per-workload tuning; YOLO/Whisper regressions
S19            1.056  YOLO reverted; Whisper best_W; best mean VR
S20            THIS   segment-based generation; structural jump fix

USAGE
-----
    python timegan_s20.py
    python timegan_s20.py --workloads whisper yolo
    python timegan_s20.py --var-reg 0.3 --fm-stat 1.0 --adv-epochs 150

OUTPUT
------
    outputs/phase4/timegan_s20/{run_tag}/plots/
    models/phase4/timegan_s20/{run_tag}/{workload}/
    outputs/phase4/timegan_s20/{run_tag}/results.json

    run_tag = s20_seg_vr{var_reg*10}_fm{fm_stat*10}_ae{adv_epochs}
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

# S20: segment-based constants
# Each 720-step trace is split into N_PHASES segments of SEGMENT_LEN steps
DEFAULT_PHASE_BOUNDARIES = [0, 96, 180, 300, 420, 600]
N_PHASES     = 6
SEGMENT_LEN  = 120  # each segment is 120 timesteps (one phase, resampled)
FULL_SEQ_LEN = 720  # full trace = N_PHASES * SEGMENT_LEN

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

# S20 WORKLOAD_OVERRIDES
# Same as S19 -- fm_stat, var_reg, smooth per workload.
# All use checkpoint_by="best_W" (segment sequences are short enough
# that adversarial training should engage for all workloads including Whisper).
WORKLOAD_OVERRIDES = {
    "bert":      {"lambda_smooth": 0.05, "n_disc_steps": 2,
                  "lambda_fm_stat": 1.5,  "lambda_var_reg": 0.3,
                  "checkpoint_by": "best_W"},
    "gpt2":      {"lambda_smooth": 0.1,  "n_disc_steps": 2,
                  "lambda_fm_stat": 1.0,  "lambda_var_reg": 0.3,
                  "checkpoint_by": "best_W"},
    "resnet152": {"lambda_smooth": 0.1,  "n_disc_steps": 2,
                  "lambda_fm_stat": 1.0,  "lambda_var_reg": 0.3,
                  "checkpoint_by": "best_W"},
    "whisper":   {"lambda_smooth": 0.05, "n_disc_steps": 1,
                  "lambda_fm_stat": 0.5,  "lambda_var_reg": 0.5,
                  "checkpoint_by": "best_W"},
    "yolo":      {"lambda_smooth": 0.1,  "n_disc_steps": 2,
                  "lambda_fm_stat": 1.0,  "lambda_var_reg": 0.3,
                  "checkpoint_by": "best_W"},
}

REFERENCE_VR = {
    "bert":      {"lstm": 0.594, "s9": 1.213, "s19": 1.226},
    "gpt2":      {"lstm": 0.676, "s9": 0.664, "s19": 1.011},
    "resnet152": {"lstm": 0.607, "s9": 0.324, "s19": 1.024},
    "whisper":   {"lstm": 0.748, "s9": 0.743, "s19": 0.788},
    "yolo":      {"lstm": 0.438, "s9": 1.446, "s19": 1.228},
}

REFERENCE_JUMP = {
    "bert":      {"s8":  3.9, "s9":  8.7, "s19": 0.083},
    "gpt2":      {"s8":  3.0, "s9":  7.2, "s19": 0.158},
    "resnet152": {"s8":  6.5, "s9": 17.1, "s19": 0.095},
    "whisper":   {"s8": 42.1, "s9": 26.2, "s19": 0.285},
    "yolo":      {"s8": 10.6, "s9": 11.2, "s19": 0.122},
}

# ------------------------------------------------------------------
# Data loading (identical to S19)
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
    """Identical to S19 -- interpolate between adjacent replica counts."""
    train_r = replica_counts[train_idx]
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
# S20 CORE: segment_traces
# ------------------------------------------------------------------

def segment_traces(traces, replica_counts, phase_boundaries, n_phases, seg_len):
    """
    Reshape (N, 720, M) traces into (N*n_phases, seg_len, M) segments.

    Each full trace is split at phase boundaries. Each 720-step phase span
    is resampled to exactly seg_len=120 timesteps via linear interpolation.
    This is needed because the raw phase boundaries are irregular:
      [0, 96, 180, 300, 420, 600, 720]  -> spans: [96, 84, 120, 120, 180, 120]
    After resampling all spans to 120 steps, each phase is comparable.

    Returns
    -------
    segments       : (N * n_phases, seg_len, M)  float32
    seg_rc         : (N * n_phases,)              int32   replica_count per segment
    seg_phase_idx  : (N * n_phases,)              int32   phase index (0..5)
    seg_trace_idx  : (N * n_phases,)              int32   original trace index
    """
    N, T, M  = traces.shape
    # build boundary list including end
    bounds = list(phase_boundaries) + [T]
    assert len(bounds) == n_phases + 1, \
        f"Expected {n_phases + 1} boundary points, got {len(bounds)}"

    segments      = []
    seg_rc        = []
    seg_phase_idx = []
    seg_trace_idx = []

    for trace_i in range(N):
        for ph in range(n_phases):
            t_start = bounds[ph]
            t_end   = bounds[ph + 1]
            raw_seg = traces[trace_i, t_start:t_end, :]  # (span, M)
            span    = t_end - t_start
            # resample to seg_len via linear interpolation
            if span == seg_len:
                resampled = raw_seg
            else:
                # interpolate each metric independently
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

    return (np.array(segments,      dtype=np.float32),   # (N*P, seg_len, M)
            np.array(seg_rc,        dtype=np.int32),
            np.array(seg_phase_idx, dtype=np.int32),
            np.array(seg_trace_idx, dtype=np.int32))


# ------------------------------------------------------------------
# SegmentDataset -- one segment = one training sample
# ------------------------------------------------------------------

class SegmentDataset(Dataset):
    """
    Returns (segment, r_int, r_norm, phase_idx) per item.

    segment   : (SEGMENT_LEN, M)   float32
    r_int     : scalar int          replica count (1-10)
    r_norm    : scalar float        (r - 1) / 9
    phase_idx : scalar int          0..5
    """
    def __init__(self, segments, seg_rc, seg_phase_idx, indices, jitter=0):
        self.segs      = segments        # (N_seg, 120, M)
        self.rc        = seg_rc          # (N_seg,)
        self.ph        = seg_phase_idx   # (N_seg,)
        self.indices   = indices         # which segment indices to use
        self.jitter    = jitter
        self.r_max     = max(float(seg_rc.max()), 1.0)

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
# Evaluation metrics (identical to S19)
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
# GeneratorSeg
# ------------------------------------------------------------------

class GeneratorSeg(nn.Module):
    """
    S20 generator: produces one phase segment of SEGMENT_LEN=120 steps.

    Conditioning:
      r_norm    : normalized replica count (scalar per sample)
      phase_idx : scalar integer 0..5 (same value for all 120 steps)

    The phase_idx is embedded and broadcast to all timesteps, providing
    the generator with a constant signal about which load phase to produce.
    This is the key change from S19: in S19, phase_ids varied within each
    720-step trace; in S20, phase_idx is constant throughout a segment.
    """
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
        # N_PHASES + 1 embeddings (0..5 plus padding index 6)
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
        """
        r_norm    : (B,)  float  normalized replica count
        phase_idx : (B,)  long   phase index 0..5
        z         : (B, latent_dim) noise, sampled if None
        """
        B      = r_norm.shape[0]
        T      = self.seg_len
        device = r_norm.device

        if z is None:
            z = torch.randn(B, self.latent_dim, device=device)

        r_emb  = self.r_embed(r_norm.unsqueeze(-1))         # (B, r_emb_dim)
        ph_emb = self.ph_embed(phase_idx)                   # (B, ph_emb_dim)

        # h/c initialised from z + r_emb + ph_emb (phase-aware init)
        zrp = torch.cat([z, r_emb, ph_emb], dim=-1)
        h0  = self.h_init(zrp)
        c0  = self.c_init(zrp)
        h0  = h0.view(B, self.n_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        c0  = c0.view(B, self.n_layers, self.hidden_dim).permute(1, 0, 2).contiguous()

        # decoder input: z + r_emb + ph_emb broadcast to all T steps
        z_exp  = z.unsqueeze(1).expand(-1, T, -1)
        r_exp  = r_emb.unsqueeze(1).expand(-1, T, -1)
        ph_exp = ph_emb.unsqueeze(1).expand(-1, T, -1)
        dec_in = torch.cat([z_exp, r_exp, ph_exp], dim=-1)  # (B, T, latent+r+ph)

        dec_out, _ = self.dec_rnn(dec_in, (h0, c0))
        return self.out_act(self.out_fc(dec_out))            # (B, T, M)

    @torch.no_grad()
    def generate_segment(self, r_norm_val, phase_idx_val, device, n_samples=1):
        """Generate n_samples segments for a given (r, phase)."""
        self.eval()
        r_norm    = torch.full((n_samples,), r_norm_val,
                               dtype=torch.float32, device=device)
        ph_idx    = torch.full((n_samples,), phase_idx_val,
                               dtype=torch.long, device=device)
        out       = self(r_norm, ph_idx)
        return out.cpu().numpy()   # (n_samples, seg_len, M)

    @torch.no_grad()
    def generate_trace(self, r_norm_val, device, n_samples=1):
        """
        Generate a full 720-step trace by concatenating 6 phase segments.
        Each segment is independently sampled from P(seg | r, phase_idx).
        Phase boundaries become explicit concatenation points -- this is
        the structural fix for jump ratio.
        """
        self.eval()
        segments = []
        for ph in range(N_PHASES):
            seg = self.generate_segment(r_norm_val, ph, device, n_samples)
            segments.append(seg)            # (n_samples, SEGMENT_LEN, M)
        full_trace = np.concatenate(segments, axis=1)  # (n_samples, FULL_SEQ_LEN, M)
        return full_trace


# ------------------------------------------------------------------
# Discriminator (unchanged, sees 120-step segments during training)
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
# S20 fm_stat: per-(phase_idx, r_int) matching
# ------------------------------------------------------------------

def compute_segment_fm_stat(fake, target, r_int, phase_idx_batch, device):
    """
    S20 simplified fm_stat: each training sample is a single-phase segment.
    For each unique (phase_idx, r_int) combination in the batch, match
    mean and std of fake vs target.

    No phase masking needed since the entire segment is one phase.

    fake          : (B, SEGMENT_LEN, M)
    target        : (B, SEGMENT_LEN, M)
    r_int         : (B,) integer replica counts
    phase_idx_batch: (B,) integer phase indices 0..5
    """
    total_loss = torch.tensor(0.0, device=device)
    count      = 0

    unique_r  = torch.unique(r_int)
    unique_ph = torch.unique(phase_idx_batch)

    for r_val in unique_r:
        for ph_val in unique_ph:
            mask = (r_int == r_val) & (phase_idx_batch == ph_val)
            if mask.sum() < 2:
                continue

            real_seg = target[mask]     # (n, SEGMENT_LEN, M)
            fake_seg = fake[mask]       # (n, SEGMENT_LEN, M)

            # match mean across (samples, time) per metric
            real_mean = real_seg.mean(dim=(0, 1))
            fake_mean = fake_seg.mean(dim=(0, 1))
            total_loss = total_loss + F.mse_loss(fake_mean, real_mean.detach())

            # match std across (samples, time) per metric
            real_std = real_seg.std(dim=(0, 1)) + 1e-8
            fake_std = fake_seg.std(dim=(0, 1)) + 1e-8
            log_std  = torch.log(fake_std / real_std.detach())
            total_loss = total_loss + (log_std ** 2).mean()

            count += 1

    return total_loss / max(count, 1)


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train(generator, discriminator, train_dl, val_dl,
          base_cfg, stat_warmup, adv_epochs, gen_cfg,
          lambda_var_reg, lambda_fm_stat, lambda_smooth,
          n_disc_steps_wl, device,
          warmup_patience=15, min_delta=1e-4,
          checkpoint_by="best_W"):
    """
    S20 training loop.

    Same WGAN structure as S19 but:
      - Batches contain (segment, r_int, r_norm, phase_idx) tuples
      - Generator forward: generator(r_norm, phase_idx) -> (B, SEGMENT_LEN, M)
      - fm_stat: compute_segment_fm_stat (no phase masking)
      - val_stat: computed on val segments

    checkpoint_by="best_W"    default -- best W-dist checkpoint
    checkpoint_by="best_stat" override -- best val_stat checkpoint
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
          f"lambda_fm_stat={lambda_fm_stat} (per-segment)  "
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

            # Per-segment per-replica fm_stat (warmup + adversarial)
            # S20: simplified -- no phase masking, each segment is one phase
            if lambda_fm_stat > 0:
                fm_stat_loss = compute_segment_fm_stat(
                    fake, target, r_int, phase_idx, device)
                loss_G    = loss_G + lambda_fm_stat * fm_stat_loss
                ep_fmstat += fm_stat_loss.item()

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
                # val_stat = fm_stat on val segments
                vs = compute_segment_fm_stat(
                    fake_v, target_v, r_int_v, phase_idx_v, device)
                val_stat_sum += vs.item()
                n_val += 1
        val_stat = val_stat_sum / max(n_val, 1)

        if n_batches > 0:
            ep_adv_g /= n_batches
            ep_loss_d /= n_batches
            ep_wdist  /= n_batches
            ep_vreg   /= n_batches
            ep_sm     /= n_batches
            ep_fmstat /= n_batches

        sched_G.step(val_stat)

        # Checkpoint tracking
        if warmup_done and ep_wdist > best_wdist:
            best_wdist = ep_wdist
            best_G     = {k: v.cpu().clone() for k, v in generator.state_dict().items()}
            best_D     = {k: v.cpu().clone() for k, v in discriminator.state_dict().items()}

        if val_stat < best_stat - min_delta:
            best_stat    = val_stat
            best_G_stat  = {k: v.cpu().clone() for k, v in generator.state_dict().items()}
            patience_c   = 0
        else:
            patience_c += 1

        # Warmup early stopping
        if in_warmup and patience_c >= warmup_patience:
            actual_warmup = epoch
            print(f"    [warmup] Patience at epoch {epoch}. Starting adv early.")

        history.append({
            "epoch": epoch, "val_stat": val_stat,
            "adv_g": ep_adv_g, "disc": ep_loss_d, "wdist": ep_wdist,
            "var_reg": ep_vreg, "fm_stat": ep_fmstat, "smooth": ep_sm,
        })

        if epoch % 10 == 1 or epoch == total_epochs:
            phase_str = "warmup" if in_warmup else "   adv"
            print(f"    ep {epoch:4d} [{phase_str}]  "
                  f"val_stat={val_stat:.5f}  "
                  f"G={ep_adv_g:.4f}  D={ep_loss_d:.4f}  W={ep_wdist:.4f}  "
                  f"vr={ep_vreg:.4f}  fm={ep_fmstat:.4f}  sm={ep_sm:.4f}")

    # Load checkpoint
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
# Evaluate -- generate full 720-step traces from 6 concatenated segments
# ------------------------------------------------------------------

def evaluate(generator, raw_kept, replica_counts, phase_seq,
             train_idx_orig, kept_names, norm_params, n_gen, device):
    """
    Evaluation for S20 segment-based model.

    For each unique r_val in training set:
      - Call generator.generate_trace(r_norm, device, n_samples=n_gen)
      - This generates 6 independent segments and concatenates them
      - Result: (n_gen, FULL_SEQ_LEN, M) per r_val

    Then compute VR, autocorr, jump_ratio on the full 720-step traces.
    Jump ratio is expected to improve because boundaries are concatenation
    points between independently-sampled segments.
    """
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
# Visualisation (adapted for S20 -- plot full 720-step traces)
# ------------------------------------------------------------------

def build_phase_sequence(full_seq_len, boundaries):
    """Rebuild 720-step phase sequence for plotting."""
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

    # Training loss plot
    ax_loss = axes[len(plot_metrics)]
    if history:
        epochs  = [h["epoch"] for h in history]
        vs_vals = [h["val_stat"] for h in history]
        g_vals  = [h["adv_g"]   for h in history]
        d_vals  = [h["disc"]    for h in history]
        w_vals  = [h["wdist"]   for h in history]
        vr_vals = [h["var_reg"] for h in history]
        fm_vals = [h["fm_stat"] for h in history]
        sm_vals = [h["smooth"]  for h in history]

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
    fname    = f"{workload}_{run_label.replace(' ', '_')}.png"
    path     = save_dir / fname
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TimeGAN S20 segment-based")
    parser.add_argument("--workloads",    nargs="+", default=WORKLOADS)
    parser.add_argument("--var-reg",      type=float, default=0.3)
    parser.add_argument("--fm-stat",      type=float, default=1.0)
    parser.add_argument("--adv-epochs",   type=int,   default=150)
    parser.add_argument("--stat-warmup",  type=int,   default=20)
    parser.add_argument("--jitter",       type=int,   default=4,
                        help="jitter in steps (S20: smaller since segments are 120 steps)")
    parser.add_argument("--aug-interp",   action="store_true", default=True)
    parser.add_argument("--smooth",       type=float, default=None)
    args = parser.parse_args()

    torch.manual_seed(GEN_CFG["seed"])
    np.random.seed(GEN_CFG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_tag  = (f"s20_seg_vr{int(args.var_reg * 10):02d}"
                f"_fm{int(args.fm_stat * 10):02d}"
                f"_ae{args.adv_epochs}")
    out_dir  = Path(f"outputs/phase4/timegan_s20/{run_tag}")
    model_dir = Path(f"models/phase4/timegan_s20/{run_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    boundaries = DEFAULT_PHASE_BOUNDARIES  # [0, 96, 180, 300, 420, 600]
    # full boundary list for segment_traces
    full_boundaries = boundaries + [FULL_SEQ_LEN]  # [0, 96, 180, 300, 420, 600, 720]

    print("=" * 70)
    print("TimeGAN S20 - Segment-Based Generation")
    print("=" * 70)
    print(f"Device          : {device}")
    print(f"Architecture    : GeneratorSeg (segment-based, NO encoder)")
    print(f"Segment length  : {SEGMENT_LEN} steps x {N_PHASES} phases = {FULL_SEQ_LEN}")
    print(f"Warmup          : statistical only (fm_stat + var_reg, {args.stat_warmup} epochs)")
    print(f"Var regression  : lambda={args.var_reg}  log-ratio")
    print(f"Segment FM      : lambda_fm_stat={args.fm_stat}  (per-workload overrides applied)")
    print(f"Adv epochs      : {args.adv_epochs}")
    print(f"Replica interp  : {args.aug_interp}")
    print(f"Output dir      : {out_dir}")
    print("=" * 70)

    all_results = []

    for workload in args.workloads:
        wl_overrides      = WORKLOAD_OVERRIDES[workload]
        lambda_smooth     = (args.smooth if args.smooth is not None
                             else wl_overrides["lambda_smooth"])
        n_disc_steps_wl   = wl_overrides["n_disc_steps"]
        lambda_fm_stat_wl = wl_overrides.get("lambda_fm_stat",  args.fm_stat)
        lambda_var_reg_wl = wl_overrides.get("lambda_var_reg",  args.var_reg)
        checkpoint_by_wl  = wl_overrides.get("checkpoint_by",   "best_W")

        print(f"\n  --- Workload: {workload.upper()} "
              f"(smooth={lambda_smooth}  n_disc={n_disc_steps_wl}"
              f"  fm={lambda_fm_stat_wl}  vr={lambda_var_reg_wl}"
              f"  ckpt={checkpoint_by_wl}) ---")

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

        # ------- S20 CORE: segment the full traces --------
        (all_segs, all_seg_rc,
         all_seg_ph, all_seg_ti) = segment_traces(
            raw_kept, replica_counts,
            boundaries, N_PHASES, SEGMENT_LEN)
        # all_segs: (N_pods * N_PHASES, SEGMENT_LEN, M)
        # all_seg_rc:  (N_pods * N_PHASES,) replica counts
        # all_seg_ph:  (N_pods * N_PHASES,) phase indices 0..5
        # all_seg_ti:  (N_pods * N_PHASES,) original trace index

        # Build train/val segment indices from trace-level splits
        # A segment belongs to train if its source trace is in train_idx_orig
        train_set = set(train_idx_orig.tolist())
        val_set   = set(val_idx.tolist())
        seg_train_idx = np.where(
            np.isin(all_seg_ti, list(train_set)))[0]
        seg_val_idx   = np.where(
            np.isin(all_seg_ti, list(val_set)))[0]

        # Optional replica interpolation at trace level, then re-segment
        if args.aug_interp:
            (raw_aug, r_aug, _, ti_aug) = replica_interpolation(
                raw_kept, replica_counts,
                np.arange(n_pods, dtype=np.int64), train_idx_orig)
            n_interp = len(ti_aug) - len(train_idx_orig)
            if n_interp > 0:
                (aug_segs, aug_seg_rc,
                 aug_seg_ph, aug_seg_ti) = segment_traces(
                    raw_aug[len(raw_kept):],   # interpolated traces only
                    r_aug[len(replica_counts):],
                    boundaries, N_PHASES, SEGMENT_LEN)
                # merge: shift aug_seg_ti to avoid collision with real indices
                aug_seg_ti_shifted = aug_seg_ti + len(raw_kept) + 1000000
                all_segs_tr  = np.concatenate([all_segs[seg_train_idx], aug_segs])
                all_rc_tr    = np.concatenate([all_seg_rc[seg_train_idx], aug_seg_rc])
                all_ph_tr    = np.concatenate([all_seg_ph[seg_train_idx], aug_seg_ph])
                # new contiguous indices for the merged training set
                seg_train_idx_merged = np.arange(len(all_segs_tr))
            else:
                n_interp = 0
                all_segs_tr       = all_segs[seg_train_idx]
                all_rc_tr         = all_seg_rc[seg_train_idx]
                all_ph_tr         = all_seg_ph[seg_train_idx]
                seg_train_idx_merged = np.arange(len(all_segs_tr))
        else:
            n_interp = 0
            all_segs_tr       = all_segs[seg_train_idx]
            all_rc_tr         = all_seg_rc[seg_train_idx]
            all_ph_tr         = all_seg_ph[seg_train_idx]
            seg_train_idx_merged = np.arange(len(all_segs_tr))

        n_train_segs = len(seg_train_idx_merged)
        n_val_segs   = len(seg_val_idx)
        n_train_orig_segs = len(seg_train_idx)

        print(f"  Kept metrics ({n_metrics}): {kept_names}")
        print(f"  Trace splits: train={len(train_idx_orig)} orig, "
              f"val={len(val_idx)}")
        print(f"  Segments:     train={n_train_segs} "
              f"({n_train_orig_segs} orig + {n_interp * N_PHASES} interp), "
              f"val={n_val_segs}")
        print(f"  Segment shape: ({SEGMENT_LEN}, {n_metrics}) x {N_PHASES} phases")

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

        t0 = time.time()
        generator, discriminator, best_stat, n_epochs, history = train(
            generator, discriminator, dl_train, dl_val,
            BASE_CONFIG, args.stat_warmup, args.adv_epochs, GEN_CFG,
            lambda_var_reg_wl, lambda_fm_stat_wl,
            lambda_smooth, n_disc_steps_wl, device,
            checkpoint_by=checkpoint_by_wl)
        elapsed = time.time() - t0

        # Evaluate on full 720-step traces (6 segments concatenated)
        phase_seq = build_phase_sequence(FULL_SEQ_LEN, boundaries)
        (real_orig, syn_orig, real_r_arr, syn_r_arr,
         vr, vr_mean, ac_diff, jump_ratio) = evaluate(
            generator, raw_kept,
            replica_counts, phase_seq, train_idx_orig,
            kept_names, norm_params, 5, device)

        ref_vr   = REFERENCE_VR.get(workload, {})
        ref_jump = REFERENCE_JUMP.get(workload, {})
        print(f"\n  Results (s20 {run_tag}):")
        print(f"    best_stat     = {best_stat:.5f}")
        print(f"    var_ratio     = {vr_mean:.4f}  "
              f"(s19={ref_vr.get('s19','?'):.4f}  s9={ref_vr.get('s9','?'):.4f}  "
              f"lstm={ref_vr.get('lstm','?'):.4f})")
        print(f"    autocorr_diff = {ac_diff:.4f}")
        print(f"    jump_ratio    = {jump_ratio:.3f}x  "
              f"(s19={ref_jump.get('s19','?'):.3f}x  s9={ref_jump.get('s9','?'):.1f}x)")
        print(f"    epochs        = {n_epochs}  time = {elapsed:.1f}s")
        print(f"  Per-metric VR:")
        for j, m in enumerate(kept_names):
            flag = " <-- low"  if vr[j] < 0.5 else (
                   " <-- HIGH" if vr[j] > 3.0 else "")
            print(f"    {m:<22} {vr[j]:.4f}{flag}")

        run_label = f"s20 {run_tag} segB"
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
                "architecture":    "GeneratorSeg",
                "preprocessing":   "raw_normalized",
                "fm_stat_type":    "per_segment",
                "seg_len":         SEGMENT_LEN,
                "n_phases":        N_PHASES,
                **GEN_CFG,
                "disc_hidden":     DISC_CFG["hidden_dim"],
            }, fh, indent=2)

        result = {
            "workload":         workload,
            "stage":            "s20",
            "architecture":     "GeneratorSeg",
            "preprocessing":    "raw_normalized",
            "fm_stat_type":     "per_segment",
            "seg_len":          SEGMENT_LEN,
            "run_tag":          run_tag,
            "adv_epochs":       args.adv_epochs,
            "stat_warmup":      args.stat_warmup,
            "jitter":           args.jitter,
            "aug_interp":       args.aug_interp,
            "lambda_var_reg":   lambda_var_reg_wl,
            "var_loss_type":    "log_ratio",
            "lambda_fm_stat":   lambda_fm_stat_wl,
            "lambda_smooth":    lambda_smooth,
            "n_disc_steps_wl":  n_disc_steps_wl,
            "n_train_segs":     n_train_segs,
            "var_ratio_mean":   float(vr_mean),
            "var_ratio_per_metric": [float(v) for v in vr],
            "autocorr_diff":    float(ac_diff),
            "phase_jump_ratio": float(jump_ratio),
            "best_stat":        float(best_stat),
            "n_epochs":         n_epochs,
            "elapsed_s":        elapsed,
            "metrics_generated": kept_names,
            "ref_vr_lstm":      ref_vr.get("lstm"),
            "ref_vr_s9":        ref_vr.get("s9"),
            "ref_vr_s19":       ref_vr.get("s19"),
            "ref_jump_s9":      ref_jump.get("s9"),
            "ref_jump_s19":     ref_jump.get("s19"),
        }
        all_results.append(result)

    # Summary table
    print(f"\n{'=' * 72}")
    print(f"S20 SUMMARY  ({run_tag})")
    print(f"{'=' * 72}")
    print(f"  {'Workload':<12} {'VR S20':>8} {'VR S19':>8} {'VR S9':>8} "
          f"{'LSTM':>8} {'jump S20':>10} {'jump S19':>10}")
    print(f"  {'-' * 72}")
    vr_means = []
    for r in all_results:
        vr_means.append(r["var_ratio_mean"])
        ref_vr   = REFERENCE_VR.get(r["workload"], {})
        ref_jump = REFERENCE_JUMP.get(r["workload"], {})
        print(f"  {r['workload']:<12} "
              f"{r['var_ratio_mean']:>8.4f} "
              f"{ref_vr.get('s19', 0):>8.4f} "
              f"{ref_vr.get('s9',  0):>8.4f} "
              f"{ref_vr.get('lstm', 0):>8.4f} "
              f"{r['phase_jump_ratio']:>10.3f}x "
              f"{ref_jump.get('s19', 0):>10.3f}x")
    print(f"  {'-' * 72}")
    print(f"  {'MEAN':<12} {sum(vr_means)/len(vr_means):>8.4f}")
    print(f"\nTarget: mean VR > 0.8, all metrics in [0.5, 3.0], "
          f"jump ratio in [0.3, 5.0]")
    print(f"Reference: LSTM=0.612  S9=0.878 (honest balanced)  S19=1.056")

    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "results.json"
    with open(result_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\nResults: {result_path}")


if __name__ == "__main__":
    main()