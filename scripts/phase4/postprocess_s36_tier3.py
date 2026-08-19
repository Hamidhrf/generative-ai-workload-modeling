"""
Adapted from postprocess_s36.py for H100 Tier 3 data. Only
MODEL_PATHS and every data_dir default (6 function
signatures) changed to point at data/processed/tier3/unified
and models/phase4/timegan_s36_tier3. Post-processing logic
(cosine blend window=20, adaptive filter, memory
reconstruction) is byte-identical to the original -- frozen
per the S36 Tier 3 retrain plan.

S36 Post-Processing Pipeline
=============================
Improve S36 synthetic trace quality before Kwok export.

Three fixes applied in order:
  1. Phase boundary smoothing (cosine blend, wider window)
  2. Per-metric adaptive filtering (smooth metrics get smoothed, noisy ones kept)
  3. Pod memory reconstruction (replace GAN output with realistic constant + noise)

Also reconstructs all dropped metrics (gpu_memory_total, gpu_temperature, etc.)
for complete 10-metric Kwok output.

Usage:
  python postprocess_s36.py --workloads bert gpt2 resnet152 whisper yolo
  python postprocess_s36.py --replica-counts 1 5 10 --samples 5
  python postprocess_s36.py --plot  # Generate before/after comparison plots
"""

import numpy as np
import json
import argparse
from pathlib import Path
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_METRICS = [
    'pod_cpu_usage',       # 0
    'pod_memory_bytes',    # 1
    'pod_psi_cpu',         # 2
    'pod_latency_avg',     # 3
    'pod_throughput',      # 4
    'gpu_utilization',     # 5
    'gpu_memory_used',     # 6
    'gpu_memory_total',    # 7
    'gpu_power_watts',     # 8
    'gpu_temperature',     # 9
]

# S36 trains on these 7 metrics (indices in 10-metric array)
TRAINED_INDICES = [0, 1, 2, 3, 4, 5, 8]
TRAINED_NAMES = [
    'pod_cpu_usage', 'pod_memory_bytes', 'pod_psi_cpu',
    'pod_latency_avg', 'pod_throughput', 'gpu_utilization', 'gpu_power_watts'
]

# Dropped metrics to reconstruct as constants
DROPPED_INDICES = [6, 7, 9]  # gpu_memory_used, gpu_memory_total, gpu_temperature

N_PHASES = 6
SEGMENT_LEN = 120
PHASE_BOUNDARIES = [120, 240, 360, 480, 600]  # Generated trace boundaries

WORKLOADS = ["bert", "gpt2", "resnet152", "whisper", "yolo"]

GEN_CFG = {
    "hidden_dim": 128,
    "num_layers": 2,
    "latent_dim": 64,
    "replica_embed_dim": 16,
    "phase_embed_dim": 8,
    "dropout": 0.1,
}

# ---------------------------------------------------------------------------
# Metric behavior classification (from visual analysis of r=5 plots)
# ---------------------------------------------------------------------------
# "smooth" = real data is relatively smooth/stable -> apply moving average
# "noisy"  = real data has genuine high-frequency variation -> leave alone
# "flat"   = real data is near-constant -> reconstruct outside model

METRIC_BEHAVIOR = {
    'pod_cpu_usage':    'noisy',    # Real has genuine fluctuations
    'pod_memory_bytes': 'flat',     # Real is near-constant per workload
    'pod_psi_cpu':      'noisy',    # Real has spiky behavior
    'pod_latency_avg':  'noisy',    # Real has genuine noise
    'pod_throughput':   'smooth',   # Real has slow transitions, not spikes
    'gpu_utilization':  'noisy',    # Real has genuine variation
    'gpu_power_watts':  'smooth',   # Real is relatively smooth
}

# Per-workload overrides (some workloads differ from default)
WORKLOAD_OVERRIDES = {
    'whisper': {
        'pod_throughput': 'noisy',  # Whisper throughput has genuine variation
        'gpu_power_watts': 'noisy', # Whisper GPU power is more variable
    },
    'gpt2': {
        'gpu_utilization': 'smooth',  # GPT2 GPU util is more stable at high load
    },
}


# ---------------------------------------------------------------------------
# Generator (identical to S36 training architecture)
# ---------------------------------------------------------------------------

class GeneratorSeg(nn.Module):
    def __init__(self, seg_len, n_metrics, cfg):
        super().__init__()
        self.seg_len = seg_len
        self.n_metrics = n_metrics
        hidden = cfg["hidden_dim"]
        n_layers = cfg["num_layers"]
        latent_dim = cfg["latent_dim"]
        r_emb_dim = cfg["replica_embed_dim"]
        ph_emb_dim = cfg["phase_embed_dim"]
        dropout = cfg["dropout"] if n_layers > 1 else 0.0

        self.r_embed = nn.Sequential(nn.Linear(1, r_emb_dim), nn.Tanh())
        self.ph_embed = nn.Embedding(N_PHASES + 1, ph_emb_dim)

        init_in = latent_dim + r_emb_dim + ph_emb_dim
        self.h_init = nn.Sequential(nn.Linear(init_in, hidden * n_layers), nn.Tanh())
        self.c_init = nn.Sequential(nn.Linear(init_in, hidden * n_layers), nn.Tanh())

        dec_in_dim = latent_dim + r_emb_dim + ph_emb_dim
        self.dec_rnn = nn.LSTM(dec_in_dim, hidden, n_layers,
                               batch_first=True, dropout=dropout)

        self.out_fc = nn.Linear(hidden, n_metrics)
        self.out_act = nn.Sigmoid()

        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden

    def forward(self, r_norm, phase_idx, z=None):
        B = r_norm.shape[0]
        T = self.seg_len
        device = r_norm.device

        if z is None:
            z = torch.randn(B, self.latent_dim, device=device)

        r_emb = self.r_embed(r_norm.unsqueeze(-1))
        ph_emb = self.ph_embed(phase_idx)

        zrp = torch.cat([z, r_emb, ph_emb], dim=-1)
        h0 = self.h_init(zrp)
        c0 = self.c_init(zrp)
        h0 = h0.view(B, self.n_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        c0 = c0.view(B, self.n_layers, self.hidden_dim).permute(1, 0, 2).contiguous()

        z_exp = z.unsqueeze(1).expand(-1, T, -1)
        r_exp = r_emb.unsqueeze(1).expand(-1, T, -1)
        ph_exp = ph_emb.unsqueeze(1).expand(-1, T, -1)
        dec_in = torch.cat([z_exp, r_exp, ph_exp], dim=-1)

        dec_out, _ = self.dec_rnn(dec_in, (h0, c0))
        return self.out_act(self.out_fc(dec_out))


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

MODEL_PATHS = {
    "bert": "models/phase4/timegan_s36_tier3/s36_bert_vr05_fm15/generator.pt",
    "gpt2": "models/phase4/timegan_s36_tier3/s36_gpt2_vr03_fm20/generator.pt",
    "resnet152": "models/phase4/timegan_s36_tier3/s36_resnet152_vr04_fm10/generator.pt",
    "whisper": "models/phase4/timegan_s36_tier3/s36_whisper_vr05_fm05/generator.pt",
    "yolo": "models/phase4/timegan_s36_tier3/s36_yolo_vr03_fm12/generator.pt",
}


def load_normalization_params(workload, data_dir='data/processed/tier3/unified'):
    norm_path = Path(data_dir) / "combined_normalization.json"
    with open(norm_path, 'r') as f:
        norm_data = json.load(f)
    return norm_data[workload]['params']


def load_real_data(data_dir='data/processed/tier3/unified'):
    data_path = Path(data_dir) / "combined_dataset.npz"
    return np.load(data_path, allow_pickle=True)


def denormalize_trace(trace_norm, norm_params, metric_names):
    trace_denorm = np.zeros_like(trace_norm)
    for i, name in enumerate(metric_names):
        p = norm_params[name]
        trace_denorm[:, i] = trace_norm[:, i] * (p['max'] - p['min']) + p['min']
    return trace_denorm


def load_generator(workload, device='cuda'):
    ckpt_path = Path(MODEL_PATHS[workload])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"S36 checkpoint not found: {ckpt_path}")
    gen = GeneratorSeg(SEGMENT_LEN, 7, GEN_CFG).to(device)
    gen.load_state_dict(torch.load(ckpt_path, map_location=device))
    gen.eval()
    return gen


# ---------------------------------------------------------------------------
# Step 0: Raw generation (before post-processing)
# ---------------------------------------------------------------------------

def generate_raw_trace(gen, replica_count, device='cuda'):
    """Generate raw 7-metric normalized trace from S36."""
    r_norm = torch.tensor([(replica_count - 1) / 9.0],
                          dtype=torch.float32, device=device)
    phases = []
    for ph in range(N_PHASES):
        phase_idx = torch.tensor([ph], dtype=torch.long, device=device)
        with torch.no_grad():
            seg = gen(r_norm, phase_idx)
        phases.append(seg.squeeze(0).cpu().numpy())
    return np.concatenate(phases, axis=0)[:715]  # (715, 7) normalized


# ---------------------------------------------------------------------------
# Step 1: Phase boundary smoothing (cosine blend)
# ---------------------------------------------------------------------------

def cosine_blend_boundaries(trace, boundaries=None, window=20):
    """
    Smooth phase boundaries with cosine blending.

    At each boundary, blend the left and right segments over a window
    using a raised-cosine (Hann) weight curve. This eliminates sharp
    jumps while preserving the character of each phase.

    Args:
        trace: (T, M) array
        boundaries: list of boundary timestep indices
        window: total blend width (timesteps on each side of boundary)

    Returns:
        Smoothed trace (T, M)
    """
    if boundaries is None:
        boundaries = PHASE_BOUNDARIES

    out = trace.copy()
    T = trace.shape[0]

    for b in boundaries:
        half = window // 2
        lo = max(0, b - half)
        hi = min(T, b + half)
        length = hi - lo

        if length < 4:
            continue

        # Values at edges of the blend window
        left_val = trace[lo]
        right_val = trace[hi - 1]

        # Cosine weight: 0 at left edge -> 1 at right edge
        t = np.linspace(0, np.pi, length)
        weight = (1 - np.cos(t)) / 2.0  # 0 -> 1 smooth curve
        weight = weight[:, np.newaxis]   # (length, 1) for broadcasting

        out[lo:hi] = (1 - weight) * left_val + weight * right_val

    return out


# ---------------------------------------------------------------------------
# Step 2: Per-metric adaptive filtering
# ---------------------------------------------------------------------------

def adaptive_filter(trace, workload, metric_names):
    """
    Apply per-metric filtering based on known behavior.

    - 'smooth' metrics: rolling mean (window=15) to remove GAN jitter
    - 'noisy' metrics: left unchanged (GAN noise is realistic)
    - 'flat' metrics: handled separately in step 3

    Args:
        trace: (T, M) denormalized array
        workload: workload name
        metric_names: list of metric names matching trace columns

    Returns:
        Filtered trace (T, M)
    """
    out = trace.copy()
    overrides = WORKLOAD_OVERRIDES.get(workload, {})

    for i, name in enumerate(metric_names):
        behavior = overrides.get(name, METRIC_BEHAVIOR.get(name, 'noisy'))

        if behavior == 'smooth':
            # Rolling mean with edge-aware padding
            kernel_size = 15
            pad = kernel_size // 2
            col = out[:, i]
            padded = np.pad(col, pad, mode='edge')
            smoothed = np.convolve(padded, np.ones(kernel_size) / kernel_size, mode='valid')
            out[:, i] = smoothed[:len(col)]

    return out


# ---------------------------------------------------------------------------
# Step 3: Pod memory reconstruction
# ---------------------------------------------------------------------------

def compute_memory_stats(data_dir='data/processed/tier3/unified'):
    """
    Compute per-workload per-replica pod_memory_bytes statistics from real data.
    Denormalizes from the normalized dataset before computing stats.

    Returns dict: {workload: {replica_count: {'mean': float, 'std': float}}}
    """
    data = load_real_data(data_dir)
    traces = data['traces']           # (N, 715, 10) NORMALIZED
    replica_counts = data['replica_counts']
    workload_ids = data['workload_ids']

    stats = {}
    for wid, wname in enumerate(WORKLOADS):
        # Load normalization params to denormalize pod_memory_bytes
        norm_params = load_normalization_params(wname, data_dir)
        mem_min = norm_params['pod_memory_bytes']['min']
        mem_max = norm_params['pod_memory_bytes']['max']

        stats[wname] = {}
        for r in sorted(set(replica_counts)):
            mask = (workload_ids == wid) & (replica_counts == r)
            if not mask.any():
                continue
            # pod_memory_bytes is index 1 in the 10-metric array (normalized)
            mem_norm = traces[mask][:, :, 1]  # (n_pods, 715)
            # Denormalize: x = x_norm * (max - min) + min
            mem_real = mem_norm * (mem_max - mem_min) + mem_min
            # Per-pod mean and std across time
            pod_means = mem_real.mean(axis=1)
            pod_stds = mem_real.std(axis=1)
            stats[wname][int(r)] = {
                'mean': float(pod_means.mean()),
                'std': float(pod_stds.mean()),
                'level_std': float(pod_means.std()),  # Variation between pods
            }
    return stats


def reconstruct_pod_memory(trace_length, workload, replica_count, memory_stats):
    """
    Generate realistic flat pod_memory trace from real statistics.

    Strategy: constant level + tiny Gaussian noise matching real std.
    If the exact replica count isn't in stats, interpolate from nearest.
    """
    wstats = memory_stats.get(workload, {})

    if replica_count in wstats:
        s = wstats[replica_count]
    else:
        # Find nearest available replica count
        available = sorted(wstats.keys())
        if not available:
            return np.zeros(trace_length)
        nearest = min(available, key=lambda x: abs(x - replica_count))
        s = wstats[nearest]

    mean_val = s['mean']
    noise_std = s['std']
    level_std = s['level_std']

    # Sample a pod-level offset (different pods have slightly different levels)
    pod_offset = np.random.normal(0, level_std)
    base = mean_val + pod_offset

    # Add tiny per-timestep noise
    noise = np.random.normal(0, noise_std, size=trace_length)
    return np.clip(base + noise, 0, None)


# ---------------------------------------------------------------------------
# Step 4: Reconstruct dropped metrics
# ---------------------------------------------------------------------------

def compute_dropped_metric_stats(data_dir='data/processed/tier3/unified'):
    """
    Compute per-workload statistics for all dropped metrics.
    Denormalizes from the normalized dataset before computing stats.

    Dropped: gpu_memory_used (6), gpu_memory_total (7), gpu_temperature (9)

    Returns dict: {workload: {metric_idx: {'mean': float, 'std': float}}}
    """
    data = load_real_data(data_dir)
    traces = data['traces']       # (N, 715, 10) NORMALIZED
    workload_ids = data['workload_ids']

    dropped_metric_names = {
        6: 'gpu_memory_used',
        7: 'gpu_memory_total',
        9: 'gpu_temperature',
    }

    stats = {}
    for wid, wname in enumerate(WORKLOADS):
        norm_params = load_normalization_params(wname, data_dir)
        mask = workload_ids == wid
        wtraces = traces[mask]  # (n_pods, 715, 10) normalized
        stats[wname] = {}
        for midx in DROPPED_INDICES:
            mname = dropped_metric_names[midx]
            m_min = norm_params[mname]['min']
            m_max = norm_params[mname]['max']
            # Denormalize
            vals_norm = wtraces[:, :, midx]
            vals_real = vals_norm * (m_max - m_min) + m_min
            stats[wname][midx] = {
                'mean': float(vals_real.mean()),
                'std': float(vals_real.std()),
                'per_pod_std': float(vals_real.mean(axis=1).std()),
                'temporal_std': float(vals_real.std(axis=1).mean()),
            }
    return stats


def reconstruct_dropped_metrics(trace_length, workload, dropped_stats):
    """
    Reconstruct dropped metrics as realistic constants + noise.

    Returns dict: {metric_idx: array of shape (trace_length,)}
    """
    result = {}
    wstats = dropped_stats.get(workload, {})

    for midx in DROPPED_INDICES:
        s = wstats.get(midx, {'mean': 0, 'temporal_std': 0, 'per_pod_std': 0})
        base = s['mean'] + np.random.normal(0, s['per_pod_std'])
        noise = np.random.normal(0, s['temporal_std'], size=trace_length)
        result[midx] = np.clip(base + noise, 0, None)

    return result


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def postprocess_trace(raw_norm_7, workload, replica_count, norm_params,
                      memory_stats, dropped_stats, blend_window=20):
    """
    Full post-processing pipeline for one synthetic trace.

    Input:  raw_norm_7  - (715, 7) normalized trace from generator
    Output: trace_full  - (715, 10) denormalized trace ready for Kwok

    Pipeline:
      1. Cosine-blend phase boundaries (in normalized space)
      2. Denormalize to real units
      3. Adaptive per-metric filtering
      4. Replace pod_memory with reconstructed constant
      5. Fill dropped metrics with reconstructed constants
    """
    T = raw_norm_7.shape[0]

    # Step 1: Boundary smoothing in normalized space
    smoothed_norm = cosine_blend_boundaries(raw_norm_7, window=blend_window)

    # Step 2: Denormalize
    denorm_7 = denormalize_trace(smoothed_norm, norm_params, TRAINED_NAMES)

    # Step 3: Adaptive filtering
    filtered_7 = adaptive_filter(denorm_7, workload, TRAINED_NAMES)

    # Step 4: Build full 10-metric trace
    trace_full = np.zeros((T, 10))

    for i, midx in enumerate(TRAINED_INDICES):
        if midx == 1:  # pod_memory_bytes -> reconstruct
            trace_full[:, 1] = reconstruct_pod_memory(
                T, workload, replica_count, memory_stats)
        else:
            trace_full[:, midx] = filtered_7[:, i]

    # Step 5: Fill dropped metrics
    dropped_vals = reconstruct_dropped_metrics(T, workload, dropped_stats)
    for midx, vals in dropped_vals.items():
        trace_full[:, midx] = vals

    return trace_full


# ---------------------------------------------------------------------------
# Batch generation with post-processing
# ---------------------------------------------------------------------------

def generate_postprocessed_traces(workload, replica_count, n_samples=1,
                                  blend_window=20, device='cuda',
                                  data_dir='data/processed/tier3/unified'):
    """
    Generate n_samples post-processed traces for a workload at given replica count.

    Returns: (n_samples, 715, 10) array in real units
    """
    gen = load_generator(workload, device)
    norm_params = load_normalization_params(workload, data_dir)
    memory_stats = compute_memory_stats(data_dir)
    dropped_stats = compute_dropped_metric_stats(data_dir)

    traces = []
    for _ in range(n_samples):
        raw = generate_raw_trace(gen, replica_count, device)
        processed = postprocess_trace(
            raw, workload, replica_count, norm_params,
            memory_stats, dropped_stats, blend_window)
        traces.append(processed)

    return np.array(traces)


# ---------------------------------------------------------------------------
# Comparison plotting (before/after)
# ---------------------------------------------------------------------------

def plot_before_after(workload, replica_count, output_dir, device='cuda',
                      data_dir='data/processed/tier3/unified'):
    """
    Generate side-by-side: Real vs Raw S36 vs Post-processed S36.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    gen = load_generator(workload, device)
    norm_params = load_normalization_params(workload, data_dir)
    memory_stats = compute_memory_stats(data_dir)
    dropped_stats = compute_dropped_metric_stats(data_dir)

    # Generate raw trace
    raw_norm = generate_raw_trace(gen, replica_count, device)

    # Raw denormalized (no post-processing)
    raw_denorm = denormalize_trace(raw_norm, norm_params, TRAINED_NAMES)

    # Post-processed
    processed = postprocess_trace(
        raw_norm, workload, replica_count, norm_params,
        memory_stats, dropped_stats, blend_window=20)

    # Get real trace for comparison
    data = load_real_data(data_dir)
    wid = WORKLOADS.index(workload)
    mask = (data['workload_ids'] == wid) & (data['replica_counts'] == replica_count)
    matching = data['traces'][mask]
    if len(matching) == 0:
        print(f"  No real traces for {workload} r={replica_count}, skipping plot")
        return
    real_trace = matching[np.random.randint(len(matching))]  # (715, 10)

    # Denormalize real trace trained metrics for comparison
    real_norm_7 = real_trace[:, TRAINED_INDICES]
    real_denorm_7 = denormalize_trace(real_norm_7, norm_params, TRAINED_NAMES)

    # Plot metrics
    plot_metrics = [
        {'trained_idx': 0, 'full_idx': 0, 'name': 'CPU Usage', 'unit': 'cores'},
        {'trained_idx': 2, 'full_idx': 2, 'name': 'CPU Pressure (PSI)', 'unit': 'fraction'},
        {'trained_idx': 3, 'full_idx': 3, 'name': 'Latency (Avg)', 'unit': 'seconds'},
        {'trained_idx': 4, 'full_idx': 4, 'name': 'Throughput', 'unit': 'req/s'},
        {'trained_idx': 1, 'full_idx': 1, 'name': 'Pod Memory', 'unit': 'bytes', 'scale': 1e6, 'scale_label': 'MB'},
        {'trained_idx': 5, 'full_idx': 5, 'name': 'GPU Utilization', 'unit': '%'},
        {'trained_idx': 6, 'full_idx': 8, 'name': 'GPU Power', 'unit': 'W'},
    ]

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
    time_min = np.arange(715) * 5 / 60.0

    for pi, m in enumerate(plot_metrics):
        row, col = pi // 3, pi % 3
        ax = fig.add_subplot(gs[row, col])

        scale = m.get('scale', 1.0)
        full_idx = m['full_idx']
        trained_idx = m['trained_idx']

        real_vals = real_denorm_7[:, trained_idx] / scale
        raw_vals = raw_denorm[:, trained_idx] / scale
        post_vals = processed[:, full_idx] / scale

        ax.plot(time_min, real_vals, 'b-', lw=1.0, alpha=0.6, label='Real')
        ax.plot(time_min, raw_vals, color='red', lw=0.8, alpha=0.5, label='Raw S36')
        ax.plot(time_min, post_vals, color='green', lw=1.2, alpha=0.8, label='Post-processed')

        # Mark phase boundaries
        for b in PHASE_BOUNDARIES:
            bmin = b * 5 / 60.0
            ax.axvline(bmin, color='gray', ls='--', alpha=0.3, lw=0.5)

        label = m.get('scale_label', m['unit'])
        ax.set_title(m['name'], fontsize=11, fontweight='bold')
        ax.set_xlabel('Time (min)', fontsize=9)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=8)

        if pi == 0:
            ax.legend(fontsize=7, loc='best')

    # Hide unused subplots
    for i in [7, 8]:
        ax = fig.add_subplot(gs[i // 3, i % 3])
        ax.axis('off')

    fig.suptitle(
        f"Post-Processing Effect: {workload.upper()} (r={replica_count})\n"
        f"Blue=Real, Red=Raw S36, Green=Post-processed",
        fontsize=13, fontweight='bold', y=0.995)

    out_path = Path(output_dir) / f"{workload}_r{replica_count}_postprocess.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='S36 Post-Processing Pipeline')
    parser.add_argument('--workloads', nargs='+', default=WORKLOADS)
    parser.add_argument('--replica-counts', nargs='+', type=int, default=[1, 5, 10])
    parser.add_argument('--samples', type=int, default=1,
                        help='Number of synthetic traces per workload/replica')
    parser.add_argument('--blend-window', type=int, default=20,
                        help='Cosine blend window width at phase boundaries')
    parser.add_argument('--plot', action='store_true',
                        help='Generate before/after comparison plots')
    parser.add_argument('--output-dir', type=str,
                        default='outputs/phase4/postprocessed/s36')
    parser.add_argument('--plot-dir', type=str,
                        default='outputs/phase4/comparison_plots/s36_postprocessed')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("S36 POST-PROCESSING PIPELINE")
    print("=" * 80)
    print(f"Workloads:     {args.workloads}")
    print(f"Replicas:      {args.replica_counts}")
    print(f"Samples each:  {args.samples}")
    print(f"Blend window:  {args.blend_window} timesteps")
    print(f"Output:        {output_dir}")
    print("=" * 80)

    # Precompute stats (once for all)
    print("\nComputing real data statistics...")
    memory_stats = compute_memory_stats()
    dropped_stats = compute_dropped_metric_stats()
    print("  Done.")

    if args.plot:
        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nGenerating before/after plots -> {plot_dir}")
        for r in args.replica_counts:
            print(f"\n  Replica count r={r}:")
            for w in args.workloads:
                plot_before_after(w, r, plot_dir, args.device)
        print("\nPlots complete!")

    # Generate and save post-processed traces
    print("\nGenerating post-processed traces...")
    for w in args.workloads:
        gen = load_generator(w, args.device)
        norm_params = load_normalization_params(w)

        for r in args.replica_counts:
            print(f"  {w.upper()} r={r} ({args.samples} samples)...", end='', flush=True)
            traces = []
            for _ in range(args.samples):
                raw = generate_raw_trace(gen, r, args.device)
                processed = postprocess_trace(
                    raw, w, r, norm_params,
                    memory_stats, dropped_stats, args.blend_window)
                traces.append(processed)

            traces = np.array(traces)  # (n_samples, 715, 10)
            save_path = output_dir / f"{w}_r{r}_s36_postprocessed.npy"
            np.save(save_path, traces)
            print(f" saved ({traces.shape})")

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print(f"Traces: {output_dir}")
    if args.plot:
        print(f"Plots:  {args.plot_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()