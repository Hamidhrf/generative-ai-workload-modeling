"""
S27 Boundary Smoothing Comparison
==================================
Compare S27 generated traces with and without post-processing smoothing.

Generates:
- VR comparison (raw vs smoothed)
- Side-by-side plots showing smoothing effect
- Quantitative discontinuity reduction metrics

Usage:
    python compare_s27_smoothing.py --workloads bert gpt2 resnet152 whisper yolo
    python compare_s27_smoothing.py --workload bert  # Single workload
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Import smoothing utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from boundary_smoothing import smooth_phase_boundaries

# Data paths
DATA_COMBINED = Path("data/processed/phase4/unified/combined_dataset.npz")
NORM_COMBINED = Path("data/processed/phase4/unified/combined_normalization.json")

WORKLOADS_ALL = ["bert", "gpt2", "resnet152", "whisper", "yolo"]

ALL_METRICS = [
    "pod_cpu_usage", "pod_memory_bytes", "pod_psi_cpu",
    "pod_latency_avg", "pod_throughput",
    "gpu_utilization", "gpu_memory_used", "gpu_memory_total",
    "gpu_power_watts", "gpu_temperature",
]

# S27 drop config (must match training exactly)
DROP_PER_WORKLOAD = {
    "bert":      {"gpu_memory_total", "gpu_memory_used", "gpu_temperature"},
    "gpt2":      {"gpu_memory_total", "gpu_temperature"},
    "resnet152": {"gpu_memory_total", "gpu_memory_used", "gpu_temperature"},
    "whisper":   {"gpu_memory_total", "gpu_memory_used", "pod_throughput", "gpu_temperature"},  # 4 dropped, 6 kept
    "yolo":      {"gpu_memory_total", "gpu_memory_used", "gpu_temperature"},
}

# Phase config
DEFAULT_PHASE_BOUNDARIES = [0, 96, 180, 300, 420, 600]
N_PHASES = 6
SEGMENT_LEN = 120

GEN_CFG = {
    "hidden_dim": 128,
    "num_layers": 2,
    "latent_dim": 64,
    "replica_embed_dim": 16,
    "phase_embed_dim": 8,
    "dropout": 0.1,
}


class GeneratorSeg(nn.Module):
    """S27 generator."""
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
    
    @torch.no_grad()
    def generate_trace(self, r_norm_val, device, n_samples=1):
        """Generate full trace."""
        self.eval()
        segments = []
        for ph in range(N_PHASES):
            r_norm = torch.full((n_samples,), r_norm_val, dtype=torch.float32, device=device)
            ph_idx = torch.full((n_samples,), ph, dtype=torch.long, device=device)
            seg = self(r_norm, ph_idx)
            segments.append(seg.cpu().numpy())
        full_trace = np.concatenate(segments, axis=1)
        return full_trace[:, :715, :]


def denormalize(traces_norm, norm_params, kept_names):
    """Denormalize traces."""
    traces = traces_norm.copy()
    for i, metric in enumerate(kept_names):
        if metric in norm_params:
            traces[:, :, i] = traces[:, :, i] * norm_params[metric]["std"] + norm_params[metric]["mean"]
    return traces


def compute_vr_per_metric(real_traces, syn_traces):
    """Compute variance ratio per metric."""
    real_var = real_traces.var(axis=(0, 1))
    syn_var = syn_traces.var(axis=(0, 1))
    vr = syn_var / (real_var + 1e-8)
    return vr


def measure_boundary_discontinuity(traces, boundaries):
    """Measure discontinuity magnitude at boundaries."""
    discontinuities = []
    for boundary in boundaries:
        if boundary > 0 and boundary < traces.shape[1] - 1:
            # Average across all samples and metrics
            jump = np.abs(traces[:, boundary, :] - traces[:, boundary-1, :])
            discontinuities.append(jump.mean())
    return np.mean(discontinuities)


def compare_workload(workload_name, device, window_size=5):
    """Compare raw vs smoothed traces for one workload."""
    print(f"\n{'='*80}")
    print(f"SMOOTHING COMPARISON: {workload_name.upper()}")
    print(f"{'='*80}")
    
    # Load data
    data = np.load(DATA_COMBINED, allow_pickle=True)
    with open(NORM_COMBINED) as f:
        norm = json.load(f)
    
    workload_id = WORKLOADS_ALL.index(workload_name)
    all_traces = data["traces"]
    all_rc = data["replica_counts"]
    all_wl = data["workload_ids"]
    val_idx = data["val_idx"]
    
    # Filter workload
    wl_mask = (all_wl == workload_id)
    traces = all_traces[wl_mask]
    rc = all_rc[wl_mask]
    
    # Get kept metrics
    drop = DROP_PER_WORKLOAD.get(workload_name, set())
    kept_idx = [i for i, m in enumerate(ALL_METRICS) if m not in drop]
    kept_names = [ALL_METRICS[i] for i in kept_idx]
    n_metrics = len(kept_names)
    
    traces = traces[:, :, kept_idx].astype(np.float32)
    
    # Validation set
    orig_to_new = {}
    new_idx = 0
    for old_idx in range(len(all_traces)):
        if wl_mask[old_idx]:
            orig_to_new[old_idx] = new_idx
            new_idx += 1
    
    new_val_idx = np.array([orig_to_new[i] for i in val_idx if i in orig_to_new])
    
    real_traces_val = traces[new_val_idx]
    real_rc_val = rc[new_val_idx]
    
    print(f"Metrics ({n_metrics}): {kept_names}")
    print(f"Validation: {len(new_val_idx)} pods")
    
    # Load S27 model
    # S27 structure: models/phase4/timegan_s27/{run_tag}/{workload}/
    import glob
    base_dir = Path("models/phase4/timegan_s27")
    
    # Find any run tag directory
    run_tags = [d for d in base_dir.glob("s27_seg_*") if d.is_dir()]
    
    if not run_tags:
        print(f"ERROR: No S27 run directories found in {base_dir}")
        return None
    
    # Use first run tag, then look for workload subdirectory
    model_dir = run_tags[0] / workload_name
    
    if not model_dir.exists():
        print(f"ERROR: No S27 model found for {workload_name} in {run_tags[0]}")
        return None
    
    print(f"Model: {model_dir}")
    
    generator = GeneratorSeg(SEGMENT_LEN, n_metrics, GEN_CFG).to(device)
    generator.load_state_dict(torch.load(model_dir / "generator.pt", map_location=device))
    generator.eval()
    
    # Generate traces (raw)
    print("\nGenerating traces...")
    unique_r = np.unique(real_rc_val)
    n_per_r = 5  # Generate 5 traces per replica count
    
    syn_traces_raw = []
    syn_traces_smooth = []
    
    for r_val in unique_r:
        r_norm = (r_val - 1.0) / 9.0
        traces_gen = generator.generate_trace(r_norm, device, n_samples=n_per_r)
        
        for trace in traces_gen:
            syn_traces_raw.append(trace)
            # Apply smoothing
            # Generated traces have boundaries at 120-timestep intervals (not training boundaries)
            actual_boundaries = [120, 240, 360, 480, 600]  # 10, 20, 30, 40, 50 min
            smooth_trace = smooth_phase_boundaries(trace, actual_boundaries, window_size)
            syn_traces_smooth.append(smooth_trace)
    
    syn_traces_raw = np.array(syn_traces_raw)
    syn_traces_smooth = np.array(syn_traces_smooth)
    
    print(f"Generated: {len(syn_traces_raw)} raw, {len(syn_traces_smooth)} smooth")
    
    # Denormalize all
    real_denorm = denormalize(real_traces_val, norm[workload_name], kept_names)
    syn_raw_denorm = denormalize(syn_traces_raw, norm[workload_name], kept_names)
    syn_smooth_denorm = denormalize(syn_traces_smooth, norm[workload_name], kept_names)
    
    # Compute VR
    vr_raw = compute_vr_per_metric(real_denorm, syn_raw_denorm)
    vr_smooth = compute_vr_per_metric(real_denorm, syn_smooth_denorm)
    
    mean_vr_raw = np.mean(vr_raw)
    mean_vr_smooth = np.mean(vr_smooth)
    
    # Measure boundary discontinuity
    actual_boundaries = [120, 240, 360, 480, 600]  # Generated traces use these
    disc_real = measure_boundary_discontinuity(real_denorm, actual_boundaries)
    disc_raw = measure_boundary_discontinuity(syn_raw_denorm, actual_boundaries)
    disc_smooth = measure_boundary_discontinuity(syn_smooth_denorm, actual_boundaries)
    
    print(f"\nVARIANCE RATIO:")
    print(f"  Raw:     {mean_vr_raw:.3f}")
    print(f"  Smooth:  {mean_vr_smooth:.3f}")
    print(f"  Change:  {mean_vr_smooth - mean_vr_raw:+.3f}")
    
    print(f"\nBOUNDARY DISCONTINUITY:")
    print(f"  Real:    {disc_real:.4f}")
    print(f"  Raw:     {disc_raw:.4f} ({disc_raw/disc_real*100:.1f}% of real)")
    print(f"  Smooth:  {disc_smooth:.4f} ({disc_smooth/disc_real*100:.1f}% of real)")
    print(f"  Reduction: {(1 - disc_smooth/disc_raw)*100:.1f}%")
    
    # Create comparison plots
    plot_dir = Path(f"outputs/phase4/smoothing_comparison/{workload_name}")
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Side-by-side for r=5
    r_mask = (real_rc_val == 5)
    if r_mask.sum() > 0:
        plot_comparison(
            real_denorm[r_mask][0], 
            syn_raw_denorm[0],
            syn_smooth_denorm[0],
            kept_names,
            workload_name,
            plot_dir / f"{workload_name}_r5_comparison.png"
        )
    
    # Plot 2: Per-metric VR comparison
    plot_vr_comparison(vr_raw, vr_smooth, kept_names, workload_name,
                       plot_dir / f"{workload_name}_vr_comparison.png")
    
    return {
        "workload": workload_name,
        "vr_raw": float(mean_vr_raw),
        "vr_smooth": float(mean_vr_smooth),
        "vr_change": float(mean_vr_smooth - mean_vr_raw),
        "disc_real": float(disc_real),
        "disc_raw": float(disc_raw),
        "disc_smooth": float(disc_smooth),
        "disc_reduction_pct": float((1 - disc_smooth/disc_raw)*100),
        "vr_per_metric_raw": vr_raw.tolist(),
        "vr_per_metric_smooth": vr_smooth.tolist(),
        "metric_names": kept_names,
    }


def plot_comparison(real_trace, raw_trace, smooth_trace, metric_names, workload, save_path):
    """Plot side-by-side comparison for 4 key metrics."""
    # Select 4 representative metrics
    key_metrics = []
    for m in ["pod_cpu_usage", "gpu_utilization", "pod_latency_avg", "pod_throughput"]:
        if m in metric_names:
            key_metrics.append(metric_names.index(m))
    
    if len(key_metrics) < 4:
        key_metrics = list(range(min(4, len(metric_names))))
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    time_min = np.arange(715) * 5 / 60.0
    
    # Actual generated boundaries (6 phases × 120 timesteps)
    actual_boundaries = [120, 240, 360, 480, 600]
    
    for i, metric_idx in enumerate(key_metrics):
        ax = axes[i]
        
        # Plot traces
        ax.plot(time_min, real_trace[:, metric_idx], 'k-', alpha=0.5, lw=1.5, label='Real')
        ax.plot(time_min, raw_trace[:, metric_idx], 'b-', alpha=0.7, lw=1.2, label='Raw (with spikes)')
        ax.plot(time_min, smooth_trace[:, metric_idx], 'r-', alpha=0.7, lw=1.2, label='Smoothed')
        
        # Mark actual boundaries (not training boundaries)
        for b in actual_boundaries:
            b_min = b * 5 / 60.0
            ax.axvline(b_min, color='gray', linestyle='--', alpha=0.3, lw=1)
        
        ax.set_ylabel(metric_names[metric_idx], fontsize=10)
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend(loc='upper right', fontsize=9)
            ax.set_title(f'{workload.upper()} - Boundary Smoothing Effect (r=5)', fontsize=12, fontweight='bold')
        
        if i == 3:
            ax.set_xlabel('Time (minutes)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_vr_comparison(vr_raw, vr_smooth, metric_names, workload, save_path):
    """Plot per-metric VR comparison."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    ax.bar(x - width/2, vr_raw, width, label='Raw', alpha=0.7, color='blue')
    ax.bar(x + width/2, vr_smooth, width, label='Smoothed', alpha=0.7, color='red')
    
    # Target line
    ax.axhline(0.8, color='green', linestyle='--', lw=2, alpha=0.5, label='Target (0.8)')
    ax.axhline(1.0, color='gray', linestyle='-', lw=1, alpha=0.3, label='Ideal (1.0)')
    
    ax.set_xlabel('Metric', fontsize=11)
    ax.set_ylabel('Variance Ratio', fontsize=11)
    ax.set_title(f'{workload.upper()} - Variance Ratio Comparison (Raw vs Smoothed)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="S27 Smoothing Comparison")
    parser.add_argument("--workloads", nargs='+', type=str,
                        default=WORKLOADS_ALL,
                        help="Workloads to compare")
    parser.add_argument("--window_size", type=int, default=5,
                        help="Smoothing window size (default: 5)")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*80)
    print("S27 BOUNDARY SMOOTHING COMPARISON")
    print("="*80)
    print(f"Device: {device}")
    print(f"Window size: {args.window_size}")
    print(f"Workloads: {args.workloads}")
    print("="*80)
    
    results = {}
    
    for wl in args.workloads:
        result = compare_workload(wl, device, args.window_size)
        if result:
            results[wl] = result
    
    # Summary
    out_dir = Path("outputs/phase4/smoothing_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "smoothing_results.json", "w") as f:
        json.dump({
            "window_size": args.window_size,
            "workloads": results,
            "summary": {
                "mean_vr_raw": np.mean([r["vr_raw"] for r in results.values()]),
                "mean_vr_smooth": np.mean([r["vr_smooth"] for r in results.values()]),
                "mean_disc_reduction": np.mean([r["disc_reduction_pct"] for r in results.values()]),
            }
        }, f, indent=2)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\n{'Workload':<15} {'VR Raw':<10} {'VR Smooth':<10} {'Change':<10} {'Disc Reduction'}")
    print("-"*80)
    for wl, res in results.items():
        print(f"{wl:<15} {res['vr_raw']:<10.3f} {res['vr_smooth']:<10.3f} "
              f"{res['vr_change']:+<10.3f} {res['disc_reduction_pct']:.1f}%")
    
    summary = {
        "mean_vr_raw": np.mean([r["vr_raw"] for r in results.values()]),
        "mean_vr_smooth": np.mean([r["vr_smooth"] for r in results.values()]),
        "mean_disc_reduction": np.mean([r["disc_reduction_pct"] for r in results.values()]),
    }
    
    print("-"*80)
    print(f"{'MEAN':<15} {summary['mean_vr_raw']:<10.3f} {summary['mean_vr_smooth']:<10.3f} "
          f"{summary['mean_vr_smooth'] - summary['mean_vr_raw']:+<10.3f} {summary['mean_disc_reduction']:.1f}%")
    
    print("\n" + "="*80)
    print("RESULTS SAVED")
    print("="*80)
    print(f"JSON: {out_dir / 'smoothing_results.json'}")
    print(f"Plots: {out_dir / '<workload>/'}")
    
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    if summary["mean_vr_smooth"] > summary["mean_vr_raw"] + 0.05:
        print("WARNING: Smoothing INCREASES variance ratio")
        print("  -> May indicate smoothing is too aggressive")
        print("  -> Try smaller window_size (e.g., 3)")
    elif abs(summary["mean_vr_smooth"] - summary["mean_vr_raw"]) < 0.02:
        print("GOOD: Smoothing preserves VR quality")
        print(f"  -> Discontinuity reduced by {summary['mean_disc_reduction']:.1f}%")
        print("  -> Recommend using smoothing in S33 eval")
    else:
        print("INFO: Smoothing slightly changes VR")
        print("  -> Evaluate trade-off: VR vs visual quality")


if __name__ == "__main__":
    main()