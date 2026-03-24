"""
S29 Comparison Plots - Unified Cross-Workload Model
====================================================
Generates per-workload comparison plots for S29 unified model

Usage:
  python comparison_plots_s29.py --replica-counts 1 5 10
  python comparison_plots_s29.py --workloads bert gpt2 --replica-counts 5
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import torch
import argparse
import sys

# Import S29 model and utilities
sys.path.append('scripts/phase4/timegan')
from timegan_s29 import (
    GeneratorSegUnified, GEN_CFG, WORKLOADS, N_PHASES,
    SEGMENT_LEN, denormalize_unified, load_combined_data,
    get_unified_kept_metrics
)

# Plot configuration (8 metrics)
PLOT_METRICS = [
    {'idx': 0, 'name': 'CPU Usage', 'unit': 'cores'},
    {'idx': 2, 'name': 'CPU Pressure (PSI)', 'unit': '%', 'scale': 100},
    {'idx': 3, 'name': 'Latency (Avg)', 'unit': 'seconds'},
    {'idx': 4, 'name': 'Throughput', 'unit': 'req/s'},
    {'idx': 1, 'name': 'Pod Memory', 'unit': 'MB', 'scale': 1e6},
    {'idx': 5, 'name': 'GPU Utilization', 'unit': '%'},
    {'idx': 6, 'name': 'GPU Memory Used', 'unit': 'MB'},
    {'idx': 7, 'name': 'GPU Power', 'unit': 'W'},
]


def load_s29_generator(model_dir, device='cuda'):
    """Load trained S29 unified generator"""
    model_path = Path(model_dir) / 'generator.pt'
    
    if not model_path.exists():
        raise FileNotFoundError(f"S29 model not found: {model_path}")
    
    kept_idx, kept_names = get_unified_kept_metrics()
    n_metrics = len(kept_idx)
    
    gen = GeneratorSegUnified(SEGMENT_LEN, n_metrics, GEN_CFG).to(device)
    gen.load_state_dict(torch.load(model_path, map_location=device))
    gen.eval()
    
    return gen, kept_names


def generate_s29_trace(generator, workload_name, replica_count, 
                       kept_names, norm_params, device='cuda'):
    """Generate trace with S29 unified model"""
    
    workload_id = WORKLOADS.index(workload_name)
    r_norm = (replica_count - 1.0) / 9.0
    
    # Generate trace (conditioned on workload_id)
    trace_norm = generator.generate_trace(r_norm, workload_id, device, n_samples=1)[0]
    
    # Trim to 715 timesteps (real data length)
    trace_norm = trace_norm[:715]
    
    # Denormalize
    wl_ids = np.array([workload_id], dtype=np.int32)
    trace_denorm = denormalize_unified(
        trace_norm[np.newaxis, :, :], wl_ids, kept_names, norm_params
    )[0]
    
    return trace_denorm


def get_real_trace(workload_name, replica_count, raw_traces, 
                   replica_counts, workload_ids, kept_names, norm_params):
    """Get random real trace for comparison"""
    
    workload_id = WORKLOADS.index(workload_name)
    
    # Filter by workload and replica count
    mask = (workload_ids == workload_id) & (replica_counts == replica_count)
    matching = raw_traces[mask]
    
    if len(matching) == 0:
        raise ValueError(f"No real traces for {workload_name} r={replica_count}")
    
    # Random sample
    idx = np.random.randint(len(matching))
    trace_norm = matching[idx]
    
    # Denormalize
    wl_ids = np.array([workload_id], dtype=np.int32)
    trace_denorm = denormalize_unified(
        trace_norm[np.newaxis, :, :], wl_ids, kept_names, norm_params
    )[0]
    
    return trace_denorm


def create_comparison_plot(workload_name, replica_count, generator, 
                          raw_traces, replica_counts, workload_ids,
                          kept_names, norm_params, output_dir, device='cuda'):
    """Create single comparison plot"""
    
    print(f"  {workload_name.upper()} r={replica_count}...", end='')
    
    # Generate synthetic
    synth_trace = generate_s29_trace(
        generator, workload_name, replica_count, 
        kept_names, norm_params, device
    )
    
    # Get real
    real_trace = get_real_trace(
        workload_name, replica_count, raw_traces,
        replica_counts, workload_ids, kept_names, norm_params
    )
    
    # Create plot
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    time_steps = np.arange(715) * 5 / 60.0
    
    for plot_idx, metric_info in enumerate(PLOT_METRICS):
        row = plot_idx // 3
        col = plot_idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        idx = metric_info['idx']
        scale = metric_info.get('scale', 1.0)
        
        real_vals = real_trace[:, idx] / scale
        synth_vals = synth_trace[:, idx] / scale
        
        ax.plot(time_steps, real_vals, 'b-', lw=1.2, alpha=0.7, label='Real')
        ax.plot(time_steps, synth_vals, color='red', lw=1.2, alpha=0.7, label='S29')
        
        ax.set_title(metric_info['name'], fontsize=11, fontweight='bold')
        ax.set_xlabel('Time (min)', fontsize=9)
        ax.set_ylabel(metric_info['unit'], fontsize=9)
        ax.grid(True, alpha=0.3)
        
        if plot_idx == 0:
            ax.legend(fontsize=8)
    
    # Hide last subplot
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    
    fig.suptitle(f"S29 Unified: {workload_name.upper()} (r={replica_count})", 
                 fontsize=14, fontweight='bold', y=0.995)
    
    output_path = output_dir / f"{workload_name}_r{replica_count}_s29.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(" saved")


def main():
    parser = argparse.ArgumentParser(description='S29 comparison plots')
    parser.add_argument('--model-dir', type=str,
                        default='models/phase4/timegan_s29/s29_unified_vr04_fm12_ae150')
    parser.add_argument('--workloads', nargs='+', type=str,
                        default=['bert', 'gpt2', 'resnet152', 'whisper', 'yolo'])
    parser.add_argument('--replica-counts', nargs='+', type=int, default=[1, 5, 10])
    parser.add_argument('--output-dir', type=str, 
                        default='outputs/phase4/comparison_plots/s29')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("S29 COMPARISON PLOTS - Unified Cross-Workload Model")
    print("=" * 70)
    print(f"Model: {args.model_dir}")
    print(f"Device: {device}")
    print(f"Workloads: {args.workloads}")
    print(f"Replica counts: {args.replica_counts}")
    print(f"Total plots: {len(args.workloads) * len(args.replica_counts)}")
    print("=" * 70)
    
    # Load S29 generator
    print("\nLoading S29 model...")
    generator, kept_names = load_s29_generator(args.model_dir, device)
    print(f"Metrics ({len(kept_names)}): {kept_names}")
    
    # Load real data
    print("\nLoading real data...")
    data, norm_params = load_combined_data()
    
    kept_idx, _ = get_unified_kept_metrics()
    raw_traces = data['traces'][:, :, kept_idx].astype(np.float32)
    replica_counts = data['replica_counts']
    workload_ids = data['workload_ids']
    
    print(f"Total pods: {len(raw_traces)}")
    
    # Generate plots
    print("\nGenerating plots...")
    for r in args.replica_counts:
        print(f"\nReplica count r={r}:")
        for workload in args.workloads:
            try:
                create_comparison_plot(
                    workload, r, generator,
                    raw_traces, replica_counts, workload_ids,
                    kept_names, norm_params, output_dir, device
                )
            except Exception as e:
                print(f"  ERROR {workload}: {e}")
    
    print("\n" + "=" * 70)
    print(f"Done! Plots saved to: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()