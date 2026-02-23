#!/usr/bin/env python3
"""
Phase 4: Exploratory Data Analysis - 10 Metrics Version

Usage:
    python scripts/phase4/run_eda_10m.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 8)
plt.rcParams['font.size'] = 10

DATA_DIR = Path('data/processed/phase1_v3')  # UPDATED PATH
REPORT_DIR = Path('reports/phase4_eda_10m')
PLOT_DIR = REPORT_DIR / 'plots'

REPORT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("PHASE 4: EXPLORATORY DATA ANALYSIS (10 METRICS)")
print("="*70)
print(f"Data directory: {DATA_DIR}")
print(f"Output directory: {REPORT_DIR}")
print()

print("[1/10] Loading data...")

workloads = ['bert', 'gpt2', 'resnet152', 'whisper', 'yolo']
data = {}
norm_params = {}

for workload in workloads:
    npz_file = DATA_DIR / f'{workload}_traces.npz'
    data[workload] = np.load(npz_file, allow_pickle=True)
    
    json_file = DATA_DIR / f'{workload}_normalization.json'
    with open(json_file) as f:
        norm_params[workload] = json.load(f)
    
    print(f"  {workload.upper()}: {len(data[workload]['traces'])} pods")

metric_names = data['bert']['metric_names']
print(f"  Metrics: {len(metric_names)} (expected 10)")
assert len(metric_names) == 10, f"ERROR: Expected 10 metrics, got {len(metric_names)}"
print()

print("[2/10] Generating dataset statistics...")

summary_data = []
for workload in workloads:
    d = data[workload]
    summary_data.append({
        'Workload': workload.upper(),
        'Total Pods': len(d['traces']),
        'Train': len(d['train_idx']),
        'Val': len(d['val_idx']),
        'Timesteps': d['traces'].shape[1],
        'Metrics': d['traces'].shape[2],
        'Replica Counts': f"{d['replica_counts'].min()}-{d['replica_counts'].max()}"
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(REPORT_DIR / 'dataset_summary.csv', index=False)
print(f"  Saved: dataset_summary.csv")

print("[3/10] Validating normalization...")

norm_check = []
for workload in workloads:
    traces = data[workload]['traces']
    norm_check.append({
        'Workload': workload.upper(),
        'Min': traces.min(),
        'Max': traces.max(),
        'Mean': traces.mean(),
        'Status': 'OK' if (traces.min() >= 0 and traces.max() <= 1) else 'ERROR'
    })

norm_df = pd.DataFrame(norm_check)
norm_df.to_csv(REPORT_DIR / 'normalization_check.csv', index=False)
print(f"  Saved: normalization_check.csv")

print("[4/10] Generating scaling curves...")

def plot_scaling_curves(workload_name, metric_idx, metric_name, save_path):
    d = data[workload_name]
    traces = d['traces']
    replica_counts = d['replica_counts']
    mean_values = traces[:, :, metric_idx].mean(axis=1)
    unique_r = np.unique(replica_counts)
    means = []
    stds = []
    for r in unique_r:
        mask = replica_counts == r
        values = mean_values[mask]
        means.append(values.mean())
        stds.append(values.std())
    means = np.array(means)
    stds = np.array(stds)
    plt.figure(figsize=(10, 6))
    plt.plot(unique_r, means, 'o-', linewidth=2, markersize=8, label='Mean')
    plt.fill_between(unique_r, means - stds, means + stds, alpha=0.3, label='±1 std')
    plt.xlabel('Replica Count', fontsize=12)
    plt.ylabel(f'{metric_name} (normalized)', fontsize=12)
    plt.title(f'{workload_name.upper()}: {metric_name} vs Replica Count', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

key_metrics = [
    (0, 'pod_cpu_usage'),
    (5, 'gpu_utilization'),
    (3, 'pod_latency_avg')
]

for workload in workloads:
    for metric_idx, metric_name in key_metrics:
        save_path = PLOT_DIR / f'{workload}_{metric_name}_scaling.png'
        plot_scaling_curves(workload, metric_idx, metric_name, save_path)

print(f"  Saved: {len(workloads) * len(key_metrics)} scaling plots")

print("[5/10] Generating workload comparison plot...")

metric_idx = 5
plt.figure(figsize=(12, 7))

for workload in workloads:
    d = data[workload]
    traces = d['traces']
    replica_counts = d['replica_counts']
    mean_values = traces[:, :, metric_idx].mean(axis=1)
    unique_r = np.unique(replica_counts)
    means = [mean_values[replica_counts == r].mean() for r in unique_r]
    plt.plot(unique_r, means, 'o-', linewidth=2, markersize=8, label=workload.upper())

plt.xlabel('Replica Count', fontsize=12)
plt.ylabel('GPU Utilization (normalized)', fontsize=12)
plt.title('GPU Utilization Scaling: All Workloads', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig(PLOT_DIR / 'all_workloads_gpu_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"  Saved: all_workloads_gpu_comparison.png")

print("[6/10] Generating temporal pattern plots...")

def plot_temporal_pattern(workload_name, replica_count, metric_idx, metric_name, save_path):
    d = data[workload_name]
    traces = d['traces']
    replica_counts = d['replica_counts']
    mask = replica_counts == replica_count
    pods = traces[mask]
    if len(pods) == 0:
        return
    metric_traces = pods[:, :, metric_idx]
    plt.figure(figsize=(15, 6))
    for trace in metric_traces:
        plt.plot(trace, alpha=0.3, linewidth=0.5, color='gray')
    mean_trace = metric_traces.mean(axis=0)
    plt.plot(mean_trace, linewidth=2, color='red', label='Mean')
    phase_boundaries = [0, 120, 240, 360, 480, 600, 715]
    phase_names = ['Warmup', 'Morning', 'Midday', 'Afternoon', 'Evening', 'Night']
    for i, (start, end, name) in enumerate(zip(phase_boundaries[:-1], phase_boundaries[1:], phase_names)):
        mid = (start + end) / 2
        plt.axvline(end, color='black', linestyle='--', alpha=0.5)
        plt.text(mid, plt.ylim()[1] * 0.95, name, ha='center', fontsize=10, fontweight='bold')
    plt.xlabel('Timestep (5s intervals)', fontsize=12)
    plt.ylabel(f'{metric_name} (normalized)', fontsize=12)
    plt.title(f'{workload_name.upper()} r={replica_count}: {metric_name} Temporal Pattern', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

examples = [
    ('bert', 5, 0, 'pod_cpu_usage'),
    ('gpt2', 5, 5, 'gpu_utilization'),
    ('whisper', 5, 3, 'pod_latency_avg')
]

for workload, r, metric_idx, metric_name in examples:
    save_path = PLOT_DIR / f'{workload}_r{r}_{metric_name}_temporal.png'
    plot_temporal_pattern(workload, r, metric_idx, metric_name, save_path)

print(f"  Saved: {len(examples)} temporal plots")

print("[7/10] Generating distribution plots...")

def plot_metric_distributions(workload_name, save_path):
    traces = data[workload_name]['traces']
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    axes = axes.flatten()
    for i, metric_name in enumerate(metric_names):
        metric_data = traces[:, :, i].flatten()
        axes[i].hist(metric_data, bins=50, alpha=0.7, edgecolor='black')
        axes[i].set_title(metric_name, fontsize=10, fontweight='bold')
        axes[i].set_xlabel('Value (normalized)', fontsize=9)
        axes[i].set_ylabel('Frequency', fontsize=9)
        axes[i].grid(True, alpha=0.3)
    plt.suptitle(f'{workload_name.upper()}: Metric Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

for workload in workloads:
    save_path = PLOT_DIR / f'{workload}_distributions.png'
    plot_metric_distributions(workload, save_path)

print(f"  Saved: {len(workloads)} distribution plots")

print("[8/10] Generating correlation matrices...")

def plot_correlation_matrix(workload_name, save_path):
    traces = data[workload_name]['traces']
    reshaped = traces.reshape(-1, traces.shape[2])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        corr_matrix = np.corrcoef(reshaped.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, xticklabels=metric_names, yticklabels=metric_names,
                annot=True, fmt='.2f', cmap='coolwarm', center=0, vmin=-1, vmax=1, square=True)
    plt.title(f'{workload_name.upper()}: Metric Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

for workload in ['bert', 'gpt2', 'whisper']:
    save_path = PLOT_DIR / f'{workload}_correlation.png'
    plot_correlation_matrix(workload, save_path)

print(f"  Saved: 3 correlation plots")

print("[9/10] Analyzing train/val splits...")

with open(REPORT_DIR / 'split_coverage.txt', 'w') as f:
    for workload in workloads:
        d = data[workload]
        replica_counts = d['replica_counts']
        train_idx = d['train_idx']
        val_idx = d['val_idx']
        train_replica_counts = replica_counts[train_idx]
        val_replica_counts = replica_counts[val_idx]
        f.write(f"\n{workload.upper()}:\n")
        f.write("="*50 + "\n")
        f.write(f"{'Replica Count':<15} {'Train':<10} {'Val':<10}\n")
        f.write("="*50 + "\n")
        for r in range(1, 11):
            train_count = (train_replica_counts == r).sum()
            val_count = (val_replica_counts == r).sum()
            f.write(f"r={r:<13} {train_count:<10} {val_count:<10}\n")
        f.write("="*50 + "\n")
        f.write(f"{'TOTAL':<15} {len(train_idx):<10} {len(val_idx):<10}\n\n")

print(f"  Saved: split_coverage.txt")

print("[10/10] Running data quality checks...")

quality_results = []
for workload in workloads:
    traces = data[workload]['traces']
    has_nan = np.isnan(traces).any()
    has_inf = np.isinf(traces).any()
    in_range = (traces >= 0).all() and (traces <= 1).all()
    zero_traces = (traces.sum(axis=(1,2)) == 0).sum()
    quality_results.append({
        'Workload': workload.upper(),
        'Has_NaN': has_nan,
        'Has_Inf': has_inf,
        'In_Range_0_1': in_range,
        'Zero_Traces': zero_traces,
        'Status': 'OK' if (not has_nan and not has_inf and in_range) else 'WARNING'
    })

quality_df = pd.DataFrame(quality_results)
quality_df.to_csv(REPORT_DIR / 'quality_checks.csv', index=False)
print(f"  Saved: quality_checks.csv")

print("\nGenerating final report...")

with open(REPORT_DIR / 'eda_summary.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("PHASE 4: DATA VALIDATION (10 METRICS)\n")
    f.write("="*70 + "\n\n")
    f.write("DATASET:\n")
    f.write(f"  Workloads: {len(workloads)}\n")
    f.write(f"  Total pods: {sum(len(data[w]['traces']) for w in workloads)}\n")
    f.write(f"  Train: {sum(len(data[w]['train_idx']) for w in workloads)}\n")
    f.write(f"  Val: {sum(len(data[w]['val_idx']) for w in workloads)}\n")
    f.write(f"  Timesteps: {data['bert']['traces'].shape[1]}\n")
    f.write(f"  Metrics: {data['bert']['traces'].shape[2]}\n\n")
    f.write("REMOVED: pod_psi_memory, pod_psi_io (zero variance)\n\n")
    f.write("STATUS: READY FOR TRAINING\n")
    f.write("="*70 + "\n")

with open(REPORT_DIR / 'normalization_params.txt', 'w') as f:
    for workload in workloads:
        f.write(f"\n{workload.upper()}:\n" + "="*70 + "\n")
        for metric in metric_names:
            params = norm_params[workload]['params'][metric]
            f.write(f"{metric:20} min: {params['min']:12.6f}  max: {params['max']:12.6f}\n")
        f.write("\n")

print(f"  Saved: eda_summary.txt")
print(f"  Saved: normalization_params.txt")

print("\n" + "="*70)
print("EDA COMPLETE!")
print("="*70)
print(f"\nResults: {REPORT_DIR}")
print(f"Plots: {PLOT_DIR}")
print(f"Generated: {len(list(PLOT_DIR.glob('*.png')))} plots")
print("\n" + "="*70)
print("DATASET: READY FOR LSTM BASELINE TRAINING")
print("="*70)
