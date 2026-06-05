#!/usr/bin/env python3
"""
S27 Quality Evaluation - Comprehensive Assessment
Generates multiple plot types to evaluate synthetic trace quality.

Plot Types:
1. Correlation scatter plots - Real vs Synthetic per metric with Pearson r
2. Distribution comparisons - Histograms showing data distribution overlap
3. Variance ratio bar chart - Color-coded quality indicator
4. Autocorrelation plots - Temporal patterns within each trace (NOT correlation between real/synth)
5. Statistical accuracy summary - MAE, RMSE, Pearson r, VR
6. Time series overlays - Direct visual comparison

Autocorrelation Explanation:
- Measures correlation of a time series with ITSELF at different time lags
- Shows if model preserves temporal dependencies (e.g., if CPU at t=10 correlates with CPU at t=20)
- We plot autocorrelation curves for real and synthetic separately, then compare shapes

Usage:
    python evaluate_s27_quality.py --workload bert --replica-counts 1 5 10
    python evaluate_s27_quality.py --workload all --replica-counts 5
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import torch
import torch.nn as nn
import argparse
import json
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error


ALL_METRICS = [
    'pod_cpu_usage', 'pod_memory_bytes', 'pod_psi_cpu', 'pod_latency_avg',
    'pod_throughput', 'gpu_utilization', 'gpu_memory_used', 'gpu_power_watts'
]

# OLD system (raw data files have 10 metrics)
ALL_METRICS_OLD = [
    'pod_cpu_usage', 'pod_memory_bytes', 'pod_psi_cpu', 'pod_latency_avg',
    'pod_throughput', 'gpu_utilization', 'gpu_memory_used', 'gpu_memory_total',
    'gpu_power_watts', 'gpu_temperature'
]

# Mapping from new 8-metric index to old 10-metric index
NEW_TO_OLD_INDEX = {
    0: 0,  # pod_cpu_usage
    1: 1,  # pod_memory_bytes
    2: 2,  # pod_psi_cpu
    3: 3,  # pod_latency_avg
    4: 4,  # pod_throughput
    5: 5,  # gpu_utilization
    6: 6,  # gpu_memory_used
    7: 8,  # gpu_power_watts (was at index 8 in old system)
}

METRIC_UNITS = {
    0: 'cores',
    1: 'MB',
    2: '%',  # PSI as percentage
    3: 'seconds',
    4: 'req/s',
    5: '%',
    6: 'MB',
    7: 'W'
}

S27_TRAINED = {
    'bert':      [0, 1, 2, 3, 4, 5, 7],
    'gpt2':      [0, 1, 2, 3, 4, 5, 6, 7],
    'resnet152': [0, 1, 2, 3, 4, 5, 7],
    'whisper':   [0, 1, 2, 3, 5, 7],
    'yolo':      [0, 1, 2, 3, 4, 5, 7],
}

N_PHASES = 6
SEGMENT_LEN = 120
GEN_CFG = {"hidden_dim": 128, "num_layers": 2, "latent_dim": 64,
           "replica_embed_dim": 16, "phase_embed_dim": 8, "dropout": 0.1}


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
        self.dec_rnn = nn.LSTM(init_in, hidden, n_layers, batch_first=True, dropout=dropout)
        self.out_fc = nn.Linear(hidden, n_metrics)
        self.out_act = nn.Sigmoid()
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden
    
    def forward(self, r_norm, phase_idx, z=None):
        B = r_norm.shape[0]
        device = r_norm.device
        if z is None:
            z = torch.randn(B, self.latent_dim, device=device)
        r_emb = self.r_embed(r_norm.unsqueeze(-1))
        ph_emb = self.ph_embed(phase_idx)
        merged = torch.cat([z, r_emb, ph_emb], dim=-1)
        h0 = self.h_init(merged).view(self.n_layers, B, self.hidden_dim).contiguous()
        c0 = self.c_init(merged).view(self.n_layers, B, self.hidden_dim).contiguous()
        merged_exp = merged.unsqueeze(1).expand(-1, self.seg_len, -1)
        rnn_out, _ = self.dec_rnn(merged_exp, (h0, c0))
        return self.out_act(self.out_fc(rnn_out))


def load_normalization_params(workload, data_dir='data/processed/phase4/raw'):
    norm_path = Path(data_dir) / f"{workload}_normalization.json"
    with open(norm_path, 'r') as f:
        norm_data = json.load(f)
    return norm_data['params']


def denormalize_trace(trace_norm, norm_params, metric_indices, use_old_system=False):
    """Denormalize trace from [0,1] to original scale.
    
    Args:
        trace_norm: Normalized trace
        norm_params: Normalization parameters
        metric_indices: Indices of metrics
        use_old_system: If True, use ALL_METRICS_OLD (10 metrics), else ALL_METRICS (8 metrics)
    """
    trace_denorm = np.zeros_like(trace_norm)
    metric_list = ALL_METRICS_OLD if use_old_system else ALL_METRICS
    
    for i, idx in enumerate(metric_indices):
        metric_name = metric_list[idx]
        params = norm_params[metric_name]
        min_val = params['min']
        max_val = params['max']
        trace_denorm[:, i] = trace_norm[:, i] * (max_val - min_val) + min_val
    return trace_denorm


def load_generator(workload, device='cpu'):
    base = Path('models/phase4/timegan_s27/s27_seg_vr03_fm10_ae150')
    ckpt_path = base / workload / 'generator.pt'
    n_metrics = len(S27_TRAINED[workload])
    gen = GeneratorSeg(SEGMENT_LEN, n_metrics, GEN_CFG).to(device)
    gen.load_state_dict(torch.load(ckpt_path, map_location=device))
    gen.eval()
    return gen


def load_real_data(workload, data_dir='data/processed/phase4/raw'):
    trace_path = Path(data_dir) / f"{workload}_traces.npz"
    data = np.load(trace_path, allow_pickle=True)
    return data


def generate_trace_s27(workload, replica_count, device='cpu'):
    gen = load_generator(workload, device)
    real_data = load_real_data(workload)
    norm_params = load_normalization_params(workload)
    
    r_norm = torch.tensor([(replica_count - 1) / 9.0], dtype=torch.float32, device=device)
    phases = []
    for ph in range(N_PHASES):
        phase_idx = torch.tensor([ph], dtype=torch.long, device=device)
        seg = gen(r_norm, phase_idx)
        phases.append(seg.squeeze(0).cpu().detach().numpy())
    
    trace_norm = np.concatenate(phases, axis=0)[:715]
    trained_indices = S27_TRAINED[workload]
    # Denormalize using new 8-metric system (default use_old_system=False)
    trace_denorm_partial = denormalize_trace(trace_norm, norm_params, trained_indices)
    
    trace_full = np.zeros((715, 8))
    for i, idx in enumerate(trained_indices):
        trace_full[:, idx] = trace_denorm_partial[:, i]
    
    # Reconstruct missing metrics
    for new_idx in range(8):
        if new_idx not in trained_indices:
            # Map new 8-metric index to old 10-metric index for accessing raw data
            old_idx = NEW_TO_OLD_INDEX[new_idx]
            all_vals = real_data['traces'][:, :, old_idx]
            # Use new metric name from 8-metric system
            params = norm_params[ALL_METRICS[new_idx]]
            all_vals_denorm = all_vals * (params['max'] - params['min']) + params['min']
            trace_full[:, new_idx] = np.mean(all_vals_denorm)
    
    return trace_full


def get_real_traces(workload, replica_count):
    """Load real traces and extract only the 8 metrics we use (excluding gpu_memory_total, gpu_temperature)."""
    data = load_real_data(workload)
    norm_params = load_normalization_params(workload)
    traces_norm = data['traces']  # Shape: (N, 715, 10) - has all 10 metrics
    replica_counts = data['replica_counts']
    mask = (replica_counts == replica_count)
    matching = traces_norm[mask]  # Shape: (n_pods, 715, 10)
    
    traces_denorm = []
    for trace_norm in matching:
        # Denormalize all 10 metrics first (using old 10-metric system)
        trace_denorm_10 = denormalize_trace(trace_norm, norm_params, list(range(10)), use_old_system=True)
        
        # Extract only the 8 metrics we care about using the mapping
        trace_denorm_8 = np.zeros((715, 8), dtype=np.float32)
        for new_idx in range(8):
            old_idx = NEW_TO_OLD_INDEX[new_idx]
            trace_denorm_8[:, new_idx] = trace_denorm_10[:, old_idx]
        
        traces_denorm.append(trace_denorm_8)
    
    return np.array(traces_denorm)


def apply_psi_percentage(trace, metric_idx):
    """Convert PSI from fraction to percentage for display."""
    if metric_idx == 2:  # pod_psi_cpu
        return trace * 100.0
    return trace


def compute_variance_ratio(real_traces, synth_traces, metric_idx):
    real_var = np.var(real_traces[:, :, metric_idx])
    synth_var = np.var(synth_traces[:, :, metric_idx])
    if real_var < 1e-10:
        return 0.0
    return synth_var / real_var


def compute_autocorrelation(trace, max_lag=50):
    """
    Compute autocorrelation of a single time series.
    
    Autocorrelation measures how correlated a signal is with ITSELF at different time lags.
    For example: Does CPU usage at time t correlate with CPU usage at time t+10?
    
    This is NOT correlation between real and synthetic traces.
    We compute separate autocorrelation curves for real and synthetic, then compare shapes.
    """
    mean_val = np.mean(trace)
    c0 = np.sum((trace - mean_val) ** 2)
    if c0 < 1e-10:
        return np.zeros(max_lag)
    acf = []
    for lag in range(max_lag):
        if lag == 0:
            acf.append(1.0)
        else:
            c_lag = np.sum((trace[:-lag] - mean_val) * (trace[lag:] - mean_val))
            acf.append(c_lag / c0)
    return np.array(acf)


def create_correlation_plots(workload, replica_count, real_traces, synth_traces, output_dir):
    """Correlation scatter plots: Real vs Synthetic values per metric."""
    print(f"  Creating correlation plots...")
    
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    trained_indices = S27_TRAINED[workload]
    
    for plot_idx in range(8):
        row = plot_idx // 3
        col = plot_idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        real_flat = real_traces[:, :, plot_idx].flatten()
        synth_flat = synth_traces[:, :, plot_idx].flatten()
        
        # Apply PSI percentage conversion
        real_flat = apply_psi_percentage(real_flat, plot_idx)
        synth_flat = apply_psi_percentage(synth_flat, plot_idx)
        
        # Ensure equal number of points for scatter plot
        n_real = len(real_flat)
        n_synth = len(synth_flat)
        n_sample = min(n_real, n_synth, 1000)  # Max 1000 points for performance
        
        if n_real != n_synth or n_sample < n_real:
            # Sample equal number from both
            if n_real > n_sample:
                idx_real = np.random.choice(n_real, n_sample, replace=False)
                real_flat = real_flat[idx_real]
            if n_synth > n_sample:
                idx_synth = np.random.choice(n_synth, n_sample, replace=False)
                synth_flat = synth_flat[idx_synth]
            # If one has fewer points, sample with replacement from the smaller one
            if len(real_flat) < n_sample:
                real_flat = np.random.choice(real_flat, n_sample, replace=True)
            if len(synth_flat) < n_sample:
                synth_flat = np.random.choice(synth_flat, n_sample, replace=True)
        
        ax.scatter(real_flat, synth_flat, alpha=0.3, s=5, c='blue')
        
        min_val = min(real_flat.min(), synth_flat.min())
        max_val = max(real_flat.max(), synth_flat.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1.5, label='Perfect')
        
        if len(real_flat) > 1 and np.std(real_flat) > 1e-10 and np.std(synth_flat) > 1e-10:
            r, p_val = pearsonr(real_flat, synth_flat)
            ax.text(0.05, 0.95, f'r={r:.3f}', transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        title_suffix = '' if plot_idx in trained_indices else ' (recon)'
        ax.set_title(f"{ALL_METRICS[plot_idx]}{title_suffix}", fontsize=10, fontweight='bold')
        ax.set_xlabel(f'Real ({METRIC_UNITS[plot_idx]})', fontsize=8)
        ax.set_ylabel(f'Synthetic ({METRIC_UNITS[plot_idx]})', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    
    # Hide last subplot
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    
    fig.suptitle(f"S27 Correlation: {workload.upper()} (r={replica_count})",
                 fontsize=14, fontweight='bold', y=0.995)
    
    output_path = output_dir / f"{workload}_r{replica_count}_correlation.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_path}")


def create_distribution_plots(workload, replica_count, real_traces, synth_traces, output_dir):
    print(f"  Creating distribution plots...")
    
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    trained_indices = S27_TRAINED[workload]
    
    for plot_idx in range(8):
        row = plot_idx // 3
        col = plot_idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        real_flat = real_traces[:, :, plot_idx].flatten()
        synth_flat = synth_traces[:, :, plot_idx].flatten()
        
        real_flat = apply_psi_percentage(real_flat, plot_idx)
        synth_flat = apply_psi_percentage(synth_flat, plot_idx)
        
        ax.hist(real_flat, bins=30, alpha=0.5, label='Real', density=True, color='blue')
        ax.hist(synth_flat, bins=30, alpha=0.5, label='Synthetic', density=True, color='orange')
        
        title_suffix = '' if plot_idx in trained_indices else ' (recon)'
        ax.set_title(f"{ALL_METRICS[plot_idx]}{title_suffix}", fontsize=10, fontweight='bold')
        ax.set_xlabel(f'{METRIC_UNITS[plot_idx]}', fontsize=8)
        ax.set_ylabel('Density', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    
    fig.suptitle(f"S27 Distributions: {workload.upper()} (r={replica_count})",
                 fontsize=14, fontweight='bold', y=0.995)
    
    output_path = output_dir / f"{workload}_r{replica_count}_distributions.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_path}")


def create_variance_ratio_plot(workload, replica_count, real_traces, synth_traces, output_dir):
    print(f"  Creating variance ratio plot...")
    
    vr_values = []
    metric_names = []
    colors_list = []
    
    trained_indices = S27_TRAINED[workload]
    
    for idx in range(8):
        vr = compute_variance_ratio(real_traces, synth_traces, idx)
        vr_values.append(vr)
        suffix = '' if idx in trained_indices else ' (recon)'
        metric_names.append(ALL_METRICS[idx] + suffix)
        
        if vr >= 0.8 and vr <= 1.2:
            colors_list.append('green')
        elif vr >= 0.5:
            colors_list.append('orange')
        else:
            colors_list.append('red')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.barh(metric_names, vr_values, color=colors_list, alpha=0.7)
    
    ax.axvline(x=1.0, color='blue', linestyle='--', linewidth=2, label='Perfect (VR=1.0)')
    ax.axvline(x=0.8, color='green', linestyle=':', linewidth=1.5, alpha=0.7, label='Target (VR≥0.8)')
    
    ax.set_xlabel('Variance Ratio (Synthetic / Real)', fontsize=12, fontweight='bold')
    ax.set_title(f"S27 Variance Ratio: {workload.upper()} (r={replica_count})",
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    for i, (bar, val) in enumerate(zip(bars, vr_values)):
        ax.text(val + 0.05, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
               va='center', fontsize=9)
    
    plt.tight_layout()
    output_path = output_dir / f"{workload}_r{replica_count}_variance_ratio.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_path}")


def create_autocorrelation_plot(workload, replica_count, real_traces, synth_traces, output_dir):
    """
    Autocorrelation plots show temporal patterns WITHIN each trace.
    
    For each metric, we:
    1. Take one real trace → compute its autocorrelation curve
    2. Take one synthetic trace → compute its autocorrelation curve
    3. Plot both curves to compare if model preserves temporal dependencies
    
    High autocorrelation at lag=10 means: value at time t is correlated with value at time t+10
    """
    print(f"  Creating autocorrelation plots...")
    
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    max_lag = 50
    
    for plot_idx in range(8):
        row = plot_idx // 3
        col = plot_idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        real_trace = real_traces[0, :, plot_idx]
        synth_trace = synth_traces[0, :, plot_idx]
        
        real_acf = compute_autocorrelation(real_trace, max_lag)
        synth_acf = compute_autocorrelation(synth_trace, max_lag)
        
        ax.plot(range(max_lag), real_acf, 'b-', label='Real', lw=1.5, alpha=0.7)
        ax.plot(range(max_lag), synth_acf, 'r--', label='Synthetic', lw=1.5, alpha=0.7)
        
        ax.set_title(f"{ALL_METRICS[plot_idx]}", fontsize=10, fontweight='bold')
        ax.set_xlabel('Lag (timesteps)', fontsize=8)
        ax.set_ylabel('Autocorrelation', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.5, 1.1])
    
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    
    fig.suptitle(f"S27 Autocorrelation (Temporal Patterns): {workload.upper()} (r={replica_count})",
                 fontsize=14, fontweight='bold', y=0.995)
    
    output_path = output_dir / f"{workload}_r{replica_count}_autocorrelation.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_path}")


def create_accuracy_summary(workload, replica_count, real_traces, synth_traces, output_dir):
    print(f"  Creating accuracy summary...")
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = [['Metric', 'Real Mean', 'Synth Mean', 'Real Std', 'Synth Std',
                   'VR', 'MAE', 'RMSE', 'Pearson r']]
    
    for idx in range(8):
        real_flat = real_traces[:, :, idx].flatten()
        synth_flat = synth_traces[:, :, idx].flatten()
        
        # Apply PSI percentage for statistics
        real_flat_display = apply_psi_percentage(real_flat, idx)
        synth_flat_display = apply_psi_percentage(synth_flat, idx)
        
        # Mean and std can use all data
        real_mean = np.mean(real_flat_display)
        synth_mean = np.mean(synth_flat_display)
        real_std = np.std(real_flat_display)
        synth_std = np.std(synth_flat_display)
        vr = compute_variance_ratio(real_traces, synth_traces, idx)
        
        # For MAE, RMSE, Pearson r - need equal sizes
        n_real = len(real_flat_display)
        n_synth = len(synth_flat_display)
        n_compare = min(n_real, n_synth)
        
        if n_real != n_synth:
            # Sample equal number from both
            idx_real = np.random.choice(n_real, n_compare, replace=False)
            idx_synth = np.random.choice(n_synth, n_compare, replace=False)
            real_compare = real_flat_display[idx_real]
            synth_compare = synth_flat_display[idx_synth]
        else:
            real_compare = real_flat_display
            synth_compare = synth_flat_display
        
        mae = mean_absolute_error(real_compare, synth_compare)
        rmse = np.sqrt(mean_squared_error(real_compare, synth_compare))
        
        if np.std(real_compare) > 1e-10 and np.std(synth_compare) > 1e-10:
            r, _ = pearsonr(real_compare, synth_compare)
        else:
            r = 0.0
        
        table_data.append([
            ALL_METRICS[idx],
            f'{real_mean:.4f}',
            f'{synth_mean:.4f}',
            f'{real_std:.4f}',
            f'{synth_std:.4f}',
            f'{vr:.3f}',
            f'{mae:.4f}',
            f'{rmse:.4f}',
            f'{r:.3f}'
        ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.12, 0.10, 0.10, 0.10, 0.10, 0.08, 0.10, 0.10, 0.10])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    for i in range(9):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for row in range(1, 9):
        vr_val = float(table_data[row][5])
        if vr_val >= 0.8 and vr_val <= 1.2:
            table[(row, 5)].set_facecolor('#C8E6C9')
        elif vr_val >= 0.5:
            table[(row, 5)].set_facecolor('#FFECB3')
        else:
            table[(row, 5)].set_facecolor('#FFCDD2')
    
    ax.set_title(f"S27 Accuracy Summary: {workload.upper()} (r={replica_count})",
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = output_dir / f"{workload}_r{replica_count}_accuracy_summary.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_path}")


def create_timeseries_overlay(workload, replica_count, real_traces, synth_traces, output_dir):
    print(f"  Creating time series overlay...")
    
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    time_steps = np.arange(715) * 5 / 60.0
    
    for plot_idx in range(8):
        row = plot_idx // 3
        col = plot_idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        real_trace = real_traces[0, :, plot_idx]
        synth_trace = synth_traces[0, :, plot_idx]
        
        real_trace = apply_psi_percentage(real_trace, plot_idx)
        synth_trace = apply_psi_percentage(synth_trace, plot_idx)
        
        ax.plot(time_steps, real_trace, 'b-', lw=1.0, alpha=0.6, label='Real')
        ax.plot(time_steps, synth_trace, 'r-', lw=1.0, alpha=0.6, label='Synthetic')
        
        ax.set_title(f"{ALL_METRICS[plot_idx]}", fontsize=10, fontweight='bold')
        ax.set_xlabel('Time (min)', fontsize=8)
        ax.set_ylabel(f'{METRIC_UNITS[plot_idx]}', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    
    fig.suptitle(f"S27 Time Series: {workload.upper()} (r={replica_count})",
                 fontsize=14, fontweight='bold', y=0.995)
    
    output_path = output_dir / f"{workload}_r{replica_count}_timeseries.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='S27 Quality Evaluation')
    parser.add_argument('--workload', type=str, required=True)
    parser.add_argument('--replica-counts', nargs='+', type=int, default=[1, 5, 10])
    parser.add_argument('--output-dir', type=str, default='outputs/phase4/quality_evaluation')
    parser.add_argument('--n-synth', type=int, default=10,
                        help='Number of synthetic traces per replica count')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.workload == 'all':
        workloads = ['bert', 'gpt2', 'resnet152', 'whisper', 'yolo']
    else:
        workloads = [args.workload]
    
    print("=" * 80)
    print("S27 Quality Evaluation - Comprehensive Assessment")
    print("=" * 80)
    print(f"Workloads: {workloads}")
    print(f"Replica counts: {args.replica_counts}")
    print(f"Output: {output_dir}")
    print("=" * 80)
    
    for workload in workloads:
        for r in args.replica_counts:
            print(f"\n{workload.upper()} r={r}:")
            
            real_traces = get_real_traces(workload, r)
            print(f"  Real traces: {real_traces.shape}")
            
            print(f"  Generating {args.n_synth} synthetic traces...")
            synth_traces = []
            for i in range(args.n_synth):
                trace = generate_trace_s27(workload, r, device=args.device)
                synth_traces.append(trace)
            synth_traces = np.array(synth_traces)
            print(f"  Synthetic traces: {synth_traces.shape}")
            
            create_correlation_plots(workload, r, real_traces, synth_traces, output_dir)
            create_distribution_plots(workload, r, real_traces, synth_traces, output_dir)
            create_variance_ratio_plot(workload, r, real_traces, synth_traces, output_dir)
            create_autocorrelation_plot(workload, r, real_traces, synth_traces, output_dir)
            create_accuracy_summary(workload, r, real_traces, synth_traces, output_dir)
            create_timeseries_overlay(workload, r, real_traces, synth_traces, output_dir)
    
    print("\n" + "=" * 80)
    print(f"Done! All plots saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()