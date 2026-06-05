"""
S28 Comparison Plots - FINAL CORRECTED
Uses workload-specific normalization parameters from JSON files

Usage:
  python comparison_plots_s28_FINAL.py --replica-counts 1 5 10
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import torch
import torch.nn as nn
import argparse
import json

# 8 plot metrics (no temperature)
PLOT_METRICS = [
    {'idx': 0, 'name': 'CPU Usage', 'unit': 'cores'},
    {'idx': 2, 'name': 'CPU Pressure (PSI)', 'unit': 'fraction'},
    {'idx': 3, 'name': 'Latency (Avg)', 'unit': 'seconds'},
    {'idx': 4, 'name': 'Throughput', 'unit': 'req/s'},
    {'idx': 1, 'name': 'Pod Memory', 'unit': 'MB', 'scale': 1e6},
    {'idx': 5, 'name': 'GPU Utilization', 'unit': '%'},
    {'idx': 6, 'name': 'GPU Memory Used', 'unit': 'MB'},
    {'idx': 8, 'name': 'GPU Power', 'unit': 'W'},
]

# Metric index to name mapping
METRIC_NAMES = [
    'pod_cpu_usage',
    'pod_memory_bytes',
    'pod_psi_cpu',
    'pod_latency_avg',
    'pod_throughput',
    'gpu_utilization',
    'gpu_memory_used',
    'gpu_memory_total',
    'gpu_power_watts',
    'gpu_temperature'
]

# S28 trained ALL 8 metrics for ALL workloads
S28_TRAINED = [0, 1, 2, 3, 4, 5, 6, 8]

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
    """Load workload-specific normalization parameters"""
    norm_path = Path(data_dir) / f"{workload}_normalization.json"
    with open(norm_path, 'r') as f:
        norm_data = json.load(f)
    return norm_data['params']


def denormalize_trace(trace_norm, norm_params, metric_indices):
    """Denormalize using workload-specific parameters"""
    trace_denorm = np.zeros_like(trace_norm)
    
    for i, idx in enumerate(metric_indices):
        metric_name = METRIC_NAMES[idx]
        params = norm_params[metric_name]
        min_val = params['min']
        max_val = params['max']
        
        trace_denorm[:, i] = trace_norm[:, i] * (max_val - min_val) + min_val
    
    return trace_denorm


def load_generator(workload, device='cuda'):
    """Load S28 generator"""
    base = Path('models/phase4/timegan_s28/s28_seg_vr03_fm10_ae150')
    ckpt_path = base / workload / 'generator.pt'
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"S28 checkpoint: {ckpt_path}")
    
    n_metrics = len(S28_TRAINED)  # 8 for all workloads
    gen = GeneratorSeg(SEGMENT_LEN, n_metrics, GEN_CFG).to(device)
    gen.load_state_dict(torch.load(ckpt_path, map_location=device))
    gen.eval()
    return gen


def load_real_data(workload, data_dir='data/processed/phase4/raw'):
    """Load normalized traces"""
    trace_path = Path(data_dir) / f"{workload}_traces.npz"
    data = np.load(trace_path, allow_pickle=True)
    return data


def generate_trace_s28(workload, replica_count, device='cuda'):
    """Generate synthetic trace and denormalize properly"""
    gen = load_generator(workload, device)
    norm_params = load_normalization_params(workload)
    
    # Generate normalized trace (all 8 metrics)
    r_norm = torch.tensor([(replica_count - 1) / 9.0], dtype=torch.float32, device=device)
    
    phases = []
    for ph in range(N_PHASES):
        phase_idx = torch.tensor([ph], dtype=torch.long, device=device)
        seg = gen(r_norm, phase_idx)
        phases.append(seg.squeeze(0).cpu().detach().numpy())
    
    trace_norm = np.concatenate(phases, axis=0)[:715]
    
    # Denormalize all 8 metrics
    trace_denorm = denormalize_trace(trace_norm, norm_params, S28_TRAINED)
    
    # Build full 10-metric trace (add dummy for idx 7, 9)
    trace_full = np.zeros((715, 10))
    
    for i, idx in enumerate(S28_TRAINED):
        trace_full[:, idx] = trace_denorm[:, i]
    
    return trace_full


def get_real_trace_sample(workload, replica_count):
    """Get random real trace and denormalize properly"""
    data = load_real_data(workload)
    norm_params = load_normalization_params(workload)
    
    traces_norm = data['traces']
    replica_counts = data['replica_counts']
    
    mask = (replica_counts == replica_count)
    matching = traces_norm[mask]
    
    if len(matching) == 0:
        raise ValueError(f"No real traces for {workload} r={replica_count}")
    
    idx = np.random.randint(len(matching))
    trace_norm = matching[idx]
    
    # Denormalize all 10 metrics
    trace_denorm = denormalize_trace(trace_norm, norm_params, list(range(10)))
    
    return trace_denorm


def create_comparison_plot(workload, replica_count, output_dir):
    """Create comparison plot"""
    
    print(f"  {workload.upper()} r={replica_count}...", end='')
    
    real_trace = get_real_trace_sample(workload, replica_count)
    synth_trace = generate_trace_s28(workload, replica_count)
    
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
        ax.plot(time_steps, synth_vals, color='darkorange', lw=1.2, alpha=0.7, label='S28')
        
        ax.set_title(metric_info['name'], fontsize=11, fontweight='bold')
        ax.set_xlabel('Time (min)', fontsize=9)
        ax.set_ylabel(metric_info['unit'], fontsize=9)
        ax.grid(True, alpha=0.3)
        
        if plot_idx == 0:
            ax.legend(fontsize=8)
    
    # Hide last subplot
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    
    fig.suptitle(f"S28: {workload.upper()} (r={replica_count})", 
                 fontsize=14, fontweight='bold', y=0.995)
    
    output_path = output_dir / f"{workload}_r{replica_count}_s28.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f" saved")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replica-counts', nargs='+', type=int, default=[1, 5, 10])
    parser.add_argument('--output-dir', type=str, default='outputs/phase4/comparison_plots/s28')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    workloads = ['bert', 'gpt2', 'resnet152', 'whisper', 'yolo']
    
    print("=" * 70)
    print("S28 Comparison Plots - FINAL (Workload-Specific Normalization)")
    print("=" * 70)
    print(f"Replica counts: {args.replica_counts}")
    print(f"Total: {len(workloads) * len(args.replica_counts)} plots")
    print("=" * 70)
    
    for r in args.replica_counts:
        print(f"\nReplica count r={r}:")
        for workload in workloads:
            try:
                create_comparison_plot(workload, r, output_dir)
            except Exception as e:
                print(f"  ERROR {workload}: {e}")
    
    print("\n" + "=" * 70)
    print(f"Done! Plots: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()