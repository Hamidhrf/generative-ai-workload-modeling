"""
S35 Comparison Plots
====================
Visualize real vs S35 synthetic traces for all 5 workloads.

S35 = S34 architecture + 4x augmented dataset

Usage:
  python comparison_plots_s35.py --replica-counts 1 5 10
  python comparison_plots_s35.py --workloads bert gpt2
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import torch
import torch.nn as nn
import argparse
import json

# 7 plot metrics (S34/S35 trained metrics - no gpu_memory_used, gpu_memory_total, gpu_temperature)
PLOT_METRICS = [
    {'idx': 0, 'name': 'CPU Usage', 'unit': 'cores'},
    {'idx': 2, 'name': 'CPU Pressure (PSI)', 'unit': 'fraction'},
    {'idx': 3, 'name': 'Latency (Avg)', 'unit': 'seconds'},
    {'idx': 4, 'name': 'Throughput', 'unit': 'req/s'},
    {'idx': 1, 'name': 'Pod Memory', 'unit': 'MB', 'scale': 1e6},
    {'idx': 5, 'name': 'GPU Utilization', 'unit': '%'},
    {'idx': 8, 'name': 'GPU Power', 'unit': 'W'},
]

# Metric index to name mapping (10 total metrics)
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

# S35: All 5 workloads have same 7 metrics (dropped gpu_memory_total, gpu_memory_used, gpu_temperature)
S35_TRAINED_INDICES = [0, 1, 2, 3, 4, 5, 8]  # Indices in 10-metric array
S35_TRAINED_NAMES = ['pod_cpu_usage', 'pod_memory_bytes', 'pod_psi_cpu', 
                     'pod_latency_avg', 'pod_throughput', 'gpu_utilization', 'gpu_power_watts']

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
    """S35 generator (identical to S34)."""
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


def load_normalization_params(workload, data_dir='data/processed/phase4/unified'):
    """Load combined normalization parameters"""
    norm_path = Path(data_dir) / "combined_normalization_s35_augmented.json"
    with open(norm_path, 'r') as f:
        norm_data = json.load(f)
    return norm_data[workload]['params']  # Access params wrapper


def denormalize_trace(trace_norm, norm_params, metric_names):
    """Denormalize using minmax normalization (not zscore)"""
    trace_denorm = np.zeros_like(trace_norm)
    
    for i, metric_name in enumerate(metric_names):
        params = norm_params[metric_name]
        min_val = params['min']
        max_val = params['max']
        
        # Minmax denormalization: x = x_norm * (max - min) + min
        trace_denorm[:, i] = trace_norm[:, i] * (max_val - min_val) + min_val
    
    return trace_denorm


def load_generator(workload, device='cuda'):
    """Load S35 generator"""
    model_paths = {
        "bert": "models/phase4/timegan_s35/s35_bert_vr05_fm15/generator.pt",
        "gpt2": "models/phase4/timegan_s35/s35_gpt2_vr03_fm20/generator.pt",
        "resnet152": "models/phase4/timegan_s35/s35_resnet152_vr04_fm10/generator.pt",
        "whisper": "models/phase4/timegan_s35/s35_whisper_vr05_fm05/generator.pt",
        "yolo": "models/phase4/timegan_s35/s35_yolo_vr03_fm12/generator.pt",
    }
    
    ckpt_path = Path(model_paths[workload])
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"S35 checkpoint not found: {ckpt_path}")
    
    n_metrics = 7  # All S35 workloads have 7 metrics
    gen = GeneratorSeg(SEGMENT_LEN, n_metrics, GEN_CFG).to(device)
    gen.load_state_dict(torch.load(ckpt_path, map_location=device))
    gen.eval()
    return gen


def load_real_data(data_dir='data/processed/phase4/unified'):
    """Load augmented dataset for S35"""
    data_path = Path(data_dir) / "combined_dataset_s35_augmented.npz"
    data = np.load(data_path, allow_pickle=True)
    return data


def generate_trace_s35(workload, replica_count, device='cuda'):
    """Generate synthetic trace and denormalize"""
    gen = load_generator(workload, device)
    norm_params = load_normalization_params(workload)
    
    # Generate normalized trace (7 metrics)
    r_norm = torch.tensor([(replica_count - 1) / 9.0], dtype=torch.float32, device=device)
    
    phases = []
    for ph in range(N_PHASES):
        phase_idx = torch.tensor([ph], dtype=torch.long, device=device)
        seg = gen(r_norm, phase_idx)
        phases.append(seg.squeeze(0).cpu().detach().numpy())
    
    trace_norm = np.concatenate(phases, axis=0)[:715]
    
    # Denormalize 7 metrics
    trace_denorm = denormalize_trace(trace_norm, norm_params, S35_TRAINED_NAMES)
    
    # Build full 10-metric trace (with zeros for dropped metrics)
    trace_full = np.zeros((715, 10))
    
    for i, idx in enumerate(S35_TRAINED_INDICES):
        trace_full[:, idx] = trace_denorm[:, i]
    
    return trace_full


def get_real_trace_sample(workload, replica_count):
    """Get random real trace and denormalize"""
    data = load_real_data()
    norm_params = load_normalization_params(workload)
    
    workloads_all = ["bert", "gpt2", "resnet152", "whisper", "yolo"]
    workload_id = workloads_all.index(workload)
    
    traces_norm = data['traces']
    replica_counts = data['replica_counts']
    workload_ids = data['workload_ids']
    
    # Filter for this workload and replica count
    mask = (workload_ids == workload_id) & (replica_counts == replica_count)
    matching = traces_norm[mask]
    
    if len(matching) == 0:
        raise ValueError(f"No real traces for {workload} r={replica_count}")
    
    # Get random sample
    idx = np.random.randint(len(matching))
    trace_norm = matching[idx]
    
    # Extract only the 7 trained metrics
    trace_norm_7 = trace_norm[:, S35_TRAINED_INDICES]
    
    # Denormalize
    trace_denorm = denormalize_trace(trace_norm_7, norm_params, S35_TRAINED_NAMES)
    
    # Build full 10-metric trace
    trace_full = np.zeros((715, 10))
    for i, idx in enumerate(S35_TRAINED_INDICES):
        trace_full[:, idx] = trace_denorm[:, i]
    
    return trace_full


def create_comparison_plot(workload, replica_count, output_dir):
    """Create comparison plot for S35"""
    
    print(f"  {workload.upper()} r={replica_count}...", end='', flush=True)
    
    try:
        real_trace = get_real_trace_sample(workload, replica_count)
        synth_trace = generate_trace_s35(workload, replica_count)
    except Exception as e:
        print(f" ERROR: {e}")
        return
    
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    time_steps = np.arange(715) * 5 / 60.0  # Convert to minutes
    
    for plot_idx, metric_info in enumerate(PLOT_METRICS):
        row = plot_idx // 3
        col = plot_idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        idx = metric_info['idx']
        scale = metric_info.get('scale', 1.0)
        
        real_vals = real_trace[:, idx] / scale
        synth_vals = synth_trace[:, idx] / scale
        
        ax.plot(time_steps, real_vals, 'b-', lw=1.2, alpha=0.7, label='Real')
        ax.plot(time_steps, synth_vals, color='darkorange', lw=1.2, alpha=0.7, label='S35')
        
        ax.set_title(metric_info['name'], fontsize=11, fontweight='bold')
        ax.set_xlabel('Time (min)', fontsize=9)
        ax.set_ylabel(metric_info['unit'], fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        
        if plot_idx == 0:
            ax.legend(fontsize=8, loc='best')
    
    # Hide last 2 subplots (only 7 metrics)
    for i in [7, 8]:
        row = i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])
        ax.axis('off')
    
    fig.suptitle(f"S35 (4x Augmented): {workload.upper()} (r={replica_count})", 
                 fontsize=14, fontweight='bold', y=0.995)
    
    output_path = output_dir / f"{workload}_r{replica_count}_s35.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f" saved")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replica-counts', nargs='+', type=int, default=[1, 5, 10])
    parser.add_argument('--workloads', nargs='+', type=str, 
                        default=['bert', 'gpt2', 'resnet152', 'whisper', 'yolo'])
    parser.add_argument('--output-dir', type=str, default='outputs/phase4/comparison_plots/s35')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("S35 COMPARISON PLOTS (4x Augmented Dataset)")
    print("=" * 80)
    print(f"Workloads: {args.workloads}")
    print(f"Replica counts: {args.replica_counts}")
    print(f"Metrics: 7 (same as S34)")
    print(f"Total plots: {len(args.workloads) * len(args.replica_counts)}")
    print("=" * 80)
    
    for r in args.replica_counts:
        print(f"\nReplica count r={r}:")
        for workload in args.workloads:
            create_comparison_plot(workload, r, output_dir)
    
    print("\n" + "=" * 80)
    print(f"COMPLETE! Plots saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()