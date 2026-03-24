#!/usr/bin/env python3
"""
S33 Comparison Plots: Real vs Raw vs Smoothed
GPU-bound workloads only (BERT, GPT2, ResNet152, YOLO)

Usage:
  python comparison_plots_s33.py --replica-counts 1 5 10
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import argparse
import json

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from boundary_smoothing import smooth_phase_boundaries

# S33 workloads (GPU-bound only)
WORKLOADS = ['bert', 'gpt2', 'resnet152', 'yolo']
ACTUAL_BOUNDARIES = [120, 240, 360, 480, 600]
WINDOW_SIZE = 5

# S33: All 4 workloads use SAME 7 metrics
S33_TRAINED = {
    'bert':      [0, 1, 2, 3, 4, 5, 8],
    'gpt2':      [0, 1, 2, 3, 4, 5, 8],
    'resnet152': [0, 1, 2, 3, 4, 5, 8],
    'yolo':      [0, 1, 2, 3, 4, 5, 8],
}

S33_RECONSTRUCT = {
    'bert':      [6],
    'gpt2':      [6],
    'resnet152': [6],
    'yolo':      [6],
}

# 7 plot metrics (S33 trained)
PLOT_METRICS = [
    {'idx': 0, 'name': 'CPU Usage', 'unit': 'cores'},
    {'idx': 2, 'name': 'CPU Pressure (PSI)', 'unit': 'fraction'},
    {'idx': 3, 'name': 'Latency (Avg)', 'unit': 'seconds'},
    {'idx': 4, 'name': 'Throughput', 'unit': 'req/s'},
    {'idx': 1, 'name': 'Pod Memory', 'unit': 'MB', 'scale': 1e6},
    {'idx': 5, 'name': 'GPU Utilization', 'unit': '%'},
    {'idx': 8, 'name': 'GPU Power', 'unit': 'W'},
]

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
    """Load S33 generator"""
    base = Path('models/phase4/timegan_s33')
    
    # Find S33 model directory (e.g., s33_bert_vr05_fm15)
    model_dirs = list(base.glob(f's33_{workload}_*'))
    if not model_dirs:
        raise FileNotFoundError(f"No S33 model for {workload}")
    
    ckpt_path = model_dirs[0] / 'generator.pt'
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"S33 checkpoint: {ckpt_path}")
    
    n_metrics = len(S33_TRAINED[workload])
    gen = GeneratorSeg(SEGMENT_LEN, n_metrics, GEN_CFG).to(device)
    gen.load_state_dict(torch.load(ckpt_path, map_location=device))
    gen.eval()
    return gen


def load_real_data(workload, data_dir='data/processed/phase4/raw'):
    """Load normalized traces"""
    trace_path = Path(data_dir) / f"{workload}_traces.npz"
    data = np.load(trace_path, allow_pickle=True)
    return data


def reconstruct_metric(idx, real_data_norm, norm_params, workload):
    """Reconstruct missing metric using mean of real data"""
    T = 715
    
    # Get all traces for this metric
    all_traces_norm = real_data_norm['traces'][:, :, idx]
    
    # Denormalize to get actual values
    metric_name = METRIC_NAMES[idx]
    params = norm_params[metric_name]
    min_val = params['min']
    max_val = params['max']
    
    all_traces_denorm = all_traces_norm * (max_val - min_val) + min_val
    
    # Return constant mean value
    return np.full(T, all_traces_denorm.mean())


def generate_trace_s33(workload, replica_count, device='cuda'):
    """Generate synthetic trace and denormalize properly"""
    gen = load_generator(workload, device)
    real_data = load_real_data(workload)
    norm_params = load_normalization_params(workload)
    
    # Generate normalized trace (trained metrics only)
    r_norm = torch.tensor([(replica_count - 1) / 9.0], dtype=torch.float32, device=device)
    
    phases = []
    for ph in range(N_PHASES):
        phase_idx = torch.tensor([ph], dtype=torch.long, device=device)
        seg = gen(r_norm, phase_idx)
        phases.append(seg.squeeze(0).cpu().detach().numpy())
    
    trace_norm = np.concatenate(phases, axis=0)[:715]
    
    # Denormalize trained metrics
    trained_indices = S33_TRAINED[workload]
    trace_denorm_partial = denormalize_trace(trace_norm, norm_params, trained_indices)
    
    # Build full 10-metric trace
    trace_full = np.zeros((715, 10))
    
    for i, idx in enumerate(trained_indices):
        trace_full[:, idx] = trace_denorm_partial[:, i]
    
    # Reconstruct missing metrics
    for idx in S33_RECONSTRUCT[workload]:
        trace_full[:, idx] = reconstruct_metric(idx, real_data, norm_params, workload)
    
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
    """Create 3-way comparison plot: Real vs Raw vs Smoothed"""
    
    print(f"  {workload.upper()} r={replica_count}...", end='', flush=True)
    
    real_trace = get_real_trace_sample(workload, replica_count)
    raw_trace = generate_trace_s33(workload, replica_count)
    smooth_trace = smooth_phase_boundaries(raw_trace, ACTUAL_BOUNDARIES, WINDOW_SIZE)
    
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
        raw_vals = raw_trace[:, idx] / scale
        smooth_vals = smooth_trace[:, idx] / scale
        
        ax.plot(time_steps, real_vals, 'b-', lw=1.2, alpha=0.6, label='Real')
        ax.plot(time_steps, raw_vals, color='orange', lw=1.2, alpha=0.5, label='Raw')
        ax.plot(time_steps, smooth_vals, 'g-', lw=1.2, alpha=0.7, label='Smoothed')
        
        # Mark boundaries
        for b in ACTUAL_BOUNDARIES:
            b_min = b * 5 / 60.0
            ax.axvline(b_min, color='red', linestyle='--', alpha=0.2, linewidth=0.8)
        
        ax.set_title(metric_info['name'], fontsize=11, fontweight='bold')
        ax.set_xlabel('Time (min)', fontsize=9)
        ax.set_ylabel(metric_info['unit'], fontsize=9)
        ax.grid(True, alpha=0.3)
        
        if plot_idx == 0:
            ax.legend(fontsize=8, loc='upper right')
    
    # Hide last 2 subplots
    for i in [7, 8]:
        row = i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])
        ax.axis('off')
    
    fig.suptitle(f"S33: {workload.upper()} (r={replica_count}) - Real vs Raw vs Smoothed", 
                 fontsize=14, fontweight='bold', y=0.995)
    
    output_path = output_dir / f"{workload}_r{replica_count}_s33.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f" saved")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replica-counts', nargs='+', type=int, default=[1, 5, 10])
    parser.add_argument('--output-dir', type=str, default='outputs/phase4/timegan_s33/comparison_plots')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("S33 Comparison Plots (GPU-Bound + Smoothing)")
    print("=" * 80)
    print(f"Workloads: {WORKLOADS}")
    print(f"Replica counts: {args.replica_counts}")
    print(f"Smoothing: window={WINDOW_SIZE}, boundaries={ACTUAL_BOUNDARIES}")
    print(f"Total: {len(WORKLOADS) * len(args.replica_counts)} plots")
    print("=" * 80)
    
    for r in args.replica_counts:
        print(f"\nReplica count r={r}:")
        for workload in WORKLOADS:
            try:
                create_comparison_plot(workload, r, output_dir)
            except Exception as e:
                print(f"  ERROR {workload}: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n" + "=" * 80)
    print(f"Done! Plots saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()


# S33 uses same 7 metrics for all 4 GPU-bound workloads
DROP_PER_WORKLOAD = {
    "bert": {"gpu_memory_total", "gpu_memory_used", "gpu_temperature"},
    "gpt2": {"gpu_memory_total", "gpu_memory_used", "gpu_temperature"},
    "resnet152": {"gpu_memory_total", "gpu_memory_used", "gpu_temperature"},
    "yolo": {"gpu_memory_total", "gpu_memory_used", "gpu_temperature"},
}

# Generator config (from S27/S33)
GEN_CFG = {
    "hidden_dim": 128,
    "num_layers": 2,
    "latent_dim": 64,
    "replica_embed_dim": 16,
    "phase_embed_dim": 8,
    "dropout": 0.1,
    "seed": 42,
}
N_PHASES = 6
SEGMENT_LEN = 120


class GeneratorSeg(torch.nn.Module):
    def __init__(self, n_features, cfg):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = cfg["hidden_dim"]
        self.num_layers = cfg["num_layers"]
        self.latent_dim = cfg["latent_dim"]
        
        self.replica_embed = torch.nn.Embedding(11, cfg["replica_embed_dim"])
        self.phase_embed = torch.nn.Embedding(N_PHASES, cfg["phase_embed_dim"])
        
        total_input = cfg["latent_dim"] + cfg["replica_embed_dim"] + cfg["phase_embed_dim"]
        self.lstm = torch.nn.LSTM(total_input, cfg["hidden_dim"], cfg["num_layers"], 
                                   batch_first=True, dropout=cfg["dropout"])
        self.fc = torch.nn.Linear(cfg["hidden_dim"], n_features)
        self.activation = torch.nn.Sigmoid()

    def forward(self, z, replica_count, phase_idx):
        r_emb = self.replica_embed(replica_count)
        p_emb = self.phase_embed(phase_idx)
        r_exp = r_emb.unsqueeze(1).expand(-1, z.size(1), -1)
        p_exp = p_emb.unsqueeze(1).expand(-1, z.size(1), -1)
        x = torch.cat([z, r_exp, p_exp], dim=2)
        out, _ = self.lstm(x)
        out = self.fc(out)
        return self.activation(out)


def load_data():
    print("Loading dataset...")
    data = np.load(DATA_PATH, allow_pickle=True)
    with open(NORM_PATH, "r") as f:
        norm = json.load(f)
    
    all_metrics = norm["metrics"]
    traces = data["traces"]
    replicas = data["replica_counts"]
    workload_ids = data["workload_ids"]
    
    return {
        "traces": traces,
        "replicas": replicas,
        "workload_ids": workload_ids,
        "all_metrics": all_metrics,
    }


def get_workload_mask(workload_ids, workload_name):
    workload_map = {"bert": 0, "gpt2": 1, "resnet152": 2, "whisper": 3, "yolo": 4}
    return workload_ids == workload_map[workload_name]


def filter_metrics(traces, all_metrics, drop_set):
    keep_indices = [i for i, m in enumerate(all_metrics) if m not in drop_set]
    return traces[:, :, keep_indices], [all_metrics[i] for i in keep_indices]


def load_generator(workload, n_features, device):
    # Find S33 model
    model_dirs = list(S33_BASE.glob(f"s33_{workload}_*"))
    if not model_dirs:
        raise FileNotFoundError(f"No S33 model for {workload}")
    
    model_path = model_dirs[0] / "generator.pt"
    print(f"  Loading: {model_path}")
    
    gen = GeneratorSeg(n_features, GEN_CFG).to(device)
    gen.load_state_dict(torch.load(model_path, map_location=device))
    gen.eval()
    return gen


def generate_traces(gen, replica_count, n_samples, n_features, device):
    """Generate full 720-timestep traces by stitching 6 phases"""
    gen.eval()
    traces = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            segments = []
            for phase_idx in range(N_PHASES):
                z = torch.randn(1, SEGMENT_LEN, GEN_CFG["latent_dim"], device=device)
                r = torch.tensor([replica_count], device=device)
                p = torch.tensor([phase_idx], device=device)
                seg = gen(z, r, p)
                segments.append(seg.cpu().numpy()[0])
            
            full_trace = np.concatenate(segments, axis=0)
            traces.append(full_trace)
    
    return np.array(traces)


def plot_comparison(real_traces, raw_traces, smooth_traces, workload, replica_count, metric_names):
    """Plot 3x3 grid: Real vs Raw vs Smoothed for first 7 metrics"""
    n_metrics = min(7, len(metric_names))
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(f'{workload.upper()} r={replica_count}: Real vs Raw vs Smoothed', 
                 fontsize=16, fontweight='bold')
    
    for idx in range(9):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        if idx < n_metrics:
            metric = metric_names[idx]
            
            # Plot all real traces (light)
            for trace in real_traces:
                ax.plot(trace[:, idx], 'b-', alpha=0.2, linewidth=0.5)
            
            # Plot all raw traces (medium)
            for trace in raw_traces:
                ax.plot(trace[:, idx], 'orange', alpha=0.3, linewidth=0.5)
            
            # Plot all smoothed traces (dark)
            for trace in smooth_traces:
                ax.plot(trace[:, idx], 'g-', alpha=0.4, linewidth=0.5)
            
            # Mark boundaries
            for b in ACTUAL_BOUNDARIES:
                ax.axvline(b, color='red', linestyle='--', alpha=0.3, linewidth=0.8)
            
            ax.set_title(metric, fontsize=10, fontweight='bold')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Normalized Value')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.1, 1.1)
            
            # Legend only on first plot
            if idx == 0:
                ax.legend(['Real', 'Raw', 'Smoothed'], loc='upper right', fontsize=8)
        else:
            ax.axis('off')
    
    plt.tight_layout()
    filename = OUTPUT_DIR / f"{workload}_r{replica_count}_comparison.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Load data
    data = load_data()
    
    for workload in WORKLOADS:
        print(f"\n{'='*80}")
        print(f"WORKLOAD: {workload.upper()}")
        print(f"{'='*80}")
        
        # Get workload-specific data
        mask = get_workload_mask(data["workload_ids"], workload)
        w_traces = data["traces"][mask]
        w_replicas = data["replicas"][mask]
        
        # Filter metrics
        drop_set = DROP_PER_WORKLOAD[workload]
        w_traces, metric_names = filter_metrics(w_traces, data["all_metrics"], drop_set)
        n_features = len(metric_names)
        print(f"Metrics ({n_features}): {metric_names}")
        
        # Load generator
        gen = load_generator(workload, n_features, device)
        
        # Process each replica level
        for r in REPLICA_LEVELS:
            print(f"\n  Replica count: {r}")
            
            # Get real traces
            r_mask = w_replicas == r
            real_traces = w_traces[r_mask]
            print(f"    Real traces: {len(real_traces)}")
            
            if len(real_traces) == 0:
                print(f"    SKIP: No real traces for r={r}")
                continue
            
            # Generate synthetic traces
            n_gen = min(10, len(real_traces))
            raw_traces = generate_traces(gen, r, n_gen, n_features, device)
            print(f"    Generated: {n_gen} raw traces")
            
            # Apply smoothing
            smooth_traces = np.array([
                smooth_phase_boundaries(trace, ACTUAL_BOUNDARIES, WINDOW_SIZE)
                for trace in raw_traces
            ])
            print(f"    Smoothed: {len(smooth_traces)} traces")
            
            # Plot
            plot_comparison(real_traces, raw_traces, smooth_traces, 
                          workload, r, metric_names)
    
    print(f"\n{'='*80}")
    print("ALL PLOTS SAVED")
    print(f"{'='*80}")
    print(f"Directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()