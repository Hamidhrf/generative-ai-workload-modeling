"""
S31 Comparison Plots - GPU-Bound Continuity-Aware Model
========================================================
Generates per-workload comparison plots for S31 model
Shows: Real vs S31 (continuous LSTM state flow)

Usage:
  python comparison_plots_s31.py --replica-counts 1 5 10
  python comparison_plots_s31.py --workloads bert gpt2 --replica-counts 5
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import torch
import json
import argparse

# S31 configuration
WORKLOADS = ["bert", "gpt2", "resnet152", "whisper", "yolo"]
VALID_WORKLOAD_IDS = [0, 1, 2, 4]  # Exclude whisper (ID=3)
VALID_WORKLOAD_NAMES = ["bert", "gpt2", "resnet152", "yolo"]
N_WORKLOADS = 4

DATA_PATH = Path("data/processed/phase4/unified/combined_dataset.npz")
NORM_PATH = Path("data/processed/phase4/unified/combined_normalization.json")

# Model architecture params
SEG_LEN = 120
N_PHASES = 6
PHASE_BOUNDARIES = [0, 96, 180, 300, 420, 600]
HIDDEN_DIM = 128
LATENT_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.1

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


class GeneratorContinuous(torch.nn.Module):
    """S31 generator with LSTM state continuity."""
    def __init__(self, seg_len, n_metrics, hidden_dim, latent_dim, num_layers, dropout):
        super().__init__()
        self.seg_len = seg_len
        self.n_metrics = n_metrics
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        self.r_embed = torch.nn.Sequential(torch.nn.Linear(1, 16), torch.nn.Tanh())
        self.wl_embed = torch.nn.Embedding(N_WORKLOADS, 16)
        self.ph_embed = torch.nn.Embedding(N_PHASES, 8)
        
        lstm_in = latent_dim + 16 + 16 + 8
        self.lstm = torch.nn.LSTM(lstm_in, hidden_dim, num_layers,
                                  batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        self.out_fc = torch.nn.Linear(hidden_dim, n_metrics)
        self.out_act = torch.nn.Sigmoid()
    
    def forward(self, r_norm, wl_id, ph_idx, z=None, h=None):
        B = r_norm.shape[0]
        device = r_norm.device
        
        if z is None:
            z = torch.randn(B, self.latent_dim, device=device)
        
        r_emb = self.r_embed(r_norm.unsqueeze(-1))
        wl_emb = self.wl_embed(wl_id)
        ph_emb = self.ph_embed(ph_idx)
        
        z_seq = z.unsqueeze(1).expand(-1, self.seg_len, -1)
        r_seq = r_emb.unsqueeze(1).expand(-1, self.seg_len, -1)
        wl_seq = wl_emb.unsqueeze(1).expand(-1, self.seg_len, -1)
        ph_seq = ph_emb.unsqueeze(1).expand(-1, self.seg_len, -1)
        
        lstm_in = torch.cat([z_seq, r_seq, wl_seq, ph_seq], dim=-1)
        
        if h is None:
            out, h_new = self.lstm(lstm_in)
        else:
            out, h_new = self.lstm(lstm_in, h)
        
        output = self.out_act(self.out_fc(out))
        return output, h_new
    
    @torch.no_grad()
    def generate_trace(self, r_norm_val, wl_id_val, device, n_samples=1):
        """Generate full 715-step trace with continuous LSTM state."""
        self.eval()
        segments = []
        
        # Single latent z for entire trace
        z = torch.randn(n_samples, self.latent_dim, device=device)
        
        # Generate all 6 phases with state continuity
        h = None
        for ph in range(N_PHASES):
            r_norm = torch.full((n_samples,), r_norm_val, dtype=torch.float32, device=device)
            wl_id = torch.full((n_samples,), wl_id_val, dtype=torch.long, device=device)
            ph_idx = torch.full((n_samples,), ph, dtype=torch.long, device=device)
            
            seg, h = self(r_norm, wl_id, ph_idx, z, h)  # Continue LSTM state!
            segments.append(seg.cpu().numpy())
        
        full_trace = np.concatenate(segments, axis=1)  # (n_samples, 720, M)
        return full_trace[:, :715, :]  # Trim to 715


def load_data():
    """Load and filter dataset for GPU-bound workloads."""
    data = np.load(DATA_PATH, allow_pickle=True)
    with open(NORM_PATH) as f:
        norm_params = json.load(f)
    
    raw_traces = data['traces']
    replica_counts = data['replica_counts']
    workload_ids = data['workload_ids']
    
    # Filter GPU-bound only
    valid_mask = np.isin(workload_ids, VALID_WORKLOAD_IDS)
    filtered_traces = raw_traces[valid_mask]
    filtered_rc = replica_counts[valid_mask]
    filtered_wl = workload_ids[valid_mask]
    
    # Remap workload IDs: 0,1,2,4 -> 0,1,2,3
    wl_map = {0: 0, 1: 1, 2: 2, 4: 3}
    filtered_wl = np.array([wl_map[w] for w in filtered_wl])
    
    return filtered_traces, filtered_rc, filtered_wl, norm_params


def denormalize(traces, workload_ids, norm_params):
    """Denormalize traces using per-workload params."""
    N, T, M = traces.shape
    out = np.zeros_like(traces, dtype=np.float64)
    
    for i in range(N):
        wl_id = int(workload_ids[i])
        wl_name = VALID_WORKLOAD_NAMES[wl_id]
        
        # Map back to original workload name for norm params
        orig_wl_name = WORKLOADS[VALID_WORKLOAD_IDS[wl_id]]
        wl_norm = norm_params[orig_wl_name]["params"]
        
        # Get metric names
        metric_names = list(wl_norm.keys())[:M]
        
        for j, mname in enumerate(metric_names):
            if mname not in wl_norm:
                continue
            mn = wl_norm[mname].get("min", 0.0)
            mx = wl_norm[mname].get("max", 1.0)
            out[i, :, j] = traces[i, :, j] * (mx - mn) + mn
    
    return out


def load_s31_generator(model_dir, device='cuda'):
    """Load trained S31 generator."""
    model_path = Path(model_dir) / 'generator.pt'
    
    if not model_path.exists():
        raise FileNotFoundError(f"S31 model not found: {model_path}")
    
    # Load model
    _, _, _, norm_params = load_data()
    n_metrics = 10  # Fixed for this dataset
    
    gen = GeneratorContinuous(SEG_LEN, n_metrics, HIDDEN_DIM, LATENT_DIM,
                              NUM_LAYERS, DROPOUT).to(device)
    gen.load_state_dict(torch.load(model_path, map_location=device))
    gen.eval()
    
    return gen


def generate_s31_trace(generator, workload_name, replica_count, 
                       norm_params, device='cuda'):
    """Generate trace with S31 continuous model."""
    
    workload_id = VALID_WORKLOAD_NAMES.index(workload_name)
    r_norm = (replica_count - 1.0) / 9.0
    
    # Generate trace
    trace_norm = generator.generate_trace(r_norm, workload_id, device, n_samples=1)[0]
    
    # Denormalize
    wl_ids = np.array([workload_id], dtype=np.int32)
    trace_denorm = denormalize(
        trace_norm[np.newaxis, :, :], wl_ids, norm_params
    )[0]
    
    return trace_denorm


def get_real_trace(workload_name, replica_count, raw_traces, 
                   replica_counts, workload_ids, norm_params):
    """Get random real trace for comparison."""
    
    workload_id = VALID_WORKLOAD_NAMES.index(workload_name)
    
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
    trace_denorm = denormalize(
        trace_norm[np.newaxis, :, :], wl_ids, norm_params
    )[0]
    
    return trace_denorm


def create_comparison_plot(workload_name, replica_count, generator, 
                          raw_traces, replica_counts, workload_ids,
                          norm_params, output_dir, device='cuda'):
    """Create single comparison plot: Real vs S31."""
    
    print(f"  {workload_name.upper()} r={replica_count}...", end='')
    
    # Generate synthetic
    synth_trace = generate_s31_trace(
        generator, workload_name, replica_count, 
        norm_params, device
    )
    
    # Get real
    real_trace = get_real_trace(
        workload_name, replica_count, raw_traces,
        replica_counts, workload_ids, norm_params
    )
    
    # Create plot
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    time_steps = np.arange(715) * 5 / 60.0
    
    # Add vertical lines for phase boundaries
    phase_times = np.array(PHASE_BOUNDARIES[1:]) * 5 / 60.0  # Skip 0
    
    for plot_idx, metric_info in enumerate(PLOT_METRICS):
        row = plot_idx // 3
        col = plot_idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        idx = metric_info['idx']
        scale = metric_info.get('scale', 1.0)
        
        real_vals = real_trace[:, idx] / scale
        synth_vals = synth_trace[:, idx] / scale
        
        ax.plot(time_steps, real_vals, 'b-', lw=1.2, alpha=0.7, label='Real')
        ax.plot(time_steps, synth_vals, color='red', lw=1.2, alpha=0.7, label='S31')
        
        # Add phase boundaries
        for pt in phase_times:
            ax.axvline(pt, color='gray', linestyle='--', alpha=0.3, lw=0.8)
        
        ax.set_title(metric_info['name'], fontsize=11, fontweight='bold')
        ax.set_xlabel('Time (min)', fontsize=9)
        ax.set_ylabel(metric_info['unit'], fontsize=9)
        ax.grid(True, alpha=0.3)
        
        if plot_idx == 0:
            ax.legend(fontsize=8)
    
    # Hide last subplot
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    
    # Add note about continuity
    ax.text(0.5, 0.5, 
            'S31: Continuous LSTM State\nAcross Phase Boundaries\n(dashed lines)',
            transform=ax.transAxes, ha='center', va='center',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    fig.suptitle(f"S31 GPU-Bound: {workload_name.upper()} (r={replica_count})", 
                 fontsize=14, fontweight='bold', y=0.995)
    
    output_path = output_dir / f"{workload_name}_r{replica_count}_s31.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(" saved")


def main():
    parser = argparse.ArgumentParser(description='S31 comparison plots')
    parser.add_argument('--model-dir', type=str,
                        default='models/phase4/timegan_s31/s31_gpu_bound_cont03')
    parser.add_argument('--workloads', nargs='+', type=str,
                        default=['bert', 'gpt2', 'resnet152', 'yolo'])
    parser.add_argument('--replica-counts', nargs='+', type=int, default=[1, 5, 10])
    parser.add_argument('--output-dir', type=str, 
                        default='outputs/phase4/comparison_plots/s31')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("S31 COMPARISON PLOTS - GPU-Bound Continuity-Aware Model")
    print("=" * 70)
    print(f"Model: {args.model_dir}")
    print(f"Device: {device}")
    print(f"Workloads: {args.workloads}")
    print(f"Replica counts: {args.replica_counts}")
    print(f"Total plots: {len(args.workloads) * len(args.replica_counts)}")
    print("=" * 70)
    
    # Load S31 generator
    print("\nLoading S31 model...")
    generator = load_s31_generator(args.model_dir, device)
    print("Generator loaded (continuous LSTM state flow)")
    
    # Load real data
    print("\nLoading real data (GPU-bound only)...")
    raw_traces, replica_counts, workload_ids, norm_params = load_data()
    
    print(f"Total pods: {len(raw_traces)}")
    print(f"Metrics: 10")
    
    # Validate workloads
    for wl in args.workloads:
        if wl not in VALID_WORKLOAD_NAMES:
            print(f"WARNING: {wl} not in GPU-bound workloads, skipping")
    
    valid_workloads = [w for w in args.workloads if w in VALID_WORKLOAD_NAMES]
    
    # Generate plots
    print("\nGenerating plots...")
    for r in args.replica_counts:
        print(f"\nReplica count r={r}:")
        for workload in valid_workloads:
            try:
                create_comparison_plot(
                    workload, r, generator,
                    raw_traces, replica_counts, workload_ids,
                    norm_params, output_dir, device
                )
            except Exception as e:
                print(f"  ERROR {workload}: {e}")
    
    print("\n" + "=" * 70)
    print(f"Done! Plots saved to: {output_dir}")
    print("\nInspect for:")
    print("  - Smooth transitions at phase boundaries (gray dashed lines)")
    print("  - Variance levels (should match real data)")
    print("  - Temporal patterns preservation")
    print("=" * 70)


if __name__ == '__main__':
    main()