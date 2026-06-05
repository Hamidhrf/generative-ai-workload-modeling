"""
FINAL Production Comparison Plots - Phase 4 S21 Model
Creates 3x3 grid plots with 9 useful metrics (drops gpu_memory_total)

Usage:
  # Single plot
  python comparison_plots_FINAL.py --workload bert --replica-count 5

  # Main results (5 workloads at r=5)
  python comparison_plots_FINAL.py --replica-counts 5

  # All 25 plots
  python comparison_plots_FINAL.py --replica-counts 1 3 5 8 10
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import torch
import torch.nn as nn

# Final metric configuration - 9 metrics (drop gpu_memory_total)
PLOT_METRICS = [
    {'idx': 0, 'name': 'CPU Usage', 'unit': 'cores', 'key': 'pod_cpu_usage'},
    {'idx': 2, 'name': 'CPU Pressure (PSI)', 'unit': 'fraction', 'key': 'pod_psi_cpu'},
    {'idx': 3, 'name': 'Latency (Avg)', 'unit': 'seconds', 'key': 'pod_latency_avg'},
    {'idx': 4, 'name': 'Throughput', 'unit': 'req/s', 'key': 'pod_throughput'},
    {'idx': 1, 'name': 'Pod Memory', 'unit': 'GB', 'scale': 1e9, 'key': 'pod_memory_bytes'},
    {'idx': 5, 'name': 'GPU Utilization', 'unit': '%', 'key': 'gpu_utilization'},
    {'idx': 6, 'name': 'GPU Memory Used', 'unit': 'MB', 'key': 'gpu_memory_used'},
    {'idx': 8, 'name': 'GPU Power', 'unit': 'W', 'key': 'gpu_power_watts'},
    {'idx': 9, 'name': 'GPU Temperature', 'unit': 'C', 'key': 'gpu_temperature'},
]

# Dropped from S21 training per workload
DROPPED_CONFIG = {
    'bert': [1, 6, 7, 8, 9],
    'gpt2': [1, 7, 8, 9],
    'resnet152': [1, 6, 7, 8, 9],
    'whisper': [1, 4, 6, 7, 8, 9],  # FIX: Also drops gpu_memory_used (idx 6)
    'yolo': [1, 6, 7, 8, 9],
}

# EXACT S21 architecture from timegan_s21.py
N_PHASES = 6
SEGMENT_LEN = 120

GEN_CFG = {
    "hidden_dim":        128,
    "num_layers":        2,
    "latent_dim":        64,
    "replica_embed_dim": 16,
    "phase_embed_dim":   8,
    "dropout":           0.1,
}

class GeneratorSeg(nn.Module):
    """S21 generator from timegan_s21.py - produces one phase segment of 120 steps."""
    
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
        B      = r_norm.shape[0]
        T      = self.seg_len
        device = r_norm.device

        if z is None:
            z = torch.randn(B, self.latent_dim, device=device)

        r_emb  = self.r_embed(r_norm.unsqueeze(-1))
        ph_emb = self.ph_embed(phase_idx)

        zrp = torch.cat([z, r_emb, ph_emb], dim=-1)
        h0  = self.h_init(zrp)
        c0  = self.c_init(zrp)
        h0  = h0.view(B, self.n_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        c0  = c0.view(B, self.n_layers, self.hidden_dim).permute(1, 0, 2).contiguous()

        z_exp  = z.unsqueeze(1).expand(-1, T, -1)
        r_exp  = r_emb.unsqueeze(1).expand(-1, T, -1)
        ph_exp = ph_emb.unsqueeze(1).expand(-1, T, -1)
        dec_in = torch.cat([z_exp, r_exp, ph_exp], dim=-1)

        dec_out, _ = self.dec_rnn(dec_in, (h0, c0))
        return self.out_act(self.out_fc(dec_out))


def load_generator(checkpoint_path, output_dim, device='cpu'):
    """Load trained S21 generator with EXACT architecture from training script."""
    gen = GeneratorSeg(
        seg_len=SEGMENT_LEN,
        n_metrics=output_dim,
        cfg=GEN_CFG
    )
    state = torch.load(checkpoint_path, map_location=device)
    gen.load_state_dict(state)
    gen.to(device)
    gen.eval()
    return gen


def load_real_trace(workload, replica_count, data_dir='data/processed/phase4/raw'):
    """
    Load real trace with all 10 metrics.
    
    IMPORTANT: For r>1, this returns ONLY THE FIRST POD (index 0).
    Example: For r=5, there are 5 pods but we show only pod[0].
    
    This is for single-trace comparison. For thesis figures showing variation,
    consider averaging all pods or showing multiple traces with transparency.
    """
    trace_path = Path(data_dir) / f"{workload}_traces.npz"
    data = np.load(trace_path, allow_pickle=True)
    
    traces = data['traces']
    replica_counts = data['replica_counts']
    
    mask = replica_counts == replica_count
    matching_traces = traces[mask]
    
    if len(matching_traces) == 0:
        raise ValueError(f"No traces for {workload} r={replica_count}")
    
    real_trace = matching_traces[0]
    return real_trace, data


def compute_normalization_params(all_traces):
    """Compute normalization params from all traces."""
    n_traces, n_timesteps, n_metrics = all_traces.shape
    traces_flat = all_traces.reshape(-1, n_metrics)
    
    mins = traces_flat.min(axis=0)
    maxs = traces_flat.max(axis=0)
    ranges = maxs - mins
    
    return {
        'mins': mins,
        'maxs': maxs,
        'ranges': ranges
    }


def denormalize_trace(trace_norm, norm_params):
    """
    Denormalize from [0,1] to original scale.
    
    FIX: GPU power (idx 8) and temperature (idx 9) use corrected ranges
    based on NVIDIA A16 specs and raw CSV verification, due to preprocessing
    artifacts in stored normalization parameters.
    """
    mins = np.array(norm_params['mins'])
    maxs = np.array(norm_params['maxs'])
    ranges = maxs - mins
    trace_denorm = trace_norm * ranges + mins
    
    # FIX: Correct GPU power denormalization (idx 8)
    # Stored params: min=3-5W, max=35-60W (WRONG)
    # Correct params: min=10W, max=70W (NVIDIA A16 specs + raw CSV verified)
    trace_denorm[:, 8] = trace_norm[:, 8] * 60.0 + 10.0  # 10W idle -> 70W max
    
    # FIX: Correct GPU temperature denormalization (idx 9)  
    # Stored params: min=6-7°C, max=64-75°C (WRONG min)
    # Correct params: min=40°C, max=75°C (typical GPU behavior)
    trace_denorm[:, 9] = trace_norm[:, 9] * 35.0 + 40.0  # 40°C idle -> 75°C load
    
    return trace_denorm


def generate_synthetic_trace(generator, replica_count, device='cpu', seg_len=120):
    """Generate synthetic trace using S21 segment-based approach."""
    phase_boundaries = [0, 96, 180, 300, 420, 600]
    n_phases = 6
    
    r_norm = torch.tensor((replica_count - 1.0) / 9.0, dtype=torch.float32).to(device)
    
    segments = []
    for phase_idx in range(n_phases):
        r_norm_batch = r_norm.unsqueeze(0)  # (1,)
        phase_tensor = torch.tensor([phase_idx], dtype=torch.long).to(device)
        
        with torch.no_grad():
            seg = generator(r_norm_batch, phase_tensor, z=None)
        
        segments.append(seg.cpu().numpy()[0])
    
    full_trace = np.concatenate(segments, axis=0)[:715]
    return full_trace


def reconstruct_dropped_metrics(synth_trace_kept, real_trace_full, dropped_indices):
    """Reconstruct dropped metrics for synthetic trace."""
    synth_trace_full = np.zeros((715, 10))
    
    kept_mask = np.ones(10, dtype=bool)
    kept_mask[dropped_indices] = False
    kept_indices = np.where(kept_mask)[0]
    
    synth_trace_full[:, kept_indices] = synth_trace_kept
    
    for idx in dropped_indices:
        if idx == 7:
            synth_trace_full[:, idx] = 15360.0
        elif idx in [1, 4, 6]:
            synth_trace_full[:, idx] = np.mean(real_trace_full[:, idx])
        elif idx == 8:
            gpu_util = synth_trace_full[:, 5]
            synth_trace_full[:, idx] = 10 + (gpu_util / 100.0) * 60
        elif idx == 9:
            gpu_util = synth_trace_full[:, 5]
            synth_trace_full[:, idx] = 40 + (gpu_util / 100.0) * 20
    
    return synth_trace_full


def create_comparison_plot(real_trace, synth_trace, workload, replica_count, output_path):
    """Create 3x3 grid comparison plot."""
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    time_steps = np.arange(715)
    
    for plot_idx, metric_info in enumerate(PLOT_METRICS):
        row = plot_idx // 3
        col = plot_idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        metric_idx = metric_info['idx']
        real_values = real_trace[:, metric_idx]
        synth_values = synth_trace[:, metric_idx]
        
        scale = metric_info.get('scale', 1.0)
        real_values = real_values / scale
        synth_values = synth_values / scale
        
        ax.plot(time_steps, real_values, linewidth=1.5, color='#2E86AB', 
                label='Real', alpha=0.9)
        ax.plot(time_steps, synth_values, linewidth=1.5, color='#A23B72', 
                linestyle='--', label='Synthetic', alpha=0.85)
        
        ax.set_xlabel('Time Step', fontsize=9)
        ax.set_ylabel(f"{metric_info['name']} ({metric_info['unit']})", fontsize=9)
        ax.set_title(metric_info['name'], fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.tick_params(labelsize=8)
        
        if plot_idx == 0:
            ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    
    pod_note = " (Single Pod)" if replica_count > 1 else ""
    title = f"{workload.upper()} - r={replica_count}{pod_note} - Real vs Synthetic Comparison"
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_comparison(workload, replica_count, device='cpu',
                     model_dir='models/phase4/timegan_s21/s21_seg_vr03_fm10_ae150',
                     data_dir='data/processed/phase4/raw',
                     output_dir='outputs/phase4/comparison_plots'):
    """Complete pipeline for one comparison plot."""
    
    print(f"\n{'='*60}")
    print(f"Creating comparison: {workload.upper()} r={replica_count}")
    print(f"{'='*60}\n")
    
    # Step 1: Load real trace
    print("[1/5] Loading REAL trace (all 10 metrics)...")
    real_trace_norm, data = load_real_trace(workload, replica_count, data_dir)
    print(f"  Loaded real trace: {real_trace_norm.shape}")
    
    # Step 2: Compute normalization params and denormalize real
    print("[2/5] Computing normalization params and denormalizing REAL trace...")
    all_traces = data['traces']
    norm_params = compute_normalization_params(all_traces)
    real_trace = denormalize_trace(real_trace_norm, norm_params)
    
    # Step 3: Generate synthetic
    print("[3/5] Generating SYNTHETIC trace (kept metrics)...")
    checkpoint_path = Path(model_dir) / workload / 'generator.pt'
    
    dropped = DROPPED_CONFIG[workload]
    output_dim = 10 - len(dropped)
    
    generator = load_generator(checkpoint_path, output_dim, device)
    synth_trace_kept_norm = generate_synthetic_trace(generator, replica_count, device)
    print(f"  Generated synthetic: {synth_trace_kept_norm.shape} (kept metrics)")
    
    # Step 4: Denormalize and reconstruct synthetic
    print("[4/5] Denormalizing and reconstructing SYNTHETIC...")
    
    kept_mask = np.ones(10, dtype=bool)
    kept_mask[dropped] = False
    kept_indices = np.where(kept_mask)[0]
    
    kept_mins = norm_params['mins'][kept_indices]
    kept_maxs = norm_params['maxs'][kept_indices]
    kept_ranges = kept_maxs - kept_mins
    
    synth_trace_kept = synth_trace_kept_norm * kept_ranges + kept_mins
    synth_trace = reconstruct_dropped_metrics(synth_trace_kept, real_trace, dropped)
    print(f"  Reconstructed to: {synth_trace.shape}")
    
    # Step 5: Create plot
    print("[5/5] Creating comparison plot...")
    output_path = Path(output_dir) / workload / f"{workload}_r{replica_count}_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    create_comparison_plot(real_trace, synth_trace, workload, replica_count, output_path)
    print(f"  Saved: {output_path}")
    print(f"\nDONE!\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Create comparison plots - Phase 4 S21')
    parser.add_argument('--workload', type=str, choices=['bert', 'gpt2', 'resnet152', 'whisper', 'yolo'],
                       help='Single workload to plot')
    parser.add_argument('--replica-count', type=int, choices=[1, 3, 5, 8, 10],
                       help='Single replica count to plot')
    parser.add_argument('--replica-counts', type=int, nargs='+', choices=[1, 3, 5, 8, 10],
                       help='Multiple replica counts (generates all 5 workloads for each)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='Device for generation')
    
    args = parser.parse_args()
    
    if args.workload and args.replica_count:
        # Single plot mode
        create_comparison(args.workload, args.replica_count, args.device)
    
    elif args.replica_counts:
        # Batch mode: all workloads for specified replica counts
        workloads = ['bert', 'gpt2', 'resnet152', 'whisper', 'yolo']
        for r in args.replica_counts:
            for workload in workloads:
                try:
                    create_comparison(workload, r, args.device)
                except Exception as e:
                    print(f"ERROR: {workload} r={r}: {e}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()