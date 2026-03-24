#!/usr/bin/env python3
"""
S27 Trace Generator with Intelligent Reconstruction
Generates complete pod traces (8 metrics only - gpu_memory_total and gpu_temperature excluded).

Reconstruction Strategy:
- gpu_memory_used: Workload-specific constant (architecture-determined)
- pod_throughput: Mean from real data (Whisper only - not trained)
- gpu_power_watts: TRAINED by S27 (no reconstruction needed!)

Usage:
    # Generate single trace
    python generate_s27_traces.py --workload bert --replica-count 5
    
    # Generate multiple traces for Kwok
    python generate_s27_traces.py --workload bert --replica-count 50 --n-pods 50
    
    # Save to file
    python generate_s27_traces.py --workload bert --replica-count 10 --output traces.npz
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import argparse
import json


ALL_METRICS = [
    'pod_cpu_usage',        # 0
    'pod_memory_bytes',     # 1
    'pod_psi_cpu',          # 2
    'pod_latency_avg',      # 3
    'pod_throughput',       # 4
    'gpu_utilization',      # 5
    'gpu_memory_used',      # 6
    'gpu_power_watts',      # 7
]

# S27 trained metrics per workload (indices 0-7 only)
# Note: gpu_power_watts (index 7) is TRAINED in all workloads!
S27_TRAINED = {
    'bert':      [0, 1, 2, 3, 4, 5, 7],        # Missing: gpu_memory_used only
    'gpt2':      [0, 1, 2, 3, 4, 5, 6, 7],     # All 8 metrics trained
    'resnet152': [0, 1, 2, 3, 4, 5, 7],        # Missing: gpu_memory_used only
    'whisper':   [0, 1, 2, 3, 5, 7],           # Missing: pod_throughput, gpu_memory_used
    'yolo':      [0, 1, 2, 3, 4, 5, 7],        # Missing: gpu_memory_used only
}

# GPU memory constants (MB) - architecturally determined
# These are fixed per workload regardless of replica count
GPU_MEMORY_CONSTANTS = {
    'bert': 562.0,
    'gpt2': 641.0,
    'resnet152': 554.0,
    'whisper': 1511.0,
    'yolo': 424.0,
}

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
    """S27 segment-based generator architecture."""
    
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


class S27TraceGenerator:
    """High-level interface for generating S27 synthetic traces."""
    
    def __init__(self, workload, model_dir='models/phase4/timegan_s27/s27_seg_vr03_fm10_ae150',
                 data_dir='data/processed/phase4/raw', device='cpu'):
        self.workload = workload
        self.device = torch.device(device)
        
        # Load normalization parameters
        norm_path = Path(data_dir) / f'{workload}_normalization.json'
        with open(norm_path, 'r') as f:
            norm_data = json.load(f)
        self.norm_params = norm_data['params']
        
        # Load real traces for reconstruction
        data_path = Path(data_dir) / f'{workload}_traces.npz'
        data = np.load(data_path, allow_pickle=True)
        self.real_traces = data['traces']
        
        # Determine trained indices
        self.trained_indices = S27_TRAINED[workload]
        
        print(f"Workload: {workload}")
        print(f"Trained metrics ({len(self.trained_indices)}): {[ALL_METRICS[i] for i in self.trained_indices]}")
        
        # Determine what needs reconstruction
        reconstruct_list = []
        if 6 not in self.trained_indices:
            reconstruct_list.append('gpu_memory_used (constant)')
        if 4 not in self.trained_indices:
            reconstruct_list.append('pod_throughput (mean)')
        
        if reconstruct_list:
            print(f"Reconstructed metrics: {', '.join(reconstruct_list)}")
        
        # Load S27 generator
        checkpoint_path = Path(model_dir) / workload / 'generator.pt'
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        n_metrics = len(self.trained_indices)
        self.generator = GeneratorSeg(SEGMENT_LEN, n_metrics, GEN_CFG).to(self.device)
        self.generator.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.generator.eval()
        
        print(f"Loaded S27 checkpoint: {checkpoint_path}")
    
    def denormalize_metrics(self, trace_norm, metric_indices):
        """Denormalize from [0,1] to original scale."""
        trace_denorm = np.zeros_like(trace_norm)
        
        for i, idx in enumerate(metric_indices):
            metric_name = ALL_METRICS[idx]
            params = self.norm_params[metric_name]
            min_val = params['min']
            max_val = params['max']
            trace_denorm[:, i] = trace_norm[:, i] * (max_val - min_val) + min_val
        
        return trace_denorm
    
    def reconstruct_gpu_memory_used(self):
        """Constant reconstruction - architecturally determined."""
        return GPU_MEMORY_CONSTANTS[self.workload]
    
    def reconstruct_pod_throughput(self):
        """Mean from real data (Whisper only)."""
        idx = 4
        all_values = self.real_traces[:, :, idx]
        params = self.norm_params[ALL_METRICS[idx]]
        all_values_denorm = all_values * (params['max'] - params['min']) + params['min']
        return np.mean(all_values_denorm)
    
    def reconstruct_missing_metrics(self, synth_trace_partial):
        """
        Apply reconstruction for missing metrics.
        
        Args:
            synth_trace_partial: (715, len(trained_indices)) denormalized
        
        Returns:
            synth_trace_full: (715, 8) complete trace
        """
        T = 715
        synth_trace_full = np.zeros((T, 8))
        
        # Insert trained metrics
        for i, idx in enumerate(self.trained_indices):
            synth_trace_full[:, idx] = synth_trace_partial[:, i]
        
        # Reconstruct gpu_memory_used (if not trained)
        if 6 not in self.trained_indices:
            synth_trace_full[:, 6] = self.reconstruct_gpu_memory_used()
        
        # Reconstruct pod_throughput (Whisper only)
        if 4 not in self.trained_indices:
            synth_trace_full[:, 4] = self.reconstruct_pod_throughput()
        
        return synth_trace_full
    
    def generate_trace(self, replica_count, seed=None):
        """
        Generate a single synthetic trace.
        
        Args:
            replica_count: int (1-100+)
            seed: optional random seed
        
        Returns:
            trace: (715, 8) numpy array in original scale
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Normalize replica count
        r_norm = torch.tensor([(replica_count - 1) / 9.0], dtype=torch.float32, device=self.device)
        
        # Generate 6 phase segments
        segments = []
        with torch.no_grad():
            for ph in range(N_PHASES):
                phase_idx = torch.tensor([ph], dtype=torch.long, device=self.device)
                seg = self.generator(r_norm, phase_idx)
                segments.append(seg.squeeze(0).cpu().numpy())
        
        trace_norm = np.concatenate(segments, axis=0)[:715]
        
        # Denormalize trained metrics
        trace_trained = self.denormalize_metrics(trace_norm, self.trained_indices)
        
        # Reconstruct missing metrics
        trace_full = self.reconstruct_missing_metrics(trace_trained)
        
        return trace_full
    
    def generate_multiple_traces(self, replica_count, n_pods, base_seed=42):
        """
        Generate multiple traces for Kwok simulation.
        
        Args:
            replica_count: int (1-100+)
            n_pods: number of pod traces
            base_seed: base random seed
        
        Returns:
            traces: (n_pods, 715, 8) numpy array
        """
        traces = []
        for i in range(n_pods):
            trace = self.generate_trace(replica_count, seed=base_seed + i)
            traces.append(trace)
        
        return np.array(traces)


def main():
    parser = argparse.ArgumentParser(description='S27 Trace Generator')
    parser.add_argument('--workload', type=str, required=True,
                        choices=['bert', 'gpt2', 'resnet152', 'whisper', 'yolo'])
    parser.add_argument('--replica-count', type=int, required=True,
                        help='Target replica count (1-100+)')
    parser.add_argument('--n-pods', type=int, default=1,
                        help='Number of pod traces to generate')
    parser.add_argument('--output', type=str, default=None,
                        help='Output .npz file path')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model-dir', type=str,
                        default='models/phase4/timegan_s27/s27_seg_vr03_fm10_ae150')
    parser.add_argument('--data-dir', type=str,
                        default='data/processed/phase4/raw')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("S27 Trace Generator")
    print("=" * 70)
    print(f"Workload: {args.workload}")
    print(f"Replica count: {args.replica_count}")
    print(f"Number of pods: {args.n_pods}")
    print(f"Device: {args.device}")
    print("=" * 70)
    
    # Initialize generator
    generator = S27TraceGenerator(
        workload=args.workload,
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        device=args.device
    )
    
    # Generate traces
    print(f"\nGenerating {args.n_pods} trace(s)...")
    traces = generator.generate_multiple_traces(
        replica_count=args.replica_count,
        n_pods=args.n_pods,
        base_seed=args.seed
    )
    
    print(f"Generated traces shape: {traces.shape}")
    
    # Print statistics
    print("\nGenerated Trace Statistics:")
    print(f"{'Metric':<20} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print("-" * 68)
    for i, metric in enumerate(ALL_METRICS):
        mean = traces[:, :, i].mean()
        std = traces[:, :, i].std()
        min_val = traces[:, :, i].min()
        max_val = traces[:, :, i].max()
        print(f"{metric:<20} {mean:>12.4f} {std:>12.4f} {min_val:>12.4f} {max_val:>12.4f}")
    
    # Save if output specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez_compressed(
            output_path,
            traces=traces,
            workload=args.workload,
            replica_count=args.replica_count,
            n_pods=args.n_pods,
            metric_names=np.array(ALL_METRICS),
            seed=args.seed,
            model='s27'
        )
        print(f"\nSaved traces to: {output_path}")
    else:
        print("\nNo output file specified. Traces not saved.")
    
    print("\nDone!")


if __name__ == '__main__':
    main()