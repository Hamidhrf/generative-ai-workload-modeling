#!/usr/bin/env python3
"""
Comprehensive S27 vs S33 Evaluation
====================================
Beyond VR: Autocorrelation, Cross-correlation, Distribution matching

Metrics computed:
1. Variance Ratio (VR)
2. Autocorrelation (lag-1 to lag-20)
3. Cross-metric correlation matrices
4. KL Divergence (distribution matching)
5. Wasserstein Distance
6. Boundary discontinuity
7. Per-metric statistics (mean, std, min, max)

Usage:
  python comprehensive_eval_s27_vs_s33.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import json
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon

# Add utils
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from boundary_smoothing import smooth_phase_boundaries

# Config
WORKLOADS = ['bert', 'gpt2', 'resnet152', 'yolo']
REPLICA_COUNT = 5  # r=5 for comparison
N_SYNTHETIC = 25  # Generate 25 traces per model
ACTUAL_BOUNDARIES = [120, 240, 360, 480, 600]
WINDOW_SIZE = 5

# Metric configs
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

# S27/S33 trained metrics (same for all 4 GPU-bound workloads in S33)
S27_TRAINED = {
    'bert':      [0, 1, 2, 3, 4, 5, 8],
    'gpt2':      [0, 1, 2, 3, 4, 5, 6, 8],
    'resnet152': [0, 1, 2, 3, 4, 5, 8],
    'yolo':      [0, 1, 2, 3, 4, 5, 8],
}

S33_TRAINED = {
    'bert':      [0, 1, 2, 3, 4, 5, 8],
    'gpt2':      [0, 1, 2, 3, 4, 5, 8],
    'resnet152': [0, 1, 2, 3, 4, 5, 8],
    'yolo':      [0, 1, 2, 3, 4, 5, 8],
}

# Generator architecture
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


def load_generator(workload, model_name, device='cuda'):
    """Load S27 or S33 generator"""
    if model_name == 's27':
        base = Path('models/phase4/timegan_s27/s27_seg_vr03_fm10_ae150')
        ckpt_path = base / workload / 'generator.pt'
        trained_indices = S27_TRAINED[workload]
    else:  # s33
        base = Path('models/phase4/timegan_s33')
        model_dirs = list(base.glob(f's33_{workload}_*'))
        if not model_dirs:
            raise FileNotFoundError(f"No S33 model for {workload}")
        ckpt_path = model_dirs[0] / 'generator.pt'
        trained_indices = S33_TRAINED[workload]
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint: {ckpt_path}")
    
    n_metrics = len(trained_indices)
    gen = GeneratorSeg(SEGMENT_LEN, n_metrics, GEN_CFG).to(device)
    gen.load_state_dict(torch.load(ckpt_path, map_location=device))
    gen.eval()
    
    return gen, trained_indices


def load_real_data(workload, data_dir='data/processed/phase4/raw'):
    """Load normalized traces"""
    trace_path = Path(data_dir) / f"{workload}_traces.npz"
    data = np.load(trace_path, allow_pickle=True)
    return data


def generate_traces(gen, workload, replica_count, n_samples, trained_indices, device='cuda'):
    """Generate and denormalize synthetic traces"""
    norm_params = load_normalization_params(workload)
    r_norm = torch.tensor([(replica_count - 1) / 9.0], dtype=torch.float32, device=device)
    
    all_traces = []
    for _ in range(n_samples):
        phases = []
        for ph in range(N_PHASES):
            phase_idx = torch.tensor([ph], dtype=torch.long, device=device)
            seg = gen(r_norm, phase_idx)
            phases.append(seg.squeeze(0).cpu().detach().numpy())
        
        trace_norm = np.concatenate(phases, axis=0)[:715]
        trace_denorm = denormalize_trace(trace_norm, norm_params, trained_indices)
        
        # Build full 10-metric trace (for trained metrics only)
        trace_full = np.zeros((715, len(trained_indices)))
        for i, idx in enumerate(trained_indices):
            trace_full[:, i] = trace_denorm[:, i]
        
        all_traces.append(trace_full)
    
    return np.array(all_traces), trained_indices


def get_real_traces(workload, replica_count):
    """Get real validation traces (normalized, trained metrics only)"""
    data = load_real_data(workload)
    norm_params = load_normalization_params(workload)
    
    traces_norm = data['traces']
    replica_counts = data['replica_counts']
    
    mask = (replica_counts == replica_count)
    matching = traces_norm[mask]
    
    if len(matching) == 0:
        raise ValueError(f"No real traces for {workload} r={replica_count}")
    
    # Get trained metric indices
    trained_indices = S27_TRAINED[workload]  # Same as S33 for GPU-bound
    
    # Denormalize
    all_real = []
    for trace_norm in matching:
        trace_denorm = denormalize_trace(trace_norm, norm_params, list(range(10)))
        # Extract only trained metrics
        trace_subset = trace_denorm[:, trained_indices]
        all_real.append(trace_subset)
    
    return np.array(all_real), trained_indices


def compute_autocorrelation(traces, max_lag=20):
    """
    Compute mean autocorrelation across all traces and metrics
    Returns: array of shape (max_lag,) with mean autocorr per lag
    """
    N, T, M = traces.shape
    autocorrs = []
    
    for n in range(N):
        for m in range(M):
            series = traces[n, :, m]
            series_norm = (series - series.mean()) / (series.std() + 1e-8)
            
            acf = []
            for lag in range(1, max_lag + 1):
                if lag < T:
                    corr = np.corrcoef(series_norm[:-lag], series_norm[lag:])[0, 1]
                    acf.append(corr if not np.isnan(corr) else 0.0)
                else:
                    acf.append(0.0)
            
            autocorrs.append(acf)
    
    return np.mean(autocorrs, axis=0)


def compute_cross_correlation(traces, metric_names):
    """
    Compute mean cross-correlation matrix across all traces
    Returns: (M, M) correlation matrix
    """
    N, T, M = traces.shape
    corr_matrices = []
    
    for n in range(N):
        # Compute correlation for this trace
        trace_2d = traces[n]  # (T, M)
        corr = np.corrcoef(trace_2d.T)  # (M, M)
        corr_matrices.append(corr)
    
    return np.mean(corr_matrices, axis=0)


def compute_kl_divergence(real_traces, synth_traces):
    """
    Compute KL divergence per metric (using histogram approximation)
    Returns: array of shape (M,) with KL divergence per metric
    """
    M = real_traces.shape[2]
    kl_divs = []
    
    for m in range(M):
        real_flat = real_traces[:, :, m].flatten()
        synth_flat = synth_traces[:, :, m].flatten()
        
        # Create histograms (30 bins)
        bins = np.linspace(
            min(real_flat.min(), synth_flat.min()),
            max(real_flat.max(), synth_flat.max()),
            31
        )
        
        real_hist, _ = np.histogram(real_flat, bins=bins, density=True)
        synth_hist, _ = np.histogram(synth_flat, bins=bins, density=True)
        
        # Normalize to sum to 1
        real_hist = real_hist / (real_hist.sum() + 1e-8)
        synth_hist = synth_hist / (synth_hist.sum() + 1e-8)
        
        # Add small epsilon to avoid log(0)
        real_hist = real_hist + 1e-10
        synth_hist = synth_hist + 1e-10
        
        # JS divergence (symmetric KL)
        js_div = jensenshannon(real_hist, synth_hist)
        kl_divs.append(js_div)
    
    return np.array(kl_divs)


def compute_wasserstein(real_traces, synth_traces):
    """
    Compute Wasserstein distance per metric
    Returns: array of shape (M,) with Wasserstein distance per metric
    """
    M = real_traces.shape[2]
    wass_dists = []
    
    for m in range(M):
        real_flat = real_traces[:, :, m].flatten()
        synth_flat = synth_traces[:, :, m].flatten()
        
        wass = wasserstein_distance(real_flat, synth_flat)
        wass_dists.append(wass)
    
    return np.array(wass_dists)


def compute_boundary_discontinuity(traces, boundaries):
    """
    Measure total discontinuity at boundaries
    Returns: scalar discontinuity score
    """
    total_disc = 0.0
    N, T, M = traces.shape
    
    for n in range(N):
        for boundary in boundaries:
            if boundary <= 0 or boundary >= T:
                continue
            
            before = traces[n, boundary - 1]
            after = traces[n, boundary]
            total_disc += np.sum(np.abs(after - before))
    
    return total_disc / (N * len(boundaries) * M)


def compute_variance_ratio(real_traces, synth_traces):
    """
    Compute variance ratio per metric
    Returns: array of shape (M,) with VR per metric
    """
    M = real_traces.shape[2]
    vrs = []
    
    for m in range(M):
        real_var = np.var(real_traces[:, :, m])
        synth_var = np.var(synth_traces[:, :, m])
        vr = synth_var / (real_var + 1e-8)
        vrs.append(vr)
    
    return np.array(vrs)


def evaluate_model(workload, model_name, apply_smoothing=False):
    """
    Comprehensive evaluation of a model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"  Loading {model_name.upper()} model...")
    gen, trained_indices = load_generator(workload, model_name, device)
    metric_names = [METRIC_NAMES[i] for i in trained_indices]
    
    print(f"  Loading real traces...")
    real_traces, _ = get_real_traces(workload, REPLICA_COUNT)
    
    print(f"  Generating {N_SYNTHETIC} synthetic traces...")
    synth_traces, _ = generate_traces(gen, workload, REPLICA_COUNT, N_SYNTHETIC, trained_indices, device)
    
    if apply_smoothing:
        print(f"  Applying smoothing (window={WINDOW_SIZE})...")
        synth_traces_smooth = np.array([
            smooth_phase_boundaries(trace, ACTUAL_BOUNDARIES, WINDOW_SIZE)
            for trace in synth_traces
        ])
        synth_traces = synth_traces_smooth
    
    print(f"  Computing metrics...")
    
    # 1. Variance Ratio
    vr_per_metric = compute_variance_ratio(real_traces, synth_traces)
    vr_mean = np.mean(vr_per_metric)
    
    # 2. Autocorrelation
    real_acf = compute_autocorrelation(real_traces, max_lag=20)
    synth_acf = compute_autocorrelation(synth_traces, max_lag=20)
    acf_error = np.mean(np.abs(real_acf - synth_acf))
    
    # 3. Cross-correlation
    real_corr = compute_cross_correlation(real_traces, metric_names)
    synth_corr = compute_cross_correlation(synth_traces, metric_names)
    corr_error = np.mean(np.abs(real_corr - synth_corr))
    
    # 4. Distribution matching
    kl_divs = compute_kl_divergence(real_traces, synth_traces)
    wass_dists = compute_wasserstein(real_traces, synth_traces)
    
    # 5. Boundary discontinuity
    boundary_disc = compute_boundary_discontinuity(synth_traces, ACTUAL_BOUNDARIES)
    
    # 6. Statistics
    real_mean = np.mean(real_traces, axis=(0, 1))
    synth_mean = np.mean(synth_traces, axis=(0, 1))
    real_std = np.std(real_traces, axis=(0, 1))
    synth_std = np.std(synth_traces, axis=(0, 1))
    
    return {
        'model': model_name + ('_smooth' if apply_smoothing else ''),
        'workload': workload,
        'metric_names': metric_names,
        'n_metrics': len(metric_names),
        'vr_mean': float(vr_mean),
        'vr_per_metric': vr_per_metric.tolist(),
        'acf_real': real_acf.tolist(),
        'acf_synth': synth_acf.tolist(),
        'acf_error': float(acf_error),
        'corr_real': real_corr.tolist(),
        'corr_synth': synth_corr.tolist(),
        'corr_error': float(corr_error),
        'kl_divs': kl_divs.tolist(),
        'kl_mean': float(np.mean(kl_divs)),
        'wass_dists': wass_dists.tolist(),
        'wass_mean': float(np.mean(wass_dists)),
        'boundary_disc': float(boundary_disc),
        'real_mean': real_mean.tolist(),
        'synth_mean': synth_mean.tolist(),
        'real_std': real_std.tolist(),
        'synth_std': synth_std.tolist(),
    }


def print_comparison_table(results):
    """Print comparison table"""
    print("\n" + "="*100)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("="*100)
    
    for workload in WORKLOADS:
        print(f"\n{'='*100}")
        print(f"WORKLOAD: {workload.upper()}")
        print(f"{'='*100}")
        
        # Filter results for this workload
        w_results = [r for r in results if r['workload'] == workload]
        
        if len(w_results) == 0:
            continue
        
        # Print header
        models = [r['model'] for r in w_results]
        print(f"\n{'Metric':<30} " + " ".join([f"{m:>15}" for m in models]))
        print("-"*100)
        
        # VR
        print(f"{'Variance Ratio (mean)':<30} " + " ".join([f"{r['vr_mean']:>15.3f}" for r in w_results]))
        
        # Autocorrelation error
        print(f"{'Autocorr Error (MAE)':<30} " + " ".join([f"{r['acf_error']:>15.4f}" for r in w_results]))
        
        # Cross-correlation error
        print(f"{'Cross-corr Error (MAE)':<30} " + " ".join([f"{r['corr_error']:>15.4f}" for r in w_results]))
        
        # KL divergence
        print(f"{'KL Divergence (mean)':<30} " + " ".join([f"{r['kl_mean']:>15.4f}" for r in w_results]))
        
        # Wasserstein
        print(f"{'Wasserstein (mean)':<30} " + " ".join([f"{r['wass_mean']:>15.2f}" for r in w_results]))
        
        # Boundary discontinuity
        print(f"{'Boundary Discontinuity':<30} " + " ".join([f"{r['boundary_disc']:>15.4f}" for r in w_results]))
        
        print("\n" + "-"*100)
        print("WINNER ANALYSIS:")
        print("-"*100)
        
        # Determine winners per metric
        vr_winner = min(w_results, key=lambda r: abs(r['vr_mean'] - 1.0))
        acf_winner = min(w_results, key=lambda r: r['acf_error'])
        corr_winner = min(w_results, key=lambda r: r['corr_error'])
        kl_winner = min(w_results, key=lambda r: r['kl_mean'])
        wass_winner = min(w_results, key=lambda r: r['wass_mean'])
        disc_winner = min(w_results, key=lambda r: r['boundary_disc'])
        
        print(f"  VR closest to 1.0:        {vr_winner['model']}")
        print(f"  Best Autocorrelation:     {acf_winner['model']}")
        print(f"  Best Cross-correlation:   {corr_winner['model']}")
        print(f"  Best Distribution (KL):   {kl_winner['model']}")
        print(f"  Best Distribution (Wass): {wass_winner['model']}")
        print(f"  Smoothest Boundaries:     {disc_winner['model']}")


def main():
    print("="*100)
    print("COMPREHENSIVE S27 vs S33 EVALUATION")
    print("="*100)
    print(f"Workloads: {WORKLOADS}")
    print(f"Replica count: {REPLICA_COUNT}")
    print(f"Synthetic traces per model: {N_SYNTHETIC}")
    print(f"Smoothing window: {WINDOW_SIZE}")
    print("="*100)
    
    all_results = []
    
    for workload in WORKLOADS:
        print(f"\n{'='*100}")
        print(f"EVALUATING: {workload.upper()}")
        print(f"{'='*100}")
        
        # Evaluate S27 (raw)
        print("\n[1/3] S27 (raw)")
        try:
            result_s27 = evaluate_model(workload, 's27', apply_smoothing=False)
            all_results.append(result_s27)
        except Exception as e:
            print(f"  ERROR: {e}")
        
        # Evaluate S27 (smoothed)
        print("\n[2/3] S27 (smoothed)")
        try:
            result_s27_smooth = evaluate_model(workload, 's27', apply_smoothing=True)
            all_results.append(result_s27_smooth)
        except Exception as e:
            print(f"  ERROR: {e}")
        
        # Evaluate S33 (smoothed)
        print("\n[3/3] S33 (smoothed)")
        try:
            result_s33 = evaluate_model(workload, 's33', apply_smoothing=True)
            all_results.append(result_s33)
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Print comparison
    print_comparison_table(all_results)
    
    # Save results
    output_dir = Path('outputs/phase4/comprehensive_eval')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'comprehensive_results.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*100}")
    print("RESULTS SAVED")
    print(f"{'='*100}")
    print(f"JSON: {output_path}")
    print(f"{'='*100}")


if __name__ == '__main__':
    main()