"""
S34 Evaluation with Boundary Smoothing
========================================
Evaluate S34 models (all 5 workloads) with automatic boundary smoothing.

S34 = S27 architecture + all 5 workloads + same 7 metrics

Usage:
    python eval_s34.py
    python eval_s34.py --workloads bert gpt2
    python eval_s34.py --window_size 3  # Less aggressive smoothing
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Import smoothing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from boundary_smoothing import smooth_phase_boundaries

# Paths
DATA_COMBINED = Path("data/processed/phase4/unified/combined_dataset.npz")
NORM_COMBINED = Path("data/processed/phase4/unified/combined_normalization.json")

WORKLOADS_ALL = ["bert", "gpt2", "resnet152", "whisper", "yolo"]
S34_WORKLOADS = ["bert", "gpt2", "resnet152", "whisper", "yolo"]  # All 5 workloads

ALL_METRICS = [
    "pod_cpu_usage", "pod_memory_bytes", "pod_psi_cpu",
    "pod_latency_avg", "pod_throughput",
    "gpu_utilization", "gpu_memory_used", "gpu_memory_total",
    "gpu_power_watts", "gpu_temperature",
]

# S34: All workloads have same 7 metrics
DROP_PER_WORKLOAD = {
    "bert":      {"gpu_memory_total", "gpu_memory_used", "gpu_temperature"},
    "gpt2":      {"gpu_memory_total", "gpu_memory_used", "gpu_temperature"},
    "resnet152": {"gpu_memory_total", "gpu_memory_used", "gpu_temperature"},
    "whisper":   {"gpu_memory_total", "gpu_memory_used", "gpu_temperature"},
    "yolo":      {"gpu_memory_total", "gpu_memory_used", "gpu_temperature"},
}

# Phase config
N_PHASES = 6
SEGMENT_LEN = 120
ACTUAL_BOUNDARIES = [120, 240, 360, 480, 600]  # Generated trace boundaries

GEN_CFG = {
    "hidden_dim": 128,
    "num_layers": 2,
    "latent_dim": 64,
    "replica_embed_dim": 16,
    "phase_embed_dim": 8,
    "dropout": 0.1,
}

# S27 reference VR
S27_VR = {
    "bert": 0.886,
    "gpt2": 0.946,
    "resnet152": 0.851,
    "whisper": 1.630,
    "yolo": 0.997,
}


class GeneratorSeg(nn.Module):
    """S27/S34 generator."""
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
    
    @torch.no_grad()
    def generate_trace(self, r_norm_val, device, n_samples=1):
        """Generate full trace."""
        self.eval()
        segments = []
        for ph in range(N_PHASES):
            r_norm = torch.full((n_samples,), r_norm_val, dtype=torch.float32, device=device)
            ph_idx = torch.full((n_samples,), ph, dtype=torch.long, device=device)
            seg = self(r_norm, ph_idx)
            segments.append(seg.cpu().numpy())
        full_trace = np.concatenate(segments, axis=1)
        return full_trace[:, :715, :]


def denormalize(traces_norm, norm_params, kept_names):
    """Denormalize traces."""
    traces = traces_norm.copy()
    for i, metric in enumerate(kept_names):
        if metric in norm_params:
            traces[:, :, i] = traces[:, :, i] * norm_params[metric]["std"] + norm_params[metric]["mean"]
    return traces


def compute_vr_per_metric(real_traces, syn_traces):
    """Compute variance ratio per metric."""
    real_var = real_traces.var(axis=(0, 1))
    syn_var = syn_traces.var(axis=(0, 1))
    vr = syn_var / (real_var + 1e-8)
    return vr


def evaluate_workload(workload_name, device, window_size=5, n_gen=5):
    """Evaluate single workload with smoothing."""
    print(f"\n{'='*80}")
    print(f"EVALUATING: {workload_name.upper()}")
    print(f"{'='*80}")
    
    # Load data
    data = np.load(DATA_COMBINED, allow_pickle=True)
    with open(NORM_COMBINED) as f:
        norm = json.load(f)
    
    workload_id = WORKLOADS_ALL.index(workload_name)
    all_traces = data["traces"]
    all_rc = data["replica_counts"]
    all_wl = data["workload_ids"]
    val_idx = data["val_idx"]
    
    # Filter workload
    wl_mask = (all_wl == workload_id)
    traces = all_traces[wl_mask]
    rc = all_rc[wl_mask]
    
    # Get kept metrics (same 7 for all S34 workloads)
    drop = DROP_PER_WORKLOAD.get(workload_name, set())
    kept_idx = [i for i, m in enumerate(ALL_METRICS) if m not in drop]
    kept_names = [ALL_METRICS[i] for i in kept_idx]
    n_metrics = len(kept_names)
    
    traces = traces[:, :, kept_idx].astype(np.float32)
    
    # Validation set
    orig_to_new = {}
    new_idx = 0
    for old_idx in range(len(all_traces)):
        if wl_mask[old_idx]:
            orig_to_new[old_idx] = new_idx
            new_idx += 1
    
    new_val_idx = np.array([orig_to_new[i] for i in val_idx if i in orig_to_new])
    
    real_traces_val = traces[new_val_idx]
    real_rc_val = rc[new_val_idx]
    
    print(f"Metrics ({n_metrics}): {kept_names}")
    print(f"Validation: {len(new_val_idx)} pods")
    
    # Load S34 model
    model_paths = {
        "bert": "models/phase4/timegan_s34/s34_bert_vr05_fm15/generator.pt",
        "gpt2": "models/phase4/timegan_s34/s34_gpt2_vr03_fm20/generator.pt",
        "resnet152": "models/phase4/timegan_s34/s34_resnet152_vr04_fm10/generator.pt",
        "whisper": "models/phase4/timegan_s34/s34_whisper_vr05_fm05/generator.pt",
        "yolo": "models/phase4/timegan_s34/s34_yolo_vr03_fm12/generator.pt",
    }
    
    model_path = Path(model_paths[workload_name])
    if not model_path.exists():
        print(f"ERROR: No S34 model found for {workload_name}")
        return None
    
    print(f"Model: {model_path}")
    
    generator = GeneratorSeg(SEGMENT_LEN, n_metrics, GEN_CFG).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()
    
    # Generate traces
    print(f"\nGenerating traces (window_size={window_size})...")
    unique_r = np.unique(real_rc_val)
    
    syn_traces_raw = []
    syn_traces_smooth = []
    
    for r_val in unique_r:
        r_norm = (r_val - 1.0) / 9.0
        traces_gen = generator.generate_trace(r_norm, device, n_samples=n_gen)
        
        for trace in traces_gen:
            syn_traces_raw.append(trace)
            # Apply smoothing
            smooth_trace = smooth_phase_boundaries(trace, ACTUAL_BOUNDARIES, window_size)
            syn_traces_smooth.append(smooth_trace)
    
    syn_traces_raw = np.array(syn_traces_raw)
    syn_traces_smooth = np.array(syn_traces_smooth)
    
    print(f"Generated: {len(syn_traces_raw)} raw, {len(syn_traces_smooth)} smooth")
    
    # Denormalize
    real_denorm = denormalize(real_traces_val, norm[workload_name], kept_names)
    syn_raw_denorm = denormalize(syn_traces_raw, norm[workload_name], kept_names)
    syn_smooth_denorm = denormalize(syn_traces_smooth, norm[workload_name], kept_names)
    
    # Compute VR
    vr_raw = compute_vr_per_metric(real_denorm, syn_raw_denorm)
    vr_smooth = compute_vr_per_metric(real_denorm, syn_smooth_denorm)
    
    mean_vr_raw = float(np.mean(vr_raw))
    mean_vr_smooth = float(np.mean(vr_smooth))
    
    # Check pass/fail (VR > 0.8)
    pass_raw = bool(mean_vr_raw > 0.8)
    pass_smooth = bool(mean_vr_smooth > 0.8)
    
    print(f"\nRESULTS:")
    print(f"  VR (raw):     {mean_vr_raw:.3f} {'✓ PASS' if pass_raw else '✗ FAIL'}")
    print(f"  VR (smooth):  {mean_vr_smooth:.3f} {'✓ PASS' if pass_smooth else '✗ FAIL'}")
    print(f"  S27 VR:       {S27_VR.get(workload_name, 0):.3f}")
    print(f"  Change:       {mean_vr_smooth - mean_vr_raw:+.3f}")
    
    print(f"\nPer-metric VR (smoothed):")
    for i, metric in enumerate(kept_names):
        status = "✓" if vr_smooth[i] > 0.8 else "✗"
        print(f"  {status} {metric:<22} {vr_smooth[i]:.3f}")
    
    return {
        "workload": workload_name,
        "vr_raw": mean_vr_raw,
        "vr_smooth": mean_vr_smooth,
        "vr_s27": float(S27_VR.get(workload_name, 0)),
        "pass_raw": pass_raw,
        "pass_smooth": pass_smooth,
        "vr_per_metric_smooth": [float(v) for v in vr_smooth.tolist()],
        "metric_names": kept_names,
    }


def main():
    parser = argparse.ArgumentParser(description="S34 Evaluation with Smoothing")
    parser.add_argument("--workloads", nargs='+', type=str,
                        default=S34_WORKLOADS,
                        help="Workloads to evaluate")
    parser.add_argument("--window_size", type=int, default=5,
                        help="Smoothing window size (default: 5)")
    parser.add_argument("--n_gen", type=int, default=5,
                        help="Traces to generate per replica count")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*80)
    print("S34 EVALUATION (ALL 5 WORKLOADS + SMOOTHING)")
    print("="*80)
    print(f"Device: {device}")
    print(f"Workloads: {args.workloads}")
    print(f"Window size: {args.window_size}")
    print(f"Smoothing: ENABLED")
    print("="*80)
    
    results = {}
    
    for wl in args.workloads:
        result = evaluate_workload(wl, device, args.window_size, args.n_gen)
        if result:
            results[wl] = result
    
    # Save results
    out_dir = Path("outputs/phase4/timegan_s34")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "s34_eval_results.json", "w") as f:
        json.dump({
            "window_size": args.window_size,
            "smoothing_enabled": True,
            "workloads": results,
            "summary": {
                "mean_vr_smooth": float(np.mean([r["vr_smooth"] for r in results.values()])),
                "pass_count": int(sum(1 for r in results.values() if r["pass_smooth"])),
                "total": int(len(results)),
            }
        }, f, indent=2)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY (SMOOTHED)")
    print("="*80)
    
    print(f"\n{'Workload':<15} {'VR S34':<10} {'VR S27':<10} {'Status':<10}")
    print("-"*80)
    for wl, res in results.items():
        status = "✓ PASS" if res["pass_smooth"] else "✗ FAIL"
        print(f"{wl:<15} {res['vr_smooth']:<10.3f} {res['vr_s27']:<10.3f} {status}")
    
    summary = {
        "mean_vr": float(np.mean([r["vr_smooth"] for r in results.values()])),
        "pass_count": int(sum(1 for r in results.values() if r["pass_smooth"])),
    }
    
    print("-"*80)
    print(f"{'MEAN':<15} {summary['mean_vr']:<10.3f} {'1.062':<10} "
          f"{summary['pass_count']}/{len(results)} PASS")
    
    print("\n" + "="*80)
    print("COMPARISON TO S27")
    print("="*80)
    
    for wl, res in results.items():
        diff = res["vr_smooth"] - res["vr_s27"]
        direction = "↑" if diff > 0 else "↓"
        print(f"  {wl:<15} {direction} {abs(diff):.3f} "
              f"({'better' if diff > 0 else 'worse'} than S27)")
    
    print("\n" + "="*80)
    print("RESULTS SAVED")
    print("="*80)
    print(f"JSON: {out_dir / 's34_eval_results.json'}")


if __name__ == "__main__":
    main()