"""
S29 Evaluation - Per-Workload Quality Assessment
=================================================
Evaluates unified S29 model by generating traces for each workload
and comparing against S27 baseline.

Usage:
    python evaluate_s29.py
    python evaluate_s29.py --n-gen 20
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path

# Import S29 model classes
import sys
sys.path.append('scripts/phase4/timegan')
from timegan_s29 import (
    GeneratorSegUnified, GEN_CFG, WORKLOADS, N_PHASES,
    SEGMENT_LEN, FULL_SEQ_LEN, DEFAULT_PHASE_BOUNDARIES,
    get_unified_kept_metrics, denormalize_unified, load_combined_data
)

REFERENCE_VR_S27 = {
    "bert": 0.886,
    "gpt2": 0.946,
    "resnet152": 0.851,
    "whisper": 1.630,
    "yolo": 0.997,
}

REFERENCE_VR_S21 = {
    "bert": 0.672,
    "gpt2": 0.996,
    "resnet152": 0.960,
    "whisper": 1.722,
    "yolo": 0.661,
}


def compute_variance_ratio(real, synthetic, cap=5.0):
    """Compute variance ratio per metric."""
    var_r = real.reshape(-1, real.shape[-1]).var(axis=0)
    var_s = synthetic.reshape(-1, synthetic.shape[-1]).var(axis=0)
    ratio = np.where(var_r > 1e-10,
                     np.clip(var_s / var_r, 0, cap),
                     np.ones_like(var_r))
    return ratio, float(ratio.mean())


def evaluate_workload(generator, workload_id, workload_name, 
                      real_traces, real_replica_counts,
                      norm_params, kept_names, device, n_gen=10):
    """
    Evaluate S29 model for a specific workload.
    
    Args:
        generator: Trained S29 unified model
        workload_id: 0-4 (workload index)
        workload_name: 'bert', 'gpt2', etc.
        real_traces: Real pod traces for this workload (normalized)
        real_replica_counts: Replica counts for real traces
        norm_params: Per-workload normalization params
        kept_names: List of metric names
        device: cuda/cpu
        n_gen: Number of synthetic traces per replica count
    
    Returns:
        dict with results
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {workload_name.upper()}")
    print(f"{'='*60}")
    
    unique_r = np.unique(real_replica_counts)
    print(f"  Replica counts: {unique_r.tolist()}")
    
    # Generate synthetic traces
    syn_traces_norm = []
    syn_replica_counts = []
    
    for r_val in unique_r:
        r_norm = (r_val - 1.0) / 9.0
        traces = generator.generate_trace(r_norm, workload_id, device, n_samples=n_gen)
        for trace in traces:
            syn_traces_norm.append(trace)
            syn_replica_counts.append(int(r_val))
    
    syn_traces_norm = np.array(syn_traces_norm)
    syn_replica_counts = np.array(syn_replica_counts, dtype=np.int32)
    
    print(f"  Generated: {len(syn_traces_norm)} synthetic traces")
    
    # Denormalize (per-workload normalization)
    real_workload_ids = np.full(len(real_traces), workload_id, dtype=np.int32)
    syn_workload_ids = np.full(len(syn_traces_norm), workload_id, dtype=np.int32)
    
    real_traces_denorm = denormalize_unified(
        real_traces, real_workload_ids, kept_names, norm_params)
    syn_traces_denorm = denormalize_unified(
        syn_traces_norm, syn_workload_ids, kept_names, norm_params)
    
    # Compute variance ratio
    vr_per_metric, vr_mean = compute_variance_ratio(real_traces_denorm, syn_traces_denorm)
    
    print(f"\n  Results:")
    print(f"    Mean VR: {vr_mean:.4f}")
    print(f"    Per-metric VR:")
    for i, mname in enumerate(kept_names):
        flag = " <--LOW" if vr_per_metric[i] < 0.5 else (
               " <--HIGH" if vr_per_metric[i] > 3.0 else "")
        print(f"      {mname:<22} {vr_per_metric[i]:.4f}{flag}")
    
    # Compare to S27
    ref_s27 = REFERENCE_VR_S27.get(workload_name, 0)
    ref_s21 = REFERENCE_VR_S21.get(workload_name, 0)
    delta_s27 = vr_mean - ref_s27
    delta_s21 = vr_mean - ref_s21
    
    print(f"\n  Comparison:")
    print(f"    S29:  {vr_mean:.4f}")
    print(f"    S27:  {ref_s27:.4f}  (delta: {delta_s27:+.4f})")
    print(f"    S21:  {ref_s21:.4f}  (delta: {delta_s21:+.4f})")
    
    if vr_mean > ref_s27:
        print(f"    → S29 BETTER than S27 by {delta_s27:.4f} ✓")
    else:
        print(f"    → S29 worse than S27 by {-delta_s27:.4f}")
    
    return {
        "workload": workload_name,
        "vr_mean": float(vr_mean),
        "vr_per_metric": {mname: float(vr_per_metric[i]) 
                          for i, mname in enumerate(kept_names)},
        "ref_s27": ref_s27,
        "ref_s21": ref_s21,
        "delta_s27": float(delta_s27),
        "delta_s21": float(delta_s21),
        "n_real": len(real_traces),
        "n_synthetic": len(syn_traces_norm),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate S29 per-workload")
    parser.add_argument("--model-dir", type=str,
                        default="models/phase4/timegan_s29/s29_unified_vr04_fm12_ae150")
    parser.add_argument("--n-gen", type=int, default=10,
                        help="Synthetic traces per replica count")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = Path(args.model_dir)
    
    print("="*70)
    print("S29 EVALUATION - Per-Workload Quality Assessment")
    print("="*70)
    print(f"Model: {model_dir}")
    print(f"Device: {device}")
    print(f"Synthetic samples per r: {args.n_gen}")
    
    # Load combined dataset
    data, norm_params = load_combined_data()
    raw_traces = data["traces"]
    replica_counts = data["replica_counts"]
    workload_ids = data["workload_ids"]
    train_idx = data["train_idx"]
    
    # Get unified metrics
    kept_idx, kept_names = get_unified_kept_metrics()
    n_metrics = len(kept_idx)
    raw_kept = raw_traces[:, :, kept_idx].astype(np.float32)
    
    print(f"\nDataset: {len(train_idx)} training pods")
    print(f"Metrics ({n_metrics}): {kept_names}")
    
    # Load S29 generator
    generator = GeneratorSegUnified(SEGMENT_LEN, n_metrics, GEN_CFG).to(device)
    checkpoint_path = model_dir / "generator.pt"
    generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
    generator.eval()
    print(f"\nLoaded: {checkpoint_path}")
    
    # Evaluate each workload
    all_results = []
    
    for wl_id, wl_name in enumerate(WORKLOADS):
        # Get real traces for this workload
        wl_mask = (workload_ids == wl_id) & np.isin(np.arange(len(raw_traces)), train_idx)
        wl_real_traces = raw_kept[wl_mask]
        wl_replica_counts = replica_counts[wl_mask]
        
        result = evaluate_workload(
            generator, wl_id, wl_name,
            wl_real_traces, wl_replica_counts,
            norm_params, kept_names, device, args.n_gen
        )
        all_results.append(result)
    
    # Summary
    print(f"\n{'='*70}")
    print("S29 SUMMARY - Comparison to S27")
    print(f"{'='*70}")
    print(f"{'Workload':<12} {'S29 VR':>8} {'S27 VR':>8} {'Delta':>8} {'Winner':>8}")
    print(f"{'-'*70}")
    
    s29_vrs = []
    wins = {"s29": 0, "s27": 0}
    
    for r in all_results:
        s29_vrs.append(r["vr_mean"])
        winner = "S29" if r["delta_s27"] > 0 else "S27"
        wins[winner.lower()] += 1
        
        print(f"{r['workload']:<12} "
              f"{r['vr_mean']:>8.4f} "
              f"{r['ref_s27']:>8.4f} "
              f"{r['delta_s27']:>+8.4f} "
              f"{winner:>8}")
    
    print(f"{'-'*70}")
    mean_s29 = np.mean(s29_vrs)
    mean_s27 = np.mean([r["ref_s27"] for r in all_results])
    print(f"{'MEAN':<12} {mean_s29:>8.4f} {mean_s27:>8.4f} {mean_s29 - mean_s27:>+8.4f}")
    
    print(f"\nWins: S29={wins['s29']}, S27={wins['s27']}")
    
    if mean_s29 > mean_s27:
        print(f"\n→ S29 SUPERIOR: Mean VR improved by {mean_s29 - mean_s27:.4f}")
    else:
        print(f"\n→ S27 SUPERIOR: Mean VR better by {mean_s27 - mean_s29:.4f}")
    
    # Save results
    output_dir = Path(f"outputs/phase4/timegan_s29/{model_dir.name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "stage": "s29_evaluation",
        "model_path": str(model_dir),
        "n_workloads": len(WORKLOADS),
        "mean_vr_s29": float(mean_s29),
        "mean_vr_s27": float(mean_s27),
        "delta_mean": float(mean_s29 - mean_s27),
        "wins_s29": wins["s29"],
        "wins_s27": wins["s27"],
        "per_workload_results": all_results,
    }
    
    output_path = output_dir / "s29_vs_s27_comparison.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved: {output_path}")
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()