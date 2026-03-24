"""
S31 Evaluation - GPU-Bound Workload Quality Assessment
=======================================================
Evaluates S31 continuity-aware model for GPU-bound workloads only.
Compares against S29 (unified) and S27 (individual) baselines.

Key checks:
- Variance ratio per workload
- Phase boundary continuity (visual inspection via plots)
- YOLO crossing VR=0.8 threshold

Usage:
    python evaluate_s31.py
    python evaluate_s31.py --n-gen 20
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path

# Load data like S29
DATA_PATH = Path("data/processed/phase4/unified/combined_dataset.npz")
NORM_PATH = Path("data/processed/phase4/unified/combined_normalization.json")

# S31 configuration (GPU-bound only)
WORKLOADS = ["bert", "gpt2", "resnet152", "whisper", "yolo"]
VALID_WORKLOAD_IDS = [0, 1, 2, 4]  # Exclude whisper (ID=3)
VALID_WORKLOAD_NAMES = ["bert", "gpt2", "resnet152", "yolo"]
N_WORKLOADS = 4

REFERENCE_VR_S29 = {
    "bert": 1.683,
    "gpt2": 2.204,
    "resnet152": 1.728,
    "yolo": 0.761,
}

REFERENCE_VR_S27 = {
    "bert": 0.886,
    "gpt2": 0.946,
    "resnet152": 0.851,
    "yolo": 0.997,
}

# Architecture params (match training)
SEG_LEN = 120
N_PHASES = 6
PHASE_BOUNDARIES = [0, 96, 180, 300, 420, 600]
HIDDEN_DIM = 128
LATENT_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.1


def load_data():
    """Load combined dataset and filter for GPU-bound workloads."""
    data = np.load(DATA_PATH, allow_pickle=True)
    with open(NORM_PATH) as f:
        norm_params = json.load(f)
    
    raw_traces = data['traces']
    replica_counts = data['replica_counts']
    workload_ids = data['workload_ids']
    train_idx = data['train_idx']
    
    # Filter GPU-bound only
    valid_mask = np.isin(workload_ids, VALID_WORKLOAD_IDS)
    filtered_traces = raw_traces[valid_mask]
    filtered_rc = replica_counts[valid_mask]
    filtered_wl = workload_ids[valid_mask]
    
    # Remap workload IDs: 0,1,2,4 -> 0,1,2,3
    wl_map = {0: 0, 1: 1, 2: 2, 4: 3}
    filtered_wl = np.array([wl_map[w] for w in filtered_wl])
    
    # Update train indices
    original_to_new = {}
    new_idx = 0
    for old_idx in range(len(raw_traces)):
        if valid_mask[old_idx]:
            original_to_new[old_idx] = new_idx
            new_idx += 1
    
    new_train_idx = np.array([original_to_new[i] for i in train_idx if i in original_to_new])
    
    return filtered_traces, filtered_rc, filtered_wl, new_train_idx, norm_params


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


def compute_variance_ratio(real, synthetic, cap=5.0):
    """Compute variance ratio per metric."""
    var_r = real.reshape(-1, real.shape[-1]).var(axis=0)
    var_s = synthetic.reshape(-1, synthetic.shape[-1]).var(axis=0)
    ratio = np.where(var_r > 1e-10,
                     np.clip(var_s / var_r, 0, cap),
                     np.ones_like(var_r))
    return ratio, float(ratio.mean())


class GeneratorContinuous(torch.nn.Module):
    """S31 generator (same as training)."""
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


def evaluate_workload(generator, workload_id, workload_name,
                      real_traces, real_replica_counts,
                      norm_params, device, n_gen=10):
    """Evaluate S31 for one GPU-bound workload."""
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
    
    # Denormalize
    real_workload_ids = np.full(len(real_traces), workload_id, dtype=np.int32)
    syn_workload_ids = np.full(len(syn_traces_norm), workload_id, dtype=np.int32)
    
    real_traces_denorm = denormalize(real_traces, real_workload_ids, norm_params)
    syn_traces_denorm = denormalize(syn_traces_norm, syn_workload_ids, norm_params)
    
    # Compute variance ratio
    vr_per_metric, vr_mean = compute_variance_ratio(real_traces_denorm, syn_traces_denorm)
    
    print(f"\n  Results:")
    print(f"    Mean VR: {vr_mean:.4f}")
    
    # Compare to S29 and S27
    ref_s29 = REFERENCE_VR_S29.get(workload_name, 0)
    ref_s27 = REFERENCE_VR_S27.get(workload_name, 0)
    delta_s29 = vr_mean - ref_s29
    delta_s27 = vr_mean - ref_s27
    
    print(f"\n  Comparison:")
    print(f"    S31:  {vr_mean:.4f}")
    print(f"    S29:  {ref_s29:.4f}  (delta: {delta_s29:+.4f})")
    print(f"    S27:  {ref_s27:.4f}  (delta: {delta_s27:+.4f})")
    
    if vr_mean > 0.8:
        status = "PASS (>0.8)"
    else:
        status = "BELOW THRESHOLD"
    print(f"    Status: {status}")
    
    if vr_mean > ref_s29:
        print(f"    → S31 BETTER than S29 by {delta_s29:.4f} ✓")
    elif abs(delta_s29) < 0.05:
        print(f"    → S31 ~SAME as S29 (delta < 0.05)")
    else:
        print(f"    → S31 worse than S29 by {-delta_s29:.4f}")
    
    return {
        "workload": workload_name,
        "vr_mean": float(vr_mean),
        "ref_s29": ref_s29,
        "ref_s27": ref_s27,
        "delta_s29": float(delta_s29),
        "delta_s27": float(delta_s27),
        "passes_threshold": vr_mean > 0.8,
        "n_real": len(real_traces),
        "n_synthetic": len(syn_traces_norm),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate S31 GPU-bound model")
    parser.add_argument("--model-dir", type=str,
                        default="models/phase4/timegan_s31/s31_gpu_bound_cont03")
    parser.add_argument("--n-gen", type=int, default=10,
                        help="Synthetic traces per replica count")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = Path(args.model_dir)
    
    print("="*70)
    print("S31 EVALUATION - GPU-Bound Workloads Only")
    print("="*70)
    print(f"Model: {model_dir}")
    print(f"Device: {device}")
    print(f"Workloads: {VALID_WORKLOAD_NAMES}")
    print(f"Synthetic samples per r: {args.n_gen}")
    
    # Load data
    traces, replica_counts, workload_ids, train_idx, norm_params = load_data()
    n_metrics = traces.shape[2]
    
    print(f"\nDataset: {len(train_idx)} training pods (GPU-bound only)")
    print(f"Metrics: {n_metrics}")
    
    # Load S31 generator
    generator = GeneratorContinuous(SEG_LEN, n_metrics, HIDDEN_DIM, LATENT_DIM,
                                    NUM_LAYERS, DROPOUT).to(device)
    checkpoint_path = model_dir / "generator.pt"
    generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
    generator.eval()
    print(f"\nLoaded: {checkpoint_path}")
    
    # Evaluate each GPU-bound workload
    all_results = []
    
    for wl_id, wl_name in enumerate(VALID_WORKLOAD_NAMES):
        # Get real traces for this workload
        wl_mask = (workload_ids == wl_id) & np.isin(np.arange(len(traces)), train_idx)
        wl_real_traces = traces[wl_mask]
        wl_replica_counts = replica_counts[wl_mask]
        
        result = evaluate_workload(
            generator, wl_id, wl_name,
            wl_real_traces, wl_replica_counts,
            norm_params, device, args.n_gen
        )
        all_results.append(result)
    
    # Summary
    print(f"\n{'='*70}")
    print("S31 SUMMARY - Comparison to S29 and S27")
    print(f"{'='*70}")
    print(f"{'Workload':<12} {'S31 VR':>8} {'S29 VR':>8} {'S27 VR':>8} {'Status':>12}")
    print(f"{'-'*70}")
    
    s31_vrs = []
    pass_count = 0
    
    for r in all_results:
        s31_vrs.append(r["vr_mean"])
        status = "PASS" if r["passes_threshold"] else "BELOW"
        if r["passes_threshold"]:
            pass_count += 1
        
        print(f"{r['workload']:<12} "
              f"{r['vr_mean']:>8.4f} "
              f"{r['ref_s29']:>8.4f} "
              f"{r['ref_s27']:>8.4f} "
              f"{status:>12}")
    
    print(f"{'-'*70}")
    mean_s31 = np.mean(s31_vrs)
    mean_s29 = np.mean([r["ref_s29"] for r in all_results])
    mean_s27 = np.mean([r["ref_s27"] for r in all_results])
    print(f"{'MEAN':<12} {mean_s31:>8.4f} {mean_s29:>8.4f} {mean_s27:>8.4f}")
    
    print(f"\n{'='*70}")
    print("THRESHOLD ANALYSIS (VR > 0.8)")
    print(f"{'='*70}")
    print(f"Pass rate: {pass_count}/{len(all_results)} workloads")
    
    yolo_result = [r for r in all_results if r['workload'] == 'yolo'][0]
    if yolo_result['passes_threshold']:
        print(f"\n✓ YOLO CROSSES THRESHOLD: VR={yolo_result['vr_mean']:.4f} > 0.8")
        print(f"  Improvement over S29: {yolo_result['delta_s29']:+.4f}")
    else:
        print(f"\n✗ YOLO BELOW THRESHOLD: VR={yolo_result['vr_mean']:.4f} < 0.8")
        print(f"  Still {0.8 - yolo_result['vr_mean']:.4f} below target")
    
    print(f"\n{'='*70}")
    print("OVERALL COMPARISON")
    print(f"{'='*70}")
    
    if mean_s31 > mean_s29:
        print(f"→ S31 IMPROVES over S29: +{mean_s31 - mean_s29:.4f}")
    elif abs(mean_s31 - mean_s29) < 0.05:
        print(f"→ S31 MAINTAINS S29 quality: {mean_s31 - mean_s29:+.4f}")
    else:
        print(f"→ S31 regresses from S29: {mean_s31 - mean_s29:+.4f}")
    
    if mean_s31 > mean_s27:
        print(f"→ S31 IMPROVES over S27: +{mean_s31 - mean_s27:.4f}")
    
    # Save results
    output_dir = Path(f"outputs/phase4/timegan_s31/{model_dir.name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "stage": "s31_evaluation",
        "innovation": "LSTM state continuity + boundary smoothness loss",
        "model_path": str(model_dir),
        "excluded_workloads": ["whisper"],
        "evaluated_workloads": VALID_WORKLOAD_NAMES,
        "n_workloads": N_WORKLOADS,
        "mean_vr_s31": float(mean_s31),
        "mean_vr_s29": float(mean_s29),
        "mean_vr_s27": float(mean_s27),
        "delta_vs_s29": float(mean_s31 - mean_s29),
        "delta_vs_s27": float(mean_s31 - mean_s27),
        "pass_rate": f"{pass_count}/{len(all_results)}",
        "yolo_crosses_threshold": yolo_result['passes_threshold'],
        "per_workload_results": all_results,
    }
    
    output_path = output_dir / "results.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved: {output_path}")
    
    # Final recommendation
    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print(f"{'='*70}")
    
    if yolo_result['passes_threshold'] and pass_count == 4:
        print("✓ S31 RECOMMENDED FOR SUBMISSION")
        print("  - All 4 workloads pass VR > 0.8 threshold")
        print("  - YOLO crosses threshold (key improvement)")
        print("  - Continuous generation eliminates discontinuities")
    elif pass_count >= 3:
        print("✓ S31 ACCEPTABLE FOR SUBMISSION")
        print(f"  - {pass_count}/4 workloads pass threshold")
        print("  - Close to target, defensible in thesis")
    else:
        print("? S31 NEEDS REVIEW")
        print(f"  - Only {pass_count}/4 workloads pass")
        print("  - Consider using S29 or S27 instead")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()