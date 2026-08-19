"""
Adapted from validate_s36.py for H100 Tier 3 data. Changed:
the postprocess_s36_tier3 import (so synthetic generation
uses Tier 3 models/data, not just this file's own data_dir
defaults -- generate_postprocessed_traces() is called without
a data_dir override at every call site in this file, so it
falls back to whatever postprocess_s36(_tier3) hardcodes),
get_real_stats' and compute_wasserstein_distances' data_dir
defaults, and the --output-dir CLI default. Post-processing
parameters, thresholds, plot styling, and seeds are untouched
-- frozen per the S36 Tier 3 retrain plan.

S36 Validation Suite
====================
Validate post-processed S36 synthetic traces against real data,
and explore extrapolation behavior beyond the training range.

Three components:
  A. Physical constraint clamping (enforce metric bounds)
  B. Scaling curve plots:
     - Validated zone (r=1-10): real vs synthetic with error bars
     - Extrapolation zone (r=15,20,30,50): synthetic only, shaded background
  C. Distribution overlap metrics (Wasserstein distance for r=1-10)

Usage:
  python validate_s36.py
  python validate_s36.py --workloads bert gpt2
  python validate_s36.py --n-samples 10
  python validate_s36.py --no-extrapolation  # Skip r>10
"""

import numpy as np
import json
import argparse
from pathlib import Path
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Import the post-processing pipeline
import sys
sys.path.insert(0, str(Path(__file__).parent))
from postprocess_s36_tier3 import (
    generate_postprocessed_traces,
    load_real_data,
    load_normalization_params,
    denormalize_trace,
    WORKLOADS,
    ALL_METRICS,
    TRAINED_INDICES,
    TRAINED_NAMES,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Available replica counts in real data (validated zone)
REPLICA_COUNTS = [1, 2, 3, 5, 6, 8, 10]

# Extrapolation replica counts (no real data - exploratory)
EXTRAPOLATION_COUNTS = [15, 20, 30, 50]

# Whisper doesn't have r=8
REPLICA_COUNTS_PER_WORKLOAD = {
    'bert': [1, 2, 3, 5, 6, 8, 10],
    'gpt2': [1, 2, 3, 5, 6, 8, 10],
    'resnet152': [1, 2, 3, 5, 6, 8, 10],
    'whisper': [1, 2, 3, 5, 6, 10],
    'yolo': [1, 2, 3, 5, 6, 8, 10],
}

# Physical bounds for clamping (metric_index: (min, max))
METRIC_BOUNDS = {
    0: (0, None),       # pod_cpu_usage: >= 0 (cores)
    1: (0, None),       # pod_memory_bytes: >= 0
    2: (0, 1.0),        # pod_psi_cpu: [0, 1]
    3: (0, None),       # pod_latency_avg: >= 0
    4: (0, None),       # pod_throughput: >= 0
    5: (0, 100.0),      # gpu_utilization: [0, 100]
    6: (0, None),       # gpu_memory_used: >= 0
    7: (0, None),       # gpu_memory_total: >= 0
    8: (0, None),       # gpu_power_watts: >= 0
    9: (0, None),       # gpu_temperature: >= 0
}

# Metrics to include in scaling curves and distribution analysis
# (only the 7 trained metrics, in a readable order)
ANALYSIS_METRICS = [
    {'idx': 0, 'name': 'CPU Usage', 'unit': 'cores'},
    {'idx': 2, 'name': 'CPU Pressure (PSI)', 'unit': 'fraction'},
    {'idx': 3, 'name': 'Latency (Avg)', 'unit': 'seconds'},
    {'idx': 4, 'name': 'Throughput', 'unit': 'req/s'},
    {'idx': 1, 'name': 'Pod Memory', 'unit': 'MB', 'scale': 1e6},
    {'idx': 5, 'name': 'GPU Utilization', 'unit': '%'},
    {'idx': 8, 'name': 'GPU Power', 'unit': 'W'},
]


# ---------------------------------------------------------------------------
# A. Physical Constraint Clamping
# ---------------------------------------------------------------------------

def clamp_trace(trace):
    """
    Enforce physical bounds on a (T, 10) trace.
    Returns clamped trace and count of violations per metric.
    """
    clamped = trace.copy()
    violations = {}

    for midx, (lo, hi) in METRIC_BOUNDS.items():
        col = clamped[:, midx]
        n_violations = 0

        if lo is not None:
            below = col < lo
            n_violations += below.sum()
            col[below] = lo

        if hi is not None:
            above = col > hi
            n_violations += above.sum()
            col[above] = hi

        clamped[:, midx] = col
        if n_violations > 0:
            violations[ALL_METRICS[midx]] = int(n_violations)

    return clamped, violations


def clamp_batch(traces):
    """
    Clamp a batch of traces (N, T, 10).
    Returns clamped traces and aggregated violations.
    """
    all_violations = {}
    for i in range(traces.shape[0]):
        traces[i], v = clamp_trace(traces[i])
        for k, cnt in v.items():
            all_violations[k] = all_violations.get(k, 0) + cnt
    return traces, all_violations


# ---------------------------------------------------------------------------
# B. Scaling Curve Plots
# ---------------------------------------------------------------------------

def get_real_stats(data_dir='data/processed/tier3/unified'):
    """
    Compute per-workload per-replica mean and std for each metric from real data.
    Returns denormalized statistics.
    """
    data = load_real_data(data_dir)
    traces = data['traces']           # (N, 715, 10) normalized
    replica_counts = data['replica_counts']
    workload_ids = data['workload_ids']

    stats = {}
    for wid, wname in enumerate(WORKLOADS):
        norm_params = load_normalization_params(wname, data_dir)
        stats[wname] = {}

        for r in REPLICA_COUNTS_PER_WORKLOAD[wname]:
            mask = (workload_ids == wid) & (replica_counts == r)
            if not mask.any():
                continue

            matched = traces[mask]  # (n_pods, 715, 10)
            n_pods = matched.shape[0]

            # Denormalize each trained metric
            metric_means = {}
            metric_stds = {}

            for m in ANALYSIS_METRICS:
                midx = m['idx']
                scale = m.get('scale', 1.0)

                if midx in TRAINED_INDICES:
                    mname = ALL_METRICS[midx]
                    p = norm_params[mname]
                    vals_norm = matched[:, :, midx]
                    vals_real = vals_norm * (p['max'] - p['min']) + p['min']
                else:
                    vals_real = matched[:, :, midx]

                # Per-pod temporal mean, then mean/std across pods
                pod_means = vals_real.mean(axis=1) / scale
                metric_means[midx] = float(pod_means.mean())
                metric_stds[midx] = float(pod_means.std())

            stats[wname][r] = {'means': metric_means, 'stds': metric_stds,
                               'n_pods': n_pods}

    return stats


def get_synthetic_stats(workloads, replica_list, n_samples=5, device='cuda'):
    """
    Generate synthetic traces and compute per-workload per-replica stats.
    replica_list can include both validated (r=1-10) and extrapolation (r>10) counts.
    """
    stats = {}
    for wname in workloads:
        stats[wname] = {}
        for r in replica_list:
            print(f"    {wname.upper()} r={r}...", end='', flush=True)
            traces = generate_postprocessed_traces(
                wname, r, n_samples=n_samples, device=device)
            traces, _ = clamp_batch(traces)  # Apply clamping

            metric_means = {}
            metric_stds = {}

            for m in ANALYSIS_METRICS:
                midx = m['idx']
                scale = m.get('scale', 1.0)
                # Per-sample temporal mean
                sample_means = traces[:, :, midx].mean(axis=1) / scale
                metric_means[midx] = float(sample_means.mean())
                metric_stds[midx] = float(sample_means.std())

            stats[wname][r] = {'means': metric_means, 'stds': metric_stds,
                               'n_pods': n_samples}
            print(f" done")

    return stats


def plot_scaling_curves(real_stats, synth_stats, workloads, output_dir):
    """
    Plot scaling curves: metric mean vs replica count.
    Shows validated zone (r=1-10) with real+synthetic, and
    extrapolation zone (r>10) with synthetic only + shaded background.
    """
    for wname in workloads:
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

        validated_replicas = REPLICA_COUNTS_PER_WORKLOAD[wname]
        extrap_replicas = [r for r in sorted(synth_stats[wname].keys())
                           if r > 10]
        all_synth_replicas = sorted(synth_stats[wname].keys())

        for pi, m in enumerate(ANALYSIS_METRICS):
            row, col = pi // 3, pi % 3
            ax = fig.add_subplot(gs[row, col])
            midx = m['idx']

            # Shaded extrapolation zone
            if extrap_replicas:
                ax.axvspan(10.5, max(extrap_replicas) + 2,
                           alpha=0.08, color='red', label='Extrapolation zone')
                ax.axvline(10.5, color='red', ls=':', alpha=0.4, lw=1)

            # Real data points (validated zone only)
            real_means = [real_stats[wname][r]['means'][midx]
                          for r in validated_replicas]
            real_stds = [real_stats[wname][r]['stds'][midx]
                         for r in validated_replicas]
            ax.errorbar(validated_replicas, real_means, yerr=real_stds,
                        fmt='bo-', lw=1.5, capsize=3, markersize=5,
                        label='Real', alpha=0.8, zorder=3)

            # Synthetic data points (all replicas including extrapolation)
            synth_means = [synth_stats[wname][r]['means'][midx]
                           for r in all_synth_replicas]
            synth_stds = [synth_stats[wname][r]['stds'][midx]
                          for r in all_synth_replicas]
            ax.errorbar(all_synth_replicas, synth_means, yerr=synth_stds,
                        fmt='gs--', lw=1.5, capsize=3, markersize=5,
                        label='Synthetic', alpha=0.8, zorder=2)

            ax.set_title(m['name'], fontsize=11, fontweight='bold')
            ax.set_xlabel('Replica Count', fontsize=9)
            ax.set_ylabel(m['unit'], fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)

            if pi == 0:
                ax.legend(fontsize=7, loc='best')

        # Hide unused subplots
        for i in [7, 8]:
            ax = fig.add_subplot(gs[i // 3, i % 3])
            ax.axis('off')

        fig.suptitle(
            f"Scaling Curves: {wname.upper()}\n"
            f"Validated (r=1-10) + Extrapolation (r>10, shaded)",
            fontsize=13, fontweight='bold', y=0.995)

        out_path = output_dir / f"{wname}_scaling_curves.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# C. Distribution Overlap (Wasserstein Distance)
# ---------------------------------------------------------------------------

def compute_wasserstein_distances(workloads, n_samples=10, device='cuda',
                                  data_dir='data/processed/tier3/unified'):
    """
    Compute per-metric Wasserstein distance between real and synthetic
    temporal distributions at each replica count.

    For each (workload, replica, metric):
      - Flatten all real pod traces into one distribution
      - Flatten all synthetic traces into one distribution
      - Compute Wasserstein-1 distance

    Returns nested dict: {workload: {replica: {metric_name: distance}}}
    """
    data = load_real_data(data_dir)
    traces = data['traces']
    replica_counts = data['replica_counts']
    workload_ids = data['workload_ids']

    results = {}
    for wname in workloads:
        wid = WORKLOADS.index(wname)
        norm_params = load_normalization_params(wname, data_dir)
        results[wname] = {}

        for r in REPLICA_COUNTS_PER_WORKLOAD[wname]:
            print(f"    {wname.upper()} r={r}...", end='', flush=True)

            # Get real traces (denormalized)
            mask = (workload_ids == wid) & (replica_counts == r)
            real_norm = traces[mask]  # (n_pods, 715, 10)

            # Generate synthetic
            synth = generate_postprocessed_traces(
                wname, r, n_samples=n_samples, device=device)
            synth, _ = clamp_batch(synth)

            distances = {}
            for m in ANALYSIS_METRICS:
                midx = m['idx']
                scale = m.get('scale', 1.0)

                # Denormalize real
                if midx in TRAINED_INDICES:
                    mname = ALL_METRICS[midx]
                    p = norm_params[mname]
                    real_vals = real_norm[:, :, midx] * (p['max'] - p['min']) + p['min']
                else:
                    real_vals = real_norm[:, :, midx]

                real_flat = (real_vals / scale).flatten()
                synth_flat = (synth[:, :, midx] / scale).flatten()

                wd = wasserstein_distance(real_flat, synth_flat)
                distances[m['name']] = float(wd)

            results[wname][r] = distances
            print(f" done")

    return results


def plot_wasserstein_heatmap(wd_results, workloads, output_dir):
    """
    Create heatmap of Wasserstein distances per workload.
    Rows = replica counts, Columns = metrics.
    """
    metric_names = [m['name'] for m in ANALYSIS_METRICS]

    for wname in workloads:
        replicas = sorted(wd_results[wname].keys())
        n_replicas = len(replicas)
        n_metrics = len(metric_names)

        matrix = np.zeros((n_replicas, n_metrics))
        for ri, r in enumerate(replicas):
            for mi, mname in enumerate(metric_names):
                matrix[ri, mi] = wd_results[wname][r].get(mname, 0)

        # Normalize each column to [0, 1] for comparable coloring
        col_max = matrix.max(axis=0, keepdims=True)
        col_max[col_max == 0] = 1
        matrix_norm = matrix / col_max

        fig, ax = plt.subplots(figsize=(12, 5))
        im = ax.imshow(matrix_norm, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

        ax.set_xticks(range(n_metrics))
        ax.set_xticklabels(metric_names, rotation=30, ha='right', fontsize=9)
        ax.set_yticks(range(n_replicas))
        ax.set_yticklabels([f"r={r}" for r in replicas], fontsize=10)

        # Annotate with actual values
        for ri in range(n_replicas):
            for mi in range(n_metrics):
                val = matrix[ri, mi]
                color = 'white' if matrix_norm[ri, mi] > 0.6 else 'black'
                ax.text(mi, ri, f"{val:.3f}", ha='center', va='center',
                        fontsize=7, color=color)

        ax.set_title(f"Wasserstein Distance: {wname.upper()}\n(lower = more similar to real)",
                     fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Normalized distance', shrink=0.8)

        out_path = output_dir / f"{wname}_wasserstein_heatmap.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_path}")


def plot_combined_scaling(real_stats, synth_stats, workloads, output_dir):
    """
    Single combined figure: all workloads on one plot per metric.
    Shows validated + extrapolation zones.
    """
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    colors = {'bert': '#1f77b4', 'gpt2': '#ff7f0e', 'resnet152': '#2ca02c',
              'whisper': '#d62728', 'yolo': '#9467bd'}

    # Find max extrapolation replica across all workloads
    max_extrap = max(
        max(synth_stats[w].keys()) for w in workloads
    )

    for pi, m in enumerate(ANALYSIS_METRICS):
        row, col = pi // 3, pi % 3
        ax = fig.add_subplot(gs[row, col])
        midx = m['idx']

        # Extrapolation zone shading
        if max_extrap > 10:
            ax.axvspan(10.5, max_extrap + 2, alpha=0.08, color='red')
            ax.axvline(10.5, color='red', ls=':', alpha=0.4, lw=1)

        for wname in workloads:
            c = colors[wname]

            # Real (validated only)
            validated = REPLICA_COUNTS_PER_WORKLOAD[wname]
            real_means = [real_stats[wname][r]['means'][midx] for r in validated]
            ax.plot(validated, real_means, 'o-', color=c, lw=1.5,
                    markersize=4, alpha=0.8, label=f'{wname} (real)')

            # Synthetic (all including extrapolation)
            all_r = sorted(synth_stats[wname].keys())
            synth_means = [synth_stats[wname][r]['means'][midx] for r in all_r]
            ax.plot(all_r, synth_means, 's--', color=c, lw=1.0,
                    markersize=3, alpha=0.5, label=f'{wname} (synth)')

        ax.set_title(m['name'], fontsize=11, fontweight='bold')
        ax.set_xlabel('Replica Count', fontsize=9)
        ax.set_ylabel(m['unit'], fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

        if pi == 0:
            ax.legend(fontsize=5, loc='best', ncol=2)

    for i in [7, 8]:
        ax = fig.add_subplot(gs[i // 3, i % 3])
        ax.axis('off')

    fig.suptitle(
        "Scaling Curves: All Workloads\n"
        "Solid=Real (r=1-10), Dashed=Synthetic, Red zone=Extrapolation",
        fontsize=14, fontweight='bold', y=0.995)

    out_path = output_dir / "all_workloads_scaling_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Summary Report
# ---------------------------------------------------------------------------

def generate_report(real_stats, synth_stats, wd_results, clamp_violations,
                    workloads, output_dir):
    """
    Generate JSON summary report with all validation results.
    Includes both validated (r=1-10) and extrapolation (r>10) stats.
    """
    report = {
        'model': 'S36 (spectral normalization)',
        'post_processing': 'cosine blend (w=20) + adaptive filter + memory reconstruction',
        'workloads': {},
    }

    for wname in workloads:
        w_report = {
            'validated_replicas': {},
            'extrapolation_replicas': {},
        }

        all_wd = []

        for r in sorted(synth_stats[wname].keys()):
            r_report = {}
            is_validated = r in real_stats.get(wname, {})

            # Scaling curve comparison (only for validated replicas)
            if is_validated:
                for m in ANALYSIS_METRICS:
                    midx = m['idx']
                    real_val = real_stats[wname][r]['means'][midx]
                    synth_val = synth_stats[wname][r]['means'][midx]
                    if real_val != 0:
                        pct_error = abs(synth_val - real_val) / abs(real_val) * 100
                    else:
                        pct_error = 0
                    r_report[m['name']] = {
                        'real_mean': round(real_val, 6),
                        'synth_mean': round(synth_val, 6),
                        'pct_error': round(pct_error, 2),
                    }
            else:
                # Extrapolation: synth only
                for m in ANALYSIS_METRICS:
                    midx = m['idx']
                    synth_val = synth_stats[wname][r]['means'][midx]
                    r_report[m['name']] = {
                        'synth_mean': round(synth_val, 6),
                        'note': 'extrapolation (no real data)',
                    }

            # Wasserstein distances (validated only)
            if r in wd_results.get(wname, {}):
                r_report['wasserstein'] = wd_results[wname][r]
                for mname, wd in wd_results[wname][r].items():
                    all_wd.append(wd)

            if is_validated:
                w_report['validated_replicas'][r] = r_report
            else:
                w_report['extrapolation_replicas'][r] = r_report

        # Mean Wasserstein across validated replicas
        if all_wd:
            w_report['mean_wasserstein_all'] = round(np.mean(all_wd), 6)

        # Clamping violations
        if wname in clamp_violations:
            w_report['clamp_violations'] = clamp_violations[wname]

        report['workloads'][wname] = w_report

    report_path = output_dir / 'validation_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved: {report_path}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='S36 Validation Suite')
    parser.add_argument('--workloads', nargs='+', default=WORKLOADS)
    parser.add_argument('--n-samples', type=int, default=5,
                        help='Synthetic samples per replica count')
    parser.add_argument('--output-dir', type=str,
                        default='outputs/phase4/validation/s36_tier3')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--skip-wasserstein', action='store_true',
                        help='Skip Wasserstein computation (slow)')
    parser.add_argument('--no-extrapolation', action='store_true',
                        help='Skip r>10 extrapolation')
    parser.add_argument('--extrapolation-counts', nargs='+', type=int,
                        default=[15, 20, 30, 50],
                        help='Replica counts for extrapolation (default: 15 20 30 50)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build full replica list
    extrap_counts = [] if args.no_extrapolation else args.extrapolation_counts

    print("=" * 80)
    print("S36 VALIDATION SUITE")
    print("=" * 80)
    print(f"Workloads:      {args.workloads}")
    print(f"Samples:        {args.n_samples} per replica count")
    print(f"Validated:      r={REPLICA_COUNTS}")
    print(f"Extrapolation:  r={extrap_counts}" if extrap_counts else "Extrapolation:  DISABLED")
    print(f"Output:         {output_dir}")
    print("=" * 80)

    # --- A. Compute real data statistics ---
    print("\n[A] Computing real data statistics...")
    real_stats = get_real_stats()
    print("  Done.")

    # --- B. Generate synthetic for validated + extrapolation replicas ---
    print("\n[B] Generating synthetic traces (validated zone r=1-10)...")
    # Build per-workload full replica list
    full_replica_lists = {}
    for wname in args.workloads:
        validated = REPLICA_COUNTS_PER_WORKLOAD[wname]
        full_replica_lists[wname] = sorted(set(validated + extrap_counts))

    synth_stats = {}
    for wname in args.workloads:
        synth_stats.update(
            get_synthetic_stats([wname], full_replica_lists[wname],
                                args.n_samples, args.device))

    # Track clamping violations across all replica counts
    print("\n[B.1] Checking clamping violations...")
    clamp_violations = {}
    for wname in args.workloads:
        wv = {}
        for r in full_replica_lists[wname]:
            traces = generate_postprocessed_traces(
                wname, r, n_samples=args.n_samples, device=args.device)
            _, v = clamp_batch(traces)
            if v:
                wv[r] = v
        if wv:
            clamp_violations[wname] = wv
            print(f"  {wname}: {wv}")
        else:
            print(f"  {wname}: no violations")

    # --- B.2 Scaling curve plots ---
    print("\n[B.2] Generating scaling curve plots...")
    plot_scaling_curves(real_stats, synth_stats, args.workloads, output_dir)
    plot_combined_scaling(real_stats, synth_stats, args.workloads, output_dir)

    # --- C. Wasserstein distances (validated zone only) ---
    wd_results = {}
    if not args.skip_wasserstein:
        print("\n[C] Computing Wasserstein distances (r=1-10 only)...")
        wd_results = compute_wasserstein_distances(
            args.workloads, n_samples=args.n_samples, device=args.device)
        plot_wasserstein_heatmap(wd_results, args.workloads, output_dir)
    else:
        print("\n[C] Skipping Wasserstein distances (--skip-wasserstein)")

    # --- D. Summary report ---
    print("\n[D] Generating validation report...")
    report = generate_report(real_stats, synth_stats, wd_results,
                             clamp_violations, args.workloads, output_dir)

    # Print summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    for wname in args.workloads:
        wr = report['workloads'][wname]
        mean_wd = wr.get('mean_wasserstein_all', 'N/A')
        n_violations = sum(
            sum(v.values()) for v in wr.get('clamp_violations', {}).values()
        )
        print(f"  {wname.upper():12s}  Mean WD: {mean_wd}  "
              f"Clamp violations: {n_violations}")

    if extrap_counts:
        print(f"\n  Extrapolation points generated: {extrap_counts}")
        print(f"  Check scaling curve plots to assess extrapolation quality.")
        print(f"  Look for: smooth continuation, no collapse, no explosion.")

    print("\n" + "=" * 80)
    print(f"COMPLETE! Results in: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()