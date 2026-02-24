#!/usr/bin/env python3
"""
Phase 4 Preprocessing - Three Transformation Methods
Master Thesis - Generative AI Workload Modeling
Fachhochschule Dortmund

Builds on the Phase 1 v3 preprocessor and adds three transformation
strategies designed to fix the flat-trace generation problem observed
in Phase 2 (TimeVAE/TimeGAN) and Phase 4b (LSTM baseline).

The core problem:
    - Raw pod traces are mostly flat (stable resource usage by design)
    - Generative models collapse to outputting the mean (safe, flat output)
    - Variance ratios end up near 0 for most metrics

Three methods implemented:
    METHOD 1 - First-order differences
        Feed change-per-timestep instead of raw values.
        Flat regions become zeros (no signal to confuse the model).
        Transitions become the actual training signal.
        Reconstruction: cumsum(output) + initial_value

    METHOD 2 - Per-trace zero-mean normalization
        Subtract each trace's own mean before training.
        Model learns deviation patterns, not absolute levels.
        The per-trace mean is stored as an extra conditioning variable
        alongside replica_count, so it is not lost.

    METHOD 3 - Short sliding windows (60 timesteps = 5 minutes)
        Replaces full 715-timestep traces with overlapping windows.
        Turns 275 pod traces into ~3000+ training samples.
        Local dynamics become prominent; flatness becomes minority signal.
        Window stride is configurable.

    METHOD 4 - Combined: differences + short windows
        Method 1 and 3 together. This is the recommended starting point
        for TimeGAN. More training samples AND dynamics as the signal.

Package-level awareness:
    Each saved file also stores experiment_id per sample so that at
    generation time you can group generated traces by experiment and
    verify that r=70 produces 70 internally-consistent traces.

Output structure:
    data/processed/phase4/{method}/{workload}_traces.npz
    data/processed/phase4/{method}/{workload}_normalization.json
    data/processed/phase4/{method}/dataset_summary.json

Usage:
    # All methods, all workloads
    python scripts/phase4/preprocess_phase4.py

    # Single method
    python scripts/phase4/preprocess_phase4.py --methods diff windows

    # Single workload
    python scripts/phase4/preprocess_phase4.py --workloads bert gpt2

    # Custom window config
    python scripts/phase4/preprocess_phase4.py --window-size 60 --window-stride 30
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------
# Paths and constants  (same as original preprocessor)
# ------------------------------------------------------------------

RAW_DATA_DIR   = Path("data/raw/phase1_v3")
OUTPUT_BASE    = Path("data/processed/phase4")

WORKLOAD_CONFIGS = {
    "bert":      {"replicas": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
    "gpt2":      {"replicas": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
    "resnet152": {"replicas": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
    "whisper":   {"replicas": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
    "yolo":      {"replicas": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
}

POD_METRICS        = ["pod_cpu_usage", "pod_memory_bytes", "pod_psi_cpu",
                       "pod_latency_avg", "pod_throughput"]
SYSTEM_GPU_METRICS = ["gpu_utilization", "gpu_memory_used", "gpu_memory_total",
                       "gpu_power_watts", "gpu_temperature"]
ALL_METRICS        = POD_METRICS + SYSTEM_GPU_METRICS  # 10 total

METHODS = ["raw", "diff", "zeromean", "windows", "diff_windows"]


# ------------------------------------------------------------------
# Raw data loading  (identical to original preprocessor)
# ------------------------------------------------------------------

def load_pod_metric_csv(csv_path):
    df = pd.read_csv(csv_path)
    pivoted = df.pivot(index="timestamp", columns="pod", values="value").reset_index()
    pod_cols = [col for col in pivoted.columns if col != "timestamp"]
    pivoted.columns = ["timestamp"] + [f"pod_{i}" for i in range(len(pod_cols))]
    return pivoted


def load_system_metric_csv(csv_path):
    return pd.read_csv(csv_path)[["timestamp", "value"]].copy()


def load_experiment_data(workload, replica_count):
    exp_dir   = RAW_DATA_DIR / f"{workload}_r{replica_count}"
    csv_files = list(exp_dir.glob("*.csv"))
    timestamp = "_".join(csv_files[0].stem.split("_")[-2:])
    data = {}
    for metric in POD_METRICS:
        csv_path = exp_dir / f"{workload}_r{replica_count}_{metric}_{timestamp}.csv"
        if csv_path.exists():
            data[metric] = load_pod_metric_csv(csv_path)
    for metric in SYSTEM_GPU_METRICS:
        csv_path = exp_dir / f"{workload}_r{replica_count}_{metric}_{timestamp}.csv"
        if csv_path.exists():
            data[metric] = load_system_metric_csv(csv_path)
    return data


def align_timestamps(data):
    all_ts    = [set(df["timestamp"]) for df in data.values()]
    common_ts = sorted(list(set.intersection(*all_ts)))
    print(f"    Aligning to {len(common_ts)} timestamps")
    return {
        m: df[df["timestamp"].isin(common_ts)].sort_values("timestamp").reset_index(drop=True)
        for m, df in data.items()
    }


def create_pod_traces(data, replica_count, workload):
    """
    Returns raw (unnormalized) traces: (num_pods, num_timesteps, 10)
    and metadata list.
    """
    first_df  = data[POD_METRICS[0]]
    pod_cols  = [c for c in first_df.columns if c.startswith("pod_")]
    num_pods  = len(pod_cols)
    num_ts    = len(first_df)

    if num_pods != replica_count:
        print(f"    Note: expected {replica_count} pods, found {num_pods} (partial experiment kept)")

    traces   = np.zeros((num_pods, num_ts, len(ALL_METRICS)))
    metadata = [
        {
            "workload":       workload,
            "replica_count":  replica_count,
            "pod_index":      i,
            "num_pods":       num_pods,
            "experiment_id":  f"{workload}_r{replica_count}",
        }
        for i in range(num_pods)
    ]

    for metric_idx, metric in enumerate(POD_METRICS):
        if metric in data:
            df = data[metric]
            for i in range(num_pods):
                pod_col = f"pod_{i}"
                if pod_col in df.columns:
                    traces[i, :, metric_idx] = df[pod_col].values

    for metric_idx, metric in enumerate(SYSTEM_GPU_METRICS, start=len(POD_METRICS)):
        if metric in data:
            # Divide system-level metric equally among pods (same as original)
            per_pod = data[metric]["value"].values / replica_count
            for i in range(num_pods):
                traces[i, :, metric_idx] = per_pod

    return traces, metadata


def load_all_raw_traces(workload):
    """
    Loads all experiments for a workload and returns:
        raw_traces  : (N_pods, 715, 10)  unnormalized
        replica_counts: (N_pods,)
        metadata    : list of N_pods dicts
    """
    all_traces, all_r, all_meta = [], [], []

    for r in WORKLOAD_CONFIGS[workload]["replicas"]:
        print(f"  Loading {workload} r={r} ...")
        try:
            data   = align_timestamps(load_experiment_data(workload, r))
            traces, metadata = create_pod_traces(data, r, workload)
            all_traces.append(traces)
            all_r.extend([r] * len(traces))
            all_meta.extend(metadata)
            print(f"    {len(traces)} pods loaded")
        except Exception as e:
            print(f"    ERROR loading {workload} r={r}: {e}")

    raw_traces     = np.nan_to_num(np.concatenate(all_traces, axis=0), nan=0.0)
    replica_counts = np.array(all_r)
    print(f"  Total: {len(raw_traces)} pod traces, shape {raw_traces.shape}")
    return raw_traces, replica_counts, all_meta


# ------------------------------------------------------------------
# MinMax normalization  (same logic as original, per-workload)
# ------------------------------------------------------------------

def minmax_normalize(traces):
    """
    Normalizes to [0, 1] per metric across all pods/timesteps.
    Returns normalized traces and norm_params dict.
    """
    normalized  = traces.copy()
    norm_params = {"method": "minmax", "metric_names": ALL_METRICS, "params": {}}

    for i, metric in enumerate(ALL_METRICS):
        min_val = traces[:, :, i].min()
        max_val = traces[:, :, i].max()
        if max_val - min_val > 1e-8:
            normalized[:, :, i] = (traces[:, :, i] - min_val) / (max_val - min_val)
        else:
            normalized[:, :, i] = 0.0
        norm_params["params"][metric] = {"min": float(min_val), "max": float(max_val)}

    return normalized, norm_params


# ------------------------------------------------------------------
# Transformation methods
# ------------------------------------------------------------------

def apply_differencing(traces):
    """
    METHOD 1: First-order differences.

    Input : (N, T, M)  normalized traces
    Output: (N, T-1, M) difference traces  +  (N, M) initial values

    The initial values (first timestep of each trace) are stored
    separately so the original trace can be reconstructed exactly:
        reconstructed = cumsum(diff_trace, axis=1) + initial_values[:, np.newaxis, :]

    Also stores per-trace mean of the ORIGINAL trace as extra context.
    """
    diffs          = np.diff(traces, axis=1)                    # (N, T-1, M)
    initial_values = traces[:, 0, :]                            # (N, M)
    trace_means    = traces.mean(axis=1)                        # (N, M)  -- extra context
    return diffs, initial_values, trace_means


def apply_zeromean(traces):
    """
    METHOD 2: Per-trace zero-mean normalization.

    Input : (N, T, M)  normalized traces
    Output: (N, T, M)  zero-meaned traces  +  (N, M) per-trace means

    The per-trace mean encodes the absolute contention level and is
    stored as an additional conditioning variable for the model.
    """
    trace_means    = traces.mean(axis=1, keepdims=True)         # (N, 1, M)
    zeromean       = traces - trace_means                        # (N, T, M)
    return zeromean, trace_means.squeeze(1)                     # (N, T, M), (N, M)


def apply_windows(traces, replica_counts, metadata, window_size, stride):
    """
    METHOD 3: Sliding windows.

    Input : (N, T, M) traces
    Output: (N_windows, window_size, M) windows
            (N_windows,) replica_counts per window
            list of window metadata

    Each window records which original pod trace it came from
    (source_pod_idx) and its position in the trace (window_start).
    This is important for package-level generation: windows from the
    same experiment_id belong to the same contention environment.
    """
    N, T, M         = traces.shape
    win_traces      = []
    win_r           = []
    win_meta        = []

    for pod_idx in range(N):
        pod_trace = traces[pod_idx]         # (T, M)
        r_val     = replica_counts[pod_idx]
        pod_meta  = metadata[pod_idx]

        start = 0
        while start + window_size <= T:
            win_traces.append(pod_trace[start: start + window_size])
            win_r.append(r_val)
            win_meta.append({
                **pod_meta,
                "window_start":   start,
                "window_end":     start + window_size,
                "source_pod_idx": pod_idx,
            })
            start += stride

    win_traces = np.array(win_traces)       # (N_windows, window_size, M)
    win_r      = np.array(win_r)
    return win_traces, win_r, win_meta


def apply_diff_windows(traces, replica_counts, metadata, window_size, stride):
    """
    METHOD 4 (Combined): Differencing then sliding windows.

    Differences are computed on full traces first, then windowed.
    This gives shorter windows where the signal is purely dynamics.
    Window size applies to the differenced sequence (length T-1).
    """
    diffs, initial_values, trace_means = apply_differencing(traces)
    # diffs: (N, T-1, M)
    win_diffs, win_r, win_meta = apply_windows(
        diffs, replica_counts, metadata, window_size, stride
    )
    # Attach initial_values and trace_means to metadata for reconstruction
    for i, wm in enumerate(win_meta):
        src = wm["source_pod_idx"]
        wm["initial_values"] = initial_values[src].tolist()
        wm["trace_means"]    = trace_means[src].tolist()

    return win_diffs, win_r, win_meta


# ------------------------------------------------------------------
# Save dataset
# ------------------------------------------------------------------

def make_split(n_samples, random_state=42):
    indices              = np.arange(n_samples)
    train_idx, val_idx   = train_test_split(indices, train_size=0.9, random_state=random_state)
    return train_idx, val_idx


def save_dataset(output_dir, workload, method, traces, replica_counts,
                 metadata, norm_params, extra_arrays=None):
    """
    Saves .npz and normalization JSON.
    extra_arrays: dict of additional arrays to include in .npz
                  (e.g. initial_values for diff method, trace_means for zeromean)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    train_idx, val_idx = make_split(len(traces))

    save_dict = dict(
        traces         = traces,
        replica_counts = replica_counts,
        train_idx      = train_idx,
        val_idx        = val_idx,
        metric_names   = np.array(ALL_METRICS),
        metadata       = np.array(metadata, dtype=object),
    )
    if extra_arrays:
        save_dict.update(extra_arrays)

    npz_path = output_dir / f"{workload}_traces.npz"
    np.savez_compressed(npz_path, allow_pickle=True, **save_dict)

    norm_path = output_dir / f"{workload}_normalization.json"
    with open(norm_path, "w") as f:
        json.dump(norm_params, f, indent=2)

    summary = {
        "workload":        workload,
        "method":          method,
        "n_samples":       int(len(traces)),
        "n_train":         int(len(train_idx)),
        "n_val":           int(len(val_idx)),
        "trace_shape":     list(traces.shape),
        "replica_distribution": {
            str(int(r)): int((replica_counts == r).sum())
            for r in sorted(np.unique(replica_counts))
        },
    }

    print(f"    Saved {npz_path}  ({len(traces)} samples, shape {traces.shape})")
    return summary


# ------------------------------------------------------------------
# Per-method processing
# ------------------------------------------------------------------

def process_method_raw(workload, raw_traces, replica_counts, metadata, output_dir):
    """Baseline: same as original preprocessor, just saved under phase4/raw/"""
    normalized, norm_params = minmax_normalize(raw_traces)
    return save_dataset(output_dir, workload, "raw",
                        normalized, replica_counts, metadata, norm_params)


def process_method_diff(workload, raw_traces, replica_counts, metadata, output_dir):
    """Method 1: first-order differences on normalized traces."""
    normalized, norm_params = minmax_normalize(raw_traces)
    diffs, initial_values, trace_means = apply_differencing(normalized)

    # Re-normalize differences to [-1, 1] per metric for stable training
    diff_norm_params = {"diff_min": {}, "diff_max": {}}
    diffs_scaled = diffs.copy()
    for i, metric in enumerate(ALL_METRICS):
        d_min = diffs[:, :, i].min()
        d_max = diffs[:, :, i].max()
        span  = max(d_max - d_min, 1e-8)
        diffs_scaled[:, :, i] = (diffs[:, :, i] - d_min) / span  # [0,1]
        diff_norm_params["diff_min"][metric] = float(d_min)
        diff_norm_params["diff_max"][metric] = float(d_max)

    combined_norm = {**norm_params, **diff_norm_params}

    return save_dataset(output_dir, workload, "diff",
                        diffs_scaled, replica_counts, metadata, combined_norm,
                        extra_arrays={
                            "initial_values": initial_values,   # (N, 10)
                            "trace_means":    trace_means,      # (N, 10)
                        })


def process_method_zeromean(workload, raw_traces, replica_counts, metadata, output_dir):
    """Method 2: per-trace zero-mean normalization."""
    normalized, norm_params = minmax_normalize(raw_traces)
    zeromean, per_trace_means = apply_zeromean(normalized)

    # Clip to [-1, 1] (deviations can go negative after subtracting mean)
    zeromean_clipped = np.clip(zeromean, -1.0, 1.0)

    return save_dataset(output_dir, workload, "zeromean",
                        zeromean_clipped, replica_counts, metadata, norm_params,
                        extra_arrays={
                            "per_trace_means": per_trace_means,  # (N, 10)
                        })


def process_method_windows(workload, raw_traces, replica_counts, metadata,
                            output_dir, window_size, stride):
    """Method 3: sliding windows on normalized traces."""
    normalized, norm_params = minmax_normalize(raw_traces)
    win_traces, win_r, win_meta = apply_windows(
        normalized, replica_counts, metadata, window_size, stride
    )
    norm_params["window_size"] = window_size
    norm_params["window_stride"] = stride

    return save_dataset(output_dir, workload, "windows",
                        win_traces, win_r, win_meta, norm_params)


def process_method_diff_windows(workload, raw_traces, replica_counts, metadata,
                                 output_dir, window_size, stride):
    """Method 4 (Combined): differences + sliding windows."""
    normalized, norm_params = minmax_normalize(raw_traces)
    win_diffs, win_r, win_meta = apply_diff_windows(
        normalized, replica_counts, metadata, window_size, stride
    )

    # Normalize difference windows
    diff_norm_params = {"diff_min": {}, "diff_max": {}}
    win_scaled = win_diffs.copy()
    for i, metric in enumerate(ALL_METRICS):
        d_min = win_diffs[:, :, i].min()
        d_max = win_diffs[:, :, i].max()
        span  = max(d_max - d_min, 1e-8)
        win_scaled[:, :, i] = (win_diffs[:, :, i] - d_min) / span
        diff_norm_params["diff_min"][metric] = float(d_min)
        diff_norm_params["diff_max"][metric] = float(d_max)

    combined_norm = {**norm_params, **diff_norm_params,
                     "window_size": window_size, "window_stride": stride}

    return save_dataset(output_dir, workload, "diff_windows",
                        win_scaled, win_r, win_meta, combined_norm)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 4 Preprocessing")
    parser.add_argument("--workloads", nargs="+", default=list(WORKLOAD_CONFIGS.keys()),
                        help="Workloads to process")
    parser.add_argument("--methods", nargs="+", default=METHODS,
                        choices=METHODS,
                        help="Transformation methods to apply")
    parser.add_argument("--window-size",   type=int, default=60,
                        help="Window length in timesteps (default: 60 = 5 minutes at 5s intervals)")
    parser.add_argument("--window-stride", type=int, default=30,
                        help="Window stride in timesteps (default: 30 = 50 percent overlap)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("=" * 60)
    print("Phase 4 Preprocessing")
    print(f"Workloads : {args.workloads}")
    print(f"Methods   : {args.methods}")
    print(f"Window    : size={args.window_size}, stride={args.window_stride}")
    print("=" * 60)

    all_summaries = []

    for workload in args.workloads:
        print(f"\n{'='*60}")
        print(f"Workload: {workload.upper()}")
        print(f"{'='*60}")

        raw_traces, replica_counts, metadata = load_all_raw_traces(workload)

        for method in args.methods:
            print(f"\n  -- Method: {method} --")
            output_dir = OUTPUT_BASE / method

            if method == "raw":
                summary = process_method_raw(
                    workload, raw_traces, replica_counts, metadata, output_dir)

            elif method == "diff":
                summary = process_method_diff(
                    workload, raw_traces, replica_counts, metadata, output_dir)

            elif method == "zeromean":
                summary = process_method_zeromean(
                    workload, raw_traces, replica_counts, metadata, output_dir)

            elif method == "windows":
                summary = process_method_windows(
                    workload, raw_traces, replica_counts, metadata, output_dir,
                    args.window_size, args.window_stride)

            elif method == "diff_windows":
                summary = process_method_diff_windows(
                    workload, raw_traces, replica_counts, metadata, output_dir,
                    args.window_size, args.window_stride)

            all_summaries.append(summary)

    # Save combined summary
    summary_path = OUTPUT_BASE / "preprocessing_summary.json"
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"\n{'Method':<14} {'Workload':<12} {'Samples':>8} {'Train':>7} {'Val':>6} {'Shape'}")
    print("-" * 65)
    for s in all_summaries:
        shape_str = "x".join(str(d) for d in s["trace_shape"])
        print(f"  {s['method']:<12} {s['workload']:<12} {s['n_samples']:>7} "
              f"{s['n_train']:>7} {s['n_val']:>5}  ({shape_str})")

    print(f"\nSummary saved: {summary_path}")
    print("\nOutput directories:")
    for method in args.methods:
        print(f"  data/processed/phase4/{method}/")

if __name__ == "__main__":
    main()