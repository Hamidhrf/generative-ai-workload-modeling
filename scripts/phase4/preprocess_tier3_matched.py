#!/usr/bin/env python3
# E1 matched-size control (NOT committed). Adapted from preprocess_tier3.py
# with ONE substantive change: WORKLOAD_CONFIGS replicas restricted to
# r=1..7 (28 traces/workload, matching Tier 2's r-range and trace count),
# OUTPUT_DIR retargeted to data/processed/tier3_matched (does not touch
# data/processed/tier3/). RAW_DATA_DIR is unchanged (data/raw/extension_tier3
# -- same source CSVs as full Tier 3) and gpu_utilization computation is
# unchanged (whole-GPU / replica_count imputation, same as full Tier 3, NOT
# Tier 2's per-slice source) -- this holds the sharing-mode/counter axis
# constant so only dataset size differs vs full Tier 3. Everything else is
# byte-identical to preprocess_tier3.py.
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import json
from sklearn.model_selection import train_test_split

RAW_DATA_DIR = Path("data/raw/extension_tier3")
OUTPUT_DIR = Path("data/processed/tier3_matched")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WORKLOAD_CONFIGS = {
    'bert': {'replicas': [1,2,3,4,5,6,7]},
    'gpt2': {'replicas': [1,2,3,4,5,6,7]},
    'resnet152': {'replicas': [1,2,3,4,5,6,7]},
    'whisper': {'replicas': [1,2,3,4,5,6,7]},
    'yolo': {'replicas': [1,2,3,4,5,6,7]}
}

POD_METRICS = ['pod_cpu_usage','pod_memory_bytes','pod_psi_cpu','pod_latency_avg','pod_throughput']
SYSTEM_GPU_METRICS = ['gpu_utilization','gpu_memory_used','gpu_memory_total','gpu_power_watts','gpu_temperature']
ALL_METRICS = POD_METRICS + SYSTEM_GPU_METRICS

# Front-gaps observed on Tier 3 topped out at 65s (whisper r=9). 180s
# (~5% of the 715-step trace) is a loud-but-not-hard-fail tripwire for
# anything materially worse than what's been diagnosed.
MAX_FRONT_GAP_S = 180

# left-fill bookkeeping for the end-of-run SUMMARY block
LEFT_FILL_EXPERIMENTS = []
LEFT_FILL_SERIES_COUNT = 0
MAX_FRONT_GAP_OBSERVED = 0.0

def load_pod_metric_csv(csv_path):
    df = pd.read_csv(csv_path)
    pivoted = df.pivot(index='timestamp', columns='pod', values='value').reset_index()
    pod_cols = [col for col in pivoted.columns if col != 'timestamp']
    pivoted.columns = ['timestamp'] + [f'pod_{i}' for i in range(len(pod_cols))]
    return pivoted

def load_system_metric_csv(csv_path):
    return pd.read_csv(csv_path)[['timestamp','value']].copy()

def load_experiment_data(workload, replica_count):
    exp_dir = RAW_DATA_DIR / f"{workload}_r{replica_count}"
    csv_files = list(exp_dir.glob("*.csv"))
    timestamp = '_'.join(csv_files[0].stem.split('_')[-2:])
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

def align_timestamps(data, workload=None, replica_count=None):
    global LEFT_FILL_SERIES_COUNT, MAX_FRONT_GAP_OBSERVED
    union_ts = sorted(set.union(*[set(df['timestamp']) for df in data.values()]))
    print(f"  Aligning to {len(union_ts)} timestamps (union)")
    union_first_dt = pd.to_datetime(union_ts[0])

    experiment_had_fill = False
    aligned = {}
    for metric, df in data.items():
        reindexed = df.set_index('timestamp').reindex(union_ts)
        for col in reindexed.columns:
            series = reindexed[col]
            first_valid_label = series.first_valid_index()
            if first_valid_label is None:
                continue
            n_leading = series.index.get_loc(first_valid_label)
            if n_leading > 0:
                first_val = series.loc[first_valid_label]
                front_gap_s = (pd.to_datetime(first_valid_label) - union_first_dt).total_seconds()
                pod_label = col if metric in POD_METRICS else 'system'
                print(f"LEFT_FILL: {workload} r={replica_count} {metric} pod={pod_label} "
                      f"fill_n={n_leading} first_val={first_val:.4f} gap_s={front_gap_s:.1f}")
                if front_gap_s > MAX_FRONT_GAP_S:
                    print(f"  WARNING: front_gap {front_gap_s:.1f}s exceeds MAX_FRONT_GAP_S="
                          f"{MAX_FRONT_GAP_S}s for {workload} r={replica_count} {metric} "
                          f"pod={pod_label} -- continuing anyway")
                reindexed[col] = series.bfill()
                experiment_had_fill = True
                LEFT_FILL_SERIES_COUNT += 1
                MAX_FRONT_GAP_OBSERVED = max(MAX_FRONT_GAP_OBSERVED, front_gap_s)

        remaining_nan = int(reindexed.isna().sum().sum())
        if remaining_nan > 0:
            print(f"  WARNING: {remaining_nan} unresolved NaN remain in {metric} for "
                  f"{workload} r={replica_count} after left-fill -- possible back-gap, not filled")

        aligned[metric] = reindexed.reset_index().sort_values('timestamp').reset_index(drop=True)

    if experiment_had_fill:
        LEFT_FILL_EXPERIMENTS.append(f"{workload} r={replica_count}")

    return aligned

def create_pod_traces(data, replica_count):
    first_df = data[POD_METRICS[0]]
    pod_cols = [c for c in first_df.columns if c.startswith('pod_')]
    num_pods = len(pod_cols)  # Use ACTUAL pod count from CSV
    num_timesteps = len(first_df)
    print(f"  Creating: {num_pods} pods, {num_timesteps} timesteps", end='')

    if num_pods != replica_count:
        print(f" (expected {replica_count}, found {num_pods})")
    else:
        print()

    traces = np.zeros((num_pods, num_timesteps, len(ALL_METRICS)))
    metadata = [{'replica_count': replica_count, 'pod_index': i, 'num_pods': num_pods} for i in range(num_pods)]

    metric_idx = 0
    for metric in POD_METRICS:
        if metric in data:
            df = data[metric]
            for i in range(num_pods):
                pod_col = f'pod_{i}'
                if pod_col in df.columns:
                    traces[i, :, metric_idx] = df[pod_col].values
                else:
                    print(f"    WARNING: Missing {pod_col} for {metric}")
        metric_idx += 1

    for metric in SYSTEM_GPU_METRICS:
        if metric in data:
            per_pod = data[metric]['value'].values / replica_count
            for i in range(num_pods):
                traces[i, :, metric_idx] = per_pod
        metric_idx += 1

    return traces, metadata

def normalize_traces(traces):
    norm_params = {'method': 'minmax', 'metric_names': ALL_METRICS, 'params': {}}
    normalized = traces.copy()
    for i in range(traces.shape[2]):
        min_val, max_val = traces[:,:,i].min(), traces[:,:,i].max()
        if max_val - min_val > 1e-8:
            normalized[:,:,i] = (traces[:,:,i] - min_val) / (max_val - min_val)
        else:
            normalized[:,:,i] = 0.0
        norm_params['params'][ALL_METRICS[i]] = {'min': float(min_val), 'max': float(max_val)}
    return normalized, norm_params

def process_workload(workload):
    print(f"\n{'='*60}\nProcessing {workload.upper()}\n{'='*60}")
    all_traces, all_r, all_meta = [], [], []
    for r in WORKLOAD_CONFIGS[workload]['replicas']:
        print(f"\nLoading {workload} r={r}...")
        try:
            data = align_timestamps(load_experiment_data(workload, r), workload, r)
            traces, metadata = create_pod_traces(data, r)
            all_traces.append(traces)
            all_r.extend([r] * len(traces))
            for m in metadata: m['workload'] = workload
            all_meta.extend(metadata)
            print(f"  ✓ Loaded {len(traces)} pods")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    all_traces = np.nan_to_num(np.concatenate(all_traces, axis=0), nan=0.0)
    all_r = np.array(all_r)
    print(f"\n{workload.upper()}: {len(all_traces)} traces, shape {all_traces.shape}")
    normalized, norm_params = normalize_traces(all_traces)
    indices = np.arange(len(normalized))
    train_idx, val_idx = train_test_split(indices, train_size=0.9, random_state=42)
    print(f"Split: train={len(train_idx)}, val={len(val_idx)}")
    np.savez_compressed(OUTPUT_DIR / f"{workload}_traces.npz", traces=normalized, replica_counts=all_r,
                        train_idx=train_idx, val_idx=val_idx, metadata=np.array(all_meta, dtype=object),
                        metric_names=np.array(ALL_METRICS), allow_pickle=True)
    with open(OUTPUT_DIR / f"{workload}_normalization.json", 'w') as f:
        json.dump(norm_params, f, indent=2)
    print(f"✓ Saved!")

print("="*60 + "\nPreprocessing (10 metrics, Tier 3 MATCHED r=1..7)\n" + "="*60)
for w in WORKLOAD_CONFIGS.keys():
    process_workload(w)
print("\n" + "="*60 + "\nDone!\n" + "="*60)
print(f"SUMMARY: {len(LEFT_FILL_EXPERIMENTS)} experiments required left-fill: {LEFT_FILL_EXPERIMENTS}")
print(f"SUMMARY: total (metric, pod) series left-filled: {LEFT_FILL_SERIES_COUNT}")
print(f"SUMMARY: max front_gap observed: {MAX_FRONT_GAP_OBSERVED:.1f} seconds")
