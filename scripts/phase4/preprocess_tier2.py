#!/usr/bin/env python3
# Adapted from preprocess_tier3.py for Tier 2 H100 MIG data. Paths
# retargeted extension_tier3 -> extension_tier2, tier3 -> tier2, replica
# range r=1..10 -> r=1..7. Substantive change: gpu_utilization is sourced
# from the per-slice CSV (native per-pod DCGM attribution under MIG,
# TIER2_NOTES.md Section 6.2/8.1) instead of the aggregated whole-GPU CSV,
# so it is loaded and stacked like a per-pod metric (no /replica_count
# imputation -- see TIER2_NOTES.md Section 5.5, 8.2, 8.3). All other
# metric branches are byte-identical to Tier 3.
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import json
from sklearn.model_selection import train_test_split

RAW_DATA_DIR = Path("data/raw/extension_tier2")
OUTPUT_DIR = Path("data/processed/tier2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WORKLOAD_CONFIGS = {
    'bert': {'replicas': [1,2,3,4,5,6,7]},
    'gpt2': {'replicas': [1,2,3,4,5,6,7]},
    'resnet152': {'replicas': [1,2,3,4,5,6,7]},
    'whisper': {'replicas': [1,2,3,4,5,6,7]},
    'yolo': {'replicas': [1,2,3,4,5,6,7]}
}

POD_METRICS = ['pod_cpu_usage','pod_memory_bytes','pod_psi_cpu','pod_latency_avg','pod_throughput']
# gpu_utilization is per-pod (per-slice source) on Tier 2; the remaining
# four GPU metrics stay aggregated whole-GPU CSVs with /replica_count
# imputation, same as Tier 3.
AGG_GPU_METRICS = ['gpu_memory_used','gpu_memory_total','gpu_power_watts','gpu_temperature']
ALL_METRICS = POD_METRICS + ['gpu_utilization'] + AGG_GPU_METRICS

# Metrics that are per-pod shaped after loading (pivoted to pod_0..pod_{n-1}
# columns) -- used only to label the front-gap audit print, not to change
# fill logic (which is identical across pod-shaped and system-shaped data).
PER_POD_METRICS = POD_METRICS + ['gpu_utilization']

# Front-gaps observed on Tier 3 topped out at 65s (whisper r=9). 180s
# (~5% of the 715-step trace) is a loud-but-not-hard-fail tripwire for
# anything materially worse than what's been diagnosed. Unchanged from
# Tier 3 -- Tier 2 magnitude is TBD, reported in the SUMMARY block below.
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

def load_gpu_utilization_per_slice_csv(csv_path):
    # Native per-pod MIG attribution (TIER2_NOTES.md 5.5/8.1): DCGM
    # populates the `pod` label directly on per-instance series. Rows
    # with an empty pod are idle (unassigned) slices -- drop them (all
    # r slices are active at r=r, so this is a no-op at r=7 and drops
    # (7-r) idle-slice rows at r<7). Values are already
    # DCGM_FI_PROF_GR_ENGINE_ACTIVE * 100 (0..100 per slice, TIER2_NOTES.md
    # 6.2) -- no further scaling.
    df = pd.read_csv(csv_path)
    df = df[df['pod'].notna() & (df['pod'] != '')]
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
    gpu_util_path = exp_dir / f"{workload}_r{replica_count}_gpu_utilization_per_slice_{timestamp}.csv"
    if gpu_util_path.exists():
        data['gpu_utilization'] = load_gpu_utilization_per_slice_csv(gpu_util_path)
    for metric in AGG_GPU_METRICS:
        csv_path = exp_dir / f"{workload}_r{replica_count}_{metric}_{timestamp}.csv"
        if csv_path.exists():
            data[metric] = load_system_metric_csv(csv_path)
    return data

# Tier 2 (H100 MIG) can exhibit the same front-gaps in pod_latency_avg /
# pod_throughput at high replica counts that Tier 3 does: these two
# metrics are computed from completed inference requests, and under
# contention the first request can take tens of seconds to complete.
# Meanwhile cAdvisor and DCGM metrics (including the per-slice
# gpu_utilization source) report from t=0 regardless.
#
# Policy unchanged from Tier 3: take the union of metric timestamps,
# forward-fill leading NaN per (metric, pod) with the first observed
# value (implemented as bfill on the reindexed series).
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
                pod_label = col if metric in PER_POD_METRICS else 'system'
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

        # Trailing/mid-series NaN are NOT filled -- bfill only resolves
        # leading gaps (nothing after the last valid value to pull from).
        # Any survivor here is a different failure mode we haven't
        # diagnosed; flag it loudly rather than silently zero-filling.
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

    # gpu_utilization: per-pod (per-slice source), no /replica_count --
    # the value is already the individual slice's measurement.
    if 'gpu_utilization' in data:
        df = data['gpu_utilization']
        for i in range(num_pods):
            pod_col = f'pod_{i}'
            if pod_col in df.columns:
                traces[i, :, metric_idx] = df[pod_col].values
            else:
                print(f"    WARNING: Missing {pod_col} for gpu_utilization")
    metric_idx += 1

    for metric in AGG_GPU_METRICS:
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
    # Truncate 720->715 to match A16/Tier 3 seq_len that S36's
    # frozen recipe expects; last 5 timesteps are Phase 5
    # cooldown tail.
    all_traces = all_traces[:, :715, :]
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

print("="*60 + "\nPreprocessing (10 metrics, Tier 2 MIG)\n" + "="*60)
for w in WORKLOAD_CONFIGS.keys():
    process_workload(w)
print("\n" + "="*60 + "\nDone!\n" + "="*60)
print(f"SUMMARY: {len(LEFT_FILL_EXPERIMENTS)} experiments required left-fill: {LEFT_FILL_EXPERIMENTS}")
print(f"SUMMARY: total (metric, pod) series left-filled: {LEFT_FILL_SERIES_COUNT}")
print(f"SUMMARY: max front_gap observed: {MAX_FRONT_GAP_OBSERVED:.1f} seconds")
