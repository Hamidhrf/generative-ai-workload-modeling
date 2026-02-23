#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import json
from sklearn.model_selection import train_test_split

RAW_DATA_DIR = Path("data/raw/phase1_v3")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WORKLOAD_CONFIGS = {
    'bert': {'replicas': [1,2,3,4,5,6,7,8,9,10]},
    'gpt2': {'replicas': [1,2,3,4,5,6,7,8,9,10]},
    'resnet152': {'replicas': [1,2,3,4,5,6,7,8,9,10]},
    'whisper': {'replicas': [1,2,3,4,5,6,7,8,9,10]},
    'yolo': {'replicas': [1,2,3,4,5,6,7,8,9,10]}
}

POD_METRICS = ['pod_cpu_usage','pod_memory_bytes','pod_psi_cpu','pod_latency_avg','pod_throughput']
SYSTEM_GPU_METRICS = ['gpu_utilization','gpu_memory_used','gpu_memory_total','gpu_power_watts','gpu_temperature']
ALL_METRICS = POD_METRICS + SYSTEM_GPU_METRICS

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

def align_timestamps(data):
    all_ts = [set(df['timestamp']) for df in data.values()]
    common_ts = sorted(list(set.intersection(*all_ts)))
    print(f"  Aligning to {len(common_ts)} timestamps")
    return {m: df[df['timestamp'].isin(common_ts)].sort_values('timestamp').reset_index(drop=True) for m, df in data.items()}

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
            data = align_timestamps(load_experiment_data(workload, r))
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

print("="*60 + "\nPreprocessing (10 metrics)\n" + "="*60)
for w in WORKLOAD_CONFIGS.keys():
    process_workload(w)
print("\n" + "="*60 + "\nDone!\n" + "="*60)
