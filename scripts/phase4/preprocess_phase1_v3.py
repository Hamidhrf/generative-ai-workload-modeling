#!/usr/bin/env python3
"""
Phase 1 v3 Data Preprocessing Script - Complete r=1-10 Coverage

Handles CSV format:
- Pod metrics: timestamp, value, pod (one row per pod per timestamp)
- GPU metrics: timestamp, value, [metadata] (one row per timestamp)

Processes 50 experiments across 5 AI workloads (complete r=1-10 for each):
- BERT: 10 experiments (r=1,2,3,4,5,6,7,8,9,10) = 55 pods
- GPT2: 10 experiments (r=1,2,3,4,5,6,7,8,9,10) = 55 pods
- ResNet152: 10 experiments (r=1,2,3,4,5,6,7,8,9,10) = 55 pods
- Whisper: 10 experiments (r=1,2,3,4,5,6,7,8,9,10) = 55 pods
- YOLO: 10 experiments (r=1,2,3,4,5,6,7,8,9,10) = 55 pods

Total: 275 pod traces

Extracts 12 metrics per pod:
- 7 pod-level metrics (CPU, memory, PSI, latency, throughput)
- 5 GPU metrics (divided by replica_count for per-pod approximation)

Note: app_latency_p50/p95/p99 are NOT included in training data.
They will be COMPUTED from pod_latency_avg after trace generation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import json
from sklearn.model_selection import train_test_split

# Configuration
RAW_DATA_DIR = Path("data/raw/phase1_v3")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Workload configurations - All workloads have complete r=1-10 coverage
WORKLOAD_CONFIGS = {
    'bert': {'replicas': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
    'gpt2': {'replicas': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
    'resnet152': {'replicas': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
    'whisper': {'replicas': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
    'yolo': {'replicas': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
}

# 12 metrics to extract
POD_METRICS = [
    'pod_cpu_usage',
    'pod_memory_bytes',
    'pod_psi_cpu',
    'pod_psi_memory',
    'pod_psi_io',
    'pod_latency_avg',
    'pod_throughput'
]

SYSTEM_GPU_METRICS = [
    'gpu_utilization',
    'gpu_memory_used',
    'gpu_memory_total',
    'gpu_power_watts',
    'gpu_temperature'
]

ALL_METRICS = POD_METRICS + SYSTEM_GPU_METRICS


def load_pod_metric_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load a pod-level metric CSV in long format.
    
    Format: timestamp, value, pod
    
    Returns:
        DataFrame with columns: timestamp, pod_0, pod_1, ..., pod_N
    """
    df = pd.read_csv(csv_path)
    
    # Pivot: rows=timestamp, columns=pod, values=value
    pivoted = df.pivot(index='timestamp', columns='pod', values='value')
    
    # Reset index to make timestamp a column
    pivoted = pivoted.reset_index()
    
    # Rename pod columns to pod_0, pod_1, etc.
    pod_cols = [col for col in pivoted.columns if col != 'timestamp']
    new_cols = ['timestamp'] + [f'pod_{i}' for i in range(len(pod_cols))]
    pivoted.columns = new_cols
    
    return pivoted


def load_system_metric_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load a system-level metric CSV.
    
    Format: timestamp, value, [metadata columns...]
    
    Returns:
        DataFrame with columns: timestamp, value
    """
    df = pd.read_csv(csv_path)
    
    # Keep only timestamp and value
    return df[['timestamp', 'value']].copy()


def load_experiment_data(workload: str, replica_count: int) -> Dict[str, pd.DataFrame]:
    """
    Load all CSV files for a single experiment.
    
    Returns:
        Dictionary with metric_name -> DataFrame
        Pod-level DataFrames: columns = [timestamp, pod_0, pod_1, ...]
        System-level DataFrames: columns = [timestamp, value]
    """
    exp_dir = RAW_DATA_DIR / f"{workload}_r{replica_count}"
    
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
    
    # Find timestamp from directory
    csv_files = list(exp_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {exp_dir}")
    
    # Extract timestamp (format: YYYYMMDD_HHMMSS)
    first_file = csv_files[0].stem
    parts = first_file.split('_')
    timestamp = '_'.join(parts[-2:])
    
    data = {}
    
    # Load pod-level metrics
    for metric in POD_METRICS:
        csv_path = exp_dir / f"{workload}_r{replica_count}_{metric}_{timestamp}.csv"
        
        if not csv_path.exists():
            print(f"  WARNING: Missing {metric}")
            continue
        
        try:
            df = load_pod_metric_csv(csv_path)
            data[metric] = df
        except Exception as e:
            print(f"  WARNING: Error loading {metric}: {e}")
    
    # Load system-level GPU metrics
    for metric in SYSTEM_GPU_METRICS:
        csv_path = exp_dir / f"{workload}_r{replica_count}_{metric}_{timestamp}.csv"
        
        if not csv_path.exists():
            print(f"  WARNING: Missing {metric}")
            continue
        
        try:
            df = load_system_metric_csv(csv_path)
            data[metric] = df
        except Exception as e:
            print(f"  WARNING: Error loading {metric}: {e}")
    
    return data


def align_timestamps(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Align all metrics to common timestamps (intersection).
    
    Handles cases where different metrics have different timestamp ranges.
    """
    # Get all unique timestamps from all metrics
    all_timestamps = []
    for metric_df in data.values():
        all_timestamps.append(set(metric_df['timestamp']))
    
    # Find intersection (common timestamps)
    common_timestamps = set.intersection(*all_timestamps)
    common_timestamps = sorted(list(common_timestamps))
    
    print(f"  Aligning to {len(common_timestamps)} common timestamps")
    
    # Filter each metric to common timestamps
    aligned_data = {}
    for metric, df in data.items():
        aligned_df = df[df['timestamp'].isin(common_timestamps)].copy()
        aligned_df = aligned_df.sort_values('timestamp').reset_index(drop=True)
        aligned_data[metric] = aligned_df
    
    return aligned_data


def create_pod_traces(data: Dict[str, pd.DataFrame], replica_count: int) -> Tuple[np.ndarray, List[dict]]:
    """
    Create per-pod traces from aligned experiment data.
    
    Returns:
        traces: (num_pods, timesteps, 12)
        metadata: List of metadata dicts for each pod
    """
    # Determine number of pods from first pod metric
    first_metric = POD_METRICS[0]
    if first_metric not in data:
        raise ValueError(f"Missing required metric: {first_metric}")
    
    first_df = data[first_metric]
    pod_cols = [col for col in first_df.columns if col.startswith('pod_')]
    num_pods = len(pod_cols)
    num_timesteps = len(first_df)
    
    print(f"  Creating traces: {num_pods} pods, {num_timesteps} timesteps")
    
    # Validate replica count
    if abs(num_pods - replica_count) > 1:
        print(f"  WARNING: Pod count ({num_pods}) != replica count ({replica_count})")
    
    # Initialize traces
    traces = np.zeros((num_pods, num_timesteps, len(ALL_METRICS)))
    
    # Metadata
    metadata = []
    for pod_idx in range(num_pods):
        metadata.append({
            'replica_count': replica_count,
            'pod_index': pod_idx,
            'num_pods': num_pods
        })
    
    # Fill traces
    metric_idx = 0
    
    # 1. Pod-level metrics
    for metric in POD_METRICS:
        if metric not in data:
            print(f"  WARNING: Missing {metric}, filling with zeros")
            metric_idx += 1
            continue
        
        df = data[metric]
        for pod_idx in range(num_pods):
            pod_col = f'pod_{pod_idx}'
            if pod_col in df.columns:
                traces[pod_idx, :, metric_idx] = df[pod_col].values
            else:
                print(f"  WARNING: {metric} missing {pod_col}, filling with zeros")
        
        metric_idx += 1
    
    # 2. System GPU metrics (approximate per-pod by dividing)
    for metric in SYSTEM_GPU_METRICS:
        if metric not in data:
            print(f"  WARNING: Missing {metric}, filling with zeros")
            metric_idx += 1
            continue
        
        df = data[metric]
        system_values = df['value'].values
        
        # Approximate per-pod usage
        per_pod_values = system_values / replica_count
        
        # Same value for all pods
        for pod_idx in range(num_pods):
            traces[pod_idx, :, metric_idx] = per_pod_values
        
        metric_idx += 1
    
    return traces, metadata


def normalize_traces(traces: np.ndarray, method: str = 'minmax') -> Tuple[np.ndarray, Dict]:
    """
    Normalize traces to [0, 1] range.
    
    Args:
        traces: (num_samples, timesteps, 12)
        method: 'minmax' or 'zscore'
    
    Returns:
        normalized_traces, norm_params
    """
    num_metrics = traces.shape[2]
    norm_params = {
        'method': method,
        'metric_names': ALL_METRICS,
        'params': {}
    }
    
    normalized = traces.copy()
    
    if method == 'minmax':
        for i in range(num_metrics):
            metric_data = traces[:, :, i]
            min_val = metric_data.min()
            max_val = metric_data.max()
            
            if max_val - min_val > 1e-8:
                normalized[:, :, i] = (metric_data - min_val) / (max_val - min_val)
            else:
                normalized[:, :, i] = 0.0
            
            norm_params['params'][ALL_METRICS[i]] = {
                'min': float(min_val),
                'max': float(max_val)
            }
    
    return normalized, norm_params


def create_splits(traces: np.ndarray, replica_counts: np.ndarray, 
                  metadata: List[dict], train_ratio: float = 0.9, 
                  val_ratio: float = 0.1, random_state: int = 42) -> Dict:
    """
    Create train/val splits (90/10) for small datasets.
    
    Following practices for generative modeling with limited samples
    (Esteban et al., 2017), we use 90% for training and 10% for validation
    (early stopping only). Primary evaluation is generation quality, not
    test set accuracy.
    """
    indices = np.arange(len(traces))
    
    # Check if stratification is possible
    unique, counts = np.unique(replica_counts, return_counts=True)
    min_samples = counts.min()
    use_stratify = min_samples >= 2
    
    if not use_stratify:
        print(f"  NOTE: Small dataset ({len(traces)} samples), using random split")
    
    # Single split: train vs val
    train_idx, val_idx = train_test_split(
        indices,
        train_size=train_ratio,
        stratify=replica_counts if use_stratify else None,
        random_state=random_state
    )
    
    return {
        'train': train_idx,
        'val': val_idx
    }


def process_workload(workload: str) -> None:
    """
    Process all experiments for a single workload.
    """
    print(f"\n{'='*60}")
    print(f"Processing {workload.upper()}")
    print(f"{'='*60}")
    
    config = WORKLOAD_CONFIGS[workload]
    replica_counts_list = config['replicas']
    
    all_traces = []
    all_replica_counts = []
    all_metadata = []
    
    # Load all experiments
    for r in replica_counts_list:
        print(f"\nLoading {workload} r={r}...")
        try:
            # Load raw data
            data = load_experiment_data(workload, r)
            
            # Align timestamps
            data = align_timestamps(data)
            
            # Create traces
            traces, metadata = create_pod_traces(data, r)
            
            # Add to collection
            all_traces.append(traces)
            all_replica_counts.extend([r] * len(traces))
            
            # Add workload to metadata
            for m in metadata:
                m['workload'] = workload
            all_metadata.extend(metadata)
            
            print(f"  ✓ Loaded {len(traces)} pod traces")
            
        except Exception as e:
            print(f"  ERROR loading {workload} r={r}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(all_traces) == 0:
        raise ValueError(f"No traces loaded for {workload}")
    
    # Concatenate
    all_traces = np.concatenate(all_traces, axis=0)
    all_replica_counts = np.array(all_replica_counts)
    
    print(f"\n{workload.upper()} Summary:")
    print(f"  Total pod traces: {len(all_traces)}")
    print(f"  Trace shape: {all_traces.shape}")
    print(f"  Replica counts: {np.unique(all_replica_counts)}")
    
    # Handle NaN/Inf
    if np.isnan(all_traces).any():
        print(f"  WARNING: Found NaN values, replacing with 0")
        all_traces = np.nan_to_num(all_traces, nan=0.0)
    
    if np.isinf(all_traces).any():
        print(f"  WARNING: Found Inf values, clipping")
        all_traces = np.nan_to_num(all_traces, posinf=1e10, neginf=-1e10)
    
    # Normalize
    print(f"\nNormalizing...")
    normalized_traces, norm_params = normalize_traces(all_traces, method='minmax')
    
    # Create splits
    print(f"Creating 90/10 split...")
    splits = create_splits(normalized_traces, all_replica_counts, all_metadata)
    
    train_size = len(splits['train'])
    val_size = len(splits['val'])
    
    print(f"  Train: {train_size} ({train_size/len(all_traces)*100:.1f}%)")
    print(f"  Val:   {val_size} ({val_size/len(all_traces)*100:.1f}%)")
    
    # Save
    output_file = OUTPUT_DIR / f"{workload}_traces.npz"
    print(f"\nSaving to {output_file}...")
    
    np.savez_compressed(
        output_file,
        traces=normalized_traces,
        replica_counts=all_replica_counts,
        train_idx=splits['train'],
        val_idx=splits['val'],
        metadata=np.array(all_metadata, dtype=object),
        metric_names=np.array(ALL_METRICS),
        allow_pickle=True
    )
    
    # Save normalization params
    norm_file = OUTPUT_DIR / f"{workload}_normalization.json"
    with open(norm_file, 'w') as f:
        json.dump(norm_params, f, indent=2)
    
    print(f"✓ {workload.upper()} complete!")


def main():
    """Main preprocessing pipeline."""
    print("="*60)
    print("Phase 1 v3 Data Preprocessing")
    print("="*60)
    print(f"\nRaw data: {RAW_DATA_DIR}")
    print(f"Output: {OUTPUT_DIR}\n")
    
    # Process each workload
    for workload in WORKLOAD_CONFIGS.keys():
        try:
            process_workload(workload)
        except Exception as e:
            print(f"\nERROR processing {workload}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*60)
    print("Preprocessing Complete!")
    print("="*60)
    
    # Summary
    print("\nOutput files:")
    for workload in WORKLOAD_CONFIGS.keys():
        trace_file = OUTPUT_DIR / f"{workload}_traces.npz"
        if trace_file.exists():
            size_mb = trace_file.stat().st_size / (1024 * 1024)
            print(f"  ✓ {trace_file.name} ({size_mb:.2f} MB)")
        else:
            print(f"  ✗ {trace_file.name} (FAILED)")


if __name__ == "__main__":
    main()