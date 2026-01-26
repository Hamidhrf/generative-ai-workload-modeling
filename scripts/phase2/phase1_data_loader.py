#!/usr/bin/env python3
"""
Phase 1 Data Loader - Pod-Level Traces with Metadata

Loads individual pod traces from Phase 1 experiments with full metadata tracking.
Each experiment directory contains 15 CSV files (one per metric) that must be merged.

Returns:
    List of (trace, metadata) tuples:
    - trace: (715, 15) numpy array
    - metadata: dict with workload, replica_count, pod_id, experiment_dir

Author: Hamidreza Fathollahzadeh
Date: January 22, 2026
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import re

# Define the 15 metrics in order
METRIC_ORDER = [
    'cpu_psi',
    'cpu_usage', 
    'gpu_memory',
    'gpu_power',
    'gpu_temperature',
    'gpu_utilization',
    'inference_latency_avg',
    'inference_latency_p50',
    'inference_latency_p95',
    'inference_latency_p99',
    'inference_throughput',
    'inference_total',
    'io_psi',
    'memory_psi',
    'memory_usage'
]

# Per-pod metrics (have 'pod' column)
PER_POD_METRICS = [
    'cpu_psi',
    'cpu_usage',
    'inference_latency_avg',
    'io_psi',
    'memory_psi',
    'memory_usage'
]

# System-level metrics (no 'pod' column, shared across all pods)
SYSTEM_METRICS = [
    'gpu_memory',
    'gpu_power',
    'gpu_temperature',
    'gpu_utilization',
    'inference_latency_p50',
    'inference_latency_p95',
    'inference_latency_p99',
    'inference_throughput',
    'inference_total'
]


def extract_experiment_info(exp_dir_name: str) -> Tuple[str, int]:
    """
    Extract workload name and replica count from experiment directory name.
    
    Args:
        exp_dir_name: e.g., 'resnet50_r3', 'distilbert_r10'
    
    Returns:
        (workload, replica_count): e.g., ('resnet50', 3)
    """
    # Match pattern: workload_rN
    match = re.match(r'(.+)_r(\d+)', exp_dir_name)
    if match:
        workload = match.group(1)
        replica_count = int(match.group(2))
        return workload, replica_count
    else:
        raise ValueError(f"Cannot parse experiment directory name: {exp_dir_name}")


def find_metric_file(exp_dir: Path, metric_name: str) -> str:
    """
    Find the CSV file for a given metric in experiment directory.
    
    Args:
        exp_dir: Path to experiment directory
        metric_name: e.g., 'cpu_usage', 'gpu_memory'
    
    Returns:
        Path to CSV file
    """
    pattern = str(exp_dir / f"*{metric_name}*.csv")
    files = glob.glob(pattern)
    
    if len(files) == 0:
        raise FileNotFoundError(f"No file found for metric '{metric_name}' in {exp_dir}")
    elif len(files) > 1:
        raise ValueError(f"Multiple files found for metric '{metric_name}' in {exp_dir}: {files}")
    
    return files[0]


def load_metric_data(csv_file: str, metric_name: str) -> pd.DataFrame:
    """
    Load a single metric CSV file.
    
    Args:
        csv_file: Path to CSV file
        metric_name: Name of the metric
    
    Returns:
        DataFrame with columns: timestamp, value, pod (if per-pod metric)
    """
    df = pd.read_csv(csv_file)
    
    # Keep only essential columns
    if metric_name in PER_POD_METRICS:
        # Per-pod metric - needs pod identifier
        essential_cols = ['timestamp', 'value', 'pod']
    else:
        # System-level metric - no pod column
        essential_cols = ['timestamp', 'value']
    
    df = df[essential_cols].copy()
    
    # Rename value column to metric name
    df.rename(columns={'value': metric_name}, inplace=True)
    
    return df


def merge_experiment_data(exp_dir: Path) -> pd.DataFrame:
    """
    Merge all 15 metric CSVs for an experiment into a single DataFrame.
    
    Args:
        exp_dir: Path to experiment directory
    
    Returns:
        DataFrame with columns: timestamp, pod, metric1, metric2, ..., metric15
        - For multi-pod: each timestamp appears N times (once per pod)
        - For single pod: each timestamp appears once
    """
    print(f"  Loading experiment: {exp_dir.name}")
    
    # Step 1: Load per-pod metrics (these have pod identifiers)
    per_pod_dfs = []
    for metric in PER_POD_METRICS:
        csv_file = find_metric_file(exp_dir, metric)
        df = load_metric_data(csv_file, metric)
        per_pod_dfs.append(df)
    
    # Merge per-pod metrics on timestamp + pod
    merged = per_pod_dfs[0]
    for df in per_pod_dfs[1:]:
        merged = merged.merge(df, on=['timestamp', 'pod'], how='outer')
    
    # Step 2: Load system-level metrics (no pod column)
    system_dfs = []
    for metric in SYSTEM_METRICS:
        csv_file = find_metric_file(exp_dir, metric)
        df = load_metric_data(csv_file, metric)
        system_dfs.append(df)
    
    # Merge system metrics on timestamp only
    system_merged = system_dfs[0]
    for df in system_dfs[1:]:
        system_merged = system_merged.merge(df, on='timestamp', how='outer')
    
    # Step 3: Merge per-pod and system data
    # System metrics are replicated for each pod
    merged = merged.merge(system_merged, on='timestamp', how='left')
    
    # Sort by timestamp and pod for consistency
    merged = merged.sort_values(['timestamp', 'pod']).reset_index(drop=True)
    
    print(f"    Loaded {len(merged)} rows")
    
    return merged


def extract_pod_traces(merged_df: pd.DataFrame, exp_dir_name: str) -> List[Tuple[np.ndarray, Dict]]:
    """
    Extract individual pod traces from merged experiment data.
    
    Args:
        merged_df: Merged DataFrame with all metrics
        exp_dir_name: Name of experiment directory
    
    Returns:
        List of (trace, metadata) tuples
    """
    workload, replica_count = extract_experiment_info(exp_dir_name)
    
    # Get unique pods
    unique_pods = merged_df['pod'].unique()
    print(f"    Found {len(unique_pods)} pods")
    
    pod_traces = []
    
    for pod_id in unique_pods:
        # Extract data for this pod
        pod_data = merged_df[merged_df['pod'] == pod_id].copy()
        
        # Sort by timestamp
        pod_data = pod_data.sort_values('timestamp').reset_index(drop=True)
        
        # Extract metric values in correct order
        trace_values = []
        for metric in METRIC_ORDER:
            if metric in pod_data.columns:
                trace_values.append(pod_data[metric].values)
            else:
                raise ValueError(f"Metric '{metric}' not found in data for pod {pod_id}")
        
        # Stack into (n_timesteps, n_metrics) array
        trace = np.column_stack(trace_values)
        
        # Create metadata
        metadata = {
            'workload': workload,
            'replica_count': replica_count,
            'pod_id': pod_id,
            'experiment_dir': exp_dir_name,
            'n_timesteps': trace.shape[0],
            'n_metrics': trace.shape[1]
        }
        
        pod_traces.append((trace, metadata))
        print(f"      Pod {pod_id[:20]:20s} → shape {trace.shape}")
    
    return pod_traces


def load_phase1_data(data_dir: str = '~/generative-ai-workload-modeling/data/raw/phase1',
                     verbose: bool = True) -> List[Tuple[np.ndarray, Dict]]:
    """
    Load all Phase 1 pod-level traces with metadata.
    
    Args:
        data_dir: Path to Phase 1 data directory
        verbose: Print loading progress
    
    Returns:
        List of (trace, metadata) tuples:
        - trace: (n_timesteps, 15) numpy array
        - metadata: dict with workload, replica_count, pod_id, experiment_dir
    
    Example:
        >>> data = load_phase1_data()
        >>> print(f"Loaded {len(data)} pod traces")
        >>> trace, meta = data[0]
        >>> print(f"Shape: {trace.shape}, Workload: {meta['workload']}, r={meta['replica_count']}")
    """
    data_path = Path(data_dir).expanduser()
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    
    # Get all experiment directories
    exp_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    
    if verbose:
        print("="*80)
        print("PHASE 1 DATA LOADER - POD-LEVEL TRACES")
        print("="*80)
        print(f"\nData directory: {data_path}")
        print(f"Found {len(exp_dirs)} experiment directories\n")
    
    all_pod_traces = []
    
    for exp_dir in exp_dirs:
        try:
            # Merge all 15 metric CSVs
            merged_df = merge_experiment_data(exp_dir)
            
            # Extract individual pod traces
            pod_traces = extract_pod_traces(merged_df, exp_dir.name)
            
            all_pod_traces.extend(pod_traces)
            
        except Exception as e:
            print(f"  ❌ Error processing {exp_dir.name}: {e}")
            continue
    
    if verbose:
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Total pod traces loaded: {len(all_pod_traces)}")
        
        # Summary by workload
        workload_counts = {}
        for _, meta in all_pod_traces:
            wl = meta['workload']
            workload_counts[wl] = workload_counts.get(wl, 0) + 1
        
        print("\nPod traces per workload:")
        for wl, count in sorted(workload_counts.items()):
            print(f"  {wl:15s}: {count:3d} pod traces")
        
        # Summary by replica count
        print("\nPod traces per replica count:")
        replica_counts = {}
        for _, meta in all_pod_traces:
            r = meta['replica_count']
            replica_counts[r] = replica_counts.get(r, 0) + 1
        
        for r in sorted(replica_counts.keys()):
            print(f"  r={r:2d}: {replica_counts[r]:3d} pod traces")
        
        # Shape summary
        if all_pod_traces:
            first_trace, _ = all_pod_traces[0]
            print(f"\nTrace shape: {first_trace.shape} (timesteps, metrics)")
            print(f"Expected: (~715, 15)")
        
        print("\n" + "="*80)
    
    return all_pod_traces


def save_pod_traces(pod_traces: List[Tuple[np.ndarray, Dict]], 
                    output_dir: str = '~/generative-ai-workload-modeling/data/processed/phase1'):
    """
    Save pod traces to numpy files for faster loading.
    
    Args:
        pod_traces: List of (trace, metadata) tuples
        output_dir: Directory to save processed data
    """
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving {len(pod_traces)} pod traces to {output_path}")
    
    # Save traces and metadata separately
    traces = [trace for trace, _ in pod_traces]
    metadata = [meta for _, meta in pod_traces]
    
    np.save(output_path / 'pod_traces.npy', traces, allow_pickle=True)
    np.save(output_path / 'pod_metadata.npy', metadata, allow_pickle=True)
    
    print(f"✓ Saved to:")
    print(f"  - pod_traces.npy ({len(traces)} traces)")
    print(f"  - pod_metadata.npy ({len(metadata)} metadata dicts)")


def load_saved_pod_traces(input_dir: str = '~/generative-ai-workload-modeling/data/processed/phase1') -> List[Tuple[np.ndarray, Dict]]:
    """
    Load pre-saved pod traces from numpy files.
    
    Args:
        input_dir: Directory with saved data
    
    Returns:
        List of (trace, metadata) tuples
    """
    input_path = Path(input_dir).expanduser()
    
    traces = np.load(input_path / 'pod_traces.npy', allow_pickle=True)
    metadata = np.load(input_path / 'pod_metadata.npy', allow_pickle=True)
    
    return list(zip(traces, metadata))


# Example usage and testing
if __name__ == "__main__":
    print("Loading Phase 1 pod-level data...\n")
    
    # Load all pod traces
    pod_traces = load_phase1_data()
    
    print("\n" + "="*80)
    print("EXAMPLE POD TRACES")
    print("="*80)
    
    # Show first 5 pod traces
    for i, (trace, meta) in enumerate(pod_traces[:5]):
        print(f"\nPod {i+1}:")
        print(f"  Workload: {meta['workload']}")
        print(f"  Replica count: {meta['replica_count']}")
        print(f"  Pod ID: {meta['pod_id'][:40]}...")
        print(f"  Experiment: {meta['experiment_dir']}")
        print(f"  Trace shape: {trace.shape}")
        print(f"  First timestep values: {trace[0, :5]}...")
    
    # Save for faster loading next time
    save_pod_traces(pod_traces)
    
    print("\n" + "="*80)
    print("✓ Data loading complete!")
    print("="*80)
    print("\nYou now have:")
    print(f"  • {len(pod_traces)} individual pod traces")
    print(f"  • Each with shape (timesteps, 15 metrics)")
    print(f"  • Full metadata (workload, replica_count, pod_id)")
    print(f"  • Ready for TimeVAE/TimeGAN training with conditioning!")