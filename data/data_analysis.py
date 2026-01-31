#!/usr/bin/env python3
"""
Data Characteristics Deep Analysis
===================================

Before trying more models, let's understand:
1. What is the ACTUAL variance in each workload's data?
2. Is the data really flat, or are we losing signal during normalization?
3. Which metrics have the most information to learn?
4. Should we use different metrics for different workloads?

Author: Hamidreza Fathollahzadeh
Date: January 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from collections import defaultdict

# Configuration
DATA_PATH = '/home/hamid/generative-ai-workload-modeling/data/processed/phase1/pod_traces_normalized.npy'
METADATA_PATH = '/home/hamid/generative-ai-workload-modeling/data/processed/phase1/pod_metadata.json'
OUTPUT_DIR = '/home/hamid/generative-ai-workload-modeling/outputs/data_analysis'

ALL_METRICS = ['cpu_psi', 'cpu_usage', 'cpu_num', 'cpu_temperature', 
               'gpu_power', 'gpu_utilization', 'gpu_temperature',
               'memory_usage', 'memory_psi', 'latency', 'latency_p50',
               'latency_p95', 'latency_p99', 'success_rate', 'throughput']


def load_data():
    """Load data and metadata"""
    data = np.load(DATA_PATH)
    
    try:
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print("Creating synthetic metadata...")
        metadata = []
        for r, count in [(1,1), (2,2), (3,3), (6,6), (10,10)]:
            for _ in range(count):
                metadata.append({'workload': 'resnet50', 'replica_count': r})
        for r, count in [(1,1), (2,2), (6,6), (10,10)]:
            for _ in range(count):
                metadata.append({'workload': 'distilbert', 'replica_count': r})
        for r, count in [(1,1), (2,2), (3,3), (5,5), (8,8)]:
            for _ in range(count):
                metadata.append({'workload': 'whisper', 'replica_count': r})
    
    return data, metadata


def compute_signal_strength(values):
    """
    Compute how much 'signal' is in the data
    High signal = values change over time
    Low signal = values are nearly constant
    """
    # Standard deviation
    std = np.std(values)
    
    # Range
    val_range = np.max(values) - np.min(values)
    
    # Coefficient of variation (normalized std)
    mean = np.mean(values)
    cv = std / (mean + 1e-8)
    
    # First-order difference variance (captures local changes)
    diff = np.diff(values.flatten())
    diff_var = np.var(diff)
    
    # Autocorrelation at lag 1 (captures temporal structure)
    autocorr = np.corrcoef(values.flatten()[:-1], values.flatten()[1:])[0, 1]
    
    return {
        'std': std,
        'range': val_range,
        'cv': cv,
        'diff_var': diff_var,
        'autocorr': autocorr if not np.isnan(autocorr) else 0
    }


def analyze_workload(traces, workload_name):
    """Deep analysis of one workload"""
    print(f"\n{'='*60}")
    print(f"{workload_name.upper()} ANALYSIS")
    print(f"{'='*60}")
    
    n_traces = len(traces)
    seq_len = traces.shape[1]
    n_metrics = traces.shape[2]
    
    print(f"Traces: {n_traces}, Sequence length: {seq_len}, Metrics: {n_metrics}")
    
    results = {}
    
    for m_idx, metric in enumerate(ALL_METRICS):
        vals = traces[:, :, m_idx]
        
        # Signal strength
        signal = compute_signal_strength(vals)
        
        # Per-trace temporal variance (what the model needs to learn)
        per_trace_vars = [np.var(vals[i]) for i in range(n_traces)]
        avg_temporal_var = np.mean(per_trace_vars)
        
        # Between-trace variance (differences between pods/replicas)
        between_trace_var = np.var(np.mean(vals, axis=1))
        
        results[metric] = {
            'signal': signal,
            'temporal_var': avg_temporal_var,
            'between_trace_var': between_trace_var,
            'min': np.min(vals),
            'max': np.max(vals),
            'mean': np.mean(vals)
        }
        
        # Print metrics with HIGH signal (potential for learning)
        if signal['cv'] > 0.05 or signal['diff_var'] > 0.0001:
            print(f"\n  {metric} [HIGH SIGNAL]:")
        else:
            print(f"\n  {metric} [LOW SIGNAL]:")
        
        print(f"    Range: [{np.min(vals):.4f}, {np.max(vals):.4f}]")
        print(f"    CV (coef of var): {signal['cv']:.4f}")
        print(f"    Temporal variance: {avg_temporal_var:.6f}")
        print(f"    Diff variance: {signal['diff_var']:.6f}")
    
    return results


def find_best_metrics(results):
    """Find which metrics have the most signal for each workload"""
    print(f"\n{'='*60}")
    print("BEST METRICS FOR EACH WORKLOAD")
    print("(Ranked by coefficient of variation)")
    print("="*60)
    
    for wl, metrics in results.items():
        print(f"\n{wl.upper()}:")
        
        # Sort by CV (coefficient of variation)
        sorted_metrics = sorted(
            metrics.items(),
            key=lambda x: x[1]['signal']['cv'],
            reverse=True
        )
        
        print("  Top 5 metrics by signal strength:")
        for i, (metric, data) in enumerate(sorted_metrics[:5]):
            cv = data['signal']['cv']
            temp_var = data['temporal_var']
            print(f"    {i+1}. {metric}: CV={cv:.4f}, temporal_var={temp_var:.6f}")


def plot_temporal_patterns(data, metadata, output_dir):
    """Visualize actual temporal patterns in the data"""
    workload_data = defaultdict(list)
    
    for i, meta in enumerate(metadata):
        wl = meta.get('workload', 'unknown').lower()
        if 'resnet' in wl:
            wl = 'resnet50'
        elif 'distil' in wl:
            wl = 'distilbert'
        elif 'whisper' in wl:
            wl = 'whisper'
        workload_data[wl].append(data[i])
    
    # Plot first 100 timesteps for selected metrics
    metrics_to_plot = [1, 5, 7, 9]  # cpu_usage, gpu_util, memory, latency
    metric_names = ['cpu_usage', 'gpu_utilization', 'memory_usage', 'latency']
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    
    for row, wl in enumerate(['resnet50', 'distilbert', 'whisper']):
        traces = np.array(workload_data[wl])
        
        for col, (m_idx, m_name) in enumerate(zip(metrics_to_plot, metric_names)):
            ax = axes[row, col]
            
            # Plot first 3 traces, first 100 timesteps
            for i in range(min(3, len(traces))):
                ax.plot(traces[i, :100, m_idx], alpha=0.7, label=f'trace {i+1}')
            
            if row == 0:
                ax.set_title(m_name)
            if col == 0:
                ax.set_ylabel(wl)
            
            # Show variance in title
            vals = traces[:, :, m_idx]
            cv = np.std(vals) / (np.mean(vals) + 1e-8)
            ax.text(0.95, 0.95, f'CV={cv:.3f}', transform=ax.transAxes, 
                   ha='right', va='top', fontsize=8)
    
    plt.suptitle('Temporal Patterns in Raw Data (First 100 timesteps)')
    plt.tight_layout()
    plt.savefig(output_dir / 'temporal_patterns.png', dpi=150)
    plt.close()
    print(f"Saved temporal patterns plot to {output_dir / 'temporal_patterns.png'}")


def plot_distributions(data, metadata, output_dir):
    """Plot value distributions for each workload"""
    workload_data = defaultdict(list)
    
    for i, meta in enumerate(metadata):
        wl = meta.get('workload', 'unknown').lower()
        if 'resnet' in wl:
            wl = 'resnet50'
        elif 'distil' in wl:
            wl = 'distilbert'
        elif 'whisper' in wl:
            wl = 'whisper'
        workload_data[wl].append(data[i])
    
    metrics_to_plot = [1, 5, 7, 9]
    metric_names = ['cpu_usage', 'gpu_utilization', 'memory_usage', 'latency']
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    
    for row, wl in enumerate(['resnet50', 'distilbert', 'whisper']):
        traces = np.array(workload_data[wl])
        
        for col, (m_idx, m_name) in enumerate(zip(metrics_to_plot, metric_names)):
            ax = axes[row, col]
            
            vals = traces[:, :, m_idx].flatten()
            ax.hist(vals, bins=50, alpha=0.7, edgecolor='black')
            
            if row == 0:
                ax.set_title(m_name)
            if col == 0:
                ax.set_ylabel(wl)
            
            # Show range
            ax.axvline(np.mean(vals), color='r', linestyle='--', label=f'mean={np.mean(vals):.3f}')
    
    plt.suptitle('Value Distributions by Workload')
    plt.tight_layout()
    plt.savefig(output_dir / 'distributions.png', dpi=150)
    plt.close()
    print(f"Saved distributions plot to {output_dir / 'distributions.png'}")


def main():
    print("="*70)
    print("DATA CHARACTERISTICS DEEP ANALYSIS")
    print("="*70)
    
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    data, metadata = load_data()
    print(f"Loaded {len(data)} traces, shape: {data.shape}")
    
    # Split by workload
    workload_data = defaultdict(list)
    for i, meta in enumerate(metadata):
        wl = meta.get('workload', 'unknown').lower()
        if 'resnet' in wl:
            wl = 'resnet50'
        elif 'distil' in wl:
            wl = 'distilbert'
        elif 'whisper' in wl:
            wl = 'whisper'
        workload_data[wl].append(data[i])
    
    # Analyze each workload
    all_results = {}
    for wl in ['resnet50', 'distilbert', 'whisper']:
        if wl in workload_data:
            traces = np.array(workload_data[wl])
            all_results[wl] = analyze_workload(traces, wl)
    
    # Find best metrics
    find_best_metrics(all_results)
    
    # Visualizations
    print("\n\nGenerating visualizations...")
    plot_temporal_patterns(data, metadata, output_dir)
    plot_distributions(data, metadata, output_dir)
    
    # KEY INSIGHTS
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    
    print("""
QUESTION: Is the data actually flat, or are we losing information?

Based on the analysis above, check:
1. If CV (coefficient of variation) is very low (< 0.01), the data IS nearly constant
2. If temporal_var is very low, there's little change over time
3. If diff_var is very low, consecutive values are almost identical

POSSIBLE OUTCOMES:
A) Data is genuinely flat (baseline workload at constant load)
   -> Model is correct to output constants
   -> Need to reframe thesis: "model captures workload characteristics"

B) Data has signal but current 4 metrics are wrong choices
   -> Try using metrics with higher CV (cpu_psi, gpu_power, etc.)
   -> Different metrics may be better for different workloads

C) Normalization is compressing signal
   -> Use per-workload normalization
   -> Or use raw (unnormalized) data

Check the plots in: """ + str(output_dir))


if __name__ == '__main__':
    main()