#!/usr/bin/env python3
"""
Quick visualization of Phase 1 v3 experiment data
Shows comprehensive metrics: pod-level, node-level, GPU, and application
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

def visualize_experiment(workload, replicas):
    data_dir = Path(f"data/raw/phase1_v3/{workload}_r{replicas}")
    
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return
    
    # Define metrics to plot (organized by category)
    metrics_config = {
        'Pod CPU (cores)': 'pod_cpu_usage',
        'Pod Memory (GB)': 'pod_memory_bytes',
        'GPU Utilization (%)': 'gpu_utilization',
        'GPU Memory (MB)': 'gpu_memory_used',
        'Latency P95 (ms)': 'app_latency_p95',
        'Throughput (req/s)': 'app_throughput',
        'Node CPU (%)': 'node_cpu_usage',
        'PSI CPU (%)': 'pod_psi_cpu',
    }
    
    fig, axes = plt.subplots(4, 2, figsize=(16, 14))
    fig.suptitle(f'{workload.upper()} - {replicas} replica(s) | Phase 1 v3', 
                 fontsize=16, fontweight='bold')
    
    for idx, (title, metric) in enumerate(metrics_config.items()):
        ax = axes[idx // 2, idx % 2]
        
        # Find the metric file
        files = list(data_dir.glob(f"*_{metric}_*.csv"))
        if not files:
            ax.text(0.5, 0.5, f'No data for {metric}', 
                   ha='center', va='center', fontsize=10, color='red')
            ax.set_title(title, fontsize=11)
            continue
        
        df = pd.read_csv(files[0])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate minutes from start
        start_time = df['timestamp'].min()
        df['minutes'] = (df['timestamp'] - start_time).dt.total_seconds() / 60
        
        # Unit conversions
        if 'memory_bytes' in metric:
            df['value'] = df['value'] / (1024**3)  # Convert to GB
        elif 'gpu_memory' in metric:
            df['value'] = df['value']  # Already in MB
        elif 'latency' in metric:
            df['value'] = df['value'] * 1000  # Convert to ms
        elif 'node_cpu' in metric or 'psi' in metric:
            df['value'] = df['value'] * 100  # Convert to percentage
        
        # Plot based on whether it's pod-level or system-level
        if 'pod' in df.columns:
            # Per-pod lines
            for pod in df['pod'].unique():
                pod_data = df[df['pod'] == pod].sort_values('minutes')
                ax.plot(pod_data['minutes'], pod_data['value'], 
                       alpha=0.6, linewidth=1.5, label=pod.split('-')[-1][:8])
        else:
            # System-level metric
            df_sorted = df.sort_values('minutes')
            ax.plot(df_sorted['minutes'], df_sorted['value'], 
                   'b-', linewidth=2, alpha=0.8)
        
        # Styling
        ax.set_xlabel('Time (minutes)', fontsize=9)
        ax.set_ylabel(title, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlim(0, 60)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add legend for pod-level metrics
        if 'pod' in df.columns and replicas > 1:
            ax.legend(loc='upper right', fontsize=7, framealpha=0.9)
        
        # Y-axis formatting
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
    
    plt.tight_layout()
    
    # Save
    output_file = data_dir / f"{workload}_r{replicas}_visualization.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"\nVisualization saved: {output_file}")
    
    # Print summary statistics
    print(f"\nSummary Statistics for {workload} r={replicas}:")
    print("="*60)
    
    for title, metric in metrics_config.items():
        files = list(data_dir.glob(f"*_{metric}_*.csv"))
        if files:
            df = pd.read_csv(files[0])
            
            # Apply same conversions
            if 'memory_bytes' in metric:
                df['value'] = df['value'] / (1024**3)
            elif 'latency' in metric:
                df['value'] = df['value'] * 1000
            elif 'node_cpu' in metric or 'psi' in metric:
                df['value'] = df['value'] * 100
            
            mean_val = df['value'].mean()
            max_val = df['value'].max()
            min_val = df['value'].min()
            
            print(f"{title:25s}: mean={mean_val:8.2f}  max={max_val:8.2f}  min={min_val:8.2f}")
    
    print("="*60)
    
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 tools/visualize_experiment_v3.py <workload> <replicas>")
        print("\nExamples:")
        print("  python3 tools/visualize_experiment_v3.py resnet152 1")
        print("  python3 tools/visualize_experiment_v3.py bert 3")
        print("  python3 tools/visualize_experiment_v3.py whisper 8")
        sys.exit(1)
    
    workload = sys.argv[1]
    replicas = int(sys.argv[2])
    visualize_experiment(workload, replicas)