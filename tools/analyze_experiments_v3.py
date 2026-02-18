#!/usr/bin/env python3
"""
Phase 1 v3 Experiment Analysis
Comprehensive statistical analysis of all collected metrics

Analyzes all experiments and generates:
1. Summary statistics per experiment
2. Data quality report (completeness, outliers)
3. Contention analysis (resource usage vs replica count)
4. CSV summary for all experiments

Usage:
  python3 tools/analyze_experiments_v3.py
  python3 tools/analyze_experiments_v3.py --workload resnet152
  python3 tools/analyze_experiments_v3.py --workload bert --replicas 3
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys

class ExperimentAnalyzer:
    def __init__(self, data_dir="data/raw/phase1_v3"):
        self.data_dir = Path(data_dir)
        
        # Expected metrics
        self.pod_metrics = [
            'pod_cpu_usage', 'pod_memory_bytes', 'pod_psi_cpu',
            'pod_psi_memory', 'pod_psi_io', 'pod_latency_avg', 'pod_throughput'
        ]
        self.node_metrics = [
            'node_cpu_usage', 'node_memory_used_percent', 'node_memory_available_bytes',
            'node_psi_cpu', 'node_psi_memory', 'node_psi_io'
        ]
        self.gpu_metrics = [
            'gpu_utilization', 'gpu_memory_used', 'gpu_memory_total',
            'gpu_power_watts', 'gpu_temperature'
        ]
        self.app_metrics = [
            'app_latency_p50', 'app_latency_p95', 'app_latency_p99', 'app_throughput'
        ]
        
        self.all_metrics = (
            self.pod_metrics + self.node_metrics + 
            self.gpu_metrics + self.app_metrics
        )
    
    def find_experiments(self, workload=None, replicas=None):
        """Find all experiment directories matching criteria."""
        experiments = []
        
        for exp_dir in self.data_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            parts = exp_dir.name.split('_r')
            if len(parts) != 2:
                continue
            
            wl = parts[0]
            r = int(parts[1])
            
            if workload and wl != workload:
                continue
            if replicas and r != replicas:
                continue
            
            experiments.append({
                'workload': wl,
                'replicas': r,
                'path': exp_dir
            })
        
        return sorted(experiments, key=lambda x: (x['workload'], x['replicas']))
    
    def load_metric(self, exp_dir, metric_name):
        """Load a single metric CSV."""
        files = list(exp_dir.glob(f"*_{metric_name}_*.csv"))
        if not files:
            return None
        
        try:
            df = pd.read_csv(files[0])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            print(f"  [WARN] Failed to load {metric_name}: {e}")
            return None
    
    def analyze_metric(self, df, metric_name):
        """Compute statistics for a metric."""
        if df is None or len(df) == 0:
            return None
        
        stats = {
            'samples': len(df),
            'mean': df['value'].mean(),
            'std': df['value'].std(),
            'min': df['value'].min(),
            'max': df['value'].max(),
            'p50': df['value'].quantile(0.5),
            'p95': df['value'].quantile(0.95),
            'p99': df['value'].quantile(0.99),
        }
        
        # Check for issues
        stats['missing_pct'] = (df['value'].isna().sum() / len(df)) * 100
        stats['zero_pct'] = ((df['value'] == 0).sum() / len(df)) * 100
        
        # Pod-level: count unique pods
        if 'pod' in df.columns:
            stats['unique_pods'] = df['pod'].nunique()
        
        return stats
    
    def analyze_experiment(self, exp):
        """Analyze single experiment."""
        print(f"\n{'='*70}")
        print(f"  {exp['workload'].upper()} - {exp['replicas']} replica(s)")
        print(f"{'='*70}")
        
        results = {
            'workload': exp['workload'],
            'replicas': exp['replicas'],
            'metrics': {}
        }
        
        # Analyze each metric
        for metric in self.all_metrics:
            df = self.load_metric(exp['path'], metric)
            stats = self.analyze_metric(df, metric)
            
            if stats:
                results['metrics'][metric] = stats
                
                # Print summary
                if metric in self.pod_metrics:
                    print(f"  {metric:25s}: {stats['unique_pods']} pods, "
                          f"mean={stats['mean']:.4f}, max={stats['max']:.4f}")
                else:
                    print(f"  {metric:25s}: mean={stats['mean']:.4f}, "
                          f"max={stats['max']:.4f}, samples={stats['samples']}")
            else:
                print(f"  {metric:25s}: [MISSING]")
                results['metrics'][metric] = None
        
        # Data quality summary
        total_metrics = len(self.all_metrics)
        collected = sum(1 for v in results['metrics'].values() if v is not None)
        print(f"\n  Data Quality: {collected}/{total_metrics} metrics collected "
              f"({collected/total_metrics*100:.0f}%)")
        
        return results
    
    def compare_replicas(self, experiments):
        """Compare metrics across replica counts for same workload."""
        if len(experiments) < 2:
            return
        
        workload = experiments[0]['workload']
        print(f"\n{'='*70}")
        print(f"  REPLICA SCALING ANALYSIS: {workload.upper()}")
        print(f"{'='*70}\n")
        
        # Focus on key contention metrics
        key_metrics = [
            'pod_cpu_usage', 'pod_latency_avg', 
            'gpu_utilization', 'node_cpu_usage'
        ]
        
        for metric in key_metrics:
            print(f"{metric}:")
            print(f"  {'Replicas':>10s} {'Mean':>12s} {'Max':>12s} {'Std':>12s}")
            print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*12}")
            
            for exp in experiments:
                df = self.load_metric(exp['path'], metric)
                if df is not None:
                    mean = df['value'].mean()
                    max_val = df['value'].max()
                    std = df['value'].std()
                    print(f"  {exp['replicas']:>10d} {mean:>12.4f} {max_val:>12.4f} {std:>12.4f}")
            print()
    
    def export_summary(self, all_results, output_file="data/processed/phase1_v3_summary.csv"):
        """Export summary statistics to CSV."""
        rows = []
        
        for result in all_results:
            base = {
                'workload': result['workload'],
                'replicas': result['replicas']
            }
            
            for metric_name, stats in result['metrics'].items():
                if stats:
                    row = base.copy()
                    row['metric'] = metric_name
                    row.update(stats)
                    rows.append(row)
        
        df = pd.DataFrame(rows)
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print(f"\nSummary exported to: {output_path}")
        return output_path
    
    def run(self, workload=None, replicas=None):
        """Run complete analysis."""
        print("\n" + "="*70)
        print("  PHASE 1 v3 EXPERIMENT ANALYSIS")
        print("="*70)
        
        # Find experiments
        experiments = self.find_experiments(workload, replicas)
        
        if not experiments:
            print("\n[ERROR] No experiments found!")
            if workload:
                print(f"  Workload filter: {workload}")
            if replicas:
                print(f"  Replicas filter: {replicas}")
            return
        
        print(f"\nFound {len(experiments)} experiment(s)")
        
        # Analyze each
        all_results = []
        for exp in experiments:
            result = self.analyze_experiment(exp)
            all_results.append(result)
        
        # Compare replica scaling per workload
        if not workload:
            workloads = set(exp['workload'] for exp in experiments)
            for wl in sorted(workloads):
                wl_exps = [e for e in experiments if e['workload'] == wl]
                if len(wl_exps) > 1:
                    self.compare_replicas(wl_exps)
        
        # Export summary
        self.export_summary(all_results)
        
        # Final summary
        print(f"\n{'='*70}")
        print("  ANALYSIS COMPLETE")
        print(f"{'='*70}")
        print(f"Total experiments analyzed: {len(experiments)}")
        print(f"Workloads: {', '.join(sorted(set(e['workload'] for e in experiments)))}")
        print(f"Replica counts: {sorted(set(e['replicas'] for e in experiments))}")
        print()


def main():
    parser = argparse.ArgumentParser(description='Analyze Phase 1 v3 experiments')
    parser.add_argument('--workload', type=str, help='Filter by workload')
    parser.add_argument('--replicas', type=int, help='Filter by replica count')
    parser.add_argument('--data-dir', type=str, default='data/raw/phase1_v3',
                       help='Data directory (default: data/raw/phase1_v3)')
    
    args = parser.parse_args()
    
    analyzer = ExperimentAnalyzer(args.data_dir)
    analyzer.run(args.workload, args.replicas)


if __name__ == "__main__":
    main()