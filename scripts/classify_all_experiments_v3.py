#!/usr/bin/env python3
"""
Comprehensive Load Level Classification v3
FIXED: Auto-detects all experiments dynamically
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import re

class LoadClassifier:
    def __init__(self):
        self.data_dir = Path("data/raw/phase1")
        self.results = {}
        
    def discover_experiments(self):
        """Auto-discover all experiments in data directory"""
        workload_configs = {}
        
        # Find all experiment directories
        for exp_dir in self.data_dir.glob("*_r*"):
            match = re.match(r"(.+)_r(\d+)", exp_dir.name)
            if match:
                workload = match.group(1)
                replicas = int(match.group(2))
                
                if workload not in workload_configs:
                    workload_configs[workload] = []
                workload_configs[workload].append(replicas)
        
        # Sort replica counts
        for workload in workload_configs:
            workload_configs[workload].sort()
        
        return workload_configs
        
    def load_metric(self, workload, replicas, metric_name):
        """Load a specific metric CSV file"""
        exp_dir = self.data_dir / f"{workload}_r{replicas}"
        pattern = f"{workload}_r{replicas}_{metric_name}_*.csv"
        
        try:
            csv_file = list(exp_dir.glob(pattern))[0]
            df = pd.read_csv(csv_file)
            return df
        except (IndexError, FileNotFoundError):
            print(f"  ⚠ Warning: {metric_name} not found for {workload} r={replicas}")
            return None
    
    def calculate_metrics(self, workload, replicas):
        """Calculate key metrics for an experiment - FIXED AGGREGATION"""
        metrics = {}
        
        # Latency average - PER-POD metric, needs timestamp grouping
        lat_df = self.load_metric(workload, replicas, "inference_latency_avg")
        if lat_df is not None:
            # FIXED: Group by timestamp first (multiple pods per timestamp)
            lat_by_time = lat_df.groupby('timestamp')['value'].mean()
            valid_latency = lat_by_time[lat_by_time > 0]
            metrics['latency_avg_ms'] = valid_latency.mean() * 1000
            metrics['latency_p95_ms'] = valid_latency.quantile(0.95) * 1000
            metrics['latency_std_ms'] = valid_latency.std() * 1000
        
        # Latency P95 - ALREADY AGGREGATED (histogram quantile)
        lat_p95_df = self.load_metric(workload, replicas, "inference_latency_p95")
        if lat_p95_df is not None:
            metrics['latency_p95_hist_ms'] = lat_p95_df['value'].mean() * 1000
        
        # Throughput - ALREADY AGGREGATED (sum across pods)
        tp_df = self.load_metric(workload, replicas, "inference_throughput")
        if tp_df is not None:
            # This is already cluster-wide total from PromQL: sum(rate(...))
            metrics['throughput_total'] = tp_df['value'].mean()
            metrics['throughput_per_pod'] = metrics['throughput_total'] / replicas
        
        # CPU usage - PER-POD metric, needs timestamp grouping
        cpu_df = self.load_metric(workload, replicas, "cpu_usage")
        if cpu_df is not None:
            # FIXED: Group by timestamp, sum across pods at each timestamp
            cpu_by_time = cpu_df.groupby('timestamp')['value'].sum()
            metrics['cpu_total_cores'] = cpu_by_time.mean()
            metrics['cpu_per_pod_cores'] = metrics['cpu_total_cores'] / replicas
            metrics['cpu_percent_of_system'] = (metrics['cpu_total_cores'] / 16) * 100
        
        # Memory usage - PER-POD metric, needs timestamp grouping
        mem_df = self.load_metric(workload, replicas, "memory_usage")
        if mem_df is not None:
            # FIXED: Group by timestamp, sum across pods
            mem_by_time = mem_df.groupby('timestamp')['value'].sum()
            metrics['memory_total_gb'] = mem_by_time.mean() / (1024**3)
            metrics['memory_per_pod_gb'] = metrics['memory_total_gb'] / replicas
        
        # GPU utilization - DEVICE LEVEL (single time series)
        gpu_df = self.load_metric(workload, replicas, "gpu_utilization")
        if gpu_df is not None:
            metrics['gpu_util_percent'] = gpu_df['value'].mean()
        
        # GPU memory - DEVICE LEVEL
        gpu_mem_df = self.load_metric(workload, replicas, "gpu_memory")
        if gpu_mem_df is not None:
            metrics['gpu_memory_mb'] = gpu_mem_df['value'].mean()
        
        # CPU PSI - PER-POD metric, needs timestamp grouping
        cpu_psi_df = self.load_metric(workload, replicas, "cpu_psi")
        if cpu_psi_df is not None:
            # FIXED: Group by timestamp, average across pods
            psi_by_time = cpu_psi_df.groupby('timestamp')['value'].mean()
            metrics['cpu_psi_percent'] = psi_by_time.mean() * 100
        
        # Memory PSI - PER-POD metric
        mem_psi_df = self.load_metric(workload, replicas, "memory_psi")
        if mem_psi_df is not None:
            psi_by_time = mem_psi_df.groupby('timestamp')['value'].mean()
            metrics['memory_psi_percent'] = psi_by_time.mean() * 100
        
        return metrics
    
    def classify_load_resnet50(self, latency_ratio, throughput_ratio):
        """Classify load for GPU-bound workload (ResNet50)"""
        if latency_ratio < 2.0:
            return "LOW", "Minimal latency increase (<2×)"
        elif latency_ratio < 5.0:
            return "MODERATE", "Noticeable degradation (2-5× latency)"
        elif latency_ratio < 10.0:
            return "HIGH", "Significant degradation (5-10× latency)"
        else:
            return "CRITICAL", "Severe degradation (>10× latency)"
    
    def classify_load_distilbert(self, latency_ratio, cpu_psi, cpu_percent):
        """Classify load for CPU-bound workload (DistilBERT)"""
        if cpu_percent < 30 and latency_ratio < 2.0:
            return "LOW", "Low CPU usage (<30%)"
        elif cpu_percent < 60 and latency_ratio < 5.0:
            return "MODERATE", "Moderate CPU usage (30-60%)"
        elif cpu_percent < 90:
            return "HIGH", "High CPU usage (60-90%)"
        else:
            return "CRITICAL", "CPU saturation (>90%)"
    
    def classify_load_whisper(self, latency_ratio, gpu_util, cpu_percent):
        """Classify load for balanced workload (Whisper)"""
        # Combined score approach
        score = (latency_ratio * 0.4) + (gpu_util/100 * 0.3) + (cpu_percent/100 * 0.3)
        
        if score < 0.7:
            return "LOW", "Low utilization on both resources"
        elif score < 1.2:
            return "MODERATE", "Balanced moderate load"
        elif score < 1.8:
            return "HIGH", "High utilization approaching saturation"
        else:
            return "CRITICAL", "Multiple resource saturation"
    
    def analyze_workload(self, workload, replica_counts):
        """Analyze all experiments for one workload"""
        print(f"\n{'='*70}")
        print(f"  {workload.upper()} - Load Classification")
        print(f"{'='*70}\n")
        
        # Get baseline (r=1)
        baseline = self.calculate_metrics(workload, 1)
        
        if not baseline or 'latency_avg_ms' not in baseline:
            print(f"❌ ERROR: Baseline data (r=1) not found or incomplete for {workload}")
            return None
        
        print(f"Baseline (r=1) Metrics:")
        print(f"  Latency:    {baseline['latency_avg_ms']:.2f} ms")
        print(f"  Throughput: {baseline.get('throughput_per_pod', 0):.2f} inf/s per pod")
        print(f"  CPU:        {baseline.get('cpu_per_pod_cores', 0):.3f} cores per pod")
        print(f"  GPU:        {baseline.get('gpu_util_percent', 0):.1f}%")
        print(f"  Memory:     {baseline.get('memory_per_pod_gb', 0):.2f} GB per pod")
        
        # Analyze each replica count
        results = []
        for replicas in replica_counts:
            if replicas == 1:
                continue
            
            print(f"\n{'-'*70}")
            print(f"Replica Count: r={replicas}")
            print(f"{'-'*70}")
            
            metrics = self.calculate_metrics(workload, replicas)
            
            if not metrics or 'latency_avg_ms' not in metrics:
                print(f"  ❌ Data incomplete for r={replicas}")
                continue
            
            # Calculate ratios
            lat_ratio = metrics['latency_avg_ms'] / baseline['latency_avg_ms']
            tp_ratio = metrics.get('throughput_per_pod', 0) / baseline.get('throughput_per_pod', 1)
            
            print(f"\nPerformance Metrics:")
            print(f"  Latency avg:      {metrics['latency_avg_ms']:.2f} ms ({lat_ratio:.2f}× baseline)")
            print(f"  Latency p95:      {metrics.get('latency_p95_hist_ms', 0):.2f} ms")
            print(f"  Throughput/pod:   {metrics.get('throughput_per_pod', 0):.2f} inf/s ({tp_ratio:.2f}× baseline)")
            print(f"  Total throughput: {metrics.get('throughput_total', 0):.2f} inf/s")
            
            print(f"\nResource Utilization:")
            print(f"  CPU total:     {metrics.get('cpu_total_cores', 0):.2f} cores ({metrics.get('cpu_percent_of_system', 0):.1f}% of 16)")
            print(f"  CPU per pod:   {metrics.get('cpu_per_pod_cores', 0):.3f} cores")
            print(f"  GPU:           {metrics.get('gpu_util_percent', 0):.1f}%")
            print(f"  GPU memory:    {metrics.get('gpu_memory_mb', 0):.0f} MB")
            print(f"  Memory total:  {metrics.get('memory_total_gb', 0):.2f} GB")
            
            print(f"\nContention Indicators:")
            print(f"  CPU PSI:       {metrics.get('cpu_psi_percent', 0):.6f}%")
            print(f"  Memory PSI:    {metrics.get('memory_psi_percent', 0):.6f}% (0 = No memory pressure ✓)")
            
            # Classify load
            if workload == "resnet50":
                load_class, reason = self.classify_load_resnet50(lat_ratio, tp_ratio)
                workload_type = "GPU-bound"
            elif workload == "distilbert":
                load_class, reason = self.classify_load_distilbert(
                    lat_ratio, 
                    metrics.get('cpu_psi_percent', 0),
                    metrics.get('cpu_percent_of_system', 0)
                )
                workload_type = "CPU-bound"
            elif workload == "whisper":
                load_class, reason = self.classify_load_whisper(
                    lat_ratio,
                    metrics.get('gpu_util_percent', 0),
                    metrics.get('cpu_percent_of_system', 0)
                )
                workload_type = "Balanced"
            
            print(f"\n{'='*70}")
            print(f"  CLASSIFICATION: {load_class}")
            print(f"  Type: {workload_type}")
            print(f"  Reason: {reason}")
            print(f"{'='*70}")
            
            results.append({
                'replicas': replicas,
                'latency_ratio': lat_ratio,
                'throughput_ratio': tp_ratio,
                'cpu_percent': metrics.get('cpu_percent_of_system', 0),
                'gpu_percent': metrics.get('gpu_util_percent', 0),
                'load_class': load_class
            })
        
        self.results[workload] = {
            'baseline': baseline,
            'experiments': results
        }
        
        return results
    
    def print_summary(self):
        """Print comprehensive summary"""
        print(f"\n\n{'#'*70}")
        print(f"#  COMPREHENSIVE LOAD CLASSIFICATION SUMMARY")
        print(f"{'#'*70}\n")
        
        for workload in ['resnet50', 'distilbert', 'whisper']:
            if workload not in self.results:
                continue
            
            print(f"\n{workload.upper()}")
            print(f"{'='*70}")
            print(f"{'Replicas':<12} {'Latency':<15} {'Throughput':<15} {'CPU%':<10} {'GPU%':<10} {'Load Class':<15}")
            print(f"{'-'*70}")
            print(f"{'r=1':<12} {'1.00× (base)':<15} {'1.00× (base)':<15} {'-':<10} {'-':<10} {'BASELINE':<15}")
            
            for exp in self.results[workload]['experiments']:
                print(f"r={exp['replicas']:<10} "
                      f"{exp['latency_ratio']:.2f}×{'':<11} "
                      f"{exp['throughput_ratio']:.2f}×{'':<11} "
                      f"{exp['cpu_percent']:.1f}%{'':<5} "
                      f"{exp['gpu_percent']:.1f}%{'':<5} "
                      f"{exp['load_class']:<15}")
        
        print(f"\n{'='*70}\n")
        
        # Check if all load levels represented
        self.check_coverage()
    
    def check_coverage(self):
        """Check if all load levels are represented"""
        print(f"\n{'='*70}")
        print(f"  LOAD LEVEL COVERAGE ANALYSIS")
        print(f"{'='*70}\n")
        
        desired_levels = ['LOW', 'MODERATE', 'HIGH']
        
        for workload in ['resnet50', 'distilbert', 'whisper']:
            if workload not in self.results:
                continue
            
            found_levels = set()
            level_replicas = {}
            
            for exp in self.results[workload]['experiments']:
                found_levels.add(exp['load_class'])
                if exp['load_class'] not in level_replicas:
                    level_replicas[exp['load_class']] = []
                level_replicas[exp['load_class']].append(exp['replicas'])
            
            print(f"{workload.upper()}:")
            
            for level in desired_levels:
                status = "✓" if level in found_levels else "✗"
                print(f"  {status} {level:<12}", end="")
                
                if level in found_levels:
                    replicas = level_replicas[level]
                    print(f" (r={','.join(map(str, replicas))})", end="")
                else:
                    print(f" (MISSING)", end="")
                print()
            
            # Coverage percentage
            coverage = (len(found_levels & set(desired_levels)) / len(desired_levels)) * 100
            print(f"  Coverage: {coverage:.0f}%")
            
            if coverage >= 100:
                print(f"  ✓ COMPLETE - All load scenarios covered!")
            else:
                missing = set(desired_levels) - found_levels
                print(f"  ⚠ Missing: {', '.join(missing)}")
            
            print()
    
    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        # AUTO-DISCOVER all experiments
        print(f"\n{'='*70}")
        print(f"  Auto-discovering experiments in {self.data_dir}")
        print(f"{'='*70}\n")
        
        workload_configs = self.discover_experiments()
        
        for workload, replicas in sorted(workload_configs.items()):
            print(f"  {workload}: r={replicas}")
        
        print(f"\n{'='*70}\n")
        
        for workload, replicas in sorted(workload_configs.items()):
            self.analyze_workload(workload, replicas)
        
        self.print_summary()

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  Phase 1 Experiment Analysis v3: Load Level Classification")
    print("  NEW: Auto-detects all experiments")
    print("="*70)
    
    classifier = LoadClassifier()
    classifier.run_full_analysis()
    
    print("\n" + "="*70)
    print("  Analysis Complete!")
    print("="*70 + "\n")
