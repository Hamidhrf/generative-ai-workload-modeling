#!/usr/bin/env python3
"""
Single Experiment Runner - Phase 1 (v1.1 - ALL ISSUES FIXED)
Run one experiment at a time with full control

FIXES APPLIED:
1. inference_latency_avg: Uses rate() properly ✓
2. Histogram quantiles: Aggregate across pods with sum by (le) ✓
3. inference_total: Now uses rate() for consistency ✓
4. Query window: 30s buffer for scrape lag ✓
"""

import subprocess
import time
import sys
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

class SingleExperimentRunner:
    def __init__(self, workload, replicas, prometheus_url="http://172.22.174.58:30090"):
        self.workload = workload
        self.replicas = replicas
        self.prometheus_url = prometheus_url
        
        # Configuration
        self.workloads = {
            "resnet50": {
                "deployment": "k8s/workloads/resnet50-deployment.yaml",
                "app_label": "resnet50",
                "namespace": "default"
            },
            "distilbert": {
                "deployment": "k8s/workloads/distilbert-deployment.yaml",
                "app_label": "distilbert",
                "namespace": "default"
            },
            "whisper": {
                "deployment": "k8s/workloads/whisper-deployment.yaml",
                "app_label": "whisper",
                "namespace": "default"
            }
        }
        
        if workload not in self.workloads:
            print(f"Error: Unknown workload '{workload}'")
            print(f"Available: {list(self.workloads.keys())}")
            sys.exit(1)
        
        # Timing configuration
        self.startup_delay = 300  # 5 minutes for safety
        self.experiment_duration = 3600  # 60 minutes
        self.cleanup_delay = 30  # 30 seconds
        
        # Output directory
        self.data_dir = Path("data/raw/phase1")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics to collect (ALL ISSUES FIXED)
        self.metrics = {
            # Per-pod resource metrics (cAdvisor)
            'cpu_usage': f'rate(container_cpu_usage_seconds_total{{pod=~"{workload}-inference.*",container="{workload}"}}[1m])',
            'memory_usage': f'container_memory_working_set_bytes{{pod=~"{workload}-inference.*",container="{workload}"}}',
    
            # Device-level GPU metrics (DCGM Exporter)
            # NOTE: These are device-level and reflect aggregate GPU usage across ALL pods
            # With GPU time-slicing, all pods share the same GPU device
            'gpu_utilization': 'DCGM_FI_DEV_GPU_UTIL{gpu="0"}',
            'gpu_memory': 'DCGM_FI_DEV_FB_USED{gpu="0"}',
            'gpu_power': 'DCGM_FI_DEV_POWER_USAGE{gpu="0"}',
            'gpu_temperature': 'DCGM_FI_DEV_GPU_TEMP{gpu="0"}',
            
            # Per-pod pressure metrics (PSI - requires cgroup v2)
            'cpu_psi': f'rate(container_pressure_cpu_waiting_seconds_total{{pod=~"{workload}-inference.*",container="{workload}"}}[1m])',
            'memory_psi': f'rate(container_pressure_memory_waiting_seconds_total{{pod=~"{workload}-inference.*",container="{workload}"}}[1m])',
            'io_psi': f'rate(container_pressure_io_waiting_seconds_total{{pod=~"{workload}-inference.*",container="{workload}"}}[1m])',

            # Application-level inference metrics (ALL FIXED!)
            
            # FIX 1: Average latency - uses rate() properly
            'inference_latency_avg': (
                f'rate({workload}_inference_latency_seconds_sum[1m]) / '
                f'rate({workload}_inference_latency_seconds_count[1m])'
            ),
            
            # FIX 2: Histogram quantiles - aggregate across pods
            'inference_latency_p50': (
                f'histogram_quantile(0.50, '
                f'sum by (le) (rate({workload}_inference_latency_seconds_bucket[1m])))'
            ),
            'inference_latency_p95': (
                f'histogram_quantile(0.95, '
                f'sum by (le) (rate({workload}_inference_latency_seconds_bucket[1m])))'
            ),
            'inference_latency_p99': (
                f'histogram_quantile(0.99, '
                f'sum by (le) (rate({workload}_inference_latency_seconds_bucket[1m])))'
            ),
            
            # FIX 3: Throughput and total both use rate() for consistency
            'inference_throughput': f'sum(rate({workload}_inference_total[1m]))',
            'inference_total': f'sum(rate({workload}_inference_total[1m]))',  # Rate, not raw counter
        }
    
    def run_cmd(self, cmd, check=True):
        """Execute shell command"""
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, check=check
        )
        return result.stdout.strip()
    
    def print_header(self, text):
        """Print formatted header"""
        print(f"\n{'='*70}")
        print(f"  {text}")
        print(f"{'='*70}\n")
    
    def check_prerequisites(self):
        """Check system is ready for experiment"""
        self.print_header("Pre-Experiment Checks")
        
        issues = []
        
        # Check Prometheus
        try:
            response = requests.get(f"{self.prometheus_url}/-/healthy", timeout=5)
            if response.status_code == 200:
                print("✓ Prometheus is healthy")
            else:
                issues.append("Prometheus not responding correctly")
        except Exception as e:
            issues.append(f"Cannot connect to Prometheus at {self.prometheus_url}: {e}")
        
        # Check no workload pods running
        output = self.run_cmd(
            f"kubectl get pods -l 'app in (resnet50,distilbert,whisper)' --no-headers",
            check=False
        )
        if output:
            print(f"⚠ Warning: Found existing workload pods:")
            print(output)
            response = input("\nDelete them and continue? (yes/no): ")
            if response.lower() == 'yes':
                self.cleanup_all_workloads()
            else:
                issues.append("Existing workload pods must be removed first")
        else:
            print("✓ No existing workload pods")
        
        # Check memory
        output = self.run_cmd("kubectl top node --no-headers", check=False)
        if output:
            parts = output.split()
            if len(parts) >= 5:
                memory_pct = int(parts[4].replace('%', ''))
                if memory_pct > 85:
                    issues.append(f"Node memory at {memory_pct}% - too high")
                else:
                    print(f"✓ Node memory OK ({memory_pct}%)")
        
        if issues:
            print(f"\n❌ Cannot proceed due to issues:")
            for issue in issues:
                print(f"   - {issue}")
            return False
        
        print("\n✓ All pre-checks passed!")
        return True
    
    def cleanup_all_workloads(self):
        """Clean up any running workload deployments"""
        print("\nCleaning up workloads...")
        for wl in self.workloads.keys():
            self.run_cmd(
                f"kubectl delete deployment {wl}-inference --ignore-not-found=true",
                check=False
            )
        time.sleep(10)
    
    def deploy(self):
        """Deploy workload with specified replica count"""
        self.print_header(f"Deploying {self.workload} × {self.replicas} replicas")
        
        config = self.workloads[self.workload]
        
        # Apply deployment
        print(f"Applying deployment from {config['deployment']}...")
        self.run_cmd(f"kubectl apply -f {config['deployment']}")
        
        # Scale to desired replicas
        print(f"Scaling to {self.replicas} replicas...")
        self.run_cmd(
            f"kubectl scale deployment {self.workload}-inference "
            f"--replicas={self.replicas}"
        )
        
        # Wait for pods to be ready
        print(f"\nWaiting for {self.replicas} pods to be ready...")
        timeout = 300  # 5 minutes
        start = time.time()
        
        while time.time() - start < timeout:
            output = self.run_cmd(
                f"kubectl get pods -l app={config['app_label']} --no-headers",
                check=False
            )
            
            if not output:
                print("  No pods found yet...")
                time.sleep(5)
                continue
            
            lines = output.strip().split('\n')
            ready_count = 0
            
            for line in lines:
                parts = line.split()
                if len(parts) >= 3:
                    ready_status = parts[1]
                    pod_status = parts[2]
                    
                    ready_parts = ready_status.split('/')
                    if (pod_status == "Running" and 
                        ready_parts[0] == ready_parts[1]):
                        ready_count += 1
            
            print(f"  Ready: {ready_count}/{self.replicas} pods", end='\r')
            
            if ready_count == self.replicas:
                print(f"\n✓ All {self.replicas} pods are ready!")
                return True
            
            time.sleep(5)
        
        print(f"\n❌ Timeout waiting for pods")
        return False
    
    def startup_stabilization(self):
        """Wait for workload to stabilize"""
        self.print_header(
            f"Startup Stabilization ({self.startup_delay}s = "
            f"{self.startup_delay//60} minutes)"
        )
        
        print("Waiting for:")
        print("  • Model loading into GPU memory")
        print("  • Inference loop warmup")
        print("  • Metrics reporting stabilization")
        print()
        
        # Progress bar
        interval = 30  # Update every 30 seconds
        for elapsed in range(0, self.startup_delay, interval):
            remaining = self.startup_delay - elapsed
            progress = elapsed / self.startup_delay * 100
            bar_length = 40
            filled = int(bar_length * progress / 100)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            print(f"  [{bar}] {progress:5.1f}% - {remaining}s remaining", end='\r')
            time.sleep(interval)
        
        print(f"  [{'█'*40}] 100.0% - Stabilization complete!  \n")
        print("✓ Workload is now in steady state")
    
    def record_experiment(self):
        """Record 60-minute experiment"""
        self.print_header(f"Recording Experiment ({self.experiment_duration//60} minutes)")
        
        # Record start time
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=self.experiment_duration)
        
        print(f"Start time:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End time:    {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration:    {self.experiment_duration}s ({self.experiment_duration//60} minutes)")
        print()
        
        # Create experiment-specific directory
        exp_dir = self.data_dir / f"{self.workload}_r{self.replicas}"
        exp_dir.mkdir(exist_ok=True)
        
        # Save timestamps to file for reference
        timestamp_file = (
            exp_dir / 
            f"{self.workload}_r{self.replicas}_"
            f"{start_time.strftime('%Y%m%d_%H%M%S')}_timestamps.txt"
        )
        with open(timestamp_file, 'w') as f:
            f.write(f"Workload: {self.workload}\n")
            f.write(f"Replicas: {self.replicas}\n")
            f.write(f"Start: {start_time.isoformat()}\n")
            f.write(f"End: {end_time.isoformat()}\n")
            f.write(f"Duration: {self.experiment_duration}s\n")
        
        print(f"Timestamps saved to: {timestamp_file}\n")
        
        # Progress updates every 5 minutes
        interval = 300
        for elapsed in range(0, self.experiment_duration, interval):
            remaining = self.experiment_duration - elapsed
            progress = elapsed / self.experiment_duration * 100
            minutes_elapsed = elapsed // 60
            minutes_remaining = remaining // 60
            
            print(
                f"  Progress: {progress:5.1f}% | "
                f"Elapsed: {minutes_elapsed:2d} min | "
                f"Remaining: {minutes_remaining:2d} min"
            )
            
            time.sleep(interval)
        
        print(f"  Progress: 100.0% | Elapsed: {self.experiment_duration//60} min | "
              f"Remaining:  0 min")
        print("\n✓ Recording complete!")
        
        return start_time, end_time
    
    def query_prometheus(self, query, start, end):
        """
        Query Prometheus for time range
        
        FIX 4: Applies 30-second buffer to end time to account for scrape lag
        """
        # Apply temporal buffer to avoid missing tail samples
        buffered_end = end - timedelta(seconds=30)
        
        url = f"{self.prometheus_url}/api/v1/query_range"
        params = {
            'query': query,
            'start': start.timestamp(),
            'end': buffered_end.timestamp(),  # Use buffered end time
            'step': '5s'  # 5-second resolution
        }
        
        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"  ✗ Query failed: {e}")
            return None
    
    def export_to_csv(self, data, filename):
        """Export Prometheus data to CSV"""
        if not data or data.get('status') != 'success':
            return False
        
        results = data['data']['result']
        if not results:
            return False
        
        rows = []
        for result in results:
            metric = result['metric']
            for timestamp, value in result['values']:
                row = {
                    'timestamp': datetime.fromtimestamp(float(timestamp)),
                    'value': float(value)
                }
                row.update(metric)
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df = df.sort_values('timestamp')
        df.to_csv(filename, index=False)
        
        return True
    
    def collect_metrics(self, start_time, end_time):
        """Collect all metrics from Prometheus"""
        self.print_header("Collecting Metrics from Prometheus")
        
        timestamp = start_time.strftime("%Y%m%d_%H%M%S")
        
        # Create experiment-specific directory
        exp_dir = self.data_dir / f"{self.workload}_r{self.replicas}"
        exp_dir.mkdir(exist_ok=True)
        
        success_count = 0
        total_count = len(self.metrics)
        
        for metric_name, query in self.metrics.items():
            print(f"  [{metric_name:20s}] ", end='', flush=True)
            
            data = self.query_prometheus(query, start_time, end_time)
            
            if data:
                filename = (
                    exp_dir / 
                    f"{self.workload}_r{self.replicas}_{metric_name}_{timestamp}.csv"
                )
                
                if self.export_to_csv(data, filename):
                    file_size = filename.stat().st_size / 1024  # KB
                    print(f"✓ Exported ({file_size:.1f} KB)")
                    success_count += 1
                else:
                    print(f"✗ No data")
            else:
                print(f"✗ Query failed")
        
        print(f"\n✓ Collected {success_count}/{total_count} metrics")
        print(f"\nData saved to: {exp_dir}")
    
    def cleanup(self):
        """Remove deployment"""
        self.print_header("Cleanup")
        
        print(f"Deleting {self.workload} deployment...")
        self.run_cmd(
            f"kubectl delete deployment {self.workload}-inference",
            check=False
        )
        
        print(f"Waiting {self.cleanup_delay}s for cleanup...")
        time.sleep(self.cleanup_delay)
        
        print("✓ Cleanup complete")
    
    def run(self):
        """Run complete experiment"""
        print("\n" + "="*70)
        print(f"  PHASE 1 EXPERIMENT (v1.1)")
        print(f"  Workload: {self.workload}")
        print(f"  Replicas: {self.replicas}")
        print("="*70)
        
        # Calculate total time
        total_time = (
            self.startup_delay + 
            self.experiment_duration + 
            self.cleanup_delay + 
            120  # Estimate for metric collection
        )
        print(f"\nEstimated duration: {total_time//60} minutes\n")
        
        # Confirm
        response = input("Start experiment? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            return
        
        try:
            # Pre-checks
            if not self.check_prerequisites():
                return
            
            # Deploy
            if not self.deploy():
                print("❌ Deployment failed")
                return
            
            # Stabilization
            self.startup_stabilization()
            
            # Record
            start_time, end_time = self.record_experiment()
            
            # Collect metrics
            self.collect_metrics(start_time, end_time)
            
            # Cleanup
            self.cleanup()
            
            # Summary
            self.print_header("EXPERIMENT COMPLETE")
            print(f"Workload: {self.workload}")
            print(f"Replicas: {self.replicas}")
            print(f"Data location: {self.data_dir / f'{self.workload}_r{self.replicas}'}")
            print(f"\n✓ Experiment successful!")
            
        except KeyboardInterrupt:
            print("\n\n⚠ Experiment interrupted by user")
            print("Cleaning up...")
            self.cleanup()
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Error: {e}")
            self.cleanup()
            sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 tools/run_single_experiment.py <workload> <replicas>")
        print("\nExamples:")
        print("  python3 tools/run_single_experiment.py resnet50 1")
        print("  python3 tools/run_single_experiment.py distilbert 6")
        print("  python3 tools/run_single_experiment.py whisper 3")
        print("\nAvailable workloads: resnet50, distilbert, whisper")
        print("Replica counts: 1, 2, 3, 6, 8, 16")
        sys.exit(1)
    
    workload = sys.argv[1]
    replicas = int(sys.argv[2])
    
    runner = SingleExperimentRunner(workload, replicas)
    runner.run()
