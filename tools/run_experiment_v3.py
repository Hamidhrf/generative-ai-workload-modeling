#!/usr/bin/env python3
"""
Phase 1 v3 Experiment Runner
Comprehensive metrics collection with interactive UX

METRICS COLLECTED:
- Pod-level: CPU, memory, PSI (per container)
- Node-level: CPU, memory, PSI (system-wide)
- GPU: Utilization, memory, power, temperature (device-level)
- Application: Latency histograms, throughput, load phase

OUTPUT STRUCTURE:
  data/raw/phase1_v3/{app}_r{replicas}/
    - {app}_r{replicas}_{metric}_{timestamp}.csv (one per metric)
    - {app}_r{replicas}_{timestamp}_timestamps.txt (metadata)

USAGE:
  python run_experiment_v3.py resnet152 3
  python run_experiment_v3.py bert 8
"""

import os
import subprocess
import time
import sys
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

class ExperimentRunnerV3:
    def __init__(self, workload, replicas, prometheus_url=os.environ.get("PROMETHEUS_URL", "http://172.22.174.58:30090")):
        self.workload = workload
        self.replicas = replicas
        self.prometheus_url = prometheus_url
        
        # Workload configurations
        self.workloads = {
            'resnet152': {
                'deployment': 'resnet152-inference',
                'app_label': 'resnet152',
                'metrics_prefix': 'resnet152',
                'namespace': 'default'
            },
            'bert': {
                'deployment': 'bert-inference',
                'app_label': 'bert',
                'metrics_prefix': 'bert',
                'namespace': 'default'
            },
            'whisper': {
                'deployment': 'whisper-inference',
                'app_label': 'whisper',
                'metrics_prefix': 'whisper',
                'namespace': 'default'
            },
            'yolo': {
                'deployment': 'yolo-inference',
                'app_label': 'yolo',
                'metrics_prefix': 'yolo',
                'namespace': 'default'
            },
            'gpt2': {
                'deployment': 'gpt2-inference',
                'app_label': 'gpt2',
                'metrics_prefix': 'gpt2',
                'namespace': 'default'
            }
        }
        
        if workload not in self.workloads:
            print(f"Error: Unknown workload '{workload}'")
            print(f"Available: {list(self.workloads.keys())}")
            sys.exit(1)
        
        self.config = self.workloads[workload]
        
        # Timing configuration
        self.warmup_duration = 300  # 5 minutes
        self.experiment_duration = 3600  # 60 minutes
        self.cleanup_delay = 30
        
        # Output directory
        self.data_dir = Path(os.environ.get("DATA_OUTPUT_DIR", "data/raw/phase1_v3"))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Scrape interval
        self.scrape_interval = 5  # seconds
        
        # Generate metrics queries
        self.metrics = self._generate_queries()
    
    def _generate_queries(self):
        """Generate Prometheus queries for all metrics"""
        label = self.config['app_label']
        prefix = self.config['metrics_prefix']
        
        return {
            # Pod-level metrics
            'pod_cpu_usage': f'''
                sum by (pod) (
                    rate(container_cpu_usage_seconds_total{{
                        namespace="default",
                        pod=~"{label}-.*",
                        container!="POD",
                        container!=""
                    }}[1m])
                )
            ''',
            'pod_memory_bytes': f'''
                sum by (pod) (
                    container_memory_working_set_bytes{{
                        namespace="default",
                        pod=~"{label}-.*",
                        container!="POD",
                        container!=""
                    }}
                )
            ''',
            'pod_psi_cpu': f'''
                sum by (pod) (
                    rate(container_pressure_cpu_waiting_seconds_total{{
                        namespace="default",
                        pod=~"{label}-.*"
                    }}[1m])
                )
            ''',
            'pod_psi_memory': f'''
                sum by (pod) (
                    rate(container_pressure_memory_waiting_seconds_total{{
                        namespace="default",
                        pod=~"{label}-.*"
                    }}[1m])
                )
            ''',
            'pod_psi_io': f'''
                sum by (pod) (
                    rate(container_pressure_io_waiting_seconds_total{{
                        namespace="default",
                        pod=~"{label}-.*"
                    }}[1m])
                )
            ''',

            # Node-level metrics
            'node_cpu_usage': '1 - avg(rate(node_cpu_seconds_total{mode="idle"}[1m]))',
            'node_memory_used_percent': '1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)',
            'node_memory_available_bytes': 'node_memory_MemAvailable_bytes',
            'node_psi_cpu': 'rate(node_pressure_cpu_waiting_seconds_total[1m])',
            'node_psi_memory': 'rate(node_pressure_memory_waiting_seconds_total[1m])',
            'node_psi_io': 'rate(node_pressure_io_waiting_seconds_total[1m])',

            # Per-pod application metrics
            'pod_latency_avg': f'''
                rate({prefix}_inference_latency_seconds_sum{{pod=~"{label}-.*"}}[1m]) / 
                rate({prefix}_inference_latency_seconds_count{{pod=~"{label}-.*"}}[1m])
            ''',
            'pod_throughput': f'''
                rate({prefix}_inference_total{{pod=~"{label}-.*"}}[1m])
            ''',

            # GPU metrics (device-level, explicit gpu="0")
            'gpu_utilization': 'DCGM_FI_DEV_GPU_UTIL{gpu="0"}',
            'gpu_memory_used': 'DCGM_FI_DEV_FB_USED{gpu="0"}',
            'gpu_memory_total': 'DCGM_FI_DEV_FB_FREE{gpu="0"} + DCGM_FI_DEV_FB_USED{gpu="0"}',
            'gpu_power_watts': 'DCGM_FI_DEV_POWER_USAGE{gpu="0"}',
            'gpu_temperature': 'DCGM_FI_DEV_GPU_TEMP{gpu="0"}',

            # Aggregated application metrics (for reference)
            'app_latency_p50': f'''
                histogram_quantile(0.5,
                    sum(rate({prefix}_inference_latency_seconds_bucket[1m])) by (le)
                )
            ''',
            'app_latency_p95': f'''
                histogram_quantile(0.95,
                    sum(rate({prefix}_inference_latency_seconds_bucket[1m])) by (le)
                )
            ''',
            'app_latency_p99': f'''
                histogram_quantile(0.99,
                    sum(rate({prefix}_inference_latency_seconds_bucket[1m])) by (le)
                )
            ''',
            'app_throughput': f'''
                sum(rate({prefix}_inference_total[1m]))
            ''',
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
                print("[OK] Prometheus is healthy")
            else:
                issues.append("Prometheus not responding correctly")
        except Exception as e:
            issues.append(f"Cannot connect to Prometheus at {self.prometheus_url}: {e}")
        
        # Check no workload pods running
        output = self.run_cmd(
            f"kubectl get pods -l 'app in (resnet152,bert,whisper,yolo,gpt2)' --no-headers",
            check=False
        )
        if output:
            print(f"[WARN] Found existing workload pods:")
            print(output)
            response = input("\nDelete them and continue? (yes/no): ")
            if response.lower() == 'yes':
                self.cleanup_all_workloads()
            else:
                issues.append("Existing workload pods must be removed first")
        else:
            print("[OK] No existing workload pods")
        
        # Check memory
        output = self.run_cmd("kubectl top node --no-headers", check=False)
        if output:
            parts = output.split()
            if len(parts) >= 5:
                memory_pct = int(parts[4].replace('%', ''))
                if memory_pct > 85:
                    issues.append(f"Node memory at {memory_pct}% - too high")
                else:
                    print(f"[OK] Node memory OK ({memory_pct}%)")
        
        if issues:
            print(f"\n[ERROR] Cannot proceed due to issues:")
            for issue in issues:
                print(f"   - {issue}")
            return False
        
        print("\n[OK] All pre-checks passed!")
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
        self.print_header(f"Deploying {self.workload} x {self.replicas} replicas")
        
        deployment_name = self.config['deployment']
        
        # Check if deployment exists, create if not
        existing = self.run_cmd(
            f"kubectl get deployment {deployment_name} --no-headers 2>/dev/null",
            check=False
        )
        
        if not existing:
            print(f"Deployment {deployment_name} does not exist, creating...")
            yaml_path = f"k8s/workloads/{self.workload}-deployment.yaml"
            result = self.run_cmd(f"kubectl apply -f {yaml_path}", check=False)
            if "created" not in result and "configured" not in result:
                print(f"[FAIL] Could not create deployment")
                return False
            print("[OK] Deployment created")
            time.sleep(5)
        
        # Scale to desired replicas
        print(f"Scaling to {self.replicas} replicas...")
        self.run_cmd(
            f"kubectl scale deployment {deployment_name} --replicas={self.replicas}"
        )
        
        # Wait for pods to be ready
        print(f"\nWaiting for {self.replicas} pods to be ready...")
        timeout = 300
        start = time.time()
        
        while time.time() - start < timeout:
            output = self.run_cmd(
                f"kubectl get pods -l app={self.config['app_label']} --no-headers",
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
                print(f"\n[OK] All {self.replicas} pods are ready!")
                return True
            
            time.sleep(5)
        
        print(f"\n[FAIL] Timeout waiting for pods")
        return False
    
    def warmup_period(self):
        """Wait for workload to stabilize with progress bar"""
        self.print_header(
            f"Warmup Period ({self.warmup_duration}s = {self.warmup_duration//60} minutes)"
        )
        
        print("Waiting for:")
        print("  - Model loading into GPU memory")
        print("  - Inference loop warmup")
        print("  - Metrics reporting stabilization")
        print()
        
        interval = 30
        for elapsed in range(0, self.warmup_duration, interval):
            remaining = self.warmup_duration - elapsed
            progress = elapsed / self.warmup_duration * 100
            bar_length = 40
            filled = int(bar_length * progress / 100)
            bar = '#' * filled + '-' * (bar_length - filled)
            
            print(f"  [{bar}] {progress:5.1f}% - {remaining}s remaining", end='\r')
            time.sleep(interval)
        
        print(f"  [{'#'*40}] 100.0% - Warmup complete!      \n")
        print("[OK] Workload is now in steady state")
    
    def record_experiment(self):
        """Record experiment with progress updates"""
        self.print_header(
            f"Recording Experiment ({self.experiment_duration//60} minutes)"
        )
        
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=self.experiment_duration)
        
        print(f"Start time:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End time:    {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration:    {self.experiment_duration}s ({self.experiment_duration//60} minutes)")
        print()
        
        # Show Business Day load pattern
        print("Expected load pattern during recording (Business Day):")
        print("  0-8 min:   NIGHT   (0.3 req/s, sleep=3.0s)")
        print("  8-15 min:  RAMP    (2.0 req/s, sleep=0.5s)")
        print("  15-25 min: MORNING (4.0 req/s, sleep=0.25s)")
        print("  25-35 min: LUNCH   (1.5 req/s, sleep=0.67s)")
        print("  35-50 min: PEAK    (5.0 req/s, sleep=0.2s)")
        print("  50-60 min: EVENING (1.0 req/s, sleep=1.0s)")
        print()
        
        # Create experiment directory
        exp_dir = self.data_dir / f"{self.workload}_r{self.replicas}"
        exp_dir.mkdir(exist_ok=True)
        
        # Save timestamps
        timestamp = start_time.strftime("%Y%m%d_%H%M%S")
        timestamp_file = exp_dir / f"{self.workload}_r{self.replicas}_{timestamp}_timestamps.txt"
        
        with open(timestamp_file, 'w') as f:
            f.write(f"Workload: {self.workload}\n")
            f.write(f"Replicas: {self.replicas}\n")
            f.write(f"Start: {start_time.isoformat()}\n")
            f.write(f"End: {end_time.isoformat()}\n")
            f.write(f"Duration: {self.experiment_duration}s\n")
            f.write(f"Version: Phase 1 v3\n")
            f.write(f"Load Profile: NIGHT->RAMP->MORNING->LUNCH->PEAK->EVENING (Business Day)\n")
        
        print(f"Timestamps saved to: {timestamp_file}\n")
        
        # Load phase boundaries
        load_phases = [
            (8, "NIGHT"), (15, "RAMP"), (25, "MORNING"),
            (35, "LUNCH"), (50, "PEAK"), (60, "EVENING")
        ]
        
        # Progress updates every 5 minutes with load phase indicator
        interval = 300
        for elapsed in range(0, self.experiment_duration, interval):
            remaining = self.experiment_duration - elapsed
            progress = elapsed / self.experiment_duration * 100
            minutes_elapsed = elapsed // 60
            minutes_remaining = remaining // 60
            
            # Determine current load phase
            current_phase = "EVENING"
            for phase_end, phase_name in load_phases:
                if minutes_elapsed < phase_end:
                    current_phase = phase_name
                    break
            
            print(
                f"  Progress: {progress:5.1f}% | "
                f"Elapsed: {minutes_elapsed:2d} min | "
                f"Remaining: {minutes_remaining:2d} min | "
                f"Load: {current_phase}"
            )
            
            time.sleep(interval)
        
        print(f"  Progress: 100.0% | Elapsed: {self.experiment_duration//60} min | "
              f"Remaining:  0 min | Load: EVENING")
        print("\n[OK] Recording complete!")
        
        return start_time, end_time
    
    def query_prometheus(self, query, start, end):
        """Query Prometheus for time range with buffer"""
        buffered_end = end - timedelta(seconds=30)
        
        url = f"{self.prometheus_url}/api/v1/query_range"
        params = {
            'query': query,
            'start': start.timestamp(),
            'end': buffered_end.timestamp(),
            'step': f'{self.scrape_interval}s'
        }
        
        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"  [FAIL] Query failed: {e}")
            return None
    
    def export_metric_to_csv(self, data, filename):
        """Export single metric to CSV"""
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
        exp_dir = self.data_dir / f"{self.workload}_r{self.replicas}"
        exp_dir.mkdir(exist_ok=True)
        
        success_count = 0
        total_count = len(self.metrics)
        
        for metric_name, query in self.metrics.items():
            print(f"  [{metric_name:30s}] ", end='', flush=True)
            
            data = self.query_prometheus(query, start_time, end_time)
            
            if data:
                filename = exp_dir / f"{self.workload}_r{self.replicas}_{metric_name}_{timestamp}.csv"
                
                if self.export_metric_to_csv(data, filename):
                    file_size = filename.stat().st_size / 1024
                    print(f"[OK] Exported ({file_size:.1f} KB)")
                    success_count += 1
                else:
                    print(f"[WARN] No data")
            else:
                print(f"[FAIL] Query failed")
        
        print(f"\n[OK] Collected {success_count}/{total_count} metrics")
        print(f"\nData saved to: {exp_dir}")
    
    def cleanup(self):
        """Remove deployment"""
        self.print_header("Cleanup")
        
        print(f"Deleting {self.workload} deployment...")
        self.run_cmd(
            f"kubectl delete deployment {self.config['deployment']}",
            check=False
        )
        
        print(f"Waiting {self.cleanup_delay}s for cleanup...")
        time.sleep(self.cleanup_delay)
        
        print("[OK] Cleanup complete")
    
    def run(self):
        """Run complete experiment"""
        print("\n" + "="*70)
        print(f"  PHASE 1 v3 EXPERIMENT")
        print(f"  Workload: {self.workload}")
        print(f"  Replicas: {self.replicas}")
        print("="*70)
        
        total_time = (
            self.warmup_duration + 
            self.experiment_duration + 
            self.cleanup_delay + 
            120
        )
        print(f"\nEstimated duration: {total_time//60} minutes (~{total_time//3600}h {(total_time%3600)//60}m)\n")
        
        response = input("Start experiment? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            return
        
        try:
            if not self.check_prerequisites():
                return
            
            if not self.deploy():
                print("[FAIL] Deployment failed")
                return
            
            self.warmup_period()
            
            start_time, end_time = self.record_experiment()
            
            self.collect_metrics(start_time, end_time)
            
            self.cleanup()
            
            self.print_header("EXPERIMENT COMPLETE")
            print(f"Workload: {self.workload}")
            print(f"Replicas: {self.replicas}")
            print(f"Data location: {self.data_dir / f'{self.workload}_r{self.replicas}'}")
            print(f"\n[OK] Experiment successful!")
            
        except KeyboardInterrupt:
            print("\n\n[WARN] Experiment interrupted by user")
            print("Cleaning up...")
            self.cleanup()
            sys.exit(1)
        except Exception as e:
            print(f"\n[FAIL] Error: {e}")
            self.cleanup()
            sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_experiment_v3.py <workload> <replicas>")
        print("\nPhase 1 v3 - Comprehensive Metrics Collection")
        print("\nExamples:")
        print("  python run_experiment_v3.py resnet152 1")
        print("  python run_experiment_v3.py bert 3")
        print("  python run_experiment_v3.py whisper 5")
        print("  python run_experiment_v3.py yolo 8")
        print("  python run_experiment_v3.py gpt2 10")
        print("\nAvailable workloads: resnet152, bert, whisper, yolo, gpt2")
        print("Recommended replica counts: 1, 3, 5, 8, 10")
        sys.exit(1)
    
    workload = sys.argv[1]
    replicas = int(sys.argv[2])
    
    runner = ExperimentRunnerV3(workload, replicas)
    runner.run()