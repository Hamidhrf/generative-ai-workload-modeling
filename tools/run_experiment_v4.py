#!/usr/bin/env python3
"""
Tier 2 v4 Experiment Runner — MIG per-slice metrics.

Same orchestration as run_experiment_v3.py (deploy, 5-minute warmup, 60-minute
Business Day recording, cleanup). Diverges from v3 only in the GPU/DCGM query
layer (per-slice MIG metrics instead of whole-GPU) and the writer (live 5s
tick sampling instead of a single end-of-run query_range call, so mid-run
pod-crash and Prometheus-outage conditions can be detected and handled).

USAGE:
  python run_experiment_v4.py <workload> <replicas>

ENVIRONMENT:
  PROMETHEUS_URL          default http://172.22.174.66:30090
  DATA_OUTPUT_DIR         default data/raw/extension_tier2
  EXPERIMENT_AUTO_CONFIRM if "1", skips interactive confirm prompts
"""

import os
import sys
import time
import subprocess
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

SAMPLE_INTERVAL_SEC = 5
EXPERIMENT_DURATION_SEC = 3600
WARMUP_DURATION_SEC = 300
CLEANUP_TIMEOUT_SEC = 60
EXPECTED_ROWS_MIN = 700
EXPECTED_ROWS_MAX = 720
POD_READY_TIMEOUT_SEC = {"whisper": 300, "default": 180}

# Usable MiB per 1g.12gb MIG slice (Step 0 finding: DCGM reports this profile
# as "1g.11gb", a cosmetic naming bug, but usable FB is 11007 MiB regardless).
MIG_SLICE_MEMORY_MIB = 11007
# Power/temperature/energy are physical-GPU quantities replicated identically
# across every slice under MIG single strategy; GPU_I_ID=7 is an arbitrary
# but stable reference slice (confirmed present in Step 0 for this node).
REFERENCE_GPU_I_ID = "7"

WORKLOADS = {
    "bert": {"deployment": "bert-inference", "app_label": "bert"},
    "gpt2": {"deployment": "gpt2-inference", "app_label": "gpt2"},
    "resnet152": {"deployment": "resnet152-inference", "app_label": "resnet152"},
    "whisper": {"deployment": "whisper-inference", "app_label": "whisper"},
    "yolo": {"deployment": "yolo-inference", "app_label": "yolo"},
}

# Per-pod non-GPU metrics: "sum by (pod)" or raw per-pod queries return one
# series per running pod, so row count scales with replica count, not just
# tick count. Kept here (rather than only in the batch script) since it is a
# property of the query registry itself.
PER_POD_NON_GPU_METRICS = {
    "pod_cpu_usage", "pod_memory_bytes", "pod_psi_cpu", "pod_psi_io",
    "pod_psi_memory", "pod_latency_avg", "pod_throughput",
}

PER_SLICE_COLUMNS = [
    "timestamp", "value", "GPU_I_ID", "GPU_I_PROFILE",
    "pod", "namespace", "container", "DCGM_FI_DRIVER_VERSION", "UUID",
]


def _auto_confirm():
    return os.environ.get("EXPERIMENT_AUTO_CONFIRM", "").lower() in ("1", "true", "yes")


def non_gpu_queries(workload):
    label = workload
    prefix = workload
    return {
        "app_latency_p50": f'histogram_quantile(0.5, sum(rate({prefix}_inference_latency_seconds_bucket[1m])) by (le))',
        "app_latency_p95": f'histogram_quantile(0.95, sum(rate({prefix}_inference_latency_seconds_bucket[1m])) by (le))',
        "app_latency_p99": f'histogram_quantile(0.99, sum(rate({prefix}_inference_latency_seconds_bucket[1m])) by (le))',
        "app_throughput": f'sum(rate({prefix}_inference_total[1m]))',
        "node_cpu_usage": '1 - avg(rate(node_cpu_seconds_total{mode="idle"}[1m]))',
        "node_memory_available_bytes": "node_memory_MemAvailable_bytes",
        "node_memory_used_percent": "1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)",
        "node_psi_cpu": "rate(node_pressure_cpu_waiting_seconds_total[1m])",
        "node_psi_io": "rate(node_pressure_io_waiting_seconds_total[1m])",
        "node_psi_memory": "rate(node_pressure_memory_waiting_seconds_total[1m])",
        "pod_cpu_usage": f'sum by (pod) (rate(container_cpu_usage_seconds_total{{namespace="default", pod=~"{label}-.*", container!="POD", container!=""}}[1m]))',
        "pod_memory_bytes": f'sum by (pod) (container_memory_working_set_bytes{{namespace="default", pod=~"{label}-.*", container!="POD", container!=""}})',
        "pod_psi_cpu": f'sum by (pod) (rate(container_pressure_cpu_waiting_seconds_total{{namespace="default", pod=~"{label}-.*"}}[1m]))',
        "pod_psi_io": f'sum by (pod) (rate(container_pressure_io_waiting_seconds_total{{namespace="default", pod=~"{label}-.*"}}[1m]))',
        "pod_psi_memory": f'sum by (pod) (rate(container_pressure_memory_waiting_seconds_total{{namespace="default", pod=~"{label}-.*"}}[1m]))',
        "pod_latency_avg": f'rate({prefix}_inference_latency_seconds_sum{{pod=~"{label}-.*"}}[1m]) / rate({prefix}_inference_latency_seconds_count{{pod=~"{label}-.*"}}[1m])',
        "pod_throughput": f'rate({prefix}_inference_total{{pod=~"{label}-.*"}}[1m])',
    }


def gpu_agg_query_specs():
    # Order matters: gpu_power_watts must run before gpu_memory_total so its
    # harvested labels (constant physical-GPU identity) can be reused for the
    # constant-value row, which has no query of its own.
    return [
        ("gpu_power_watts", "single", f'DCGM_FI_DEV_POWER_USAGE{{GPU_I_ID="{REFERENCE_GPU_I_ID}"}}', None),
        ("gpu_temperature", "single", f'DCGM_FI_DEV_GPU_TEMP{{GPU_I_ID="{REFERENCE_GPU_I_ID}"}}', None),
        ("gpu_total_energy_consumption", "single", f'DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION{{GPU_I_ID="{REFERENCE_GPU_I_ID}"}}', None),
        ("gpu_utilization", "raw_sum", 'DCGM_FI_PROF_GR_ENGINE_ACTIVE{pod!="", namespace="default"}', 100),
        ("gpu_memory_used", "raw_sum", 'DCGM_FI_DEV_FB_USED{pod!="", namespace="default"}', 1),
        ("gpu_dram_active", "raw_sum", 'DCGM_FI_PROF_DRAM_ACTIVE{pod!="", namespace="default"}', 1),
        ("gpu_pipe_tensor_active", "raw_sum", 'DCGM_FI_PROF_PIPE_TENSOR_ACTIVE{pod!="", namespace="default"}', 1),
        ("gpu_memory_total", "constant", None, None),
    ]


def gpu_per_slice_query_specs():
    return {
        "gpu_utilization_per_slice": "DCGM_FI_PROF_GR_ENGINE_ACTIVE * 100",
        "gpu_memory_used_per_slice": "DCGM_FI_DEV_FB_USED",
        "gpu_memory_total_per_slice": "DCGM_FI_DEV_FB_USED + DCGM_FI_DEV_FB_FREE",
        "gpu_power_watts_per_slice": "DCGM_FI_DEV_POWER_USAGE",
        "gpu_temperature_per_slice": "DCGM_FI_DEV_GPU_TEMP",
        "gpu_dram_active_per_slice": "DCGM_FI_PROF_DRAM_ACTIVE",
        "gpu_pipe_tensor_active_per_slice": "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE",
        "gpu_total_energy_consumption_per_slice": "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION",
    }


class ExperimentRunnerV4:
    def __init__(self, workload, replicas):
        if workload not in WORKLOADS:
            print(f"Error: Unknown workload '{workload}'. Available: {list(WORKLOADS.keys())}")
            sys.exit(1)

        self.workload = workload
        self.replicas = replicas
        self.config = WORKLOADS[workload]
        self.deployment = self.config["deployment"]
        self.app_label = self.config["app_label"]

        self.prometheus_url = os.environ.get("PROMETHEUS_URL", "http://172.22.174.66:30090")
        self.data_dir = Path(os.environ.get("DATA_OUTPUT_DIR", "data/raw/extension_tier2"))
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.non_gpu_queries = non_gpu_queries(workload)
        self.gpu_agg_specs = gpu_agg_query_specs()
        self.gpu_per_slice_queries = gpu_per_slice_query_specs()

        self.session = requests.Session()

    # -- shell / kubectl helpers -------------------------------------------------

    def run_cmd(self, cmd, check=True):
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
        return result.stdout.strip()

    def print_header(self, text):
        print(f"\n{'=' * 70}\n  {text}\n{'=' * 70}\n")

    def get_pod_status_counts(self):
        """Returns (running_ready_count, total_pod_count) for this workload's app label."""
        output = self.run_cmd(f"kubectl get pods -l app={self.app_label} --no-headers", check=False)
        if not output:
            return 0, 0
        lines = output.strip().split("\n")
        running = 0
        for line in lines:
            parts = line.split()
            if len(parts) >= 3 and parts[2] == "Running":
                ready = parts[1].split("/")
                if len(ready) == 2 and ready[0] == ready[1]:
                    running += 1
        return running, len(lines)

    # -- deployment lifecycle -----------------------------------------------------

    def deploy(self):
        self.print_header(f"Deploying {self.workload} x {self.replicas} replicas (Tier 2 MIG)")
        manifest = f"k8s/workloads/tier2/{self.workload}-mig.yaml"
        self.run_cmd(f"kubectl apply -f {manifest}", check=False)
        self.run_cmd(f"kubectl scale deployment {self.deployment} --replicas={self.replicas}")

    def wait_for_pods_ready(self):
        timeout = POD_READY_TIMEOUT_SEC.get(self.workload, POD_READY_TIMEOUT_SEC["default"])
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            running, _ = self.get_pod_status_counts()
            print(f"  Ready: {running}/{self.replicas} pods", end="\r")
            if running == self.replicas:
                print(f"\n[OK] All {self.replicas} pods ready")
                return True
            time.sleep(5)
        print(f"\n[FAIL] Timeout waiting for pods ({timeout}s)")
        return False

    def cleanup(self):
        self.print_header("Cleanup")
        self.run_cmd(f"kubectl delete deployment {self.deployment} --ignore-not-found=true", check=False)
        start = time.monotonic()
        while time.monotonic() - start < CLEANUP_TIMEOUT_SEC:
            _, total = self.get_pod_status_counts()
            if total == 0:
                print("[OK] All pods terminated")
                return True
            time.sleep(5)
        _, total = self.get_pod_status_counts()
        print(f"[FAIL] {total} pod(s) remain after {CLEANUP_TIMEOUT_SEC}s")
        return False

    # -- warmup --------------------------------------------------------------------

    def warmup_period(self):
        self.print_header(f"Warmup Period ({WARMUP_DURATION_SEC}s = {WARMUP_DURATION_SEC // 60} minutes)")
        interval = 30
        for elapsed in range(0, WARMUP_DURATION_SEC, interval):
            remaining = WARMUP_DURATION_SEC - elapsed
            print(f"  {remaining}s remaining...", end="\r")
            time.sleep(interval)
        print("[OK] Workload is now in steady state" + " " * 20)

    # -- Prometheus query + row-building -------------------------------------------

    def instant_query(self, query):
        try:
            resp = self.session.get(
                f"{self.prometheus_url}/api/v1/query", params={"query": query}, timeout=10
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") != "success":
                return [], False
            return data["data"]["result"], True
        except Exception:
            return [], False

    @staticmethod
    def non_gpu_rows(results, ts):
        if not results:
            return [{"timestamp": ts, "value": None}]
        rows = []
        for r in results:
            row = {"timestamp": ts, "value": float(r["value"][1])}
            row.update(r["metric"])
            rows.append(row)
        return rows

    @staticmethod
    def harvest_agg_row(ts, metric, value, include_name):
        # Column order matches Tier 1's actual GPU CSV header exactly:
        # timestamp,value,DCGM_FI_DRIVER_VERSION,Hostname,UUID,[__name__,]
        # container,device,gpu,instance,job,modelName,namespace,pci_bus_id,pod
        row = {"timestamp": ts, "value": value}
        for k in ["DCGM_FI_DRIVER_VERSION", "Hostname", "UUID"]:
            row[k] = metric.get(k, "")
        if include_name:
            row["__name__"] = metric.get("__name__", "")
        row["container"] = ""
        for k in ["device", "gpu", "instance", "job", "modelName"]:
            row[k] = metric.get(k, "")
        row["namespace"] = ""
        row["pci_bus_id"] = ""
        row["pod"] = ""
        return row

    def gpu_agg_rows(self, ts, name, kind, query, multiplier, cached_labels):
        if kind == "constant":
            metric = cached_labels.get("gpu_power_watts", {})
            return [self.harvest_agg_row(ts, metric, MIG_SLICE_MEMORY_MIB, include_name=False)], True

        results, ok = self.instant_query(query)
        if not results:
            return [{"timestamp": ts, "value": None}], ok

        if kind == "single":
            metric = results[0]["metric"]
            cached_labels[name] = metric
            value = float(results[0]["value"][1])
            return [self.harvest_agg_row(ts, metric, value, include_name=True)], ok

        # raw_sum: sum values client-side (not via PromQL sum()) so the first
        # series' labels remain available to populate the aggregated row.
        value = sum(float(r["value"][1]) for r in results) * multiplier
        return [self.harvest_agg_row(ts, results[0]["metric"], value, include_name=True)], ok

    def gpu_per_slice_rows(self, ts, query):
        results, ok = self.instant_query(query)
        if not results:
            return [], ok
        rows = []
        for r in results:
            m = r["metric"]
            rows.append({
                "timestamp": ts,
                "value": float(r["value"][1]),
                "GPU_I_ID": m.get("GPU_I_ID", ""),
                "GPU_I_PROFILE": m.get("GPU_I_PROFILE", ""),
                "pod": m.get("pod", ""),
                "namespace": m.get("namespace", ""),
                "container": m.get("container", ""),
                "DCGM_FI_DRIVER_VERSION": m.get("DCGM_FI_DRIVER_VERSION", ""),
                "UUID": m.get("UUID", ""),
            })
        return rows, ok

    # -- main sampling loop ---------------------------------------------------------

    def record_experiment(self):
        self.print_header(f"Recording Experiment ({EXPERIMENT_DURATION_SEC // 60} minutes, {SAMPLE_INTERVAL_SEC}s ticks)")

        total_ticks = EXPERIMENT_DURATION_SEC // SAMPLE_INTERVAL_SEC
        start_time = datetime.now()

        non_gpu_buffers = {name: [] for name in self.non_gpu_queries}
        agg_gpu_buffers = {name: [] for name, *_ in self.gpu_agg_specs}
        per_slice_buffers = {name: [] for name in self.gpu_per_slice_queries}
        tick_timestamps = []

        consecutive_error_ticks = 0
        cached_labels = {}
        anchor = time.monotonic()

        for tick in range(total_ticks):
            now = datetime.now()
            tick_timestamps.append(now)
            tick_ok = True

            running, _ = self.get_pod_status_counts()
            if running == 0:
                self.halt(5, f"pod count dropped to zero at tick {tick} ({tick * SAMPLE_INTERVAL_SEC}s elapsed)")
            elif running < self.replicas:
                print(f"  [WARN] tick {tick}: {running}/{self.replicas} pods running, continuing")

            for name, query in self.non_gpu_queries.items():
                results, ok = self.instant_query(query)
                non_gpu_buffers[name].append(self.non_gpu_rows(results, now))
                tick_ok = tick_ok and ok

            for name, kind, query, multiplier in self.gpu_agg_specs:
                rows, ok = self.gpu_agg_rows(now, name, kind, query, multiplier, cached_labels)
                agg_gpu_buffers[name].append(rows)
                tick_ok = tick_ok and ok

            for name, query in self.gpu_per_slice_queries.items():
                rows, ok = self.gpu_per_slice_rows(now, query)
                per_slice_buffers[name].append(rows)
                tick_ok = tick_ok and ok

            consecutive_error_ticks = 0 if tick_ok else consecutive_error_ticks + 1
            if consecutive_error_ticks >= 3:
                self.halt(4, "Prometheus HTTP error on 3 consecutive sample ticks")

            if tick % 60 == 0:
                pct = tick / total_ticks * 100
                print(f"  Progress: {pct:5.1f}% | tick {tick}/{total_ticks} | elapsed {tick * SAMPLE_INTERVAL_SEC // 60} min")

            next_tick_at = anchor + (tick + 1) * SAMPLE_INTERVAL_SEC
            sleep_time = next_tick_at - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)

        print("[OK] Recording complete!")
        return start_time, tick_timestamps, non_gpu_buffers, agg_gpu_buffers, per_slice_buffers

    # -- output writing ---------------------------------------------------------------

    def write_outputs(self, start_time, tick_timestamps, non_gpu_buffers, agg_gpu_buffers, per_slice_buffers):
        self.print_header("Writing Output Files")

        ts_str = start_time.strftime("%Y%m%d_%H%M%S")
        exp_dir = self.data_dir / f"{self.workload}_r{self.replicas}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{self.workload}_r{self.replicas}"

        for name, tick_rows in {**non_gpu_buffers, **agg_gpu_buffers}.items():
            flat = [row for rows in tick_rows for row in rows]
            df = pd.DataFrame(flat)
            df.to_csv(exp_dir / f"{prefix}_{name}_{ts_str}.csv", index=False)
            print(f"  [{name:35s}] {len(flat)} rows")

        for name, tick_rows in per_slice_buffers.items():
            flat = [row for rows in tick_rows for row in rows]
            df = pd.DataFrame(flat, columns=PER_SLICE_COLUMNS)
            df.to_csv(exp_dir / f"{prefix}_{name}_{ts_str}.csv", index=False)
            print(f"  [{name:35s}] {len(flat)} rows")

        ts_file = exp_dir / f"{prefix}_{ts_str}_timestamps.txt"
        with open(ts_file, "w") as f:
            for t in tick_timestamps:
                f.write(t.strftime("%Y-%m-%d %H:%M:%S.%f") + "\n")

        print(f"\nData saved to: {exp_dir}")
        return exp_dir

    # -- halt / run -----------------------------------------------------------------

    def halt(self, code, reason):
        print(f"RUNNER FAIL: {reason}", file=sys.stderr)
        if code == 2:
            self.cleanup()
        sys.exit(code)

    def run(self):
        self.print_header("TIER 2 v4 EXPERIMENT (MIG)")
        print(f"Workload:         {self.workload}")
        print(f"Replicas:         {self.replicas}")
        print(f"Prometheus URL:   {self.prometheus_url}")
        print(f"Output dir:       {self.data_dir}")
        print(f"Sample interval:  {SAMPLE_INTERVAL_SEC}s")
        print(f"Duration:         {EXPERIMENT_DURATION_SEC}s ({EXPERIMENT_DURATION_SEC // 60} min)")

        if _auto_confirm():
            print("\nStart experiment? (yes/no): yes  [EXPERIMENT_AUTO_CONFIRM]")
        else:
            response = input("\nStart experiment? (yes/no): ")
            if response.lower() != "yes":
                print("Aborted.")
                return

        self.deploy()
        if not self.wait_for_pods_ready():
            self.halt(2, f"pods not Running within timeout for {self.workload}")

        self.warmup_period()

        start_time, tick_timestamps, non_gpu_buffers, agg_gpu_buffers, per_slice_buffers = self.record_experiment()

        self.write_outputs(start_time, tick_timestamps, non_gpu_buffers, agg_gpu_buffers, per_slice_buffers)

        if not self.cleanup():
            print("RUNNER FAIL: deployment cleanup timeout", file=sys.stderr)
            sys.exit(3)

        print(f"\nEXPERIMENT COMPLETE: {self.workload} r={self.replicas}")
        sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_experiment_v4.py <workload> <replicas>")
        print(f"Available workloads: {list(WORKLOADS.keys())}")
        sys.exit(1)

    workload_arg = sys.argv[1]
    try:
        replicas_arg = int(sys.argv[2])
    except ValueError:
        print(f"Error: replicas must be an integer, got '{sys.argv[2]}'")
        sys.exit(1)

    runner = ExperimentRunnerV4(workload_arg, replicas_arg)
    runner.run()
