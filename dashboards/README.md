# Grafana Dashboards for Data Collection

This directory contains Grafana dashboard configurations for monitoring AI inference workloads on Kubernetes.

## Dashboard Overview

### 1. System Resources Dashboard
**File:** `system-resources.json`
**Purpose:** Monitor host-level resource consumption

**Metrics Included:**
- CPU utilization (total and per-core)
- Memory usage (bytes and percentage)
- Disk I/O (read/write operations)
- Network traffic (transmit/receive)

**Use Case:** Tracking system-level resource consumption for Phase 1 workload characterization.

### 2. GPU Performance Dashboard
**File:** `gpu-performance.json`
**Purpose:** Monitor NVIDIA GPU metrics for AI workload analysis

**Metrics Included:**
- GPU utilization percentage
- GPU memory usage (bytes and percentage)
- GPU temperature (Celsius)
- GPU power consumption (Watts)
- GPU clock speeds (SM and memory clocks)

**Use Case:** Essential for thesis requirement of GPU load and memory tracking under different load states.

### 3. Container Metrics Dashboard
**File:** `container-metrics.json`
**Purpose:** Monitor pod and container-level resource usage

**Metrics Included:**
- Per-pod CPU usage
- Per-pod memory consumption
- Pod network I/O
- Pod filesystem usage
- Pod status summary table

**Use Case:** Granular analysis of individual AI application resource consumption.

### 4. System Pressure Dashboard
**File:** `system-pressure.json`
**Purpose:** PSI-based load state detection and classification

**Metrics Included:**
- CPU pressure (PSI waiting time)
- Memory pressure (PSI waiting and stalled time)
- I/O pressure (PSI waiting and stalled time)
- Load state classification (empty, modest, high)
- System pressure score

**Use Case:** Critical for thesis requirement of measuring performance under different load states using PSI metrics.

### 5. Inference Performance Dashboard
**File:** `inference-performance.json`
**Purpose:** Application-level QoS metrics and latency tracking

**Metrics Included:**
- Per-application inference latency (p50, p95, p99 percentiles)
- Request throughput (requests per second)
- Queue depth
- Comparative latency analysis
- Load correlation with performance

**Use Case:** Thesis requirement for application response time measurement under varying load conditions.

**Note:** This dashboard requires custom metrics instrumentation in AI applications.

### Example PromQL Queries

**CPU Usage Percentage:**
```promql
100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)
```

**GPU Utilization:**
```promql
dcgm_gpu_utilization{gpu="0"}
```

**Container Memory:**
```promql
container_memory_usage_bytes{pod=~"resnet50.*"}
```

**Inference Latency (p95):**
```promql
histogram_quantile(0.95, rate(inference_latency_seconds_bucket{app="resnet50"}[5m]))
```

## Data Export for Thesis Analysis

### Export Time-Series Data

**Via Prometheus API:**
```bash
# Export GPU utilization for date range
curl -G 'http://172.22.174.58:30090/api/v1/query_range' \
  --data-urlencode 'query=dcgm_gpu_utilization{gpu="0"}' \
  --data-urlencode 'start=2025-12-09T00:00:00Z' \
  --data-urlencode 'end=2025-12-09T23:59:59Z' \
  --data-urlencode 'step=15s' \
  > data/raw/gpu_utilization.json
```

## Alerting Configuration

### High Resource Utilization Alerts

Dashboards include alert conditions (red indicators) for:
- GPU utilization > 90%
- GPU temperature > 80C
- System pressure score > 10

### Backup Dashboards

Dashboards are automatically backed up via Grafana persistent volume. For additional safety:
```bash
# Backup all dashboards via API
curl -u admin:admin http://172.22.174.58:30030/api/search | \
  jq -r '.[] | select(.type=="dash-db") | .uid' | \
  xargs -I {} curl -u admin:admin \
    http://172.22.174.58:30030/api/dashboards/uid/{} \
    > dashboards/backup-{}.json
```

## References

- Prometheus Query Documentation: https://prometheus.io/docs/prometheus/latest/querying/basics/
- Grafana Dashboard Documentation: https://grafana.com/docs/grafana/latest/dashboards/
- DCGM Exporter Metrics: https://github.com/NVIDIA/dcgm-exporter
- PSI Documentation: https://www.kernel.org/doc/html/latest/accounting/psi.html
