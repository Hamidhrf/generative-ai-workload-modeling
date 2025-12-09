# Monitoring Stack for Thesis Data Collection

Complete observability stack for collecting AI workload performance metrics.

---

##  Stack Components

### 1. **Prometheus** (Time-Series Database)
- **Purpose**: Store all metrics for thesis analysis
- **Retention**: 30 days
- **Storage**: 50GB persistent volume
- **Access**: http://NODE_IP:30090

### 2. **Grafana** (Visualization)
- **Purpose**: Dashboards and data visualization
- **Storage**: 10GB persistent volume
- **Access**: http://NODE_IP:30030
- **Credentials**: admin/admin

### 3. **Node Exporter** (Host Metrics)
**Metrics Provided:**
- CPU utilization (per core, total)
- RAM usage (available, used, cached)
- Disk I/O (read/write operations)
- Network traffic (bytes sent/received)
- **PSI metrics** (pressure stall information for load detection)

### 4. **DCGM Exporter** (GPU Metrics)
**Metrics Provided:**
- GPU utilization (%)
- GPU memory usage (MB)
- GPU temperature (°C)
- GPU power consumption (W)
- GPU clock speeds (MHz)
- SM (streaming multiprocessor) activity

### 5. **kube-state-metrics** (K8s State)
**Metrics Provided:**
- Pod status (running, pending, failed)
- Resource requests/limits
- Deployment replicas
- Node status

### 6. **kubelet/cAdvisor** (Container Metrics)
**Metrics Provided:**
- Container CPU usage
- Container memory usage
- Container network I/O
- Container filesystem usage
- **Per-pod resource consumption**

### 7. **Kepler** (Power Consumption)
**Metrics Provided:**
- Pod-level power consumption (W)
- Node-level power consumption
- Energy efficiency metrics
- Carbon emissions estimates

---

##  Metrics for Thesis

### Phase 1 Requirements Mapped to Metrics:

| Thesis Requirement | Prometheus Metric | Dashboard |
|-------------------|-------------------|-----------|
| **CPU utilization** | `node_cpu_seconds_total` | System Resources |
| **Memory consumption** | `node_memory_MemAvailable_bytes` | System Resources |
| **GPU load** | `dcgm_gpu_utilization` | GPU Performance |
| **GPU memory** | `dcgm_fb_used_bytes` | GPU Performance |
| **Power usage** | `kepler_container_joules_total` | Energy Consumption |
| **Application latency** | `inference_latency_seconds` (custom) | Inference Performance |
| **PSI (load state)** | `node_pressure_cpu_waiting_seconds_total` | System Pressure |
| **Container resources** | `container_cpu_usage_seconds_total` | Container View |

---

##  Deployment

### Quick Deploy
```bash
cd ~/generative-ai-workload-modeling
chmod +x scripts/monitoring/deploy-monitoring-stack.sh
sudo ./scripts/monitoring/deploy-monitoring-stack.sh
```

### Manual Deploy
```bash
kubectl apply -f k8s/monitoring/namespace.yaml
kubectl apply -f k8s/monitoring/prometheus-config.yaml
kubectl apply -f k8s/monitoring/prometheus-deployment.yaml
kubectl apply -f k8s/monitoring/node-exporter.yaml
kubectl apply -f k8s/monitoring/dcgm-exporter.yaml
kubectl apply -f k8s/monitoring/kube-state-metrics.yaml
kubectl apply -f k8s/monitoring/kepler.yaml
kubectl apply -f k8s/monitoring/grafana-deployment.yaml
```

---

##  Verification

### Check All Pods Running
```bash
kubectl get pods -n monitoring
```

### Check Prometheus Targets
```bash
# Access Prometheus UI
# Go to: http://NODE_IP:30090/targets
# All targets should be "UP"
```

### Test Metrics Collection
```bash
# Query GPU utilization
curl -s http://NODE_IP:30090/api/v1/query?query=dcgm_gpu_utilization

# Query CPU usage
curl -s http://NODE_IP:30090/api/v1/query?query=node_cpu_seconds_total

# Query power consumption
curl -s http://NODE_IP:30090/api/v1/query?query=kepler_container_joules_total
```

---

##  Creating Grafana Dashboards

### Access Grafana
1. Open: http://NODE_IP:30030
2. Login: admin/admin
3. Change password when prompted

### Import Pre-built Dashboards
1. Click "+" → "Import"
2. Upload JSON files from `dashboards/` folder
3. Select "Prometheus" as data source

### Create Custom Dashboard
1. Click "+" → "Dashboard"
2. Add panel → Select visualization type
3. Query examples:
   ```promql
   # GPU utilization
   dcgm_gpu_utilization{gpu="0"}
   
   # CPU usage
   100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)
   
   # Container memory
   container_memory_usage_bytes{pod=~"resnet50.*"}
   
   # Power consumption
   rate(kepler_container_joules_total[5m])
   ```

---

##  Thesis Data Collection Workflow

### 1. Baseline Collection (Uncontended State)
```bash
# Ensure no workloads running
kubectl get pods --all-namespaces

# Record metrics for 10 minutes
# Export from Prometheus: http://NODE_IP:30090
```

### 2. Modest Load Collection
```bash
# Deploy 1 replica of each AI app
kubectl scale deployment resnet50-inference --replicas=1
kubectl scale deployment distilbert-inference --replicas=1
kubectl scale deployment whisper-inference --replicas=1

# Generate moderate traffic
# Record metrics for 30 minutes
```

### 3. High Load Collection
```bash
# Deploy 3 replicas of each AI app
kubectl scale deployment resnet50-inference --replicas=3
kubectl scale deployment distilbert-inference --replicas=3
kubectl scale deployment whisper-inference --replicas=3

# Generate heavy traffic
# Record metrics for 30 minutes
```

### 4. Export Data for Model Training
```bash
# From Prometheus API
curl -G http://NODE_IP:30090/api/v1/query_range \
  --data-urlencode 'query=dcgm_gpu_utilization' \
  --data-urlencode 'start=2025-12-09T00:00:00Z' \
  --data-urlencode 'end=2025-12-09T23:59:59Z' \
  --data-urlencode 'step=15s' \
  > data/raw/gpu_metrics.json
```

---

##  Troubleshooting

### Prometheus Not Scraping
```bash
# Check logs
kubectl logs -n monitoring -l app=prometheus

# Check targets
# Go to: http://NODE_IP:30090/targets
```

### GPU Metrics Missing
```bash
# Check DCGM Exporter logs
kubectl logs -n monitoring -l app=dcgm-exporter

# Verify GPU accessible
kubectl describe node | grep nvidia.com/gpu
```

### Kepler Not Working
```bash
# Check if kernel supports eBPF
uname -r  # Should be >= 4.14

# Check logs
kubectl logs -n monitoring -l app=kepler
```

### Grafana Not Loading
```bash
# Check pod status
kubectl get pods -n monitoring -l app=grafana

# Check logs
kubectl logs -n monitoring -l app=grafana
```

---

##  Storage Management

### Prometheus Storage
- **Location**: `/prometheus` in pod
- **PVC**: `prometheus-pvc` (50GB)
- **Retention**: 30 days

### Grafana Storage
- **Location**: `/var/lib/grafana` in pod
- **PVC**: `grafana-pvc` (10GB)

### Cleanup Old Data
```bash
# Prometheus automatically cleans data after 30 days
# To manually reduce retention:
kubectl edit deployment prometheus -n monitoring
# Change: --storage.tsdb.retention.time=15d
```

---

##  Next Steps

1. **Deploy AI Workloads**: Deploy ResNet50, DistilBERT, Whisper
2. **Instrument Apps**: Add latency metrics to inference scripts
3. **Create Dashboards**: Build custom Grafana dashboards
4. **Collect Data**: Run experiments under different loads
5. **Export Data**: Extract time-series for model training

---

##  Useful Queries

### GPU Queries
```promql
# GPU utilization
dcgm_gpu_utilization{gpu="0"}

# GPU memory usage (%)
(dcgm_fb_used_bytes / dcgm_fb_total_bytes) * 100

# GPU power consumption
dcgm_power_usage_watts
```

### CPU Queries
```promql
# CPU usage (%)
100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)

# CPU usage per core
rate(node_cpu_seconds_total{mode="user"}[5m])
```

### Memory Queries
```promql
# Available memory (GB)
node_memory_MemAvailable_bytes / 1024^3

# Memory usage (%)
(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100
```

### Container Queries
```promql
# CPU usage by pod
rate(container_cpu_usage_seconds_total{pod=~"resnet50.*"}[5m])

# Memory usage by pod
container_memory_usage_bytes{pod=~"resnet50.*"} / 1024^3
```

### PSI Queries (Load Detection)
```promql
# CPU pressure
rate(node_pressure_cpu_waiting_seconds_total[5m])

# Memory pressure
rate(node_pressure_memory_waiting_seconds_total[5m])
```

### Power Queries
```promql
# Power consumption per pod (Watts)
rate(kepler_container_joules_total[5m])

# Total node power
sum(rate(kepler_container_joules_total[5m]))
```

---

##  References

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [DCGM Exporter](https://github.com/NVIDIA/dcgm-exporter)
- [Kepler](https://sustainable-computing.io/)
- [Node Exporter](https://github.com/prometheus/node_exporter)