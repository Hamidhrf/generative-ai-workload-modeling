# Phase 2 EDA - Pod-Level Analysis Report
**Date**: January 23, 2026
**Dataset**: 60 Individual Pod Traces from Phase 1

---

## 1. Dataset Overview
- **Total pod traces**: 60
- **Trace shape**: (715 timesteps, 15 metrics)
- **Duration**: ~59.6 minutes per experiment
- **Sampling interval**: 5 seconds

### Workload Distribution:
- **distilbert**: 19 pod traces
- **resnet50**: 22 pod traces
- **whisper**: 19 pod traces

### Replica Count Distribution:
- **r=1**: 3 pod traces
- **r=2**: 6 pod traces
- **r=3**: 6 pod traces
- **r=5**: 5 pod traces
- **r=6**: 12 pod traces
- **r=8**: 8 pod traces
- **r=10**: 20 pod traces

---

## 2. Metric Statistics

### Overall Statistics (All 60 Pods):

| Metric | Min | Max | Mean | Std |
|--------|-----|-----|------|-----|
| cpu_psi | 1.85e-06 | 2.96e-01 | 6.21e-02 | 9.97e-02 |
| cpu_usage | 5.66e-01 | 6.21e+00 | 1.64e+00 | 1.23e+00 |
| gpu_memory | 3.85e+02 | 1.27e+04 | 4.92e+03 | 3.55e+03 |
| gpu_power | 4.06e+01 | 6.14e+01 | 5.63e+01 | 4.02e+00 |
| gpu_temperature | 6.50e+01 | 7.40e+01 | 7.10e+01 | 1.34e+00 |
| gpu_utilization | 0.00e+00 | 1.00e+02 | 8.67e+01 | 2.64e+01 |
| latency_avg | 3.05e-03 | 1.69e+00 | 2.98e-01 | 4.74e-01 |
| latency_p50 | 3.00e-03 | 1.57e+00 | 3.15e-01 | 5.10e-01 |
| latency_p95 | 4.80e-03 | 2.47e+00 | 5.98e-01 | 9.42e-01 |
| latency_p99 | 4.96e-03 | 4.32e+00 | 7.00e-01 | 1.14e+00 |
| throughput | 5.73e+00 | 3.52e+02 | 1.71e+02 | 1.36e+02 |
| total_inferences | 5.73e+00 | 3.52e+02 | 1.71e+02 | 1.36e+02 |
| io_psi | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| memory_psi | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| memory_usage | 7.37e+08 | 3.36e+09 | 1.64e+09 | 7.68e+08 |

---

## 3. Key Observations

### Contention Effects:
- **CPU usage** increases with replica count across all workloads
- **GPU utilization** remains high (>80%) showing saturation
- **Latency** shows clear degradation from r=1 to r=10
- **Throughput** increases initially but plateaus at high replica counts

### Workload Characteristics:
- **DistilBERT**: Gradual performance degradation
- **ResNet50**: Immediate contention at r>1
- **Whisper**: Similar to ResNet50, resource-intensive

### Zero-Value Metrics:
- **io_psi**: All zeros (no I/O pressure stalls)
- **memory_psi**: All zeros (no memory pressure stalls)
- These metrics are kept in the dataset for model consistency

---

## 4. Recommendations for Modeling

### Data Preprocessing:
1. **Normalization**: Use StandardScaler (save parameters for denormalization)
2. **Memory metric**: Scale is very large (~1e9), needs normalization
3. **Zero metrics**: Keep io_psi and memory_psi (model learns 'no pressure')

### Model Training:
1. **Conditioning**: Always condition on replica_count
2. **Workload encoding**: Use one-hot or embedding for workload type
3. **Validation**: Stratify by experiment (not by individual pods)

### Expected Generation:
- For r=50: Model should interpolate/extrapolate from learned patterns
- System metrics (GPU) will be shared across generated pods
- Per-pod metrics should show variability similar to training data

---

## 5. Visualizations Generated
1. `distilbert_temporal_patterns.png` - Temporal patterns for DistilBERT
2. `resnet50_temporal_patterns.png` - Temporal patterns for ResNet50
3. `whisper_temporal_patterns.png` - Temporal patterns for Whisper
4. `contention_effects.png` - Replica count vs metrics
5. `metric_distributions.png` - Distribution of all 15 metrics
6. `correlation_matrices.png` - Metric correlations per workload
7. `pod_variability.png` - Pod-to-pod variability analysis

---
