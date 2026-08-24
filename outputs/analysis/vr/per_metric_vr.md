# Per-metric VR appendix (N2) - 15 cells x 7 trained metrics

Source: stored `vr_per_metric_smooth` from the three S36 `s36_eval_results.json` files. Pure read, no re-eval.


## A16


### A16 / bert

| metric | VR (smooth) | excluded? |
|---|---|---|
| pod_cpu_usage | 1.1740 |  |
| pod_memory_bytes | 1.2621 | yes |
| pod_psi_cpu | 0.7349 |  |
| pod_latency_avg | 1.1324 |  |
| pod_throughput | 0.9313 |  |
| gpu_utilization | 0.9445 |  |
| gpu_power_watts | 0.9775 |  |

composite_7 = 1.022394 (stored vr_smooth = 1.022394) | composite_6 = 0.982442

### A16 / gpt2

| metric | VR (smooth) | excluded? |
|---|---|---|
| pod_cpu_usage | 0.8390 |  |
| pod_memory_bytes | 0.4069 | yes |
| pod_psi_cpu | 0.4753 |  |
| pod_latency_avg | 0.8699 |  |
| pod_throughput | 1.2721 |  |
| gpu_utilization | 2.7068 |  |
| gpu_power_watts | 1.3859 |  |

composite_7 = 1.136576 (stored vr_smooth = 1.136576) | composite_6 = 1.258182

### A16 / resnet152

| metric | VR (smooth) | excluded? |
|---|---|---|
| pod_cpu_usage | 0.8654 |  |
| pod_memory_bytes | 1.5011 | yes |
| pod_psi_cpu | 0.6331 |  |
| pod_latency_avg | 0.9766 |  |
| pod_throughput | 0.7682 |  |
| gpu_utilization | 0.8696 |  |
| gpu_power_watts | 1.2840 |  |

composite_7 = 0.985424 (stored vr_smooth = 0.985424) | composite_6 = 0.899485

### A16 / whisper

| metric | VR (smooth) | excluded? |
|---|---|---|
| pod_cpu_usage | 1.0496 |  |
| pod_memory_bytes | 3.1652 | yes |
| pod_psi_cpu | 1.4350 |  |
| pod_latency_avg | 1.1819 |  |
| pod_throughput | 0.9101 |  |
| gpu_utilization | 1.0472 |  |
| gpu_power_watts | 0.9884 |  |

composite_7 = 1.396778 (stored vr_smooth = 1.396778) | composite_6 = 1.102036

### A16 / yolo

| metric | VR (smooth) | excluded? |
|---|---|---|
| pod_cpu_usage | 0.7890 |  |
| pod_memory_bytes | 1.5943 | yes |
| pod_psi_cpu | 0.9648 |  |
| pod_latency_avg | 0.9316 |  |
| pod_throughput | 0.9070 |  |
| gpu_utilization | 0.8384 |  |
| gpu_power_watts | 1.2464 |  |

composite_7 = 1.038784 (stored vr_smooth = 1.038784) | composite_6 = 0.946191

## Tier3


### Tier3 / bert

| metric | VR (smooth) | excluded? |
|---|---|---|
| pod_cpu_usage | 0.9041 |  |
| pod_memory_bytes | 7.2590 | yes |
| pod_psi_cpu | 1.2646 |  |
| pod_latency_avg | 0.8755 |  |
| pod_throughput | 0.7731 |  |
| gpu_utilization | 0.9189 |  |
| gpu_power_watts | 1.2025 |  |

composite_7 = 1.885380 (stored vr_smooth = 1.885380) | composite_6 = 0.989784

### Tier3 / gpt2

| metric | VR (smooth) | excluded? |
|---|---|---|
| pod_cpu_usage | 1.2535 |  |
| pod_memory_bytes | 0.5282 | yes |
| pod_psi_cpu | 0.6074 |  |
| pod_latency_avg | 1.0207 |  |
| pod_throughput | 0.9863 |  |
| gpu_utilization | 1.3333 |  |
| gpu_power_watts | 0.7089 |  |

composite_7 = 0.919757 (stored vr_smooth = 0.919757) | composite_6 = 0.985012

### Tier3 / resnet152

| metric | VR (smooth) | excluded? |
|---|---|---|
| pod_cpu_usage | 1.1727 |  |
| pod_memory_bytes | 1.5249 | yes |
| pod_psi_cpu | 0.1836 |  |
| pod_latency_avg | 1.1323 |  |
| pod_throughput | 0.7018 |  |
| gpu_utilization | 0.9125 |  |
| gpu_power_watts | 1.8221 |  |

composite_7 = 1.064280 (stored vr_smooth = 1.064280) | composite_6 = 0.987512

### Tier3 / whisper

| metric | VR (smooth) | excluded? |
|---|---|---|
| pod_cpu_usage | 0.7984 |  |
| pod_memory_bytes | 0.3540 | yes |
| pod_psi_cpu | 0.8331 |  |
| pod_latency_avg | 0.8853 |  |
| pod_throughput | 0.7456 |  |
| gpu_utilization | 1.0290 |  |
| gpu_power_watts | 0.6670 |  |

composite_7 = 0.758915 (stored vr_smooth = 0.758915) | composite_6 = 0.826400

### Tier3 / yolo

| metric | VR (smooth) | excluded? |
|---|---|---|
| pod_cpu_usage | 0.8335 |  |
| pod_memory_bytes | 0.4321 | yes |
| pod_psi_cpu | 1.0638 |  |
| pod_latency_avg | 1.0369 |  |
| pod_throughput | 0.8745 |  |
| gpu_utilization | 0.6388 |  |
| gpu_power_watts | 2.6052 |  |

composite_7 = 1.069243 (stored vr_smooth = 1.069243) | composite_6 = 1.175432

## Tier2


### Tier2 / bert

| metric | VR (smooth) | excluded? |
|---|---|---|
| pod_cpu_usage | 0.8286 |  |
| pod_memory_bytes | 0.8325 | yes |
| pod_psi_cpu | 1.7974 |  |
| pod_latency_avg | 1.0568 |  |
| pod_throughput | 0.9002 |  |
| gpu_utilization | 0.9329 |  |
| gpu_power_watts | 1.6423 |  |

composite_7 = 1.141543 (stored vr_smooth = 1.141543) | composite_6 = 1.193049

### Tier2 / gpt2

| metric | VR (smooth) | excluded? |
|---|---|---|
| pod_cpu_usage | 0.8595 |  |
| pod_memory_bytes | 0.1126 | yes |
| pod_psi_cpu | 0.2587 |  |
| pod_latency_avg | 1.0525 |  |
| pod_throughput | 1.0616 |  |
| gpu_utilization | 0.7817 |  |
| gpu_power_watts | 0.1559 |  |

composite_7 = 0.611800 (stored vr_smooth = 0.611800) | composite_6 = 0.694996

### Tier2 / resnet152

| metric | VR (smooth) | excluded? |
|---|---|---|
| pod_cpu_usage | 0.8486 |  |
| pod_memory_bytes | 0.1191 | yes |
| pod_psi_cpu | 0.2517 |  |
| pod_latency_avg | 0.0536 |  |
| pod_throughput | 0.6232 |  |
| gpu_utilization | 0.6117 |  |
| gpu_power_watts | 0.0723 |  |

composite_7 = 0.368613 (stored vr_smooth = 0.368613) | composite_6 = 0.410191

### Tier2 / whisper

| metric | VR (smooth) | excluded? |
|---|---|---|
| pod_cpu_usage | 0.5361 |  |
| pod_memory_bytes | 1.4190 | yes |
| pod_psi_cpu | 1.3264 |  |
| pod_latency_avg | 1.3411 |  |
| pod_throughput | 0.8111 |  |
| gpu_utilization | 0.7070 |  |
| gpu_power_watts | 1.6007 |  |

composite_7 = 1.105908 (stored vr_smooth = 1.105908) | composite_6 = 1.053721

### Tier2 / yolo

| metric | VR (smooth) | excluded? |
|---|---|---|
| pod_cpu_usage | 0.9283 |  |
| pod_memory_bytes | 0.3515 | yes |
| pod_psi_cpu | 0.2092 |  |
| pod_latency_avg | 0.7684 |  |
| pod_throughput | 0.8612 |  |
| gpu_utilization | 0.7500 |  |
| gpu_power_watts | 1.5516 |  |

composite_7 = 0.774309 (stored vr_smooth = 0.774309) | composite_6 = 0.844781
