# 3-Way Cross-Tier Real-Data Wasserstein Distance

Real-vs-real distance (no model involved), computed on denormalized (raw physical unit) traces, per (workload, metric). A16=phase1_v3 (55 pods), Tier2 (28 pods, r=1..7), Tier3 (55 pods, r=1..10).

| Metric | Workload | A16-vs-Tier2 W | A16-vs-Tier3 W | Tier2-vs-Tier3 W |
|---|---|---|---|---|
| pod_cpu_usage | bert | 0.03227 | 0.02834 | 0.004094 |
| pod_cpu_usage | gpt2 | 0.2536 | 0.1023 | 0.153 |
| pod_cpu_usage | resnet152 | 0.01945 | 0.01843 | 0.001492 |
| pod_cpu_usage | whisper | 1.205 | 1.239 | 0.2022 |
| pod_cpu_usage | yolo | 0.04827 | 0.02896 | 0.01938 |
| pod_memory_bytes | bert | 2.806e+08 | 2.084e+08 | 7.569e+07 |
| pod_memory_bytes | gpt2 | 3.554e+08 | 2.688e+08 | 9.626e+07 |
| pod_memory_bytes | resnet152 | 1.687e+09 | 1.775e+09 | 8.793e+07 |
| pod_memory_bytes | whisper | 4.467e+08 | 3.412e+08 | 1.233e+08 |
| pod_memory_bytes | yolo | 1.247e+09 | 1.315e+09 | 7.058e+07 |
| pod_psi_cpu | bert | 2.245e-05 | 1.884e-05 | 1.224e-05 |
| pod_psi_cpu | gpt2 | 4.805e-05 | 3.663e-05 | 5.191e-05 |
| pod_psi_cpu | resnet152 | 9.319e-06 | 1.766e-05 | 1.955e-05 |
| pod_psi_cpu | whisper | 0.391 | 0.3466 | 0.04443 |
| pod_psi_cpu | yolo | 1.549e-05 | 8.935e-06 | 9.494e-06 |
| pod_latency_avg | bert | 0.003337 | 0.002583 | 0.0007539 |
| pod_latency_avg | gpt2 | 1.008 | 0.5483 | 0.4601 |
| pod_latency_avg | resnet152 | 0.007963 | 0.007736 | 0.0002325 |
| pod_latency_avg | whisper | 0.8874 | 0.7504 | 0.1412 |
| pod_latency_avg | yolo | 0.0165 | 0.009154 | 0.007349 |
| pod_throughput | bert | 0.1348 | 0.1363 | 0.02299 |
| pod_throughput | gpt2 | 0.6525 | 0.1909 | 0.4707 |
| pod_throughput | resnet152 | 0.08238 | 0.09437 | 0.01682 |
| pod_throughput | whisper | 0.464 | 0.2753 | 0.2416 |
| pod_throughput | yolo | 0.2065 | 0.1319 | 0.07458 |
| gpu_utilization | bert | 1.184 | 1.801 | 0.9511 |
| gpu_utilization | gpt2 | 3.614 | 2.325 | 4.665 |
| gpu_utilization | resnet152 | 1.986 | 2.556 | 0.9588 |
| gpu_utilization | whisper | 4.743 | 7.592 | 4.24 |
| gpu_utilization | yolo | 1.006 | 1.138 | 0.5651 |
| gpu_memory_used | bert | 142 | 632 | 491.3 |
| gpu_memory_used | gpt2 | 143.8 | 630.7 | 489.5 |
| gpu_memory_used | resnet152 | 46.02 | 446.2 | 492.2 |
| gpu_memory_used | whisper | 214.7 | 709.2 | 507.7 |
| gpu_memory_used | yolo | 203.4 | 286.9 | 490.4 |
| gpu_memory_total | bert | 328.5 | 1.461e+04 | 1.458e+04 |
| gpu_memory_total | gpt2 | 328.5 | 1.461e+04 | 1.458e+04 |
| gpu_memory_total | resnet152 | 328.5 | 1.461e+04 | 1.458e+04 |
| gpu_memory_total | whisper | 328.5 | 1.461e+04 | 1.458e+04 |
| gpu_memory_total | yolo | 328.5 | 1.461e+04 | 1.458e+04 |
| gpu_power_watts | bert | 17.12 | 15.88 | 2.594 |
| gpu_power_watts | gpt2 | 18.77 | 15.74 | 4.264 |
| gpu_power_watts | resnet152 | 17.04 | 15.49 | 2.933 |
| gpu_power_watts | whisper | 17.34 | 15.4 | 3.147 |
| gpu_power_watts | yolo | 17.33 | 15.68 | 3.1 |
| gpu_temperature | bert | 1.24 | 1.986 | 2.508 |
| gpu_temperature | gpt2 | 1.457 | 2.214 | 2.888 |
| gpu_temperature | resnet152 | 1.337 | 1.946 | 2.614 |
| gpu_temperature | whisper | 1.286 | 2.573 | 2.668 |
| gpu_temperature | yolo | 1.426 | 1.685 | 2.563 |
