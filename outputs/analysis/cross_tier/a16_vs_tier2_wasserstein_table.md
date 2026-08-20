# A16 vs Tier 2 Real-Data Wasserstein Distance

Real-vs-real distance (no model involved), computed on denormalized traces.

| Workload | pod_cpu_usage | pod_memory_bytes | pod_psi_cpu | pod_latency_avg | pod_throughput | gpu_utilization | gpu_memory_used | gpu_memory_total | gpu_power_watts | gpu_temperature |
|---|---|---|---|---|---|---|---|---|---|---|
| bert | 0.03227 | 2.806e+08 | 2.245e-05 | 0.003337 | 0.1348 | 1.184 | 142 | 328.5 | 17.12 | 1.24 |
| gpt2 | 0.2536 | 3.554e+08 | 4.805e-05 | 1.008 | 0.6525 | 3.614 | 143.8 | 328.5 | 18.77 | 1.457 |
| resnet152 | 0.01945 | 1.687e+09 | 9.319e-06 | 0.007963 | 0.08238 | 1.986 | 46.02 | 328.5 | 17.04 | 1.337 |
| whisper | 1.205 | 4.467e+08 | 0.391 | 0.8874 | 0.464 | 4.743 | 214.7 | 328.5 | 17.34 | 1.286 |
| yolo | 0.04827 | 1.247e+09 | 1.549e-05 | 0.0165 | 0.2065 | 1.006 | 203.4 | 328.5 | 17.33 | 1.426 |
