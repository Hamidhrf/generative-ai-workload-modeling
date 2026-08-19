# A16 vs Tier 3 Real-Data Wasserstein Distance

Real-vs-real distance (no model involved), computed on denormalized traces. 55x715=39325 samples per tier per workload per metric.

| Workload | pod_cpu_usage | pod_memory_bytes | pod_psi_cpu | pod_latency_avg | pod_throughput | gpu_utilization | gpu_memory_used | gpu_memory_total | gpu_power_watts | gpu_temperature |
|---|---|---|---|---|---|---|---|---|---|---|
| bert | 0.02834 | 2.084e+08 | 1.884e-05 | 0.002583 | 0.1363 | 1.801 | 632 | 1.461e+04 | 15.88 | 1.986 |
| gpt2 | 0.1023 | 2.688e+08 | 3.663e-05 | 0.5483 | 0.1909 | 2.325 | 630.7 | 1.461e+04 | 15.74 | 2.214 |
| resnet152 | 0.01843 | 1.775e+09 | 1.766e-05 | 0.007736 | 0.09437 | 2.556 | 446.2 | 1.461e+04 | 15.49 | 1.946 |
| whisper | 1.239 | 3.412e+08 | 0.3466 | 0.7504 | 0.2753 | 7.592 | 709.2 | 1.461e+04 | 15.4 | 2.573 |
| yolo | 0.02896 | 1.315e+09 | 8.935e-06 | 0.009154 | 0.1319 | 1.138 | 286.9 | 1.461e+04 | 15.68 | 1.685 |
