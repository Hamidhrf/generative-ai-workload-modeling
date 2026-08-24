# E1 report - matched-size control retrain of Tier 3 (NOT committed)

tier3 = committed full Tier3 (r=1..10, n=55/wl). tier3_matched = this E1 retrain (r=1..7, n=28/wl, same sharing mode + gpu_utilization imputation as tier3). tier2 = committed Tier2 MIG (r=1..7, n=28/wl).

## Q1 - composite_6 VR headline

| workload | tier3 (full) | tier3_matched | tier2 |
|---|---|---|---|
| bert | 0.9898 | 1.1795 | 1.1930 |
| gpt2 | 0.9850 | 0.6802 | 0.6950 |
| resnet152 | 0.9875 | 0.5369 | 0.4102 |
| whisper | 0.8264 | 0.2640 | 1.0537 |
| yolo | 1.1754 | 0.7881 | 0.8448 |
| **MEAN** | 0.9928 | 0.6897 | 0.8393 |

## Q2 - DECISION-CRITICAL: per-metric minmax Wasserstein, S36 vs Gaussian, contention metrics

| tier | metric | gaussian | S36 mean (K=30) | S36 beats Gaussian? |
|---|---|---|---|---|
| tier3 | gpu_utilization | 0.06557 | 0.04010 | True |
| tier3 | pod_throughput | 0.08304 | 0.07750 | True |
| tier3 | pod_cpu_usage | 0.04642 | 0.03555 | True |
| tier3_matched | gpu_utilization | 0.06709 | 0.07141 | False |
| tier3_matched | pod_throughput | 0.08451 | 0.09259 | False |
| tier3_matched | pod_cpu_usage | 0.07205 | 0.11765 | False |
| tier2 | gpu_utilization | 0.08613 | 0.11960 | False |
| tier2 | pod_throughput | 0.09025 | 0.11313 | False |
| tier2 | pod_cpu_usage | 0.07068 | 0.08702 | False |

## Q3 - real-trace resampler floor (composite_6, K=30)

| tier | n_traces/wl | mean of per-workload resampler means | mean of per-workload resampler SDs |
|---|---|---|---|
| tier3 | 55 | 0.9556 | 0.1168 |
| tier3_matched | 28 | 0.8908 | 0.1280 |
| tier2 | 28 | 0.9070 | 0.1266 |
