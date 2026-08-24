# E4 - model-vs-real Wasserstein, recomputed scale-free

PRIMARY = min-max training space (tier's own combined_normalization.json). SECONDARY = std-normalized (raw_w / real stdev). raw_pooled_w is raw-unit-only (MB convention on pod_memory_bytes), retired from paper tables, kept only for reconciliation with legacy validation_report.json figures.

Seed: torch.manual_seed(42) + np.random.seed(42), set once, deliberately - generation elsewhere in this codebase is unseeded.

## Per-metric table (5 workloads x 3 tiers x 7 metrics)

| tier | workload | metric | minmax_w (primary) | std_w (secondary) | raw_w (native units) |
|---|---|---|---|---|---|
| a16 | bert | pod_cpu_usage | 0.06188 | 0.21069 | 0.0119703 |
| a16 | bert | pod_memory_bytes | 0.01280 | 0.08048 | 4.21762e+06 |
| a16 | bert | pod_psi_cpu | 0.01539 | 0.22467 | 1.64458e-05 |
| a16 | bert | pod_latency_avg | 0.02645 | 0.25360 | 6.15733e-05 |
| a16 | bert | pod_throughput | 0.05730 | 0.16672 | 0.225032 |
| a16 | bert | gpu_utilization | 0.02931 | 0.14253 | 0.359053 |
| a16 | bert | gpu_power_watts | 0.02439 | 0.19441 | 0.932821 |
| a16 | gpt2 | pod_cpu_usage | 0.07606 | 0.30090 | 0.0633409 |
| a16 | gpt2 | pod_memory_bytes | 0.04213 | 0.22456 | 1.83339e+07 |
| a16 | gpt2 | pod_psi_cpu | 0.01907 | 0.21079 | 3.80334e-05 |
| a16 | gpt2 | pod_latency_avg | 0.04244 | 0.14758 | 0.0850652 |
| a16 | gpt2 | pod_throughput | 0.03613 | 0.21264 | 0.0440142 |
| a16 | gpt2 | gpu_utilization | 0.13364 | 0.84543 | 6.68176 |
| a16 | gpt2 | gpu_power_watts | 0.02449 | 0.17036 | 1.00538 |
| a16 | resnet152 | pod_cpu_usage | 0.05178 | 0.19141 | 0.00489703 |
| a16 | resnet152 | pod_memory_bytes | 0.01431 | 0.09096 | 1.34391e+07 |
| a16 | resnet152 | pod_psi_cpu | 0.00639 | 0.14777 | 9.40067e-06 |
| a16 | resnet152 | pod_latency_avg | 0.03167 | 0.32078 | 0.000239125 |
| a16 | resnet152 | pod_throughput | 0.09432 | 0.26470 | 0.406436 |
| a16 | resnet152 | gpu_utilization | 0.02027 | 0.26096 | 0.881557 |
| a16 | resnet152 | gpu_power_watts | 0.01953 | 0.14135 | 0.672772 |
| a16 | whisper | pod_cpu_usage | 0.02175 | 0.12372 | 0.134364 |
| a16 | whisper | pod_memory_bytes | 0.00483 | 0.17250 | 3.29263e+07 |
| a16 | whisper | pod_psi_cpu | 0.02673 | 0.14186 | 0.0165708 |
| a16 | whisper | pod_latency_avg | 0.06032 | 0.27062 | 0.121435 |
| a16 | whisper | pod_throughput | 0.02628 | 0.18649 | 0.19397 |
| a16 | whisper | gpu_utilization | 0.03140 | 0.21411 | 2.44914 |
| a16 | whisper | gpu_power_watts | 0.00838 | 0.06000 | 0.462067 |
| a16 | yolo | pod_cpu_usage | 0.05934 | 0.23818 | 0.0142925 |
| a16 | yolo | pod_memory_bytes | 0.00948 | 0.06012 | 8.54676e+06 |
| a16 | yolo | pod_psi_cpu | 0.01308 | 0.22763 | 1.97799e-05 |
| a16 | yolo | pod_latency_avg | 0.03095 | 0.13386 | 0.00162646 |
| a16 | yolo | pod_throughput | 0.07182 | 0.22577 | 0.317313 |
| a16 | yolo | gpu_utilization | 0.02263 | 0.07717 | 0.158434 |
| a16 | yolo | gpu_power_watts | 0.01154 | 0.07771 | 0.362993 |
| tier3 | bert | pod_cpu_usage | 0.03198 | 0.15084 | 0.00636873 |
| tier3 | bert | pod_memory_bytes | 0.02861 | 0.20953 | 5.05398e+07 |
| tier3 | bert | pod_psi_cpu | 0.00561 | 0.16616 | 2.22669e-05 |
| tier3 | bert | pod_latency_avg | 0.01633 | 0.17255 | 5.7583e-05 |
| tier3 | bert | pod_throughput | 0.08929 | 0.26561 | 0.395485 |
| tier3 | bert | gpu_utilization | 0.03036 | 0.13987 | 0.121425 |
| tier3 | bert | gpu_power_watts | 0.01600 | 0.10438 | 1.78552 |
| tier3 | gpt2 | pod_cpu_usage | 0.05573 | 0.20435 | 0.049251 |
| tier3 | gpt2 | pod_memory_bytes | 0.04974 | 0.34621 | 1.15044e+08 |
| tier3 | gpt2 | pod_psi_cpu | 0.00328 | 0.10786 | 4.20817e-05 |
| tier3 | gpt2 | pod_latency_avg | 0.02506 | 0.09732 | 0.0323799 |
| tier3 | gpt2 | pod_throughput | 0.06448 | 0.41978 | 0.123159 |
| tier3 | gpt2 | gpu_utilization | 0.04688 | 0.30616 | 2.32042 |
| tier3 | gpt2 | gpu_power_watts | 0.02997 | 0.20723 | 3.71039 |
| tier3 | resnet152 | pod_cpu_usage | 0.01261 | 0.30209 | 0.00516453 |
| tier3 | resnet152 | pod_memory_bytes | 0.03143 | 0.17415 | 4.41892e+07 |
| tier3 | resnet152 | pod_psi_cpu | 0.00178 | 0.07854 | 2.52191e-05 |
| tier3 | resnet152 | pod_latency_avg | 0.01761 | 0.14313 | 7.64607e-05 |
| tier3 | resnet152 | pod_throughput | 0.09647 | 0.27983 | 0.456518 |
| tier3 | resnet152 | gpu_utilization | 0.03338 | 0.16549 | 0.150217 |
| tier3 | resnet152 | gpu_power_watts | 0.02082 | 0.13459 | 2.30084 |
| tier3 | whisper | pod_cpu_usage | 0.05317 | 0.21229 | 0.158377 |
| tier3 | whisper | pod_memory_bytes | 0.03545 | 0.26898 | 1.02261e+08 |
| tier3 | whisper | pod_psi_cpu | 0.02298 | 0.08578 | 0.00882468 |
| tier3 | whisper | pod_latency_avg | 0.01537 | 0.11071 | 0.0321135 |
| tier3 | whisper | pod_throughput | 0.06431 | 0.34059 | 0.21013 |
| tier3 | whisper | gpu_utilization | 0.02044 | 0.16707 | 0.551746 |
| tier3 | whisper | gpu_power_watts | 0.02508 | 0.17425 | 3.10373 |
| tier3 | yolo | pod_cpu_usage | 0.02533 | 0.08845 | 0.0025572 |
| tier3 | yolo | pod_memory_bytes | 0.01283 | 0.08308 | 2.20928e+07 |
| tier3 | yolo | pod_psi_cpu | 0.00921 | 0.18581 | 1.21555e-05 |
| tier3 | yolo | pod_latency_avg | 0.07053 | 0.20957 | 0.00106591 |
| tier3 | yolo | pod_throughput | 0.06871 | 0.20105 | 0.308592 |
| tier3 | yolo | gpu_utilization | 0.06841 | 0.21514 | 0.171014 |
| tier3 | yolo | gpu_power_watts | 0.03204 | 0.20438 | 3.49558 |
| tier2 | bert | pod_cpu_usage | 0.05844 | 0.24247 | 0.00964557 |
| tier2 | bert | pod_memory_bytes | 0.03391 | 0.15608 | 4.06376e+07 |
| tier2 | bert | pod_psi_cpu | 0.01210 | 0.22083 | 2.56825e-05 |
| tier2 | bert | pod_latency_avg | 0.02742 | 0.37422 | 3.21647e-05 |
| tier2 | bert | pod_throughput | 0.06954 | 0.20006 | 0.296928 |
| tier2 | bert | gpu_utilization | 0.10354 | 0.29179 | 0.335271 |
| tier2 | bert | gpu_power_watts | 0.07703 | 0.39003 | 6.54304 |
| tier2 | gpt2 | pod_cpu_usage | 0.06870 | 0.25684 | 0.0468299 |
| tier2 | gpt2 | pod_memory_bytes | 0.04694 | 0.26438 | 9.53101e+07 |
| tier2 | gpt2 | pod_psi_cpu | 0.00642 | 0.15716 | 4.95973e-05 |
| tier2 | gpt2 | pod_latency_avg | 0.03931 | 0.19094 | 0.00615853 |
| tier2 | gpt2 | pod_throughput | 0.04055 | 0.15457 | 0.0778656 |
| tier2 | gpt2 | gpu_utilization | 0.09262 | 0.35421 | 2.43831 |
| tier2 | gpt2 | gpu_power_watts | 0.03452 | 0.18337 | 3.06495 |
| tier2 | resnet152 | pod_cpu_usage | 0.04320 | 0.18362 | 0.00266728 |
| tier2 | resnet152 | pod_memory_bytes | 0.04571 | 0.20943 | 6.1761e+07 |
| tier2 | resnet152 | pod_psi_cpu | 0.01057 | 0.22353 | 2.33685e-05 |
| tier2 | resnet152 | pod_latency_avg | 0.02171 | 0.32972 | 0.000143659 |
| tier2 | resnet152 | pod_throughput | 0.20783 | 0.57601 | 0.931093 |
| tier2 | resnet152 | gpu_utilization | 0.16481 | 0.46435 | 0.582892 |
| tier2 | resnet152 | gpu_power_watts | 0.09459 | 0.47540 | 7.9129 |
| tier2 | whisper | pod_cpu_usage | 0.20584 | 0.69987 | 0.624637 |
| tier2 | whisper | pod_memory_bytes | 0.02893 | 0.18813 | 7.85119e+07 |
| tier2 | whisper | pod_psi_cpu | 0.03257 | 0.12003 | 0.00881941 |
| tier2 | whisper | pod_latency_avg | 0.07019 | 0.35141 | 0.0515163 |
| tier2 | whisper | pod_throughput | 0.04955 | 0.23220 | 0.152929 |
| tier2 | whisper | gpu_utilization | 0.03931 | 0.27400 | 1.0403 |
| tier2 | whisper | gpu_power_watts | 0.03687 | 0.19409 | 3.31192 |
| tier2 | yolo | pod_cpu_usage | 0.06175 | 0.22431 | 0.00280251 |
| tier2 | yolo | pod_memory_bytes | 0.02228 | 0.11286 | 3.60478e+07 |
| tier2 | yolo | pod_psi_cpu | 0.01807 | 0.37669 | 1.97176e-05 |
| tier2 | yolo | pod_latency_avg | 0.03238 | 0.23416 | 9.78355e-05 |
| tier2 | yolo | pod_throughput | 0.19125 | 0.53431 | 0.858864 |
| tier2 | yolo | gpu_utilization | 0.19515 | 0.55905 | 0.329237 |
| tier2 | yolo | gpu_power_watts | 0.11803 | 0.58783 | 9.78423 |

## Per (tier, workload) pooled summary

| tier | workload | normalized_pooled_w (minmax) | normalized_pooled_w_std | raw_pooled_w (MB-conv) |
|---|---|---|---|---|
| a16 | bert | 0.0325 | 0.1819 | 0.8209 |
| a16 | gpt2 | 0.0534 | 0.3018 | 3.7448 |
| a16 | resnet152 | 0.0340 | 0.2026 | 2.2007 |
| a16 | whisper | 0.0257 | 0.1670 | 5.1863 |
| a16 | yolo | 0.0313 | 0.1486 | 1.3431 |
| tier3 | bert | 0.0312 | 0.1727 | 7.5498 |
| tier3 | gpt2 | 0.0393 | 0.2413 | 17.3257 |
| tier3 | resnet152 | 0.0306 | 0.1825 | 6.7289 |
| tier3 | whisper | 0.0338 | 0.1942 | 15.1895 |
| tier3 | yolo | 0.0410 | 0.1696 | 3.7245 |
| tier2 | bert | 0.0546 | 0.2679 | 6.8318 |
| tier2 | gpt2 | 0.0470 | 0.2231 | 14.4206 |
| tier2 | resnet152 | 0.0841 | 0.3517 | 10.1701 |
| tier2 | whisper | 0.0662 | 0.2942 | 11.9574 |
| tier2 | yolo | 0.0913 | 0.3756 | 6.7176 |

## Per-tier aggregate (mean across 5 workloads)

| tier | mean normalized_pooled_w (minmax) | mean normalized_pooled_w_std | mean raw_pooled_w (MB-conv) |
|---|---|---|---|
| a16 | 0.0354 | 0.2004 | 2.6592 |
| tier3 | 0.0352 | 0.1921 | 10.1037 |
| tier2 | 0.0686 | 0.3025 | 10.0195 |

## C6 ratios vs A16 (the ~5x number, recomputed per space)

| space | tier2/a16 | tier3/a16 |
|---|---|---|
| raw_pooled_w | 3.77x | 3.80x |
| normalized_pooled_w (minmax, PRIMARY) | 1.94x | 0.99x |
| normalized_pooled_w_std (SECONDARY) | 1.51x | 0.96x |

**C6 outcome, minmax (primary) space: intermediate**

**C6 outcome, std (secondary) space: intermediate**

## Tier2 vs Tier3 explicit comparison

- raw_pooled_w: tier2=10.0195, tier3=10.1037, relative gap=0.8%
- normalized_pooled_w (minmax): tier2=0.0686, tier3=0.0352, relative gap=64.4%
- normalized_pooled_w_std: tier2=0.3025, tier3=0.1921, relative gap=44.7%

raw_pooled_w is expected to reconcile with outputs/phase4/validation/s36{,_tier2,_tier3}/validation_report.json mean_wasserstein_all (~3.19 a16, ~16.93 tier2, ~16.56 tier3) only up to generation noise (unseeded elsewhere; this script seeds torch+numpy at 42 but n_samples and exact real-pod pairing differ from the original validation run) - small mismatches are expected, not errors.

