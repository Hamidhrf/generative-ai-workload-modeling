# Gaussian-Wasserstein discriminative check

S36: 30 independently seeded draws (mean+SD). Gaussian: one fixed seed (42). Both in PRIMARY min-max space, composite_6 (exclude `pod_memory_bytes`).

## Q1 - is normalized W saturated by the Gaussian the way VR is?

| tier | Gaussian pooled_6 | S36 pooled_6 (mean+-SD) | S36 below Gaussian? | relative margin |
|---|---|---|---|---|
| a16 | 0.0581 | 0.0383 +/- 0.0002 | True | +34.1% |
| tier3 | 0.0599 | 0.0357 +/- 0.0003 | True | +40.3% |
| tier2 | 0.0726 | 0.0740 +/- 0.0007 | False | -1.9% |

## Q2 - Tier2/Tier3 separation: S36-specific, or data property?

- Gaussian Tier2/Tier3 ratio: 1.2133x
- S36 Tier2/Tier3 ratio: 2.0706x

## Q3 - per-metric: where S36 beats the Gaussian

| tier | metric | Gaussian | S36 (mean) | S36 beats Gaussian? |
|---|---|---|---|---|
| a16 | pod_cpu_usage | 0.06973 | 0.05296 | True |
| a16 | pod_memory_bytes | 0.05760 | 0.01727 | True |
| a16 | pod_psi_cpu | 0.02930 | 0.01611 | True |
| a16 | pod_latency_avg | 0.03603 | 0.03738 | False |
| a16 | pod_throughput | 0.09151 | 0.05727 | True |
| a16 | gpu_utilization | 0.05467 | 0.04807 | True |
| a16 | gpu_power_watts | 0.06724 | 0.01772 | True |
| tier3 | pod_cpu_usage | 0.04642 | 0.03555 | True |
| tier3 | pod_memory_bytes | 0.05707 | 0.03287 | True |
| tier3 | pod_psi_cpu | 0.02980 | 0.00798 | True |
| tier3 | pod_latency_avg | 0.05768 | 0.02888 | True |
| tier3 | pod_throughput | 0.08304 | 0.07750 | True |
| tier3 | gpu_utilization | 0.06557 | 0.04010 | True |
| tier3 | gpu_power_watts | 0.07665 | 0.02446 | True |
| tier2 | pod_cpu_usage | 0.07068 | 0.08702 | False |
| tier2 | pod_memory_bytes | 0.07022 | 0.03826 | True |
| tier2 | pod_psi_cpu | 0.03968 | 0.01503 | True |
| tier2 | pod_latency_avg | 0.05410 | 0.03757 | True |
| tier2 | pod_throughput | 0.09025 | 0.11313 | False |
| tier2 | gpu_utilization | 0.08613 | 0.11960 | False |
| tier2 | gpu_power_watts | 0.09492 | 0.07177 | True |
