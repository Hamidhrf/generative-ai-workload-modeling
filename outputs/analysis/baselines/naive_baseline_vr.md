# E10 - naive-baseline VR under the exact S36 estimator

K=30 seeded draws per baseline. Composite_6 (exclude `pod_memory_bytes` by name) is PRIMARY, composite_7 secondary. `compute_vr_per_metric` imported unmodified from `eval_s36.py` (tier-agnostic, confirmed byte-identical across tiers).

## Batch A Task 2: Gaussian pass counts per tier

| tier | Gaussian pass one-sided (>0.8) | Gaussian pass two-sided [0.8,1.25] | S36 pass one-sided (ref) | S36 pass two-sided (ref) |
|---|---|---|---|---|
| a16 | 5/5 | 5/5 | 5/5 | 4/5 |
| tier3 | 5/5 | 5/5 | 5/5 | 5/5 |
| tier2 | 5/5 | 5/5 | 3/5 | 3/5 |

**Verdict: Gaussian passes Tier3 5/5 (one-sided). It passes every cell S36 passes on Tier3 -> VR CANNOT DISTINGUISH S36 FROM I.I.D. NOISE ON TIER3 under the one-sided rule. VR passing on Tier3 is not evidence of learned temporal/cross-pod structure there; it is consistent with the estimator's own noise floor / correct-marginals-only baseline.**

## DECISION-CRITICAL: resnet152 Tier2 under both baselines

- S36 headline: composite_6=0.4102, composite_7=0.3686 (n_traces=28)
- Real-trace resampler (noise floor): mean_c6=0.8872 SD=0.1880
- Per-metric Gaussian: mean_c6=0.9119 SD=0.0035

**S4 finding: Gaussian baseline scores NEAR 1.0 by construction on resnet152 Tier2 -> S36's 0.369/0.410 is a HARDER failure than 'MIG is different': a method with only correct marginals (no temporal/cross-pod structure) already matches the target under this estimator, so S36 undershooting variance this much needs the tougher explanation (real model/mode-collapse deficiency), not just an estimator/data quirk of this cell.**

## Per-tier resampler noise floor (composite_6)

| tier | mean of per-workload resampler means | mean of per-workload resampler SDs | max |mean-1| across workloads |
|---|---|---|---|
| a16 | 0.9179 | 0.1463 | 0.1321 |
| tier3 | 0.9582 | 0.1192 | 0.1047 |
| tier2 | 0.9070 | 0.1266 | 0.1779 |

## Per-cell table

| tier | workload | n | S36 headline c6 | S36 headline c7 | resampler mean c6 | resampler SD c6 | resampler mean c7 | gaussian mean c6 | gaussian SD c6 | gaussian mean c7 |
|---|---|---|---|---|---|---|---|---|---|---|
| a16 | bert | 55 | 0.9824 | 1.0224 | 0.8794 | 0.1014 | 0.9070 | 0.8668 | 0.0019 | 0.8859 |
| a16 | gpt2 | 55 | 1.2582 | 1.1366 | 0.9276 | 0.1438 | 0.9297 | 0.9600 | 0.0025 | 0.9657 |
| a16 | resnet152 | 55 | 0.8995 | 0.9854 | 0.8679 | 0.0788 | 0.8795 | 0.8781 | 0.0025 | 0.8954 |
| a16 | whisper | 55 | 1.1020 | 1.3968 | 1.0006 | 0.3114 | 0.9825 | 0.9996 | 0.0028 | 0.9997 |
| a16 | yolo | 55 | 0.9462 | 1.0388 | 0.9139 | 0.0964 | 0.9360 | 0.9049 | 0.0027 | 0.9187 |
| tier3 | bert | 55 | 0.9898 | 1.8854 | 0.9268 | 0.0985 | 0.9402 | 0.9263 | 0.0028 | 0.9368 |
| tier3 | gpt2 | 55 | 0.9850 | 0.9198 | 0.9989 | 0.1299 | 0.9967 | 0.9896 | 0.0018 | 0.9912 |
| tier3 | resnet152 | 55 | 0.9875 | 1.0643 | 0.9917 | 0.1100 | 0.9949 | 0.9800 | 0.0037 | 0.9832 |
| tier3 | whisper | 55 | 0.8264 | 0.7589 | 0.9782 | 0.1431 | 0.9758 | 1.0005 | 0.0034 | 1.0005 |
| tier3 | yolo | 55 | 1.1754 | 1.0692 | 0.8953 | 0.1146 | 0.9223 | 0.8829 | 0.0029 | 0.8998 |
| tier2 | bert | 28 | 1.1930 | 1.1415 | 0.8221 | 0.1076 | 0.8434 | 0.8328 | 0.0038 | 0.8564 |
| tier2 | gpt2 | 28 | 0.6950 | 0.6118 | 0.9905 | 0.1053 | 0.9996 | 0.9855 | 0.0041 | 0.9876 |
| tier2 | resnet152 | 28 | 0.4102 | 0.3686 | 0.8872 | 0.1880 | 0.8877 | 0.9119 | 0.0035 | 0.9241 |
| tier2 | whisper | 28 | 1.0537 | 1.1059 | 1.0071 | 0.1315 | 1.0069 | 1.0000 | 0.0034 | 1.0003 |
| tier2 | yolo | 28 | 0.8448 | 0.7743 | 0.8280 | 0.1007 | 0.8248 | 0.8601 | 0.0032 | 0.8797 |
