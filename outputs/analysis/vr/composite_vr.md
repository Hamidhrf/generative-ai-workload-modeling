# Composite VR re-aggregation (E2 resume)

Pure arithmetic re-aggregation of stored `vr_per_metric_smooth` vectors. `composite_7` = mean of all 7 trained metrics (must equal stored `vr_smooth`). `composite_6` = mean of 6, excluding `pod_memory_bytes` by name.

## Reconciliation check

composite_7 == stored vr_smooth to 5 decimals: **PASS for all 15/15 cells** (enforced by assertion in `vr_per_metric.py`; script would have raised otherwise).

## Spot checks

- bert A16 composite_6 = **0.982442** (expect ~0.983)
- bert Tier3 composite_6 = **0.989784** (expect ~0.990)

## Full composite_6 grid (all 15 cells, 3x5)

| tier | bert | gpt2 | resnet152 | whisper | yolo |
|---|---|---|---|---|---|
| A16 | 0.9824 | 1.2582 | 0.8995 | 1.1020 | 0.9462 |
| Tier3 | 0.9898 | 0.9850 | 0.9875 | 0.8264 | 1.1754 |
| Tier2 | 1.1930 | 0.6950 | 0.4102 | 1.0537 | 0.8448 |

- **whisper Tier3 `pod_memory_bytes` per-metric VR = 0.3540** (<= 0.513 expected) - a DEFLATION, confirming it is NOT the same inflation artifact as bert Tier3 (7.259, a >7x inflation).
- **A16 resnet152 composite_6 = 0.899485** (authoritative value from this reconciled per-metric grid; the 0.900 figure quoted in the E10 report is this same value rounded to 3 decimals (0.8995 -> 0.900) - RECONCILES, no discrepancy).

## Per-cell composite table

| tier | workload | composite_7 | composite_6 | |c7-1| | |c6-1| | pass c7 1-sided | pass c7 2-sided | pass c6 1-sided | pass c6 2-sided |
|---|---|---|---|---|---|---|---|---|---|
| A16 | bert | 1.0224 | 0.9824 | 0.0224 | 0.0176 | True | True | True | True |
| A16 | gpt2 | 1.1366 | 1.2582 | 0.1366 | 0.2582 | True | True | True | False |
| A16 | resnet152 | 0.9854 | 0.8995 | 0.0146 | 0.1005 | True | True | True | True |
| A16 | whisper | 1.3968 | 1.1020 | 0.3968 | 0.1020 | True | False | True | True |
| A16 | yolo | 1.0388 | 0.9462 | 0.0388 | 0.0538 | True | True | True | True |
| Tier3 | bert | 1.8854 | 0.9898 | 0.8854 | 0.0102 | True | False | True | True |
| Tier3 | gpt2 | 0.9198 | 0.9850 | 0.0802 | 0.0150 | True | True | True | True |
| Tier3 | resnet152 | 1.0643 | 0.9875 | 0.0643 | 0.0125 | True | True | True | True |
| Tier3 | whisper | 0.7589 | 0.8264 | 0.2411 | 0.1736 | False | False | True | True |
| Tier3 | yolo | 1.0692 | 1.1754 | 0.0692 | 0.1754 | True | True | True | True |
| Tier2 | bert | 1.1415 | 1.1930 | 0.1415 | 0.1930 | True | True | True | True |
| Tier2 | gpt2 | 0.6118 | 0.6950 | 0.3882 | 0.3050 | False | False | False | False |
| Tier2 | resnet152 | 0.3686 | 0.4102 | 0.6314 | 0.5898 | False | False | False | False |
| Tier2 | whisper | 1.1059 | 1.0537 | 0.1059 | 0.0537 | True | True | True | True |
| Tier2 | yolo | 0.7743 | 0.8448 | 0.2257 | 0.1552 | False | False | True | True |

## Per-tier summary

| tier | mean composite_7 | mean composite_6 | mean\|VR-1\| (c7) | mean\|VR-1\| (c6) | pass/5 c7 1-sided | pass/5 c7 2-sided | pass/5 c6 1-sided | pass/5 c6 2-sided |
|---|---|---|---|---|---|---|---|---|
| A16 | 1.1160 | 1.0377 | 0.1218 | 0.1064 | 5/5 | 4/5 | 5/5 | 4/5 |
| Tier3 | 1.1395 | 0.9928 | 0.2680 | 0.0773 | 4/5 | 3/5 | 5/5 | 5/5 |
| Tier2 | 0.8004 | 0.8393 | 0.2985 | 0.2594 | 2/5 | 2/5 | 3/5 | 3/5 |

## Tier-step deltas in mean|VR-1| under composite_6

- A16 -> Tier3 (hardware step): 0.0773 - 0.1064 = **-0.0291**
- Tier3 -> Tier2 (sharing step): 0.2594 - 0.0773 = **+0.1820**

## Flips: composite_7 vs composite_6 (threshold fixed)

| tier | workload | one-sided flip | two-sided flip |
|---|---|---|---|
| A16 | gpt2 | same | True->False |
| A16 | whisper | same | False->True |
| Tier3 | bert | same | False->True |
| Tier3 | whisper | False->True | False->True |
| Tier2 | yolo | False->True | False->True |

## Flips: one-sided vs two-sided (composite fixed) - the F7 exposure

| tier | workload | composite_7 flip | composite_6 flip |
|---|---|---|---|
| A16 | gpt2 | same | True->False |
| A16 | whisper | True->False | same |
| Tier3 | bert | True->False | same |
