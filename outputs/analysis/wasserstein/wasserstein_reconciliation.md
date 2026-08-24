# Batch A Task 4 - E4 reconciliation, generation variance, exclusion

## 4a - raw pooled Wasserstein reconciliation (held-constant setup)

| tier | committed ref | method (i) pooled | i % gap | method (ii) per-r avg | ii % gap |
|---|---|---|---|---|---|
| a16 | 3.1928 | 2.4383 | -23.6% | 3.3724 | +5.6% |
| tier3 | 16.5622 | 9.8218 | -40.7% | 17.7684 | +7.3% |
| tier2 | 16.9319 | 11.0863 | -34.5% | 19.5779 | +15.6% |

method (ii) reproduces committed figures within 15%: **False**

### True committed-function reproduction attempt

- a16: {"tier_mean": 3.1457526565706875, "per_workload": {"bert": 1.6513046240346918, "gpt2": 4.634659914803009, "resnet152": 3.108389246437766, "whisper": 5.42216703662289, "yolo": 1.2374445152482518}, "pct_gap_vs_committed": -1.4735449583222437}
- tier3: {"tier_mean": 14.289827743740195, "per_workload": {"bert": 12.605134082431352, "gpt2": 18.363958336290874, "resnet152": 14.687288686141763, "whisper": 19.030078281126826, "yolo": 7.439857980908256}, "pct_gap_vs_committed": -13.720231951430398}
- tier2: {"tier_mean": 14.585799225570907, "per_workload": {"bert": 11.993907824761942, "gpt2": 17.3440094151586, "resnet152": 15.695738825797877, "whisper": 20.840242076559292, "yolo": 7.055097985576821}, "pct_gap_vs_committed": -13.856098691990217}


## 4b - 30-draw generation variance, minmax pooled_7 separation (Tier2 vs Tier3)

- Tier3: mean=0.0353 sd=0.0007
- Tier2: mean=0.0689 sd=0.0013
- Separation (diff, tier2-tier3): mean=0.0336 sd=0.0011
- Separation (ratio, tier2/tier3): mean=1.9506 sd=0.0367

## 4c - normalized pooled W (minmax) with vs without pod_memory_bytes

| tier | mean minmax7 (with memory) | mean minmax6 (excl memory) |
|---|---|---|
| a16 | 0.0353 | 0.0384 |
| tier3 | 0.0350 | 0.0355 |
| tier2 | 0.0686 | 0.0735 |
