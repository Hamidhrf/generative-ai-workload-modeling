# Batch A Task 3 - per-workload floor + floor-corrected tier-step deltas

## Per (tier, workload): S36|VR-1| vs resampler floor|VR-1|

| tier | workload | n | S36 \|VR-1\| (c6) | resampler floor \|VR-1\| | floor-corrected |
|---|---|---|---|---|---|
| a16 | bert | 55 | 0.0176 | 0.1206 | -0.1030 |
| a16 | gpt2 | 55 | 0.2582 | 0.0724 | +0.1858 |
| a16 | resnet152 | 55 | 0.1005 | 0.1321 | -0.0316 |
| a16 | whisper | 55 | 0.1020 | 0.0006 | +0.1015 |
| a16 | yolo | 55 | 0.0538 | 0.0861 | -0.0323 |
| tier3 | bert | 55 | 0.0102 | 0.0732 | -0.0630 |
| tier3 | gpt2 | 55 | 0.0150 | 0.0011 | +0.0139 |
| tier3 | resnet152 | 55 | 0.0125 | 0.0083 | +0.0042 |
| tier3 | whisper | 55 | 0.1736 | 0.0218 | +0.1518 |
| tier3 | yolo | 55 | 0.1754 | 0.1047 | +0.0707 |
| tier2 | bert | 28 | 0.1930 | 0.1779 | +0.0152 |
| tier2 | gpt2 | 28 | 0.3050 | 0.0095 | +0.2956 |
| tier2 | resnet152 | 28 | 0.5898 | 0.1128 | +0.4770 |
| tier2 | whisper | 28 | 0.0537 | 0.0071 | +0.0466 |
| tier2 | yolo | 28 | 0.1552 | 0.1720 | -0.0168 |

## Per-tier floor vs trace count

- Tier3 floor = 0.0418 (n=55)
- Tier2 floor = 0.0959 (n=28)
- Tier2/Tier3 floor ratio = 2.29x; trace-count ratio (Tier3/Tier2) = 1.96x
- Tier2 floor worse and tracks lower trace count: **True**

## Tier-step deltas against both yardsticks

| step | raw delta | generation-SD yardstick | outside gen? | resampler-floor yardstick | outside floor? |
|---|---|---|---|---|---|
| hardware (A16->Tier3) | -0.0291 | 0.0131 | True | 0.0923 | False |
| sharing (Tier3->Tier2) | +0.1820 | 0.0375 | True | 0.1046 | True |

## Floor-corrected tier-step deltas

- Floor-corrected mean|VR-1| per tier: A16=+0.0241, Tier3=+0.0355, Tier2=+0.1635
- **Corrected hardware step (A16->Tier3): +0.0115** (raw was -0.0291)
- **Corrected sharing step (Tier3->Tier2): +0.1280** (raw was +0.1820)
