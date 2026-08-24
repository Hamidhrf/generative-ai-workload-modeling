# E6 follow-up - Tier2/resnet152/fm=4.0 stability check (NOT committed)

Checkpoint: `/home/hamid/generative-ai-workload-modeling/models/phase4/timegan_s36_tier2_e6_fm4p0/resnet152/generator.pt` (already trained, no retrain). K=30 seeded draws (0..29), Gaussian fixed seed=42, minmax (PRIMARY) space.

| metric | Gaussian | S36 mean | S36 SD | wins/30 | margin (G - S36) | SD swamps margin? | verdict |
|---|---|---|---|---|---|---|---|
| gpu_utilization | 0.11786 | 0.07812 | 0.00085 | 30/30 | +0.03973 | False | REAL RECOVERY (stable) |
| pod_throughput | 0.12098 | 0.08207 | 0.00123 | 30/30 | +0.03891 | False | REAL RECOVERY (stable) |
| pod_cpu_usage | 0.07114 | 0.06139 | 0.00093 | 30/30 | +0.00974 | False | REAL RECOVERY (stable) |
