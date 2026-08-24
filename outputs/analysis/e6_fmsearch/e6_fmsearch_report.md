# E6 report - fm-weight search on low-data-budget failures (NOT committed)

Frozen S27 hyperparameters held fixed except lambda_fm_stat, searched per (tier, workload). 'beats Gaussian' = S36's minmax-space Wasserstein distance to real is SMALLER than the i.i.d. per-metric Gaussian baseline's, on that contention metric.

## All variants

| tier | workload | fm | composite_6 VR | gpu_util beats G? | throughput beats G? | cpu beats G? | #/3 |
|---|---|---|---|---|---|---|---|
| tier2 | gpt2 | 0.125 | 0.7433 | False | False | False | 0/3 |
| tier2 | gpt2 | 0.25 | 0.6182 | False | False | False | 0/3 |
| tier2 | gpt2 | 0.5 | 0.5118 | False | False | False | 0/3 |
| tier2 | resnet152 | 0.125 | 0.4977 | False | False | False | 0/3 |
| tier2 | resnet152 | 0.25 | 0.5168 | True | False | True | 2/3 |
| tier2 | resnet152 | 0.5 | 0.3788 | False | False | True | 1/3 |
| tier2 | resnet152 | 2.0 | 0.4066 | False | True | True | 2/3 |
| tier2 | resnet152 | 4.0 | 0.3956 | True | True | True | 3/3 |
| tier3_matched | gpt2 | 0.125 | 0.8575 | False | False | False | 0/3 |
| tier3_matched | gpt2 | 0.25 | 0.5101 | False | False | False | 0/3 |
| tier3_matched | gpt2 | 0.5 | 0.7737 | False | False | False | 0/3 |
| tier3_matched | resnet152 | 0.125 | 0.4704 | False | False | False | 0/3 |
| tier3_matched | resnet152 | 0.25 | 0.6917 | False | False | True | 1/3 |
| tier3_matched | resnet152 | 0.5 | 0.5402 | False | False | True | 1/3 |
| tier3_matched | resnet152 | 2.0 | 0.5982 | False | True | True | 2/3 |
| tier3_matched | resnet152 | 4.0 | 0.5608 | False | False | True | 1/3 |
| tier3_matched | whisper | 0.125 | 0.6569 | False | False | True | 1/3 |
| tier3_matched | whisper | 0.25 | 1.5061 | False | False | True | 1/3 |
| tier3_matched | whisper | 0.5 | 0.6166 | False | True | True | 2/3 |
| tier3_matched | yolo | 0.125 | 0.7933 | True | False | False | 1/3 |
| tier3_matched | yolo | 0.25 | 0.9367 | True | False | True | 2/3 |
| tier3_matched | yolo | 0.5 | 0.7799 | True | False | False | 1/3 |

## Best fm per (tier, workload)

| tier | workload | frozen c6 | best fm | best c6 | contention beats-Gaussian | recovers fully (3/3)? |
|---|---|---|---|---|---|---|
| tier2 | gpt2 | 0.6950 | 0.125 | 0.7433 | 0/3 | False |
| tier2 | resnet152 | 0.4102 | 4.0 | 0.3956 | 3/3 | True |
| tier3_matched | gpt2 | 0.6802 | 0.125 | 0.8575 | 0/3 | False |
| tier3_matched | resnet152 | 0.5369 | 2.0 | 0.5982 | 2/3 | False |
| tier3_matched | whisper | 0.2640 | 0.5 | 0.6166 | 2/3 | False |
| tier3_matched | yolo | 0.7881 | 0.25 | 0.9367 | 2/3 | False |
