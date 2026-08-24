# E11 - two-source VR variance characterization

K=30 seeded generation draws (torch.manual_seed 0..29), B=2000 between-trace bootstrap resamples. Composite = 6-metric (exclude `pod_memory_bytes` by name), composite_7 carried as secondary.

## DECISION-CRITICAL: tier-step deltas in mean|VR-1| (composite_6)

- A16 -> Tier3 (hardware step): delta=-0.0291, propagated generation SD=0.0131 -> **OUTSIDE** the propagated generation interval (|delta| > SD)
- Tier3 -> Tier2 (sharing step): delta=+0.1820, propagated generation SD=0.0375 -> **OUTSIDE** the propagated generation interval (|delta| > SD)

## DECISION-CRITICAL: resnet152 Tier2

resnet152 Tier2 (stored composite_7 headline 0.369; composite_6=0.4102): K-draw composite_6 mean=0.4274 SD=0.0081 (excludes 1.0 at +-1 generation SD). val n=2 -> INSUFFICIENT between-trace support, no CI. Frozen all-traces (n=28) composite_6=0.5636, 95% CI=[0.4813, 0.9414] (excludes 1.0). VERDICT: once the correct pooled (large-n, n=28) between-trace variance is used instead of the fragile val-only (n=2) estimate, VR=0.369-ish is DISTINGUISHABLE from 1.0.

## Per-cell table

| tier | workload | stored c6 (headline) | K-mean c6 | gen-SD c6 | stored c7 | K-mean c7 | gen-SD c7 | val n | val 95% CI (c6) | frozen all-traces n | frozen all-traces c6 | frozen all-traces 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a16 | bert | 0.9824 | 0.9821 | 0.0076 | 1.0224 | 1.0199 | 0.0076 | 6 | [0.8738, 1.9621] | 55 | 1.0531 | [0.9040, 1.9110] |
| a16 | gpt2 | 1.2582 | 1.2489 | 0.0168 | 1.1366 | 1.1287 | 0.0147 | 6 | [0.8291, 6.3135] | 55 | 1.6119 | [1.1626, 3.2674] |
| a16 | resnet152 | 0.8995 | 0.8994 | 0.0065 | 0.9854 | 0.9866 | 0.0071 | 6 | [0.7810, 2.2582] | 55 | 0.9467 | [0.7894, 1.9211] |
| a16 | whisper | 1.1020 | 1.1191 | 0.0380 | 1.3968 | 1.4034 | 0.0490 | 6 | [0.6557, 10.2015] | 55 | 1.6769 | [1.0107, 5.5080] |
| a16 | yolo | 0.9462 | 0.9574 | 0.0115 | 1.0388 | 1.0478 | 0.0107 | 6 | [0.7695, 3.3204] | 55 | 0.9583 | [0.7500, 2.5864] |
| tier3 | bert | 0.9898 | 0.9942 | 0.0057 | 1.8854 | 1.9015 | 0.0141 | 7 | [0.8435, 2.4906] | 55 | 1.2162 | [0.9356, 3.1559] |
| tier3 | gpt2 | 0.9850 | 0.9567 | 0.0201 | 0.9198 | 0.8931 | 0.0173 | 7 | [0.8293, 3.5989] | 55 | 1.0016 | [0.8357, 1.7490] |
| tier3 | resnet152 | 0.9875 | 0.9739 | 0.0614 | 1.0643 | 1.0430 | 0.0766 | 7 | [0.8157, 35.3306] | 55 | 1.1615 | [0.8776, 3.1849] |
| tier3 | whisper | 0.8264 | 0.8177 | 0.0080 | 0.7589 | 0.7531 | 0.0067 | 3 | insufficient between-trace support | 55 | 1.2058 | [0.8744, 3.0023] |
| tier3 | yolo | 1.1754 | 1.1693 | 0.0173 | 1.0692 | 1.0632 | 0.0150 | 6 | [0.9916, 34.0729] | 55 | 0.7886 | [0.7124, 1.2349] |
| tier2 | bert | 1.1930 | 1.2399 | 0.0877 | 1.1415 | 1.1878 | 0.0872 | 4 | [1.0952, 218.7326] | 28 | 1.1883 | [1.0126, 3.0078] |
| tier2 | gpt2 | 0.6950 | 0.6901 | 0.0041 | 0.6118 | 0.6074 | 0.0036 | 4 | [0.6542, 2.2358] | 28 | 0.7366 | [0.6799, 1.3178] |
| tier2 | resnet152 | 0.4102 | 0.4274 | 0.0081 | 0.3686 | 0.3836 | 0.0070 | 2 | insufficient between-trace support | 28 | 0.5636 | [0.4813, 0.9414] |
| tier2 | whisper | 1.0537 | 1.1450 | 0.0966 | 1.1059 | 1.1731 | 0.0851 | 3 | insufficient between-trace support | 28 | 0.8645 | [0.6964, 2.4448] |
| tier2 | yolo | 0.8448 | 0.7031 | 0.1555 | 0.7743 | 0.6475 | 0.1361 | 2 | insufficient between-trace support | 28 | 0.6067 | [0.5804, 0.7768] |

## Per-metric within/between decomposition (val traces)

Exact identity: pooled_var == within + between (equal trace length T=715).

| tier | workload | metric | within (temporal) | between (trace-level) | pooled (check) |
|---|---|---|---|---|---|
| a16 | bert | pod_cpu_usage | 0.0841794 | 2.49414e-05 | 0.0842043 |
| a16 | bert | pod_memory_bytes | 5.3372e-07 | 0.00786288 | 0.00786341 |
| a16 | bert | pod_psi_cpu | 0.0042844 | 2.0838e-05 | 0.00430524 |
| a16 | bert | pod_latency_avg | 0.00807527 | 0.000560826 | 0.0086361 |
| a16 | bert | pod_throughput | 0.118613 | 1.38215e-06 | 0.118615 |
| a16 | bert | gpu_utilization | 0.0401543 | 4.54557e-05 | 0.0401998 |
| a16 | bert | gpu_power_watts | 0.000465448 | 0.00470633 | 0.00517178 |
| a16 | gpt2 | pod_cpu_usage | 0.058324 | 0.00615961 | 0.0644836 |
| a16 | gpt2 | pod_memory_bytes | 1.4286e-08 | 0.0332405 | 0.0332405 |
| a16 | gpt2 | pod_psi_cpu | 0.00702659 | 0.000436365 | 0.00746296 |
| a16 | gpt2 | pod_latency_avg | 0.0233449 | 0.0516578 | 0.0750027 |
| a16 | gpt2 | pod_throughput | 0.0095768 | 0.0167726 | 0.0263494 |
| a16 | gpt2 | gpu_utilization | 0.0136606 | 0.0093411 | 0.0230017 |
| a16 | gpt2 | gpu_power_watts | 0.000534459 | 0.00829579 | 0.00883024 |
| a16 | resnet152 | pod_cpu_usage | 0.07364 | 4.70352e-05 | 0.0736871 |
| a16 | resnet152 | pod_memory_bytes | 1.61623e-09 | 0.0114445 | 0.0114445 |
| a16 | resnet152 | pod_psi_cpu | 0.00192579 | 7.58768e-06 | 0.00193338 |
| a16 | resnet152 | pod_latency_avg | 0.0076765 | 0.00119149 | 0.00886799 |
| a16 | resnet152 | pod_throughput | 0.126832 | 3.67206e-07 | 0.126832 |
| a16 | resnet152 | gpu_utilization | 0.00570565 | 7.72788e-06 | 0.00571338 |
| a16 | resnet152 | gpu_power_watts | 0.000513762 | 0.00570294 | 0.00621671 |
| a16 | whisper | pod_cpu_usage | 0.000939467 | 0.0256888 | 0.0266283 |
| a16 | whisper | pod_memory_bytes | 1.1171e-06 | 6.17589e-05 | 6.2876e-05 |
| a16 | whisper | pod_psi_cpu | 0.00147512 | 0.0126585 | 0.0141336 |
| a16 | whisper | pod_latency_avg | 0.00121514 | 0.0370152 | 0.0382303 |
| a16 | whisper | pod_throughput | 4.91595e-05 | 0.00580117 | 0.00585033 |
| a16 | whisper | gpu_utilization | 0.00678283 | 0.0054301 | 0.0122129 |
| a16 | whisper | gpu_power_watts | 0.000107668 | 0.0058633 | 0.00597097 |
| a16 | yolo | pod_cpu_usage | 0.0484447 | 0.0026097 | 0.0510545 |
| a16 | yolo | pod_memory_bytes | 1.16875e-08 | 0.00632285 | 0.00632286 |
| a16 | yolo | pod_psi_cpu | 0.00221074 | 1.88369e-05 | 0.00222958 |
| a16 | yolo | pod_latency_avg | 0.0312776 | 0.00595202 | 0.0372296 |
| a16 | yolo | pod_throughput | 0.103417 | 0.000150059 | 0.103567 |
| a16 | yolo | gpu_utilization | 0.0869569 | 0.000822495 | 0.0877794 |
| a16 | yolo | gpu_power_watts | 0.000164396 | 0.00663772 | 0.00680212 |
| tier3 | bert | pod_cpu_usage | 0.0454837 | 1.26566e-05 | 0.0454963 |
| tier3 | bert | pod_memory_bytes | 0.000289962 | 0.000792516 | 0.00108248 |
| tier3 | bert | pod_psi_cpu | 0.00117999 | 1.93518e-05 | 0.00119934 |
| tier3 | bert | pod_latency_avg | 0.00807849 | 0.0011417 | 0.00922019 |
| tier3 | bert | pod_throughput | 0.114102 | 1.11784e-05 | 0.114113 |
| tier3 | bert | gpu_utilization | 0.0406801 | 9.63131e-05 | 0.0407765 |
| tier3 | bert | gpu_power_watts | 5.69816e-05 | 0.00140664 | 0.00146363 |
| tier3 | gpt2 | pod_cpu_usage | 0.0706202 | 0.00267246 | 0.0732927 |
| tier3 | gpt2 | pod_memory_bytes | 0.000692348 | 0.0151158 | 0.0158081 |
| tier3 | gpt2 | pod_psi_cpu | 0.00124829 | 4.38663e-05 | 0.00129216 |
| tier3 | gpt2 | pod_latency_avg | 0.0309211 | 0.0299064 | 0.0608275 |
| tier3 | gpt2 | pod_throughput | 0.0132233 | 0.00686226 | 0.0200855 |
| tier3 | gpt2 | gpu_utilization | 0.0174028 | 0.00371927 | 0.0211221 |
| tier3 | gpt2 | gpu_power_watts | 0.000107878 | 0.00428636 | 0.00439424 |
| tier3 | resnet152 | pod_cpu_usage | 0.00154714 | 1.26979e-06 | 0.00154841 |
| tier3 | resnet152 | pod_memory_bytes | 1.28118e-08 | 0.0306989 | 0.030699 |
| tier3 | resnet152 | pod_psi_cpu | 0.000223844 | 7.12721e-07 | 0.000224557 |
| tier3 | resnet152 | pod_latency_avg | 0.00635609 | 0.00145839 | 0.00781448 |
| tier3 | resnet152 | pod_throughput | 0.118319 | 8.30983e-06 | 0.118327 |
| tier3 | resnet152 | gpu_utilization | 0.0420533 | 4.35681e-05 | 0.0420968 |
| tier3 | resnet152 | gpu_power_watts | 2.01598e-05 | 0.0228199 | 0.0228401 |
| tier3 | whisper | pod_cpu_usage | 0.0524759 | 0.0010818 | 0.0535577 |
| tier3 | whisper | pod_memory_bytes | 0.00361675 | 0.0145309 | 0.0181476 |
| tier3 | whisper | pod_psi_cpu | 0.0730889 | 0.0115155 | 0.0846043 |
| tier3 | whisper | pod_latency_avg | 0.0162958 | 0.00583651 | 0.0221323 |
| tier3 | whisper | pod_throughput | 0.0107657 | 0.00502571 | 0.0157914 |
| tier3 | whisper | gpu_utilization | 0.0118962 | 1.88924e-07 | 0.0118964 |
| tier3 | whisper | gpu_power_watts | 0.00010094 | 0.00109903 | 0.00119997 |
| tier3 | yolo | pod_cpu_usage | 0.0918695 | 0.00764079 | 0.0995103 |
| tier3 | yolo | pod_memory_bytes | 1.38666e-08 | 0.00780266 | 0.00780268 |
| tier3 | yolo | pod_psi_cpu | 0.00363351 | 6.7229e-05 | 0.00370074 |
| tier3 | yolo | pod_latency_avg | 0.0665656 | 0.0438294 | 0.110395 |
| tier3 | yolo | pod_throughput | 0.113626 | 6.53999e-05 | 0.113692 |
| tier3 | yolo | gpu_utilization | 0.106379 | 0.000194998 | 0.106574 |
| tier3 | yolo | gpu_power_watts | 4.02852e-06 | 0.00187702 | 0.00188105 |
| tier2 | bert | pod_cpu_usage | 0.0589125 | 4.6587e-06 | 0.0589171 |
| tier2 | bert | pod_memory_bytes | 6.25128e-08 | 0.0528902 | 0.0528903 |
| tier2 | bert | pod_psi_cpu | 0.0023908 | 5.47319e-06 | 0.00239627 |
| tier2 | bert | pod_latency_avg | 0.00381776 | 0.000439101 | 0.00425687 |
| tier2 | bert | pod_throughput | 0.120618 | 9.88682e-06 | 0.120628 |
| tier2 | bert | gpu_utilization | 0.127339 | 2.91328e-05 | 0.127368 |
| tier2 | bert | gpu_power_watts | 3.4975e-05 | 0.0284801 | 0.0285151 |
| tier2 | gpt2 | pod_cpu_usage | 0.0659589 | 3.40546e-05 | 0.065993 |
| tier2 | gpt2 | pod_memory_bytes | 6.93437e-09 | 0.0895174 | 0.0895174 |
| tier2 | gpt2 | pod_psi_cpu | 0.0016536 | 5.82089e-05 | 0.00171181 |
| tier2 | gpt2 | pod_latency_avg | 0.0252112 | 0.00362429 | 0.0288354 |
| tier2 | gpt2 | pod_throughput | 0.0701914 | 0.000239219 | 0.0704306 |
| tier2 | gpt2 | gpu_utilization | 0.0714191 | 0.000524894 | 0.0719441 |
| tier2 | gpt2 | gpu_power_watts | 0.000567678 | 0.12597 | 0.126538 |
| tier2 | resnet152 | pod_cpu_usage | 0.0551183 | 0.000330628 | 0.0554489 |
| tier2 | resnet152 | pod_memory_bytes | 1.22982e-08 | 0.194924 | 0.194924 |
| tier2 | resnet152 | pod_psi_cpu | 0.00493677 | 0.000161183 | 0.00509795 |
| tier2 | resnet152 | pod_latency_avg | 0.0206212 | 0.00333478 | 0.023956 |
| tier2 | resnet152 | pod_throughput | 0.129752 | 4.67673e-06 | 0.129757 |
| tier2 | resnet152 | gpu_utilization | 0.126014 | 0.000143296 | 0.126158 |
| tier2 | resnet152 | gpu_power_watts | 2.67088e-05 | 0.241314 | 0.241341 |
| tier2 | whisper | pod_cpu_usage | 0.100529 | 0.000641521 | 0.10117 |
| tier2 | whisper | pod_memory_bytes | 1.61013e-05 | 0.0114743 | 0.0114904 |
| tier2 | whisper | pod_psi_cpu | 0.0309002 | 0.0092116 | 0.0401118 |
| tier2 | whisper | pod_latency_avg | 0.0109923 | 0.00459303 | 0.0155853 |
| tier2 | whisper | pod_throughput | 0.0405223 | 0.00289373 | 0.0434161 |
| tier2 | whisper | gpu_utilization | 0.0167123 | 0.00117498 | 0.0178873 |
| tier2 | whisper | gpu_power_watts | 0.000266225 | 0.00198192 | 0.00224815 |
| tier2 | yolo | pod_cpu_usage | 0.0756218 | 0.000104913 | 0.0757267 |
| tier2 | yolo | pod_memory_bytes | 2.32439e-08 | 0.0106187 | 0.0106187 |
| tier2 | yolo | pod_psi_cpu | 0.00574831 | 8.51979e-05 | 0.00583351 |
| tier2 | yolo | pod_latency_avg | 0.0125097 | 0.00132673 | 0.0138364 |
| tier2 | yolo | pod_throughput | 0.128045 | 2.5147e-08 | 0.128045 |
| tier2 | yolo | gpu_utilization | 0.125423 | 1.65142e-06 | 0.125424 |
| tier2 | yolo | gpu_power_watts | 5.36036e-06 | 0.00598703 | 0.0059924 |

## Tier-level mean|VR-1| (composite_6) with propagated generation SD

| tier | mean\|VR-1\| (headline, c6) | propagated generation SD |
|---|---|---|
| a16 | 0.1064 | 0.0089 |
| tier3 | 0.0773 | 0.0096 |
| tier2 | 0.2594 | 0.0363 |
