# H100 Extension Journal

Known issue: tools/clear_system_cache.sh and tools/pre_experiment_checklist.sh use stale phase 1 v1 workload labels (resnet50, distilbert, whisper) — will miss current 5-workload pods. Deferred to H100 tooling task.

Aug 2 2026: Docker-side cleanup — removed 26 containerlab containers (clab-century-*), containerlab images (abdullahmuzlim279/k3s-serf-node, frrouting/frr), obsolete phase 1 v1 images (resnet50, distilbert). Docker builder prune reclaimed 41.32GB build cache. Total freed: 137GB (225GB → 88GB used, 96% → 38%). DiskPressure taint initially remained after Docker cleanup because CRI-O's imagefs is a separate store. Cleared after `sudo crictl rmi --prune` — the prune itself didn't free significant space (CRI-O storage held at 22GB, all in active use), but the earlier 137GB Docker reclaim was enough once kubelet re-evaluated. Final state: 96GB used / 139GB free (41%). bert-inference deployment applied at :v4, image already in CRI-O, smoke test ready to proceed.

Aug 2 2026: v4 container rebuild complete. All 5 workloads smoke-tested on A16 GPU with startup log confirming NVIDIA A16, compute_capability=(8, 6). Results:

  workload    | image                              | counter after 60s
  ------------|------------------------------------|-------------------
  bert        | hamidhrf/bert-inference:v4         | 13655 (pre-warmed)
  gpt2        | hamidhrf/gpt2-inference:v4         | 18
  resnet152   | hamidhrf/resnet152-inference:v4    | 54
  whisper     | hamidhrf/whisper-inference:v4      | 22
  yolo        | hamidhrf/yolo-inference:v4         | 23

Base image: pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime (all 5).
H100 compatibility: CUDA 12.1 runtime supports sm_90; startup log will confirm compute_capability=(9, 0) on first H100 run.
