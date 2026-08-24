# H100 Extension — Tier 1 Notes

Reference document for the H100 extension of the research project thesis. The Tier 1 phase covers five AI inference workloads (BERT-base, GPT-2, ResNet-152, Whisper-small, YOLOv8n) at replica count r=1, run on a single NVIDIA H100 NVL 94GB GPU. This document records the setup, methodology, decisions, results, and known issues in enough detail to support later thesis or paper writing without needing to reread commit histories or chat logs.

Companion file: `EXTENSION_JOURNAL.md` in the repository root holds a chronological event log. This document is organised by topic.

---

## 1. Objective

The Tier 1 phase asks a narrow question: how does the same workload set behave when the only variable changed from the A16 baseline is the GPU itself? The intent is to hold every other factor constant — inference code, batch size, request rate, load-phase schedule, metric collection, and cluster software — so any measurable difference can be attributed to hardware.

This makes Tier 1 the controlled comparison arm of the extension. Tier 2 will add partitioning (MIG single strategy at all-1g.12gb) and a full sweep of r=1 through r=7, which lets us observe scaling under smaller effective GPU slices. Tier 3, if it proceeds, will test a custom GPU-sharing scheduler.

---

## 2. Infrastructure

### 2.1 Hardware and OS

The H100 VM is named `devLab`, reachable at 172.22.174.66 on the same /24 subnet as the A16 VM (172.22.174.58). Ubuntu 26.04 LTS, kernel 7.0.0-29-generic. 16 vCPU, 62.5 GB RAM, 294 GB root filesystem. PCIe passthrough of an NVIDIA H100 NVL 94 GB card (compute capability 9.0, Hopper family, single device visible via `lspci`).

The 94 GB NVL variant matters because it uses different MIG profile naming than the 80 GB SXM variant of H100. Any MIG-related manifests or scripts reference `all-1g.12gb` and not `all-1g.10gb`. This is easy to get wrong from documentation aimed at the more common 80 GB variant.

### 2.2 GPU driver

NVIDIA driver 580.173.02, installed on the host via the Ubuntu package `nvidia-driver-580-server`. The driver was intentionally kept in the 580 series to match the A16 driver line (580.95.05), so the driver is not a variable when comparing A16 and H100 measurements. The 580 series is also required to avoid a known scheduler bug in the 570 series where MIG-partitioned and full-GPU pods coexist poorly (fix landed in 580.65.06 and later).

We chose to install the driver directly on the host rather than have the GPU Operator manage a containerised driver. The reason was practical: kernel 7.0 is recent, and prebuilt containerised driver images may not match every host kernel. A host driver install is one fewer thing that can go wrong on an unfamiliar kernel.

### 2.3 Kubernetes and container runtime

Kubernetes 1.34.0, CRI-O 1.31.5, both version-pinned via `apt-mark hold` to prevent auto-upgrades. Single-node cluster with the control-plane taint removed so workloads schedule on the same node as the API server. This mirrors the A16 configuration exactly. Calico CNI v3.29.1 with pod CIDR 10.244.0.0/16 and VXLAN encapsulation.

Kubelet configuration uses systemd cgroup driver, which is required for compatibility with CRI-O 1.31 on modern Kubernetes releases.

### 2.4 GPU Operator

NVIDIA GPU Operator v26.3.3, installed via Helm from `helm.ngc.nvidia.com`. The chart is configured with `driver.enabled=false` so the operator uses the host-installed driver rather than deploying its own. All other components are enabled: NVIDIA container toolkit, device plugin, DCGM exporter, MIG manager, node feature discovery, GPU feature discovery, node status exporter. MIG strategy is set to `single`, which is the strategy Tier 2 requires.

v26.3.0 introduced runtime MIG profile discovery via NVML, which is why the version pin matters. Older GPU Operator releases relied on hardcoded profile lists that did not include the NVL variant's `1g.12gb` profile. On this operator version the node reports `nvidia.com/mig.capable=true` and `nvidia.com/mig.config=all-disabled` at rest, with MIG manager sitting dormant until a node label triggers reconfiguration.

The GPU registers as `nvidia.com/gpu: 1` allocatable. Additional GFD labels include `nvidia.com/gpu.product=NVIDIA-H100-NVL`, `nvidia.com/gpu.family=hopper`, `nvidia.com/gpu.memory=95830`.

### 2.5 Monitoring stack

Prometheus scrapes four healthy targets in Tier 1: itself, node-exporter, kubelet-cadvisor, and the DCGM exporter installed by GPU Operator in the `gpu-operator` namespace. Two intentional omissions: Grafana (the runner queries Prometheus over HTTP directly, so no dashboard layer is needed) and kube-state-metrics (the runner never queries `kube_*` metrics). Kepler was also skipped because the runner does not consume its metrics and its eBPF probes carry an unquantified compatibility risk on the 7.0 kernel.

The Prometheus deployment uses a 50 GB PVC on the `local-path` storageclass. Storage class was installed separately (Rancher local-path-provisioner v0.0.24) and set as the cluster default. This step was not needed on A16 because that cluster had it pre-installed as part of an older deploy script.

### 2.6 Container base image

All five workload containers were rebuilt from `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime` and tagged `:v4` on Docker Hub under `hamidhrf/`. CUDA 12.1 supports sm_90, which is what H100 requires. The A16 GPU is sm_86 and also runs on CUDA 12.1, so the same container image works on both platforms. This was a deliberate choice to eliminate container-level differences from the comparison.

The rebuild also added a startup log line to each inference script that prints the GPU name and compute capability at container start. This gives every experiment a self-contained record of which hardware it ran on. On A16 the log reads `device=NVIDIA A16, compute_capability=(8, 6)`; on H100 it reads `device=NVIDIA H100 NVL, compute_capability=(9, 0)`.

---

## 3. Software stack details

The runner is `tools/run_experiment_v3.py`, unchanged in its experiment logic from the A16 baseline. Three environment variables were added to make it work across both clusters without file edits: `PROMETHEUS_URL` (defaults to the A16 IP), `DATA_OUTPUT_DIR` (defaults to `data/raw/phase1_v3`), and `EXPERIMENT_AUTO_CONFIRM` (defaults unset, meaning both interactive prompts still fire as before). When all three env vars are unset the runner produces byte-identical behaviour to the pre-extension version, so A16 Phase 1 experiments remain reproducible with the same file. When set on the H100 host, the runner writes to the correct Prometheus endpoint, the correct output directory, and skips the two interactive prompts that would otherwise block a nohup-launched run.

The Python environment on H100 is a conda environment named `tracegen`, created from `environment.yml` at the repository root, byte-identical to the A16 environment. Python 3.10.19, pandas 2.3.3, numpy 2.2.6, requests 2.32.5, PyTorch 2.9.0. Miniconda was installed at `~/miniconda3`. Anaconda's Terms of Service for the `defaults` channel had to be accepted explicitly (a newer conda requirement not present when the A16 environment was created), otherwise `conda env create` refuses to proceed non-interactively.

Two utility scripts run before each experiment: `tools/pre_experiment_checklist.sh` for a read-only system health check and `tools/clear_system_cache.sh` for dropping the Linux page cache. Both scripts had stale workload labels (they referenced the phase 1 v1 workload set: `resnet50`, `distilbert`, `whisper`) which were updated to the current five-workload set. The cache-clearing script was also refactored to invoke `sudo tee /proc/sys/vm/drop_caches` instead of `sudo sh -c 'echo 3 > ...'`, so it can run under a narrowly scoped sudoers rule.

---

## 4. Design decisions

### 4.1 Why keep workloads identical

The extension's scientific value depends on holding everything constant except the GPU. Changing the request rate, batch size, or input length on H100 would confound hardware effects with workload effects and make the results uninterpretable as a comparison. It would also invalidate any use of the A16-trained S36 model on the new data, because S36 was trained on the specific workload we ran on A16. Keeping the workload identical is what makes the H100 measurements a controlled experiment rather than a new baseline.

### 4.2 Why the driver on the host, not in the operator

Two reasons. First, a host driver install is a well-trodden path on Ubuntu with reliable package support; a containerised driver depends on the operator having a prebuilt image matching the exact host kernel, which is a coin flip on newer kernels. Second, keeping the driver installation method the same as A16 removes one variable from any cross-hardware comparison. If either platform showed an anomaly, we would not want to have to consider whether the driver install path was the cause.

### 4.3 Why v26.3.3 specifically

We needed v26.3.0 or later for runtime MIG profile discovery on the NVL variant. v26.3.3 is the latest patch release in the v26.3 line as of the H100 setup date, so it gets any bug fixes on top of the runtime discovery feature.

### 4.4 Why skip Grafana

The runner queries Prometheus HTTP API directly and writes CSVs. Grafana is a dashboard layer for humans. Nothing in the automated pipeline consumes it. Skipping it saves one deployment, one PVC, and one class of dashboard-broken distractions during experiment runs.

### 4.5 Why skip kube-state-metrics and Kepler

kube-state-metrics exposes `kube_*` metrics describing Kubernetes object state (pod counts, deployment ready state, and so on). The runner never queries any `kube_*` metric, so deploying the exporter would waste resources and add scrape targets that never resolve to useful data. Kepler exposes eBPF-based energy metrics. The runner does not query it, and its eBPF probes have not been validated against the 7.0 kernel. Both were dropped.

### 4.6 Why env vars over CLI flags

The runner had no argparse layer; it read positional arguments only. Adding argparse would have been a larger refactor with more surface area for breakage. Env vars with defaults matching the A16 configuration let us extend the runner without touching its logic and without changing the invocation pattern used in A16 Phase 1. Same reasoning applied to `EXPERIMENT_AUTO_CONFIRM`.

### 4.7 Why a standalone shell script for Tier 2

Tier 2 will run 35 experiments over two to three days of wall-clock time. A shell script running under `nohup` in a `tmux` session survives SSH drops, laptop closures, and any interruption that does not touch the H100 VM itself. Running Tier 2 through an interactive agent would need that agent to stay alive for the entire duration, would burn attention on approval clicks between experiments, and would leave no first-class artifact in the repository describing how Tier 2 was actually executed. A committed shell script is reviewable, reproducible, and does not depend on any external session.

### 4.8 Why all-1g.12gb for Tier 2 MIG

The H100 NVL supports several MIG profile combinations, but the `single` strategy for the maximum number of instances requires all slices to be the same size. The smallest per-slice size that yields the maximum instance count on NVL is `1g.12gb`, producing seven slices with roughly 10.75 GB of usable memory each. This matches the Tier 2 plan of r=1 through r=7 (one workload pod per slice) and provides an effective per-pod GPU size much closer to A16's whole GPU than to H100's whole GPU, which makes Tier 2 a useful bridge between the two platforms rather than a smaller-scale repeat of Tier 1.

---

## 5. Methodology

Each Tier 1 experiment ran a single workload at r=1 for 60 minutes of active recording, plus setup and teardown that brought total wall-clock time to roughly 65 minutes. The runner scales the deployment from zero to one replica, waits for the pod to be ready, then enters a six-phase Business Day traffic pattern (NIGHT, RAMP, MORNING, LUNCH, PEAK, EVENING) totalling 60 minutes. Metrics are pulled from Prometheus at 5-second intervals for the whole recording period, giving 715 samples per metric per experiment. At the end of recording the runner writes 22 CSV files (one per metric) plus a timestamps file, then cleans up by deleting the deployment. The cleanup step is a full `kubectl delete deployment`, not a scale-to-zero, which means each subsequent experiment needs a fresh `kubectl apply` of the workload YAML.

The 22 metrics cover pod-level CPU and memory use, PSI (pressure stall information) for CPU and memory and IO, application-side latency percentiles and throughput, node-level CPU and memory and PSI, and GPU-level utilization, memory used and total, power draw, and temperature. All metrics carry the same set of Prometheus labels (pod, gpu, job, instance, namespace, and so on) which makes joining across metrics straightforward at analysis time.

Load-phase boundaries are deterministic from experiment start time (NIGHT 0-8 min, RAMP 8-15 min, MORNING 15-25 min, LUNCH 25-35 min, PEAK 35-50 min, EVENING 50-60 min) and are not persisted as a CSV column. Phase-level analysis reconstructs them from timestamps at analysis time. This matches the A16 behaviour and is deliberate: keeping the CSV schema minimal makes cross-experiment merging cleaner.

Before each experiment we ran `pre_experiment_checklist.sh` (read-only baseline check) and `clear_system_cache.sh` (drops the Linux page cache and confirms no stale workload pods are running). Both scripts return zero when the system is in a clean starting state. Baseline before each Tier 1 experiment showed 0% GPU utilisation and 0 MiB GPU memory in use.

---

## 6. Results

Five experiments completed successfully, one per workload, over August 12-13, 2026. All runs exited cleanly with the runner's success marker in the log. All CSVs contained the expected number of rows (715) with zero null values across any column of any file.

### 6.1 Runtime

| Workload | Runtime (min) |
|---|---|
| BERT | ~65 |
| GPT-2 | ~68 |
| ResNet-152 | ~67 |
| Whisper | ~61 |
| YOLO | ~67 |

The runner's internal 60-minute recording window was exact across all runs. Variation in total wall-clock time came from setup and teardown, which depends slightly on how quickly the pod reaches `Ready` and how quickly the runner's post-collection processing finishes.

### 6.2 Peak GPU utilisation

The most informative single-number comparison between A16 and H100 at r=1 is peak GPU utilisation. Values are from the DCGM `DCGM_FI_DEV_GPU_UTIL` metric, sampled at 5-second intervals across the full 60-minute recording.

| Workload | A16 r=1 peak | H100 r=1 peak |
|---|---|---|
| BERT | 2% | 4% |
| GPT-2 | 21% | 24% |
| ResNet-152 | 3% | 4% |
| Whisper | 65% | 26% |
| YOLO | 1% | 2% |

Four of the five workloads sit in the same low-utilisation regime on both platforms. Classification and detection models barely touch either GPU at r=1, which is what one would expect from a light request rate against small models. GPT-2 shows meaningful but not saturating utilisation on both platforms, driven by its autoregressive generation loop rather than raw compute.

### 6.3 GPU memory and thermal behaviour

`DCGM_FI_DEV_FB_USED` (GPU memory in use) was flat across all runs on H100, sitting at 1192 MiB for BERT and comparable values for the others. This is expected behaviour: PyTorch's caching memory allocator claims memory at model load time and holds it for the duration of the process, regardless of actual activation memory during inference. The metric was already noted in Phase 1 as near-constant on A16 and was excluded from the S36 training set for that reason (reconstructed as a constant from real stats during post-processing).

`DCGM_FI_DEV_FB_FREE` correspondingly stayed close to the full 94 GB, confirming the workload is nowhere near memory bounds at r=1. GPU temperature held steady at 48-49°C throughout every run, which is typical of a chassis-cooled datacenter GPU under low sustained load. Power draw sat around 92-96 W (against the 400 W TDP), roughly matching the utilisation numbers.

### 6.4 Whisper: the reclassification observation

Whisper is the one workload where the H100 result differs qualitatively from A16, not just quantitatively. Peak GPU utilisation dropped from 65% to 26%.

On A16, Whisper's Phase 1 data showed a clear CPU bottleneck. Peak CPU pressure (PSI) rose sharply with replica count, hitting 0.63 at r=10 and causing at least one pod to crash from CPU starvation. The 65% GPU utilisation at r=1 on A16 was not a signal of heavy GPU work; it was the downstream effect of CPU waiting time on inference throughput. Whisper's audio pre-processing pipeline (decoding, resampling, mel-spectrogram computation) is CPU-heavy and the A16 host CPU could not keep pace, so the GPU cycled between busy and blocked in a pattern that reads as moderate utilisation in the metric.

On H100 the same code runs the same input on a beefier host CPU with more headroom. The pre-processing pipeline completes faster, the GPU inference stage is no longer starved, and the peak utilisation drops to a level that reflects the actual size of the workload relative to the GPU's compute capability. The workload is now genuinely small relative to H100, rather than being blocked by upstream CPU work.

The implication that matters for the thesis is that the resource-bound classification of a workload is not intrinsic to the workload; it is a function of the workload plus the hardware. On A16 we would call Whisper CPU-bound. On H100, running the same code with the same input, we would call it GPU-bound and under-utilised. This has practical consequences for capacity planning and for any workload characterisation model trained on a single hardware baseline. It is also directly relevant to Jan's fidelity analysis: how much of the A16-trained S36 model's implicit resource-bound assumptions transfer to H100 measurements is now an empirical question rather than a theoretical one.

### 6.5 Pattern across workloads

The transformer and attention-based generative models (GPT-2, Whisper) show meaningfully higher peak utilisation than the classification and detection models (BERT, ResNet-152, YOLO), all still well under 30% on H100 at r=1. This mirrors the A16 pattern with the exception noted for Whisper. At the r=1 request rate used here, none of these workloads exercise H100's full capability, which is consistent with the general observation that modern datacenter GPUs are over-provisioned for many production inference workloads. Tier 2's MIG partitioning is expected to bring effective per-pod GPU size closer to A16, which should raise per-pod utilisation and produce data more directly comparable to A16 across the full replica-count range.

---

## 7. Bugs found and fixed during Tier 1 setup

Several latent issues surfaced during the H100 bring-up. All were fixed and committed on the `extension-h100` branch. This section lists them so they are not rediscovered from scratch later.

`k8s/monitoring/prometheus-config.yaml` had been committed as raw Prometheus configuration (starting with `global:` and `scrape_configs:`) rather than wrapped in a Kubernetes ConfigMap manifest. Deploys on A16 must have relied on a manual `kubectl create configmap --from-file` step that was never captured in git. Wrapped the file as a proper ConfigMap so `kubectl apply -f` works standalone from now on.

The same file's DCGM scrape job pointed at the `monitoring` namespace, but GPU Operator installs its DCGM exporter in the `gpu-operator` namespace. Patched the scrape job's namespace filter to `gpu-operator`. The Service name (`nvidia-dcgm-exporter`) happens to match what GPU Operator uses, so no other change was needed.

`tools/pre_experiment_checklist.sh` and `tools/clear_system_cache.sh` both used the phase 1 v1 workload label set (`resnet50`, `distilbert`, `whisper`), which meant their "no workload pods running" check would silently return true even if pods from the current five-workload set (`bert`, `gpt2`, `resnet152`, `whisper`, `yolo`) were running. Updated the label set in both scripts.

`clear_system_cache.sh` previously used `sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'`, which would require passwordless sudo for `sh` itself. Refactored to `sudo tee /proc/sys/vm/drop_caches` so a narrowly scoped sudoers rule can enable non-interactive execution. Also removed `sudo` from the `sync` call because `sync` does not require root on Linux.

The runner had two `input()` prompts (a start confirmation and a pre-existing-pods cleanup confirmation) designed for interactive use. Added an `EXPERIMENT_AUTO_CONFIRM` environment variable that gates both prompts. When set to `1`, `true`, or `yes`, the runner answers both with `yes` automatically; when unset, the prompts fire as before.

Local-path storageclass was missing on H100. Installed rancher/local-path-provisioner v0.0.24 to match A16 and set it as the default storageclass.

FH Dortmund's network intercepts TLS to some less-common hosts including `baltocdn.com`, which prevents installing Helm from its apt repository. Installed Helm from a GitHub release tarball instead. The same interception could hit other domains during future work; hosts like `github.com` and `k8s.io` have been passing cleanly so far.

DCGM Exporter has a known upstream bug where it reports the H100 NVL MIG profile as `1g.11gb` instead of `1g.12gb` in metric labels. This is a cosmetic label issue only; the metric values are correct. It becomes relevant if any dashboard, alerting rule, or analysis script pattern-matches on the profile string.

---

## 8. Data and reproducibility

Data location on the H100 VM: `~/generative-ai-workload-modeling/data/raw/extension_tier1/<workload>_r1/` for each workload, containing 22 metric CSVs and one timestamps file. Each workload's data is committed to the `extension-h100` branch as a separate commit (bert `c0eb65a`, gpt2 `cfbd854`, resnet152 `f3ddc58`, whisper `c3516c9`, yolo `9d14dc8`).

The `.gitignore` was updated with a narrowly scoped exception for `data/raw/extension_tier1/` only. Historical A16 data under `data/raw/` remains gitignored, matching the pre-existing policy of keeping raw experimental data out of the repository. Tier 2 will need a similar exception for `data/raw/extension_tier2/`.

To reproduce a single Tier 1 experiment from a clean H100 cluster state:

```bash
# From ~/generative-ai-workload-modeling on the H100 VM
bash tools/pre_experiment_checklist.sh
bash tools/clear_system_cache.sh
kubectl apply -f k8s/workloads/<workload>-deployment.yaml
mkdir -p data/raw/extension_tier1

nohup env \
  PROMETHEUS_URL=http://172.22.174.66:30090 \
  DATA_OUTPUT_DIR=data/raw/extension_tier1 \
  EXPERIMENT_AUTO_CONFIRM=1 \
  ~/miniconda3/envs/tracegen/bin/python \
    tools/run_experiment_v3.py <workload> 1 \
  > /tmp/<workload>_r1_tier1.log 2>&1 &
```

The runner scales the deployment, records for 60 minutes, exports 22 CSVs, and deletes the deployment on completion. Total wall-clock time is roughly 65 minutes.

To replay a Phase 1 A16 experiment on the A16 VM, the same script works without any environment variables set. Its defaults preserve the original A16 behaviour (Prometheus URL 172.22.174.58:30090, output directory `data/raw/phase1_v3`, interactive prompts enabled).

---

## 9. Tier 2 plan

Tier 2 will run the same five workloads through r=1 to r=7, giving 35 experiments. The H100 will be reconfigured into MIG single strategy at profile `all-1g.12gb` (seven slices, roughly 10.75 GB usable memory each). Each workload pod requests one MIG slice via a specific resource limit rather than the `nvidia.com/gpu` request used in Tier 1.

Two artifacts are needed. A new runner variant `tools/run_experiment_v4.py` handles MIG-instance-aware Prometheus queries (DCGM per-instance labels use different keys than whole-GPU labels) and the different pod resource request. The Tier 1 runner (`v3`) stays in place unmodified so it can continue to serve both Phase 1 A16 replay and Tier 1 re-run scenarios. A batch orchestration script `tools/run_tier2_batch.sh` runs standalone under `nohup` and `tmux`, calling the pre-experiment scripts between runs, applying each workload's deployment YAML with the right MIG resource limit, invoking the v4 runner, verifying output, committing to git with a journal entry after each success, and halting cleanly on any failure with a resume-from-N mechanism via a state file.

Expected total runtime is 2 to 3 days of wall-clock time. The batch runs unattended and reports final status when it finishes or halts.

---

## 10. Open questions

The Whisper interpretation should be validated with Prof. Recker before Tier 2 completes. The reasoning about CPU-bound to GPU-bound reclassification depends on correctly reading the A16 CPU pressure behaviour; if there is a subtlety in how DCGM reports utilisation across platforms, or in how the H100 VM's CPU differs from A16's, the interpretation may need adjustment. This is the most consequential finding of Tier 1 and worth checking rather than assuming.

Tier 2 will reveal whether the whole-Tier-1 under-utilisation persists under MIG partitioning. If it does, the extension has a coherent story about H100 being over-provisioned for these workloads across configurations. If MIG raises per-pod utilisation into an A16-comparable range, MIG becomes a defensible multi-tenancy configuration for this workload class and produces data directly comparable to A16 Phase 1 (which used a shared GPU with 10 virtual slices). Either outcome is a valid finding.

The transferability of the A16-trained S36 model to H100 measurements is an empirical question that Jan's fidelity analysis is designed to answer. Tier 1 gives Jan five workloads at r=1 to compare against S36's Tier 1 predictions; Tier 2 will give him the full r=1 to r=7 range under partitioned conditions closer to A16's original training regime.

---

## Appendix: commit summary

| Commit | Purpose |
|---|---|
| 7e76825 | Unified CUDA 12.1 Dockerfiles for v4 images |
| 5252d5c | Bump workload images to v4 |
| 332defa | Task 1 completion journal entry |
| 72617ad | H100 cluster brought up on devLab |
| 7e365cf, 1917aa9, 0d3f830, 4e75cf9, b38fc9f | H100 smoke tests (bert, gpt2, resnet152, whisper, yolo) |
| 38e799b | Prometheus URL env-configurable, ConfigMap wrap, DCGM ns patch |
| 4ec9fbe | Monitoring stack deployment journal entry |
| 2632378 | Data output dir env-configurable |
| f1e239a | Pre-experiment scripts updated for 5-workload set, scoped sudo |
| 6ff19a4 | EXPERIMENT_AUTO_CONFIRM env var |
| c0eb65a | Tier 1 bert r=1 data |
| cfbd854 | Tier 1 gpt2 r=1 data |
| f3ddc58 | Tier 1 resnet152 r=1 data |
| c3516c9 | Tier 1 whisper r=1 data |
| 9d14dc8 | Tier 1 yolo r=1 data |

All commits on branch `extension-h100`, pushed to origin. Commit hashes reflect the state after the co-author trailer was stripped from history via `git filter-branch`; earlier hashes referenced in the raw journal file may differ.
