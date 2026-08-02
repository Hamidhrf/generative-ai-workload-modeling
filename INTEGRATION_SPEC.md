# INTEGRATION_SPEC.md

Technical integration reference for the TimeGAN workload-modeling repository.
Written to brief an external scheduler team that will replace the
Kubernetes-Deployment model of running workloads with a single-scheduler-pod
model that owns the GPU.

Repo root: `/home/hamid/generative-ai-workload-modeling` (branch `main`).

---

## 1. Repository map

### 1.1 Top-level tree, 3 levels deep

```
generative-ai-workload-modeling/
├── current_vm_config.yaml         # snapshot of the host VM (16 vCPU, A16 GPU, K8s v1.34)
├── environment.yml                # conda env for training/analysis code
├── README.md                      # project overview, results, figures
├── dashboards/                    # Grafana dashboard JSON
├── data/
│   ├── raw/                       # per-experiment CSV dumps from Prometheus
│   │   ├── phase1/                # not present (early runs, superseded)
│   │   ├── phase1_v2/
│   │   └── phase1_v3/             # canonical raw traces (275 pods × 5 workloads × r=1..10)
│   └── processed/
│       ├── phase1_v1/, phase1_v3/ # per-workload intermediate NPZ
│       └── phase4/                # unified/combined_dataset.npz + combined_normalization.json
├── docs/                          # thesis-related notes
├── figures/                       # PNGs used by README (architecture, scaling curves, etc.)
├── k8s/
│   ├── base/                      # cluster-wide base manifests
│   ├── gpu/                       # NVIDIA device-plugin + time-slicing configmap
│   ├── monitoring/                # Prometheus, DCGM, node-exporter, kepler, grafana, ksm
│   └── workloads/                 # one Deployment YAML per workload
├── models/
│   └── phase4/                    # per-stage checkpoints (timegan_s1 … s39, lstm, timevae)
├── outputs/                       # evaluation results per stage (VR, Wasserstein, plots)
├── reports/                       # EDA and evaluation reports
├── scripts/
│   ├── cluster-creation/          # kubeadm bootstrap + recovery scripts
│   ├── gpu-setup/                 # CRI-O + NVIDIA runtime install
│   ├── monitoring/                # helpers for enabling PSI, pod-level DCGM, etc.
│   ├── phase2/                    # archived model-selection code
│   ├── phase4/                    # canonical training pipeline (see §5)
│   ├── models/                    # duplicate/symlink layout for phase4 (largely empty)
│   ├── utils/                     # boundary_smoothing.py etc.
│   └── workloads/                 # per-workload inference container source
│       ├── bert_base/  gpt2/  resnet152/  whisper/  yolo/
└── tools/                         # experiment orchestration (run_experiment_v3.py etc.)
```

Top-level folder descriptions (one line each):

- `data/` — raw Prometheus CSV dumps and the derived NPZ training corpus.
- `dashboards/` — Grafana dashboard exports.
- `docs/`, `figures/`, `reports/` — thesis prose, figure PNGs, EDA reports.
- `k8s/` — every Kubernetes manifest applied to the cluster (base, GPU, monitoring, workloads).
- `models/` — trained checkpoint files, one subdir per generator stage.
- `outputs/` — evaluation artefacts per stage (metrics JSON, comparison plots).
- `scripts/` — everything that runs off-cluster: training, preprocessing, cluster bootstrap.
- `tools/` — experiment lifecycle scripts invoked by the operator (`run_experiment_v3.py`).
- `current_vm_config.yaml`, `environment.yml` — host and conda environment specs.

### 1.2 Where the 5 workloads are defined

| Workload   | Inference script                                      | Dockerfile                                      | K8s manifest                              |
|------------|--------------------------------------------------------|-------------------------------------------------|-------------------------------------------|
| BERT       | `scripts/workloads/bert_base/inference_variable.py`   | `scripts/workloads/bert_base/Dockerfile`        | `k8s/workloads/bert-deployment.yaml`      |
| GPT-2      | `scripts/workloads/gpt2/inference_variable.py`        | `scripts/workloads/gpt2/Dockerfile`             | `k8s/workloads/gpt2-deployment.yaml`      |
| ResNet-152 | `scripts/workloads/resnet152/inference_variable.py`   | `scripts/workloads/resnet152/Dockerfile`        | `k8s/workloads/resnet152-deployment.yaml` |
| Whisper    | `scripts/workloads/whisper/inference_variable.py`     | `scripts/workloads/whisper/Dockerfile`          | `k8s/workloads/whisper-deployment.yaml`   |
| YOLO       | `scripts/workloads/yolo/inference_variable.py`        | `scripts/workloads/yolo/Dockerfile`             | `k8s/workloads/yolo-deployment.yaml`      |

All five deployments share the same shape: `replicas: 0` at rest, `runtimeClassName: nvidia`,
one `nvidia.com/gpu: 1` request per pod, metrics on port `8000`, Prometheus scrape annotation
enabled with `path: /metrics`.

---

## 2. Workload runtime specification

Common structural facts (identical across all five workloads):

- Each container is a **self-driven inference loop** — no HTTP server, no external client.
- Load is **generated inside the container** following an identical **6-phase business-day
  profile** over 60 minutes (see table below).
- Each container exposes a `prometheus_client` HTTP endpoint on **port 8000** for
  application counters/histograms.
- Deployments start with `replicas: 0`; the runner (`tools/run_experiment_v3.py`) scales
  to the target replica count.

Shared load profile (`inference_variable.py` in each workload):

| Phase | Name     | Window (min) | Sleep between calls | Target req/s |
|-------|----------|--------------|---------------------|--------------|
| 0     | NIGHT    | 0–8          | 3.00 s              | 0.3          |
| 1     | RAMP     | 8–15         | 0.50 s              | 2.0          |
| 2     | MORNING  | 15–25        | 0.25 s              | 4.0          |
| 3     | LUNCH    | 25–35        | 0.67 s              | 1.5          |
| 4     | PEAK     | 35–50        | 0.20 s              | 5.0          |
| 5     | EVENING  | 50–60        | 1.00 s              | 1.0          |

There are no external knobs (env vars, ConfigMaps) that alter this profile — the constants
are baked into each `inference_variable.py`. Only `PYTHONUNBUFFERED=1` is set in the
Deployment manifests.

### 2.1 BERT (`bert-inference`)

- **Image:** `hamidhrf/bert-inference:v3`  (Dockerfile base: `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime`)
- **Python deps:** `transformers==4.36.0`, `filelock==3.13.1`, `prometheus-client`
- **Model:** `bert-base-uncased` from HuggingFace, loaded via
  `AutoModelForSequenceClassification` (2 labels), **fp32**, on the first available CUDA
  device.
- **Inference loop:** random token IDs of length ∈ [20, 128] fed through the model in a
  `while True:` loop with phase-based sleep (`inference_variable.py:159-212`).
- **Load generation:** self-generated random inputs; phase pattern as above.
- **App metrics exposed on :8000** — `bert_inference_total`,
  `bert_inference_latency_seconds` (histogram, buckets 0.005–0.3 s),
  `bert_load_phase`, `bert_sleep_time_seconds`, `bert_input_length_tokens`.

### 2.2 GPT-2 (`gpt2-inference`)

- **Image:** `hamidhrf/gpt2-inference:v3.1`  (base: `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime`)
- **Python deps:** `transformers==4.36.0`, `filelock==3.13.1`, `prometheus-client`
- **Model:** `gpt2` from HuggingFace, `AutoModelForCausalLM`, **fp32**.
- **Inference loop:** samples a prompt from a fixed pool of 24 prompts, calls
  `model.generate(..., max_new_tokens=50, do_sample=True, temperature=0.8, top_p=0.9)`.
- **Load generation:** self-driven, same 6-phase profile.
- **App metrics on :8000** — `gpt2_inference_total`,
  `gpt2_inference_latency_seconds` (buckets 0.01–5.0 s), `gpt2_load_phase`,
  `gpt2_sleep_time_seconds`, `gpt2_tokens_generated`.

### 2.3 ResNet-152 (`resnet152-inference`)

- **Image:** `hamidhrf/resnet152-inference:v3`  (base: `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime`)
- **Python deps:** `torchvision`, `prometheus_client`
- **Model:** `torchvision.models.resnet152` with `ResNet152_Weights.IMAGENET1K_V1`, **fp32**.
- **Inference loop:** random tensor `(1, 3, 224, 224)` per call, phase-based sleep.
- **Load generation:** self-driven, same 6-phase profile.
- **App metrics on :8000** — `resnet152_inference_total`,
  `resnet152_inference_latency_seconds` (buckets 0.005–0.3 s), `resnet152_load_phase`,
  `resnet152_sleep_time_seconds`.

### 2.4 Whisper (`whisper-inference`)

- **Image:** `hamidhrf/whisper-inference:v3`  (base: `python:3.10-slim` + `ffmpeg`, `libsndfile1`)
- **Python deps:** `torch`, `openai-whisper`, `numpy`, `scipy`, `pydub`, `prometheus_client`
- **Model:** OpenAI Whisper variant `small` (244M params), loaded via
  `whisper.load_model("small")`. Runs **fp16 on CUDA** (`fp16=torch.cuda.is_available()`),
  fp32 on CPU. Language pinned to `"en"`.
- **Inference loop:** picks a random file from a pool of 11 pre-generated MP3 files
  (5–15 s, 16 kHz, 64 kbps) and calls `model.transcribe(...)`.
- **Load generation:** self-driven; effective rate is capped by transcription latency
  (~0.5–1.5 s per call).
- **App metrics on :8000** — `whisper_inference_total`,
  `whisper_inference_latency_seconds` (buckets 0.1–10 s), `whisper_load_phase`,
  `whisper_sleep_time_seconds`, `whisper_audio_duration_seconds`.

### 2.5 YOLO (`yolo-inference`)

- **Image:** `hamidhrf/yolo-inference:v3.1`  (base: `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime`,
  system libs `libgl1-mesa-glx`, `libglib2.0-0`)
- **Python deps:** `numpy<2.0`, `ultralytics`, `prometheus_client`
- **Model:** Ultralytics `yolov8n.pt` (3M params), **fp32**.
- **Inference loop:** random uint8 image `(640, 640, 3)`, `model.predict(image, verbose=False)`.
- **Load generation:** self-driven, same 6-phase profile.
- **App metrics on :8000** — `yolo_inference_total`,
  `yolo_inference_latency_seconds` (buckets 0.01–0.5 s), `yolo_load_phase`,
  `yolo_sleep_time_seconds`, `yolo_detections_count`.

---

## 3. Metric collection stack

### 3.1 What runs

All monitoring components live under `k8s/monitoring/`:

| Component            | Manifest                                             | Kind        | Port  | Notes                                          |
|----------------------|-------------------------------------------------------|-------------|-------|------------------------------------------------|
| Prometheus           | `k8s/monitoring/prometheus-deployment.yaml` + `prometheus-config.yaml` | Deployment | 9090 (NodePort 30090) | image `prom/prometheus:v2.48.0`, 50 GB PVC     |
| Node exporter        | `k8s/monitoring/node-exporter.yaml`                  | DaemonSet   | 9100  | `prom/node-exporter:v1.7.0`, `--collector.pressure` enabled |
| DCGM exporter        | `k8s/monitoring/dcgm-exporter.yaml` (and `dcgm-exporter-pod-metrics.yaml`) | DaemonSet | 9400  | `nvcr.io/nvidia/k8s/dcgm-exporter:3.3.5-3.4.0-ubuntu22.04`, mounts `/var/lib/kubelet/pod-resources` |
| kube-state-metrics   | `k8s/monitoring/kube-state-metrics.yaml`             | Deployment  | 8080  | `kube-state-metrics:v2.10.1`                   |
| Kepler               | `k8s/monitoring/kepler.yaml`                         | DaemonSet   | 9102  | eBPF-based power estimator, privileged         |
| Grafana              | `k8s/monitoring/grafana-deployment.yaml`             | Deployment  | 3000 (NodePort 30030) | `grafana/grafana:10.2.2`                       |
| cAdvisor             | *(built into kubelet)*                                | —           | —     | scraped by Prometheus job `kubelet-cadvisor` via `/api/v1/nodes/${node}/proxy/metrics/cadvisor` |

### 3.2 Scrape interval and retention

From `k8s/monitoring/prometheus-config.yaml`:

```yaml
global:
  scrape_interval: 5s
  evaluation_interval: 5s
```

From `k8s/monitoring/prometheus-deployment.yaml`:

```yaml
- '--storage.tsdb.retention.time=30d'
- '--storage.tsdb.path=/prometheus'
```

`tools/run_experiment_v3.py:87` sets `self.scrape_interval = 5` and uses `step=5s` when
calling `/api/v1/query_range`.

### 3.3 Exact metric names collected

The full list queried per experiment lives in `tools/run_experiment_v3.py:97-187`. Grouped:

**Pod-level (per pod)**
- `pod_cpu_usage` — `sum by (pod) (rate(container_cpu_usage_seconds_total{pod=~"{workload}.*"}[1m]))`
- `pod_memory_bytes` — `sum by (pod) (container_memory_working_set_bytes{pod=~"{workload}.*"})`
- `pod_psi_cpu` — `sum by (pod) (rate(container_pressure_cpu_waiting_seconds_total{pod=~"{workload}.*"}[1m]))`
- `pod_psi_memory` — `sum by (pod) (rate(container_pressure_memory_waiting_seconds_total{pod=~"{workload}.*"}[1m]))`
- `pod_psi_io` — `sum by (pod) (rate(container_pressure_io_waiting_seconds_total{pod=~"{workload}.*"}[1m]))`
- `pod_latency_avg` — `rate({prefix}_inference_latency_seconds_sum[1m]) / rate({prefix}_inference_latency_seconds_count[1m])`
- `pod_throughput` — `rate({prefix}_inference_total[1m])`

**GPU-level (device 0)**
- `gpu_utilization` — `DCGM_FI_DEV_GPU_UTIL{gpu="0"}`
- `gpu_memory_used` — `DCGM_FI_DEV_FB_USED{gpu="0"}`
- `gpu_memory_total` — `DCGM_FI_DEV_FB_FREE{gpu="0"} + DCGM_FI_DEV_FB_USED{gpu="0"}`
- `gpu_power_watts` — `DCGM_FI_DEV_POWER_USAGE{gpu="0"}`
- `gpu_temperature` — `DCGM_FI_DEV_GPU_TEMP{gpu="0"}`

**System-level (host)**
- `node_cpu_usage` — `1 - avg(rate(node_cpu_seconds_total{mode="idle"}[1m]))`
- `node_memory_used_percent` — `1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)`
- `node_memory_available_bytes` — `node_memory_MemAvailable_bytes`
- `node_psi_cpu` — `rate(node_pressure_cpu_waiting_seconds_total[1m])`
- `node_psi_memory` — `rate(node_pressure_memory_waiting_seconds_total[1m])`
- `node_psi_io` — `rate(node_pressure_io_waiting_seconds_total[1m])`

**Aggregated app metrics (across all pods of one workload)**
- `app_latency_p50`, `app_latency_p95`, `app_latency_p99` — `histogram_quantile(q, sum(rate({prefix}_inference_latency_seconds_bucket[1m])) by (le))`
- `app_throughput` — `sum(rate({prefix}_inference_total[1m]))`

Prometheus scrape interval: **5 s**. Retention: **30 d**.

### 3.4 How PSI metrics are collected

There is **no custom PSI exporter** in this repo. PSI is delivered via two out-of-the-box paths:

- **Node-level PSI**: node-exporter v1.7.0 with `--collector.pressure` reads
  `/proc/pressure/{cpu,memory,io}` from the host and exposes
  `node_pressure_{cpu,memory,io}_waiting_seconds_total`.
- **Pod/container-level PSI**: cAdvisor embedded in the kubelet reads each container's
  cgroup-v2 `pressure` files and exposes
  `container_pressure_{cpu,memory,io}_waiting_seconds_total`, keyed by the
  `pod`/`container` labels that Prometheus attaches via kubernetes SD.

`scripts/monitoring/enable_pod_level_metrics.sh` is what switched the training pipeline
from `node_pressure_*` to `container_pressure_*` for the per-pod PSI columns.

### 3.5 Output format per experiment

**One CSV per metric, plus a text metadata file**, written to
`data/raw/phase1_v3/{workload}_r{replicas}/`.

Naming: `{workload}_r{replicas}_{metric}_{YYYYMMDD_HHMMSS}.csv`
plus `{workload}_r{replicas}_{YYYYMMDD_HHMMSS}_timestamps.txt`.

Columns:
- Pod-level CSVs: `timestamp, value, pod`
- Node CSVs: `timestamp, value, instance, job`
- GPU CSVs: `timestamp, value, DCGM_FI_DRIVER_VERSION, Hostname, UUID, __name__, device, gpu, instance, job, modelName`
- App aggregated CSVs: `timestamp, value`

Extraction code: `tools/run_experiment_v3.py:438-482` (Prometheus `/api/v1/query_range` →
pandas DataFrame → `df.to_csv`).

---

## 4. Experiment runner

### 4.1 Script and invocation

**Script:** `tools/run_experiment_v3.py`

```
python tools/run_experiment_v3.py <workload> <replicas>
```

**Parameters:**
- `workload` — one of `bert | gpt2 | resnet152 | whisper | yolo`
- `replicas` — integer, in practice 1–10

Duration, warm-up, scrape interval, cleanup delay and Prometheus URL are hard-coded in the
class constructor (`run_experiment_v3.py:78-90`); there is no `--duration`, `--seed`, or
similar CLI flag.

Hard-coded constants:
- `warmup_duration = 300`  (5 min, no metrics recorded)
- `experiment_duration = 3600`  (60 min, recorded window)
- `scrape_interval = 5`
- `cleanup_delay = 30`
- `prometheus_url = "http://172.22.174.58:30090"`

### 4.2 End-to-end lifecycle

1. **Pre-flight checks** (`:202-252`): probe Prometheus, ensure no lingering pods for the
   workload, ensure node memory < 85 %.
2. **Deploy** (`:264-331`): `kubectl apply -f k8s/workloads/{workload}-deployment.yaml`
   (idempotent), then `kubectl scale deployment {name} --replicas={replicas}`. Waits up to
   300 s for `Running`+ready.
3. **Warmup** (`:333-357`): sleep 300 s while pods stabilise (no data recorded).
4. **Experiment** (`:359-436`): sleep 3600 s. Writes a timestamps.txt file containing
   workload, replica count, ISO start/end, duration, version tag "Phase 1 v3", and the
   Business Day profile name.
5. **Metric collection** (`:438-513`): iterate over the 21-metric list, call
   Prometheus `/api/v1/query_range` with `step=5s`, dump each result as a CSV.
6. **Cleanup** (`:515-528`): `kubectl delete deployment {name}`; wait 30 s.

### 4.3 Where results land

`data/raw/phase1_v3/{workload}_r{replicas}/*.csv` (+ the timestamps `.txt` file). Each
run produces ~21 metric CSVs, ~715 rows per pod-level CSV per pod.

---

## 5. From raw traces to training data

### 5.1 Preprocessing script

**Script:** `scripts/phase4/preprocess_phase4.py`

```
python scripts/phase4/preprocess_phase4.py            # default: all workloads, all methods
python scripts/phase4/preprocess_phase4.py --workloads bert gpt2
python scripts/phase4/preprocess_phase4.py --methods raw diff windows
```

Output: `data/processed/phase4/unified/combined_dataset.npz` plus
`data/processed/phase4/unified/combined_normalization.json`. Per-workload NPZs are also
written under `data/processed/phase4/raw/`.

### 5.2 Combined dataset tensor

Loaded with `np.load(..., allow_pickle=True)`:

| Key              | Shape          | dtype    | Meaning                                          |
|------------------|----------------|----------|--------------------------------------------------|
| `traces`         | (275, 715, 10) | float64  | Per-pod minmax-normalised metric traces          |
| `replica_counts` | (275,)         | int64    | Replica count per trace, ∈ [1, 10]               |
| `workload_ids`   | (275,)         | int32    | 0=bert, 1=gpt2, 2=resnet152, 3=whisper, 4=yolo   |
| `workload_names` | (5,)           | object   | `['bert','gpt2','resnet152','whisper','yolo']`   |
| `metric_names`   | (10,)          | object   | The 10 metrics (see §5.3)                        |
| `train_idx`      | (245,)         | int64    | 90 % train split                                 |
| `val_idx`        | (30,)          | int64    | 10 % val split, split at pod level, seed 42      |
| `metadata`       | (275,)         | object   | Dicts: workload, replica_count, pod_index, …    |

275 = 5 workloads × (1+2+…+10) = 5 × 55 pods.

### 5.3 The 10 metrics ingested and the 7 that S36 trains on

Order comes from `scripts/phase4/preprocess_phase4.py:85-89`:

```python
POD_METRICS        = ["pod_cpu_usage", "pod_memory_bytes", "pod_psi_cpu",
                      "pod_latency_avg", "pod_throughput"]
SYSTEM_GPU_METRICS = ["gpu_utilization", "gpu_memory_used", "gpu_memory_total",
                      "gpu_power_watts", "gpu_temperature"]
ALL_METRICS        = POD_METRICS + SYSTEM_GPU_METRICS  # 10 total
```

**Only per-workload minmax normalisation is applied**; no log, no clipping, no
standardisation. Params (min, max per metric) are persisted in
`data/processed/phase4/unified/combined_normalization.json` under
`{workload: {method: "minmax", metric_names: [...], params: {metric: {min, max}}}}`.

**S36 drops** the three constant-across-a-run GPU metrics for every workload
(`scripts/phase4/timegan/timegan_s36.py:64-70`):

```python
DROP_PER_WORKLOAD = {
    "bert":      {"gpu_memory_total", "gpu_memory_used", "gpu_temperature"},
    "gpt2":      {"gpu_memory_total", "gpu_memory_used", "gpu_temperature"},
    "resnet152": {"gpu_memory_total", "gpu_memory_used", "gpu_temperature"},
    "whisper":   {"gpu_memory_total", "gpu_memory_used", "gpu_temperature"},
    "yolo":      {"gpu_memory_total", "gpu_memory_used", "gpu_temperature"},
}
```

Resulting **7-metric S36 tensor order** (all 5 workloads):

1. `pod_cpu_usage`
2. `pod_memory_bytes`
3. `pod_psi_cpu`
4. `pod_latency_avg`
5. `pod_throughput`
6. `gpu_utilization`
7. `gpu_power_watts`

### 5.4 Segmentation and 715 → 720 resampling

Constants in `scripts/phase4/timegan/timegan_s36.py:107-109`:

```python
DEFAULT_PHASE_BOUNDARIES = [0, 96, 180, 300, 420, 600]
N_PHASES = 6
SEGMENT_LEN = 120
```

The 715-step trace is cut on these boundaries into 6 phases whose native spans are
`[96, 84, 120, 120, 180, 115]`. Each phase is **linearly interpolated to exactly 120
timesteps** (`:196-202`), giving a `(6 × 120,)` = `(720,)` time axis for training.
Generated traces are truncated back to 715 on the way out (`:307`,
`return full_trace[:, :715, :]`).

### 5.5 Where the conditioning variables live

- **Replica count** — stored raw in `combined_dataset.npz['replica_counts']`.
  At training time it is scalar-normalised in `SegmentDataset.__getitem__`
  (`timegan_s36.py:234`): `r_norm = (r_val - 1.0) / 9.0`. Fed to the generator through
  `nn.Sequential(nn.Linear(1, 16), nn.Tanh())`.
- **Phase index** — derived on the fly by `SegmentDataset` (0..5). Fed through
  `nn.Embedding(N_PHASES + 1, ph_emb_dim=8)` (`:254`).
- **Workload identity** — stored as `workload_ids` (int32) and `workload_names` (object)
  in the combined NPZ, but **not used as a model input**: S36 trains **one separate
  generator per workload**, checkpointed under `models/phase4/timegan_s36/s36_{workload}_…/`.
  Workload is therefore an implicit condition (choice of model file), not a learned
  embedding.

---

## 6. Assumptions that would break under a custom scheduler

If the workloads stop being submitted as normal Kubernetes Deployments and are instead
handed to a single scheduler-owning pod, the following pipeline assumptions all break.
Each one needs an explicit story.

1. **Deployment-name-based Prometheus filters.** Every pod-level query uses
   `{pod=~"{workload}.*"}`, e.g. `bert-inference-*`. With a single scheduler pod owning
   all workloads, container names/pod names no longer identify the workload — pod-level
   time series would collapse into one label group.
2. **`sum by (pod)` aggregation.** Preprocessing assumes one row per pod per timestep and
   discovers `num_pods` from the first pod-level CSV
   (`preprocess_phase4.py:141-149`). If all workload activity runs inside one scheduler
   pod, `num_pods == 1` always and `replica_count` conditioning becomes fictitious.
3. **cAdvisor `container_*` metrics keyed by pod cgroups.** `container_cpu_usage_seconds_total`,
   `container_memory_working_set_bytes` and `container_pressure_*_waiting_seconds_total`
   are per-container/per-cgroup. Workloads that share a scheduler container share a
   cgroup and are indistinguishable to cAdvisor.
4. **PSI paths tied to per-pod cgroups.** `pod_psi_cpu`, `pod_psi_memory`, `pod_psi_io`
   come from `container_pressure_*` on each pod's cgroup v2 tree. A scheduler that
   multiplexes work inside its own cgroup will only expose *its own* PSI — you get
   scheduler-level pressure, not per-workload pressure. This is the single biggest gap
   for S36, which trains on `pod_psi_cpu`.
5. **DCGM pod-to-GPU mapping.** The DCGM exporter tags metrics with pod UID via
   `/var/lib/kubelet/pod-resources`. With one pod holding the GPU, every GPU sample is
   attributed to the scheduler pod — the `pod` label carries no workload information.
6. **`kubectl apply` / `kubectl scale` / `kubectl delete` lifecycle.**
   `tools/run_experiment_v3.py:264-528` drives the whole experiment via these commands
   and waits on Deployment readiness. Nothing shells to the custom scheduler.
7. **`replicas` as the load-scaling axis.** Every training sample is labelled with an
   integer `replica_count ∈ [1, 10]` and S36 uses `(r-1)/9` as its conditioning scalar.
   A custom scheduler will likely express concurrency differently (job count, slice
   share, time-slice quantum) — that new axis has no home in the current NPZ schema.
8. **Business-day load profile lives inside each workload container.** Phase timing is
   baked into `scripts/workloads/*/inference_variable.py`. If the scheduler submits
   discrete jobs instead of running long-lived pods, the internal `while True:` loop
   and its 60-minute phase schedule no longer apply, and the training data's phase
   structure loses its physical basis.
9. **`prometheus.io/scrape` annotation on port 8000.** App metrics
   (`{prefix}_inference_latency_seconds`, `{prefix}_inference_total`) are scraped
   because each Deployment sets the annotation. Jobs run inside a scheduler pod are not
   automatically scraped and would need a per-workload metrics multiplexer or a
   push-based sink.
10. **Fixed run duration of 3600 s + 300 s warm-up.** The runner assumes 715 samples
    at 5 s intervals over one hour. Jobs that finish sooner (or run to a fixed total
    of ops) will produce shorter traces that break the 715 → 720 segmentation.
11. **Node-level PSI as a system-load proxy.** `node_pressure_*` currently reflects
    contention across N pods on the box. Under a scheduler that owns the GPU
    exclusively, node-level PSI reflects the scheduler's aggregate behaviour, not the
    experiment under study.
12. **Prometheus URL and cluster assumptions.** The runner points at
    `http://172.22.174.58:30090` (K8s NodePort). If the scheduler runs on a different
    host/namespace, the URL, RBAC, and `kubelet-cadvisor` scrape job all need review.
13. **`num_pods` inferred from CSV rows.** `preprocess_phase4.py:141-149` uses
    `first_df` to determine pod count and allocates
    `traces = np.zeros((num_pods, num_ts, 10))`. If pod-level rows disappear from the
    CSVs, this loop silently produces `num_pods = 0` or `= 1` and the resulting NPZ is
    unusable.
14. **Per-pod normalisation stats.** `combined_normalization.json` is *per workload*,
    computed across all pods of that workload. If the scheduler changes the resource
    profile (e.g. via time-slicing quantum), the stored min/max no longer bracket the
    real distribution and TimeGAN outputs drift out of range.
15. **Cleanup by `kubectl delete deployment`.** The scheduler pod won't be deleted
    between experiments — the runner needs a scheduler-native "drain / reset" hook.

---

## 7. What a compatible sharing mechanism must expose

To keep `preprocess_phase4.py` and `timegan_s36.py` unchanged, the custom scheduler must
present its GPU-sharing telemetry to Prometheus in a form indistinguishable from the
current per-pod stack. Concretely:

### 7.1 Required metric surface

Emit one time series per **synthetic "pod"** — i.e. per workload instance that the
scheduler is currently servicing — with the exact metric names below and a label
`pod` whose value matches `{workload}.*` (e.g. `bert-slot-3`, `gpt2-slot-1`).

| Metric emitted                                            | Label(s)                              | Semantics                                          |
|-----------------------------------------------------------|---------------------------------------|----------------------------------------------------|
| `container_cpu_usage_seconds_total`                       | `pod=<workload>-*`, `container=…`     | CPU seconds attributed to that workload instance   |
| `container_memory_working_set_bytes`                      | `pod=<workload>-*`                    | Working-set memory attributed to that instance     |
| `container_pressure_cpu_waiting_seconds_total`            | `pod=<workload>-*`                    | Cumulative CPU PSI-waiting seconds for that instance |
| `container_pressure_memory_waiting_seconds_total`         | `pod=<workload>-*`                    | Same for memory                                    |
| `container_pressure_io_waiting_seconds_total`             | `pod=<workload>-*`                    | Same for I/O                                       |
| `{prefix}_inference_latency_seconds` (Histogram: `_sum`, `_count`, `_bucket`) | `pod=<workload>-*`  | Application latency per instance, `prefix ∈ {bert, gpt2, resnet152, whisper, yolo}` |
| `{prefix}_inference_total`                                | `pod=<workload>-*`                    | Application request counter per instance           |
| `DCGM_FI_DEV_GPU_UTIL{gpu="0"}`                           | (device-level is fine)                | Whole-device utilisation; if per-slice attribution is possible, emit per-workload copies with `pod` label |
| `DCGM_FI_DEV_POWER_USAGE{gpu="0"}`                        | (device-level)                        | Whole-device power                                 |
| `DCGM_FI_DEV_FB_USED{gpu="0"}`, `DCGM_FI_DEV_FB_FREE{gpu="0"}`, `DCGM_FI_DEV_GPU_TEMP{gpu="0"}` | (device-level) | Not used by S36, but required to keep `preprocess_phase4.py` happy |
| `node_cpu_seconds_total`, `node_memory_MemAvailable_bytes`, `node_memory_MemTotal_bytes`, `node_pressure_{cpu,memory,io}_waiting_seconds_total` | *(host-level from node-exporter)* | Keep node-exporter running as-is                   |

The rate/quantile/sum wrappers in `tools/run_experiment_v3.py:97-187` (e.g.
`rate(...[1m])`, `histogram_quantile`, `sum by (pod)`) must resolve against these series
without modification.

### 7.2 Labelling contract

- Every per-workload series **must** carry a `pod` label whose value starts with the
  workload's Deployment name prefix: `bert-*`, `gpt2-*`, `resnet152-*`, `whisper-*`,
  `yolo-*`. Anything else and the `{pod=~"{workload}.*"}` regex will drop it.
- Concurrency **must** be exposable as an integer replica-equivalent in the range
  `[1, 10]`, recorded in the experiment metadata (the runner currently takes it as a
  CLI argument). If the scheduler expresses concurrency differently, provide a
  documented mapping to that integer axis.
- Application histogram metrics must use the exact HuggingFace/Ultralytics
  `{prefix}_*` names above so that the `pod_latency_avg` and `pod_throughput` queries
  compose without change.

### 7.3 Update frequency and retention

- Scrape cadence: **5 s** (matches Prometheus `scrape_interval` and the runner's
  `step=5s`). Slower cadences will alias into the 715-timestep-per-hour grid.
- Retention: at least the duration of one experiment (**3600 s + 300 s warm-up +
  30 s cleanup buffer**). The existing Prometheus retains 30 d, so anything that
  survives one experiment window is enough.

### 7.4 Data format the pipeline consumes

- **On the wire**: standard Prometheus text exposition (or a Prometheus-compatible
  scrape endpoint). No push, no OpenMetrics extras required.
- **Per-experiment on disk**: 21 CSVs plus a `*_timestamps.txt` file, exactly as
  produced today by `tools/run_experiment_v3.py`, under
  `data/raw/phase1_v3/{workload}_r{replicas}/`. Column schema per metric family is
  documented in §3.5.
- **After preprocessing**: a single `combined_dataset.npz` matching the schema in
  §5.2 (shape `(N, 715, 10)`, per-workload minmax normalisation, `replica_counts`,
  `workload_ids`). If the scheduler substitutes a different concurrency axis,
  either (a) map it onto `replica_counts ∈ [1..10]` before writing the NPZ, or
  (b) extend the NPZ schema and retrain S36 — the current model checkpoints will
  only accept the `(r-1)/9` scalar as its replica condition.

If those five things hold — metric names, `pod` label convention, 5 s cadence,
1-hour recording window, and integer replica axis — the existing preprocessing
(`preprocess_phase4.py`) and S36 training/generation code (`timegan_s36.py`,
`postprocess_s36.py`) run against the new scheduler unmodified.
