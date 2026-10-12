# Sweep 2 Drift Diagnostics (devLab)

Collected 2026-10-10 09:11–09:20 UTC, read-only, while the Tier 3 time-slicing
sweep (`sweep2-ts`) was running bert r=1. Nothing was deployed, stopped or
changed. The question: what changed on devLab between the August sweeps and
sweep 2, with dates where available. The window of interest is
2026-08-26 → 2026-10-08.

Reference dates. From the August data file names and `TIER*_NOTES.md`:
Tier 1 2026-08-13, Tier 2 2026-08-14 → 08-15, Tier 3 2026-08-16 → ~08-18.
Sweep 2 Tier 2 ran 2026-10-08 16:41 → 2026-10-10 07:10. Sweep 2 Tier 3
started 2026-10-10 08:54.

Items that needed root were run by the operator at 10:10 UTC; the results
are in section 6 and are folded into the sections below.

## Summary of changes since August

| Date (UTC) | Change | Source | Relevance |
|---|---|---|---|
| 2026-08-22 06:48 | Kernel 7.0.0-30 installed (unattended); not booted before 08-31 | apt history | August sweeps ran before this install |
| 2026-08-31 → 09-01 | Reboots: the August boot (7.0.0-29, up since 08-10 14:16) ends 08-31; two boots on 09-01 at 10:01 and 10:11, both **7.0.0-30** | `journalctl --list-boots`, kernel log (root) | first boots after the August sweeps |
| 2026-09-06 06:49 | Kernel **7.0.0-31** installed (unattended); this is the running kernel | apt history, `uname -r` | kernel differs from August, which ran **7.0.0-29** (confirmed, section 6) |
| 2026-09-09 08:28 | `nvtop` installed by user ivan | apt history | none expected |
| 2026-09-09 13:39 | **docker.io, containerd 2.2.2, runc 1.4.0, nvidia-container-toolkit 1.20.0** installed by user ivan; `dockerd` + `containerd` active since boot | apt history, `systemctl` | second container runtime on the node; 2 Docker container shims seen at 09:12, none at 10:10 and `docker ps -a` empty (section 6) |
| 2026-09-12 06:57 | **NVIDIA driver 580.173.02 → 580.178.04** (unattended-upgrades) | apt history | already recorded in `SWEEP2_ENVIRONMENT.md`; now dated |
| 2026-09-16 09:22 → 09:23 | Reboot (boot −1 ends 09-16 09:22; current uptime since 09-16 09:23). New driver and kernel 7.0.0-31 loaded from here | `journalctl --list-boots`, `uptime -s` | |
| 2026-09-16 11:33 | podman-stack packages (netavark, skopeo, containernetworking-plugins…) installed by user kostya | apt history | |
| 2026-09-16 11:42 → 12:13 | `odm-gpu`, `gpu-gatekeeper`, `gpu-scheduler` namespaces; **kube-prometheus-stack** (second Prometheus, Grafana, node-exporter, kube-state-metrics) | k8s creation timestamps | gatekeeper paused for sweep 2; kube-prom load in section 4 |
| 2026-09-25 06:05 | Kernel 7.0.0-34 installed (unattended); never booted | apt history | |
| 2026-10-07 06:45 | Kernel **7.0.0-38** installed (unattended); `linux-image-generic` now points to it. **Not running.** The next reboot changes the kernel | apt history | reboot risk during/between sweeps |
| unknown | **GPU power limit 400 W** (default 310 W) | `nvidia-smi` | not binding for these workloads (section 3); no record in the kernel log or shell histories, date and author still unknown |
| 2026-10-09 06:48, 10-10 06:14 | unattended-upgrades ran **during the sweep** (libpng, sudo, libxml2 only) | apt history | no relevant packages; unattended-upgrades remains enabled |

No change found to: kubelet configuration (unchanged since 2026-08-10),
workload manifests (unchanged since 2026-08-16), runner scripts and their
Prometheus queries (unchanged since 2026-08-14 14:41, before the first
August Tier 2 run), the August Prometheus (same image, flags, 5 s scrape
config, jobs).

---

## 1. Host

### Boots

`journalctl --list-boots` (as user; the per-boot kernel version needs root):

```
IDX BOOT ID                          FIRST ENTRY                 LAST ENTRY
 -3 0559e45afd3741adbb56244098dc54a1 Mon 2026-08-10 13:42:24 UTC Mon 2026-08-10 14:15:58 UTC
 -2 133ac89be3e84912a1e1a1a6661f913d Mon 2026-08-10 14:26:34 UTC Mon 2026-08-31 14:35:57 UTC
 -1 3f4d21d8970f474fbcb308567220b567 Tue 2026-09-01 12:06:34 UTC Wed 2026-09-16 09:22:29 UTC
  0 b22502763e6c4997b6ab1a76e9d7a5cf Fri 2026-09-25 10:03:19 UTC Sat 2026-10-10 08:54:35 UTC
```

`uptime -s` = 2026-09-16 09:23:28. The current boot's first journal entry
visible to an unprivileged user is 09-25; the boot itself is 09-16.

- All August sweeps ran in boot −2 (2026-08-10 → 08-31).
- Sweep 2 runs in boot 0 (since 2026-09-16).

The root view (`sudo journalctl -k -b N`, section 6) indexes boots
differently, because the unprivileged list only shows boots with entries
the user can read. Root's last four boots:

| Root index | Kernel boot (UTC) | Kernel | Unprivileged view |
|---|---|---|---|
| −3 | 2026-08-10 14:16:39 | 7.0.0-29 | −2 (first user entry 14:26) |
| −2 | 2026-09-01 10:01:27 | 7.0.0-30 | not listed (short boot, ~10 min) |
| −1 | 2026-09-01 10:11:43 | 7.0.0-30 | −1 (first user entry 12:06) |
| 0 | 2026-09-16 09:23:43 | 7.0.0-31 | 0 |

So the August sweeps ran on **7.0.0-29**, and sweep 2 runs on **7.0.0-31**.

### Kernels

| Kernel | Installed | Status |
|---|---|---|
| 7.0.0-29 | 2026-08-10 12:18 | removed 2026-09-06; **the August kernel** (booted 08-10 14:16, confirmed by the kernel log) |
| 7.0.0-30 | 2026-08-22 06:48 | ran 2026-09-01 → 09-16 only (between the sweeps) |
| 7.0.0-31 | 2026-09-06 06:49 | **running** (`uname -r`) |
| 7.0.0-38 | 2026-10-07 06:45 | installed, default for the next boot |

### Package changes 2026-08-26 → 2026-10-10 (filtered)

nvidia / kernel / container runtime / kube* / containerd / runc / CPU and
power management. Full list: `/var/log/apt/history.log{,.1.gz}`.

```
2026-09-06 06:49  unattended  linux-image/modules 7.0.0-31 (remove 7.0.0-29)
2026-09-09 13:39  ivan        docker.io, containerd 2.2.2-0ubuntu1.1, runc 1.4.0-0ubuntu1
2026-09-09 13:39  ivan        nvidia-container-toolkit(+base), libnvidia-container1/-tools 1.20.0-1
2026-09-12 06:57  unattended  nvidia-*-580-server 580.173.02 -> 580.178.04 (dkms, utils, compute, firmware, ...)
2026-09-25 06:05  unattended  linux-image/modules 7.0.0-34 (remove 7.0.0-30)
2026-10-07 06:45  unattended  linux-image/modules 7.0.0-38 (remove 7.0.0-34)
```

- **No changes** to cri-o (1.31.5, since 2026-08-10), kubelet/kubeadm/kubectl
  (1.34.0, since 2026-08-10), thermald, microcode, tuned, cpupower or
  power-profiles in the window.
- Other window activity was routine security updates: openssl, glibc,
  openssh, sudo, perl, python3.14, systemd-adjacent libs, dracut, rsyslog.

## 2. CPU

devLab is a **KVM guest**: Intel Xeon 6760P, 16 vCPUs, 1 thread/core, 1
NUMA node. GPU virtualization mode is Pass-Through.

| Item | Value |
|---|---|
| Scaling governor | **not exposed**: no `/sys/devices/system/cpu/cpu*/cpufreq` in the guest |
| Current / max frequency | `/proc/cpuinfo` reports 2200 MHz on all 16 vCPUs (the nominal value seen by the guest, not live frequency) |
| Boost/turbo, EPP | **not exposed** (no `cpufreq/boost`, `intel_pstate`, `amd_pstate`) |
| Steal time | ~0 (`vmstat` st = 0; cumulative steal 151,290 jiffies vs 3.2e9 idle) |
| Load average (09:11) | 0.48 0.36 0.39 |

CPU frequency, turbo and power policy belong to the **hypervisor host** and
cannot be observed or compared from inside devLab. Host-side changes between
August and October cannot be ruled out from here; that needs the hypervisor
operator.

Top CPU consumers outside the sweep pod. `top`, 5 s sample at 09:12,
bert r=1 running, node 95.9% idle:

| Process | %CPU | Note |
|---|---|---|
| kubelet | 13.0 | restarted 08:49 by mig-manager |
| prometheus (kube-prometheus-stack, pid 120660) | 12.8 | bursty; 1 h avg 0.034 cores |
| *python (bert inference, sweep pod)* | *10.2* | *sweep workload* |
| kube-apiserver | 5.4 | |
| claude (this agent session) | 4.8 | |
| etcd | 3.0 | |
| dcgm-exporter | 2.2 | |
| calico-node | 1.8 | |
| kube-controller-manager | 1.4 | |
| gpu-scheduler (`scheduler_service.py`) | 1.4 | disabled, idle polling |
| prometheus (August, :30090) | 1.2 | |
| grafana, node_exporter, crio, ... | <1 each | |

Docker: `dockerd` (pid 1796) and `containerd` (pid 1545) have been active
since boot, with 2 `containerd-shim-runc-v2 -namespace moby` processes, so 2
Docker containers are running. They are invisible to Kubernetes. Their CPU
use did not appear among the top consumers.

At 10:10 the shims were gone (`ps` shows no `containerd-shim` process, while
`dockerd` is still up). The root `docker ps -a` / `docker stats --no-stream`
output (section 6) is empty, so the containers were not just stopped but
removed, likely started with `--rm`. What they were cannot be recovered from
here. They ran during bert r=1 of the Tier 3 sweep and possibly earlier. Ask
ivan, who installed Docker.

## 3. GPU

`nvidia-smi -q -d CLOCK,POWER,PERFORMANCE,COMPUTE` at 09:12 (time-slicing,
bert r=1 running):

| Item | Value |
|---|---|
| Driver | 580.178.04 (August: 580.173.02, changed 2026-09-12) |
| Performance state | P0 |
| Clocks (graphics/SM/mem) | 1785 / 1785 / 2619 MHz = max clocks |
| Application clocks | 1785 / 2619 = default application clocks (**not changed**) |
| Locked clocks | none indicated ("Applications Clocks Setting: Not Active"; no clock-event reason active) |
| **Power limit** | **current 400 W, requested 400 W, default 310 W**, min 200 W, max 400 W |
| Power draw | 117 W (samples 116–140 W) |
| Persistence mode | Disabled (`nvidia-persistenced --no-persistence-mode` running, as before) |
| Compute mode | Default |
| ECC | Enabled |
| Clock event (throttle) reasons | none active; counters SW power cap 0 µs, HW slowdown 0, HW thermal 0, SW thermal 0, power brake 0 |

**Power limit.** Someone raised it from the 310 W default to the 400 W
maximum; when is unknown. No systemd unit, cron job or script under `/etc`,
`/usr/local` or `/opt` sets it. Prometheus keeps 30 days (oldest data
2026-09-08) and DCGM does not export the limit metric, so it has no
history. It does **not** affect the measurements:

| Data | Peak `gpu_power_watts` |
|---|---|
| August Tier 1 | 103.9 W |
| August Tier 2 (MIG) | 148.8 W (gpt2 r=7) |
| August Tier 3 (time-slicing) | 163.8 W (whisper r=10) |
| Sweep 2 Tier 2 (MIG) | 143.6 W (gpt2 r=7) |

All peaks are about half of either limit, and the SW power-cap counter is 0.
Who set it and when: not found. The kernel log since 2026-08-01 has no
NVRM power or limit message. No user's `.bash_history` has an `nvidia-smi`
power, clock or persistence command, and `/root/.bash_history` does not
exist (section 6). It may have been set from a container, a script or a
shell whose history wasn't saved, or in the VM/host configuration.

Idle power (Prometheus, 6 h max): ~62.7 W from 09-08 to 09-23, ~63.0–63.2 W
from 09-23 to 10-08. Sweep 2 load: 93–102 W.

## 4. Kubernetes

### kubelet

- `/var/lib/kubelet/config.yaml` last modified 2026-08-10 14:56:25.
- `kubeadm-flags.env` and `/etc/default/kubelet` (`--node-ip=172.22.174.66`):
  last modified 2026-08-10.
- Live `configz`: `cpuManagerPolicy=none`, `topologyManagerPolicy=none`,
  `memoryManagerPolicy=None`, `cpuCFSQuota=true` (100 ms), no
  system/kube reserved, `cgroupDriver=systemd`, `maxPods=110`. **Unchanged
  since cluster creation.**
- `cpu_manager_state`: dated 2026-08-10 14:56:26, unreadable without root.
  With policy `none` it carries no pinning.

### Sweep pod resources vs manifest

- Running pod: `bert-inference-69f8cc488d-n69zr`, image
  `hamidhrf/bert-inference:v4`, `requests/limits nvidia.com/gpu: 1`, **no CPU
  or memory requests/limits**, QoS **BestEffort**.
- Manifest `k8s/workloads/bert-deployment.yaml`: `limits nvidia.com/gpu: 1`,
  `runtimeClassName: nvidia`. Image, env, command and args are identical to
  the live Deployment. The only `resources` difference is `1` written as an
  integer vs a string (same value).
- All five manifests have the same shape. Last change 2026-08-16 (Tier 3
  copies), before the August Tier 3 sweep.
- Because the pods are BestEffort with no CPU pinning, sweep pods share the
  16 vCPUs with every other process on the node. This was equally true in
  August. Background load that is new since August is the kube-prometheus
  stack, Docker, and gpu-scheduler.

### kube-prometheus-stack resource use

`kubectl top`: Metrics API not available. Measured from cAdvisor via the
August Prometheus instead:

| Pod | CPU, 1 h avg | Memory, 1 h peak |
|---|---|---|
| prometheus-monitoring-kube-prometheus-prometheus-0 | 0.034 cores | 1445 MB |
| monitoring-grafana | 0.012 cores | 1266 MB |
| monitoring-kube-prometheus-operator | 0.003 cores | 120 MB |
| monitoring-kube-state-metrics | 0.001 cores | 103 MB |
| alertmanager-…-0 | 0.001 cores | 168 MB |
| *for comparison: August prometheus-77df554df5* | *0.017 cores* | *5415 MB* |

- 24 h average: kube-prometheus-stack pods **0.053 cores (~0.3% of 16
  vCPUs)**; other `monitoring` pods 0.033 cores.
- For scale: sweep pods (`default`) 1.65 cores; whole node non-idle 2.17
  cores.
- The stack's Prometheus is bursty (12.8% of one vCPU in a 5 s `top`
  sample).

## 5. Measurement path

- **Prometheus at :30090.** `prom/prometheus:v2.48.0`, Deployment
  generation 1, created 2026-08-11 20:15. Retention 30d on PVC
  `prometheus-pvc`. Restarted 3 times, last at the 2026-09-16 09:23 boot.
- **Scrape config.** Global and per-job `scrape_interval` 5 s,
  `evaluation_interval` 5 s, `scrape_timeout` 5 s. Jobs: `prometheus`,
  `node-exporter`, `dcgm-exporter`, `kubelet-cadvisor`,
  `kube-state-metrics`, `ai-inference-apps`, identical to the repo
  `k8s/monitoring/prometheus-config.yaml`. ConfigMap
  `prometheus-config` was created 2026-08-11 20:20; the repo file was last
  changed 2026-08-11 (`38e799b`).
- **Latency queries are unchanged.**
  - `tools/run_experiment_v3.py` (Tier 3): last change `6ff19a4`
    2026-08-12.
  - `tools/run_experiment_v4.py` (Tier 2): added in `77391b2` 2026-08-14
    14:41, before the first August Tier 2 file (08-14 16:04); no change
    since.
  - Both use
    `histogram_quantile(q, sum(rate(<prefix>_inference_latency_seconds_bucket[1m])) by (le))`
    for p50/p95/p99, and `rate(_sum[1m]) / rate(_count[1m])` per pod for
    `pod_latency_avg`.
- **Image digests.** August did not record the `:v4` image digests (no
  digest in `TIER*_NOTES.md` or `EXTENSION_JOURNAL.md`), so "same image as
  August" can only be inferred from the tag. The sweep 2 digests are in
  `SWEEP2_ENVIRONMENT.md` section 4. All five `:v4` images in the node
  cache were built 2026-08-02 22:05–22:07 (+02:00). That is before every
  August run, and there has been no rebuild since. This is consistent with
  the same images, but it isn't a digest match.

## 6. Root-only checks (run by the operator, 2026-10-10 10:10 UTC)

Commands as given, output in `~/drift_diag_*.txt`.

1. Kernel per boot:
   `for b in -3 -2 -1 0; do echo "boot $b"; sudo journalctl -k -b $b -o short-iso --no-pager | grep -m1 'Linux version'; done > ~/drift_diag_kernels.txt 2>&1`

   ```
   boot -3  2026-08-10T14:16:39+00:00  Linux version 7.0.0-29-generic (#29-Ubuntu SMP PREEMPT_DYNAMIC Fri Jul 17 20:52:35 UTC 2026, 7.0.12)
   boot -2  2026-09-01T10:01:27+00:00  Linux version 7.0.0-30-generic (#30-Ubuntu SMP PREEMPT_DYNAMIC Fri Jul 31 18:22:54 UTC 2026, 7.0.12)
   boot -1  2026-09-01T10:11:43+00:00  Linux version 7.0.0-30-generic (#30-Ubuntu SMP PREEMPT_DYNAMIC Fri Jul 31 18:22:54 UTC 2026, 7.0.12)
   boot 0   2026-09-16T09:23:43+00:00  Linux version 7.0.0-31-generic (#31-Ubuntu SMP PREEMPT_DYNAMIC Sat Aug  1 04:26:38 UTC 2026, 7.0.14)
   ```
   Result: August = 7.0.0-29 (upstream 7.0.12); sweep 2 = 7.0.0-31
   (upstream 7.0.14).

2. Power limit, who and when:
   `{ sudo journalctl -k --since 2026-08-01 --no-pager | grep -i -E 'NVRM.*(power|limit)'; sudo grep -H -E 'nvidia-smi.*(-pl|--power-limit|-lgc|--lock-gpu-clocks|-ac |-pm )' /root/.bash_history /home/*/.bash_history; } > ~/drift_diag_powerlimit.txt 2>&1`

   ```
   grep: /root/.bash_history: No such file or directory
   ```
   Result: no kernel-log match, no history match. Origin of the 400 W limit
   is still unknown.

3. Docker containers:
   `sudo docker ps -a --format '{{.ID}} {{.Image}} {{.Status}} {{.CreatedAt}} {{.Names}}' > ~/drift_diag_docker.txt 2>&1; sudo docker stats --no-stream >> ~/drift_diag_docker.txt 2>&1`

   Output file is empty (0 bytes). Result: no Docker containers, running or
   stopped, at 10:10; the 2 shims seen at 09:12 had exited.

4. Workload image creation dates:
   `for w in bert gpt2 resnet152 whisper yolo; do sudo crictl inspecti -o go-template --template '{{.status.repoTags}} {{.info.imageSpec.created}}' docker.io/hamidhrf/$w-inference:v4; done > ~/drift_diag_images.txt 2>&1`

   ```
   [docker.io/hamidhrf/bert-inference:v4] 2026-08-02T22:05:41.298753098+02:00
   [docker.io/hamidhrf/gpt2-inference:v4] 2026-08-02T22:05:56.076672291+02:00
   [docker.io/hamidhrf/resnet152-inference:v4] 2026-08-02T22:06:04.382855068+02:00
   [docker.io/hamidhrf/whisper-inference:v4] 2026-08-02T22:06:45.648265577+02:00
   [docker.io/hamidhrf/yolo-inference:v4] 2026-08-02T22:07:41.544278213+02:00
   ```
   Result: all built 2026-08-02, before the August runs.
