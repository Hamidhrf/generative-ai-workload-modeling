#!/usr/bin/env python3
"""Tier 2 per-slice-vs-averaged diagnostic (S36 Tier 2 retrain plan, Step 2).

Quantifies the cross-pod signal that a naive /replica_count broadcast
(the imputation Tier 3 and Tier 2's aggregated GPU metrics use, and what
Tier 2's gpu_utilization deliberately avoids via native per-slice MIG
attribution) would destroy, if applied to Tier 2's real per-pod
gpu_utilization signal at r=7.

For each workload:
  1. Load data/processed/tier2/<workload>_traces.npz + normalization.json,
     denormalize gpu_utilization to raw units.
  2. Extract the r=7 subset (7 pods, 715 timesteps) -> real_per_slice[7, 715].
  3. Construct the /7 broadcast: mean across pods at each timestep,
     repeated across all 7 pods -> averaged_broadcast[7, 715]. Cross-pod
     variance of this construction is 0 at every timestep by definition.
  4. Compute the signal /7 destroys:
       mean_cross_pod_var_real = mean over t of Var_pod(real_per_slice[:, t])
       var_of_pod_means_real   = Var_pod(time-mean of real_per_slice per pod)
  5. fraction_of_total_variance = var_of_pod_means_real / Var(real_per_slice
     flattened over pods and timesteps) -- fraction of total signal
     variance that is systematic cross-pod identity structure.

Produces two paper-body figures at outputs/analysis/tier2_sanity/:
  - tier2_per_slice_vs_averaged_bert_whisper.png
  - tier2_destroyed_variance_by_workload.png

Run from repo root:
    python scripts/analysis/tier2_per_slice_vs_averaged.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WORKLOADS = ["bert", "gpt2", "resnet152", "whisper", "yolo"]
TIER2_DIR = Path("data/processed/tier2")
OUT_DIR = Path("outputs/analysis/tier2_sanity")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_r7_gpu_utilization(workload):
    data = np.load(TIER2_DIR / f"{workload}_traces.npz", allow_pickle=True)
    traces = data["traces"]
    metric_names = [str(m) for m in data["metric_names"]]
    metadata = data["metadata"]

    with open(TIER2_DIR / f"{workload}_normalization.json") as f:
        norm = json.load(f)
    gpu_idx = metric_names.index("gpu_utilization")
    p = norm["params"]["gpu_utilization"]

    r7_indices = [i for i, m in enumerate(metadata) if m["replica_count"] == 7]
    assert r7_indices == list(range(21, 28)), (
        f"{workload}: r=7 pods not at expected indices 21..27, got {r7_indices}"
    )

    denorm_gpu = traces[r7_indices, :, gpu_idx] * (p["max"] - p["min"]) + p["min"]
    return denorm_gpu  # shape (7, 715)


def compute_diagnostics(real_per_slice):
    # real_per_slice: (7, 715)
    broadcast_value = real_per_slice.mean(axis=0)  # (715,)
    averaged_broadcast = np.tile(broadcast_value, (real_per_slice.shape[0], 1))  # (7, 715)

    cross_pod_var_real = real_per_slice.var(axis=0)  # (715,) var across pods, per t
    mean_cross_pod_var_real = float(cross_pod_var_real.mean())

    per_pod_time_mean_real = real_per_slice.mean(axis=1)  # (7,)
    var_of_pod_means_real = float(per_pod_time_mean_real.var())

    total_variance = float(real_per_slice.flatten().var())
    fraction_of_total_variance = var_of_pod_means_real / total_variance

    return {
        "averaged_broadcast": averaged_broadcast,
        "mean_cross_pod_var_real": mean_cross_pod_var_real,
        "var_of_pod_means_real": var_of_pod_means_real,
        "total_variance": total_variance,
        "fraction_of_total_variance": fraction_of_total_variance,
    }


def plot_bert_whisper(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, workload in zip(axes, ["bert", "whisper"]):
        real = results[workload]["real_per_slice"]
        broadcast = results[workload]["diag"]["averaged_broadcast"]
        num_pods, num_timesteps = real.shape
        t = np.arange(num_timesteps)

        cmap = plt.get_cmap("tab10")
        for pod in range(num_pods):
            ax.plot(t, real[pod, :], color=cmap(pod % 10), lw=0.8, alpha=0.6,
                     label=f"pod {pod}" if pod < num_pods else None)
        ax.plot(t, broadcast[0, :], color="black", lw=2.2, label="/7 broadcast", zorder=10)

        ax.set_title(f"{workload.capitalize()} r=7", fontsize=13)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("GPU utilization (%)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2, loc="upper right")

    fig.suptitle("Tier 2 per-pod gpu_utilization: real per-slice vs. /7 broadcast",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "tier2_per_slice_vs_averaged_bert_whisper.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_destroyed_variance(results):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    values = [results[w]["diag"]["mean_cross_pod_var_real"] for w in WORKLOADS]
    x = np.arange(len(WORKLOADS))
    ax.bar(x, values, color="tab:blue", width=0.55, label="real per-slice (mean cross-pod var)")
    ax.axhline(0.0, color="tab:red", linestyle="--", lw=1.5,
               label="/7 broadcast (variance = 0 by construction)")

    ax.set_xticks(x)
    ax.set_xticklabels([w.capitalize() for w in WORKLOADS])
    ax.set_ylabel("Mean cross-pod variance of gpu_utilization (%$^2$)")
    ax.set_title("Tier 2: cross-pod variance destroyed by /7 broadcast imputation, r=7",
                 fontsize=13)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9)

    fig.tight_layout()
    out = OUT_DIR / "tier2_destroyed_variance_by_workload.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def print_table(results):
    print("\n## Per-slice-vs-averaged diagnostic -- gpu_utilization at r=7\n")
    header = (f"{'workload':10s} | {'mean_cross_pod_var_real':>24s} | "
              f"{'var_of_pod_means_real':>22s} | {'fraction_of_total_var':>22s}")
    print(header)
    print("-" * len(header))
    for w in WORKLOADS:
        d = results[w]["diag"]
        print(f"{w:10s} | {d['mean_cross_pod_var_real']:24.6f} | "
              f"{d['var_of_pod_means_real']:22.6f} | {d['fraction_of_total_variance']:22.6f}")


def main():
    results = {}
    for w in WORKLOADS:
        real_per_slice = load_r7_gpu_utilization(w)
        diag = compute_diagnostics(real_per_slice)
        results[w] = {"real_per_slice": real_per_slice, "diag": diag}

    print_table(results)

    p1 = plot_bert_whisper(results)
    p2 = plot_destroyed_variance(results)
    print(f"\nSaved {p1}")
    print(f"Saved {p2}")


if __name__ == "__main__":
    main()
