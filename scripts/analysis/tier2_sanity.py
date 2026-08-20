#!/usr/bin/env python3
"""Tier 2 sanity audit (S36 Tier 2 retrain plan, Step 2).

Adapted from the denormalization pattern in
scripts/analysis/a16_vs_tier3_wasserstein.py (loads processed npz +
normalization.json per tier, denormalizes to raw physical units before
computing statistics -- traces are stored min-max normalized independently
per tier, so comparing normalized values directly across tiers is not
physically meaningful).

Note: scripts/analysis/tier3_sanity.py (the script named as this script's
baseline) does NOT actually do cross-tier denormalized comparison -- it
loads raw Tier 3 CSVs via tools.load_experiment for all 5 workloads/r=1..10
and produces per-workload 2x2 plots plus peak-of-peaks tables, with no A16
involvement and no denormalization step (raw CSVs are already in raw
units). That does not match the "A16 vs Tier 3, denormalized, BERT as
representative workload" description this script is supposed to adapt.
a16_vs_tier3_wasserstein.py is the actual match for that pattern, so this
script's denormalization/table logic is adapted from there instead, while
its output shape (a per-metric MEAN audit table, not a distance table) is
new.

Produces two stdout tables:
  1. 3-column (A16, Tier 2, Tier 3) raw per-metric mean audit for BERT.
  2. Per-workload cross-pod spread audit for gpu_utilization at r=7,
     across all 5 Tier 2 workloads.

Run from repo root:
    python scripts/analysis/tier2_sanity.py
"""
import json
from pathlib import Path

import numpy as np

WORKLOADS = ["bert", "gpt2", "resnet152", "whisper", "yolo"]
A16_DIR = Path("data/processed/phase1_v3")
TIER2_DIR = Path("data/processed/tier2")
TIER3_DIR = Path("data/processed/tier3")


def load_denormalized(npz_dir, workload):
    data = np.load(npz_dir / f"{workload}_traces.npz", allow_pickle=True)
    traces = data["traces"]
    metric_names = [str(m) for m in data["metric_names"]]

    with open(npz_dir / f"{workload}_normalization.json") as f:
        norm = json.load(f)
    params = norm["params"]

    denorm = np.zeros_like(traces)
    for i, name in enumerate(metric_names):
        p = params[name]
        denorm[:, :, i] = traces[:, :, i] * (p["max"] - p["min"]) + p["min"]

    return denorm, metric_names, data["metadata"]


def print_three_tier_table():
    print("\n## 3-column raw per-metric mean audit -- BERT (A16 / Tier 2 / Tier 3)\n")

    a16_traces, a16_metrics, _ = load_denormalized(A16_DIR, "bert")
    tier2_traces, tier2_metrics, _ = load_denormalized(TIER2_DIR, "bert")
    tier3_traces, tier3_metrics, _ = load_denormalized(TIER3_DIR, "bert")

    assert a16_metrics == tier2_metrics == tier3_metrics, (
        f"metric_names order differs across tiers: "
        f"A16={a16_metrics} Tier2={tier2_metrics} Tier3={tier3_metrics}"
    )

    print(f"shapes: A16={a16_traces.shape}  Tier2={tier2_traces.shape}  "
          f"Tier3={tier3_traces.shape}\n")

    header = f"{'metric':20s} | {'A16':>14s} | {'Tier 2':>14s} | {'Tier 3':>14s}"
    print(header)
    print("-" * len(header))
    for i, metric in enumerate(a16_metrics):
        a16_mean = float(np.nanmean(a16_traces[:, :, i]))
        tier2_mean = float(np.nanmean(tier2_traces[:, :, i]))
        tier3_mean = float(np.nanmean(tier3_traces[:, :, i]))
        print(f"{metric:20s} | {a16_mean:14.4f} | {tier2_mean:14.4f} | {tier3_mean:14.4f}")

    return a16_metrics


def print_cross_pod_spread_table():
    print("\n## Cross-pod spread audit -- gpu_utilization at r=7, all workloads\n")

    header = (f"{'workload':10s} | {'pod0':>7s} {'pod1':>7s} {'pod2':>7s} {'pod3':>7s} "
              f"{'pod4':>7s} {'pod5':>7s} {'pod6':>7s} | {'mean':>7s} {'spread':>7s} {'stdev':>7s}")
    print(header)
    print("-" * len(header))

    rows = []
    for workload in WORKLOADS:
        traces, metric_names, metadata = load_denormalized(TIER2_DIR, workload)
        gpu_idx = metric_names.index("gpu_utilization")

        r7_indices = [i for i, m in enumerate(metadata) if m["replica_count"] == 7]
        assert len(r7_indices) == 7, f"{workload}: expected 7 r=7 pods, found {len(r7_indices)}"
        # r=7 pods are the last 7 in pod-order (r=1..6 = 21 pods precede them)
        assert r7_indices == list(range(21, 28)), (
            f"{workload}: r=7 pods not at expected indices 21..27, got {r7_indices}"
        )

        per_pod_means = np.array([
            float(np.nanmean(traces[i, :, gpu_idx])) for i in r7_indices
        ])
        mean_of_means = float(per_pod_means.mean())
        spread = float(per_pod_means.max() - per_pod_means.min())
        stdev = float(per_pod_means.std())

        pods_str = " ".join(f"{v:7.4f}" for v in per_pod_means)
        print(f"{workload:10s} | {pods_str} | {mean_of_means:7.4f} {spread:7.4f} {stdev:7.4f}")

        rows.append({
            "workload": workload,
            "per_pod_means": per_pod_means,
            "mean_of_means": mean_of_means,
            "spread": spread,
            "stdev": stdev,
        })

    return rows


def main():
    print_three_tier_table()
    print_cross_pod_spread_table()


if __name__ == "__main__":
    main()
