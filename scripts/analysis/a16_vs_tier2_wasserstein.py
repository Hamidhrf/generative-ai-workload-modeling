#!/usr/bin/env python3
"""
A16 vs Tier 2 real-data distributional distance (S36 Tier 2 retrain
plan, Step 7 appendix). Read-only analysis, no model involved.

For each (workload, metric), flattens all real pod-timesteps from
A16 (phase1_v3) and Tier 2 into two 1D distributions and computes
their Wasserstein-1 distance.

Traces are stored min-max normalized to [0, 1], independently per
tier (each tier's normalization.json has its own min/max fit on its
own data). Computing Wasserstein directly on those two independently-
normalized distributions would not be physically meaningful -- a
value of 0.5 in A16's normalized space is a different real quantity
than 0.5 in Tier 2's normalized space. Traces are denormalized to raw
physical units (using each tier's own normalization.json) before the
distance is computed, so the result reflects genuine hardware/dataset
distributional difference, not an artifact of independently-fit
normalization ranges.

metric_names is read from each .npz file, not hardcoded, and the two
tiers' orderings are asserted equal before pairing by index.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

WORKLOADS = ["bert", "gpt2", "resnet152", "whisper", "yolo"]
A16_DIR = Path("data/processed/phase1_v3")
TIER2_DIR = Path("data/processed/tier2")
OUT_DIR = Path("outputs/analysis/cross_tier")


def load_denormalized(npz_dir, workload):
    data = np.load(npz_dir / f"{workload}_traces.npz", allow_pickle=True)
    traces = data["traces"]  # normalized
    metric_names = [str(m) for m in data["metric_names"]]

    with open(npz_dir / f"{workload}_normalization.json") as f:
        norm = json.load(f)
    params = norm["params"]

    denorm = np.zeros_like(traces)
    for i, name in enumerate(metric_names):
        p = params[name]
        denorm[:, :, i] = traces[:, :, i] * (p["max"] - p["min"]) + p["min"]

    return denorm, metric_names


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    metric_names_ref = None

    for wl in WORKLOADS:
        a16_traces, a16_metrics = load_denormalized(A16_DIR, wl)
        tier2_traces, tier2_metrics = load_denormalized(TIER2_DIR, wl)
        assert a16_metrics == tier2_metrics, (
            f"{wl}: metric_names order differs between tiers: "
            f"{a16_metrics} vs {tier2_metrics}"
        )
        if metric_names_ref is None:
            metric_names_ref = a16_metrics

        print(f"{wl}: A16 shape={a16_traces.shape}  Tier2 shape={tier2_traces.shape}")

        results[wl] = {}
        for i, metric in enumerate(a16_metrics):
            a16_flat = a16_traces[:, :, i].flatten()
            tier2_flat = tier2_traces[:, :, i].flatten()
            wd = wasserstein_distance(a16_flat, tier2_flat)
            results[wl][metric] = float(wd)
            print(f"  {metric:20s} n_a16={a16_flat.size} n_tier2={tier2_flat.size} "
                  f"wasserstein={wd:.6g}")

    # --- JSON output ---
    json_path = OUT_DIR / "a16_vs_tier2_wasserstein.json"
    with open(json_path, "w") as f:
        json.dump({
            "description": "A16 (phase1_v3) vs Tier 2 real-data Wasserstein-1 "
                            "distance per (workload, metric), computed on "
                            "denormalized (raw-unit) traces.",
            "workloads": WORKLOADS,
            "metric_names": metric_names_ref,
            "distances": results,
        }, f, indent=2)

    # --- Markdown table output ---
    md_path = OUT_DIR / "a16_vs_tier2_wasserstein_table.md"
    lines = []
    lines.append("# A16 vs Tier 2 Real-Data Wasserstein Distance")
    lines.append("")
    lines.append("Real-vs-real distance (no model involved), computed on "
                  "denormalized traces.")
    lines.append("")
    header = "| Workload | " + " | ".join(metric_names_ref) + " |"
    sep = "|---|" + "|".join(["---"] * len(metric_names_ref)) + "|"
    lines.append(header)
    lines.append(sep)
    for wl in WORKLOADS:
        row = [wl] + [f"{results[wl][m]:.4g}" for m in metric_names_ref]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
