#!/usr/bin/env python3
"""E6 follow-up (NOT committed) - stability check on the one cell that fully
recovered in the E6 fm search: Tier2 / resnet152 / fm=4.0 (checkpoint at
models/phase4/timegan_s36_tier2_e6_fm4p0/resnet152/generator.pt, already
trained by run_e6_fmsearch.sh -- no retrain here).

e6_fmsearch_report.py's compute_contention_wasserstein() only kept the
mean across the K=30 seeded draws. This script re-runs the identical K=30
draws (same seeds 0..29, same primitives: wr.gather_real/synth_by_r/
minmax_pooled_w, wn.gaussian_minmax_per_metric, fixed Gaussian seed=42) but
keeps every draw's per-metric Wasserstein value, so per-metric win count
(how many of the 30 draws beat the Gaussian) and the SD-vs-margin
comparison can be reported directly -- distinguishing a real recovery from
n=28 generation noise.

Run from repo root:
    PYTHONPATH=$(pwd)/scripts/utils python3 scripts/analysis/e6_resnet152_tier2_fm4_confirm.py
"""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "scripts" / "utils",
          REPO_ROOT / "scripts" / "phase4",
          REPO_ROOT / "scripts" / "phase4" / "evaluation",
          REPO_ROOT / "scripts" / "analysis"):
    sys.path.insert(0, str(p))

import e6_fmsearch_report as e6  # reuses GeneratorSeg, load_e6_generator, TIER_CONFIG

TIER = "tier2"
WORKLOAD = "resnet152"
FM = 4.0
K = 30
GAUSSIAN_SEED = 42
CONTENTION_METRICS = ["gpu_utilization", "pod_throughput", "pod_cpu_usage"]

OUT_DIR = REPO_ROOT / "outputs" / "analysis" / "e6_fmsearch"
OUT_JSON = OUT_DIR / "e6_resnet152_tier2_fm4_confirm.json"
OUT_MD = OUT_DIR / "e6_resnet152_tier2_fm4_confirm.md"


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    wr = importlib.import_module("wasserstein_reconciliation")
    wn = importlib.import_module("wasserstein_normalized")
    pp = importlib.import_module(e6.TIER_CONFIG[TIER]["pp"])
    cfg = e6.TIER_CONFIG[TIER]

    names = pp.TRAINED_NAMES
    n_metrics = len(names)

    ckpt = REPO_ROOT / f"models/phase4/timegan_s36_{TIER}_e6_{e6.fmtag(FM)}/{WORKLOAD}/generator.pt"
    print(f"Checkpoint: {ckpt}  exists={ckpt.exists()}")
    generator = e6.load_e6_generator(TIER, WORKLOAD, FM, n_metrics, device="cpu")
    assert generator is not None, f"missing checkpoint {ckpt}"

    memory_stats = pp.compute_memory_stats(cfg["data_dir"])
    dropped_stats = pp.compute_dropped_metric_stats(cfg["data_dir"])
    real_by_r, norm_params = wr.gather_real(pp, cfg, WORKLOAD)

    gaussian_ws = wn.gaussian_minmax_per_metric(real_by_r, names, norm_params, seed=GAUSSIAN_SEED)

    per_metric_draws = {m: [] for m in names}
    for k in range(K):
        syn = wr.synth_by_r(pp, generator, WORKLOAD, cfg, real_by_r, memory_stats, dropped_stats, seed=k)
        _, per_metric = wr.minmax_pooled_w(real_by_r, syn, names, norm_params)
        for m in names:
            per_metric_draws[m].append(per_metric[m])
        print(f"  draw {k:2d}: " + "  ".join(f"{m}={per_metric[m]:.5f}" for m in CONTENTION_METRICS), flush=True)

    results = {}
    for m in names:
        draws = np.array(per_metric_draws[m])
        mean = float(draws.mean())
        sd = float(draws.std(ddof=1))
        g = gaussian_ws[m]
        wins = int(np.sum(draws < g))
        margin = g - mean
        # "noise swamps the margin" heuristic: SD >= |margin|
        sd_swamps_margin = sd >= abs(margin)
        results[m] = {
            "gaussian": g, "s36_mean": mean, "s36_sd": sd,
            "wins_out_of_30": wins, "margin_gaussian_minus_s36mean": margin,
            "sd_swamps_margin": sd_swamps_margin,
            "draws": draws.tolist(),
        }

    print("\n" + "=" * 70)
    print(f"E6 follow-up: {TIER}/{WORKLOAD}/fm={FM} -- K={K} draws")
    print("=" * 70)
    verdicts = {}
    for m in CONTENTION_METRICS:
        r = results[m]
        stable = (r["wins_out_of_30"] >= 25) and not r["sd_swamps_margin"]
        marginal = 10 <= r["wins_out_of_30"] <= 20
        if stable:
            verdict = "REAL RECOVERY (stable)"
        elif marginal or r["sd_swamps_margin"]:
            verdict = "N=28 NOISE (marginal / SD swamps margin)"
        else:
            verdict = "MIXED (wins skew but not decisively stable)"
        verdicts[m] = verdict
        print(f"{m:16s} gaussian={r['gaussian']:.5f}  s36_mean={r['s36_mean']:.5f}  s36_sd={r['s36_sd']:.5f}  "
              f"wins={r['wins_out_of_30']}/30  margin={r['margin_gaussian_minus_s36mean']:+.5f}  "
              f"sd_swamps_margin={r['sd_swamps_margin']}  -> {verdict}")

    out = {
        "cell": {"tier": TIER, "workload": WORKLOAD, "fm": FM, "checkpoint": str(ckpt)},
        "config": {"K": K, "gaussian_seed": GAUSSIAN_SEED},
        "per_metric": results,
        "verdicts": verdicts,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))

    lines = ["# E6 follow-up - Tier2/resnet152/fm=4.0 stability check (NOT committed)\n"]
    lines.append(f"Checkpoint: `{ckpt}` (already trained, no retrain). K={K} seeded draws (0..29), "
                  f"Gaussian fixed seed={GAUSSIAN_SEED}, minmax (PRIMARY) space.\n")
    lines.append("| metric | Gaussian | S36 mean | S36 SD | wins/30 | margin (G - S36) | SD swamps margin? | verdict |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for m in CONTENTION_METRICS:
        r = results[m]
        lines.append(f"| {m} | {r['gaussian']:.5f} | {r['s36_mean']:.5f} | {r['s36_sd']:.5f} | "
                      f"{r['wins_out_of_30']}/30 | {r['margin_gaussian_minus_s36mean']:+.5f} | "
                      f"{r['sd_swamps_margin']} | {verdicts[m]} |")
    OUT_MD.write_text("\n".join(lines) + "\n")

    print(f"\nWrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
