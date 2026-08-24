#!/usr/bin/env python3
"""E1 report (NOT committed) - matched-size control retrain of Tier 3.

Tests whether the MIG break (Tier 2, 2/5 frozen-recipe pass) is caused by
sharing mode or by dataset size, by retraining the frozen S36 recipe on
Tier 3 data restricted to r=1..7 (28 traces/workload, matching Tier 2's
train/val budget exactly) and comparing three cells per workload:
  - tier3          : committed full Tier 3 (r=1..10, n=55/workload)
  - tier3_matched  : this E1 retrain (r=1..7, n=28/workload, same sharing
                      mode + gpu_utilization imputation as full Tier 3)
  - tier2          : committed Tier 2 MIG (r=1..7, n=28/workload)

Reuses (imports, does not reimplement) the established analysis
primitives:
  - eval_s36.compute_vr_per_metric        (tier-agnostic VR estimator)
  - naive_baselines.composite / resampler_baseline / gaussian_baseline / load_real
  - wasserstein_reconciliation.gather_real / synth_by_r / minmax_pooled_w
  - wasserstein_normalized.gaussian_minmax_per_metric

Three questions, in report order:
  Q1. composite_6 VR per workload: tier3_matched vs tier3 vs tier2. Does
      tier3_matched stay near tier3 (clean) or degrade toward tier2?
  Q2. DECISION-CRITICAL: per-metric normalized (minmax) Wasserstein on the
      3 contention metrics (gpu_utilization, pod_throughput, pod_cpu_usage)
      -- does tier3_matched still beat the i.i.d. per-metric Gaussian
      baseline there (as full tier3 does), or does it collapse to
      Gaussian-level (as tier2 does)?
  Q3. tier3_matched's real-trace resampler floor (n=28, K=30 bootstrap
      draws, composite_6) vs tier2's floor (n=28) vs full tier3's floor
      (n=55) -- is the floor trace-count-driven (R4)?

Requires (produced by this E1 run, not committed):
  data/processed/tier3_matched/unified/combined_dataset.npz
  data/processed/tier3_matched/unified/combined_normalization.json
  models/phase4/timegan_s36_tier3_matched/s36_<wl>_vr*_fm*/generator.pt
  outputs/phase4/timegan_s36_tier3_matched/s36_eval_results.json  (run
    eval_s36_tier3_matched.py first)

Run from repo root:
    PYTHONPATH=$(pwd)/scripts/utils python3 scripts/analysis/e1_matched_tier3_report.py
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

WORKLOADS = ["bert", "gpt2", "resnet152", "whisper", "yolo"]
EXCLUDED_METRIC = "pod_memory_bytes"
CONTENTION_METRICS = ["gpu_utilization", "pod_throughput", "pod_cpu_usage"]
K = 30
GAUSSIAN_SEED = 42

TIER_CONFIG = {
    "tier3": {"pp": "postprocess_s36_tier3", "data_dir": "data/processed/tier3/unified",
              "replicas": list(range(1, 11))},
    "tier3_matched": {"pp": "postprocess_s36_tier3_matched", "data_dir": "data/processed/tier3_matched/unified",
                       "replicas": list(range(1, 8))},
    "tier2": {"pp": "postprocess_s36_tier2", "data_dir": "data/processed/tier2/unified",
              "replicas": list(range(1, 8))},
}
TIER_ORDER = ["tier3", "tier3_matched", "tier2"]

STORED_EVAL_RESULTS = {
    "tier3": REPO_ROOT / "outputs/phase4/timegan_s36_tier3/s36_eval_results.json",
    "tier3_matched": REPO_ROOT / "outputs/phase4/timegan_s36_tier3_matched/s36_eval_results.json",
    "tier2": REPO_ROOT / "outputs/phase4/timegan_s36_tier2/s36_eval_results.json",
}

OUT_DIR = REPO_ROOT / "outputs" / "analysis" / "e1_matched_tier3"
OUT_JSON = OUT_DIR / "e1_matched_tier3_report.json"
OUT_MD = OUT_DIR / "e1_matched_tier3_report.md"


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    nb = importlib.import_module("naive_baselines")
    wr = importlib.import_module("wasserstein_reconciliation")
    wn = importlib.import_module("wasserstein_normalized")
    ev = importlib.import_module("eval_s36")
    compute_vr = ev.compute_vr_per_metric

    pp_mods = {t: importlib.import_module(cfg["pp"]) for t, cfg in TIER_CONFIG.items()}

    # ============================================================
    # Q1: composite_6 VR headline per workload, all 3 tiers
    # ============================================================
    print("=" * 70)
    print("Q1: composite_6 VR headline (stored eval results)")
    print("=" * 70)
    q1 = {}
    for tier in TIER_ORDER:
        q1[tier] = {}
        path = STORED_EVAL_RESULTS[tier]
        if not path.exists():
            print(f"  {tier}: MISSING {path} -- run eval first")
            continue
        d = json.loads(path.read_text())
        for wl in WORKLOADS:
            w = d["workloads"][wl]
            names = w["metric_names"]
            vr7 = w["vr_per_metric_smooth"]
            c6 = nb.composite(vr7, names, exclude=EXCLUDED_METRIC)
            c7 = nb.composite(vr7, names)
            q1[tier][wl] = {"c6": c6, "c7": c7}
            print(f"  {tier:14s} {wl:10s} c6={c6:.4f} c7={c7:.4f}")

    # ============================================================
    # Q3: real-trace resampler noise floor, K=30, composite_6
    # ============================================================
    print("\n" + "=" * 70)
    print("Q3: real-trace resampler floor (composite_6, K=30)")
    print("=" * 70)
    q3 = {}
    seed_base = 0
    for tier in TIER_ORDER:
        pp = pp_mods[tier]
        cfg = TIER_CONFIG[tier]
        q3[tier] = {}
        for wl in WORKLOADS:
            real, names = nb.load_real(pp, wl, cfg["data_dir"], cfg["replicas"])
            resamp = nb.resampler_baseline(compute_vr, real, names, seed_base)
            q3[tier][wl] = {"n_traces": int(real.shape[0]), "mean_c6": resamp["mean_c6"], "sd_c6": resamp["sd_c6"]}
            print(f"  {tier:14s} {wl:10s} n={real.shape[0]:3d} resampler_c6={resamp['mean_c6']:.4f}+-{resamp['sd_c6']:.4f}")
            seed_base += 1000

    q3_tier_agg = {}
    for tier in TIER_ORDER:
        means = [q3[tier][wl]["mean_c6"] for wl in WORKLOADS]
        sds = [q3[tier][wl]["sd_c6"] for wl in WORKLOADS]
        q3_tier_agg[tier] = {
            "mean_of_means_c6": float(np.mean(means)),
            "mean_of_sds_c6": float(np.mean(sds)),
        }
        print(f"  -> {tier}: mean_of_means={q3_tier_agg[tier]['mean_of_means_c6']:.4f} "
              f"mean_of_sds={q3_tier_agg[tier]['mean_of_sds_c6']:.4f}")

    # ============================================================
    # Q2: DECISION-CRITICAL per-metric normalized (minmax) Wasserstein,
    # S36 vs i.i.d. Gaussian, focused on the 3 contention metrics.
    # ============================================================
    print("\n" + "=" * 70)
    print("Q2: per-metric minmax Wasserstein, S36 (K=30) vs Gaussian (seed=42)")
    print("=" * 70)
    q2 = {}
    for tier in TIER_ORDER:
        pp = pp_mods[tier]
        cfg = TIER_CONFIG[tier]
        print(f"--- {tier} ---", flush=True)
        memory_stats = pp.compute_memory_stats(cfg["data_dir"])
        dropped_stats = pp.compute_dropped_metric_stats(cfg["data_dir"])
        q2[tier] = {}
        for wl in WORKLOADS:
            generator = pp.load_generator(wl, device="cpu")
            real_by_r, norm_params = wr.gather_real(pp, cfg, wl)
            names = pp.TRAINED_NAMES

            gaussian_ws = wn.gaussian_minmax_per_metric(real_by_r, names, norm_params, seed=GAUSSIAN_SEED)

            per_metric_draws = {m: [] for m in names}
            for k in range(K):
                syn = wr.synth_by_r(pp, generator, wl, cfg, real_by_r, memory_stats, dropped_stats, seed=k)
                _, per_metric = wr.minmax_pooled_w(real_by_r, syn, names, norm_params)
                for m in names:
                    per_metric_draws[m].append(per_metric[m])

            s36_mean = {m: float(np.mean(v)) for m, v in per_metric_draws.items()}
            s36_sd = {m: float(np.std(v, ddof=1)) for m, v in per_metric_draws.items()}

            q2[tier][wl] = {
                "gaussian_per_metric": gaussian_ws,
                "s36_per_metric_mean": s36_mean,
                "s36_per_metric_sd": s36_sd,
                "s36_beats_gaussian": {m: bool(s36_mean[m] < gaussian_ws[m]) for m in names},
            }
            beats_contention = [q2[tier][wl]["s36_beats_gaussian"][m] for m in CONTENTION_METRICS if m in names]
            print(f"  {wl:10s} contention beats-Gaussian: "
                  f"{dict(zip(CONTENTION_METRICS, beats_contention))}", flush=True)

    # Tier-aggregated (mean over 5 workloads) per-metric comparison, contention metrics only
    q2_tier_agg = {}
    for tier in TIER_ORDER:
        q2_tier_agg[tier] = {}
        for m in CONTENTION_METRICS:
            present = [wl for wl in WORKLOADS if m in q2[tier][wl]["gaussian_per_metric"]]
            if not present:
                continue
            g = float(np.mean([q2[tier][wl]["gaussian_per_metric"][m] for wl in present]))
            s = float(np.mean([q2[tier][wl]["s36_per_metric_mean"][m] for wl in present]))
            q2_tier_agg[tier][m] = {"gaussian": g, "s36_mean": s, "s36_beats_gaussian": s < g}

    print("\nTier-aggregated contention-metric verdicts (mean over 5 workloads):")
    for tier in TIER_ORDER:
        for m in CONTENTION_METRICS:
            c = q2_tier_agg[tier].get(m)
            if c:
                print(f"  {tier:14s} {m:16s} gaussian={c['gaussian']:.5f} s36={c['s36_mean']:.5f} "
                      f"beats_gaussian={c['s36_beats_gaussian']}")

    # ============================================================
    # Write outputs
    # ============================================================
    out = {
        "config": {"K": K, "gaussian_seed": GAUSSIAN_SEED, "excluded_metric_c6": EXCLUDED_METRIC,
                   "contention_metrics": CONTENTION_METRICS, "tier_replica_ranges": {t: c["replicas"] for t, c in TIER_CONFIG.items()}},
        "q1_composite6_vr_headline": q1,
        "q3_resampler_floor_per_workload": q3,
        "q3_resampler_floor_tier_aggregate": q3_tier_agg,
        "q2_per_metric_wasserstein": q2,
        "q2_contention_tier_aggregate": q2_tier_agg,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))

    lines = ["# E1 report - matched-size control retrain of Tier 3 (NOT committed)\n"]
    lines.append("tier3 = committed full Tier3 (r=1..10, n=55/wl). tier3_matched = this E1 retrain "
                  "(r=1..7, n=28/wl, same sharing mode + gpu_utilization imputation as tier3). "
                  "tier2 = committed Tier2 MIG (r=1..7, n=28/wl).\n")

    lines.append("## Q1 - composite_6 VR headline\n")
    lines.append("| workload | tier3 (full) | tier3_matched | tier2 |")
    lines.append("|---|---|---|---|")
    for wl in WORKLOADS:
        row = [f"{q1[t].get(wl, {}).get('c6', float('nan')):.4f}" if wl in q1.get(t, {}) else "MISSING" for t in TIER_ORDER]
        lines.append(f"| {wl} | {row[0]} | {row[1]} | {row[2]} |")
    means = {t: (float(np.mean([q1[t][wl]["c6"] for wl in WORKLOADS])) if all(wl in q1.get(t, {}) for wl in WORKLOADS) else None) for t in TIER_ORDER}

    def fmt_mean(v):
        return f"{v:.4f}" if v is not None else "MISSING"

    lines.append(f"| **MEAN** | {fmt_mean(means['tier3'])} | {fmt_mean(means['tier3_matched'])} | {fmt_mean(means['tier2'])} |")

    lines.append("\n## Q2 - DECISION-CRITICAL: per-metric minmax Wasserstein, S36 vs Gaussian, contention metrics\n")
    lines.append("| tier | metric | gaussian | S36 mean (K=30) | S36 beats Gaussian? |")
    lines.append("|---|---|---|---|---|")
    for tier in TIER_ORDER:
        for m in CONTENTION_METRICS:
            c = q2_tier_agg[tier].get(m)
            if c:
                lines.append(f"| {tier} | {m} | {c['gaussian']:.5f} | {c['s36_mean']:.5f} | {c['s36_beats_gaussian']} |")

    lines.append("\n## Q3 - real-trace resampler floor (composite_6, K=30)\n")
    lines.append("| tier | n_traces/wl | mean of per-workload resampler means | mean of per-workload resampler SDs |")
    lines.append("|---|---|---|---|")
    for tier in TIER_ORDER:
        n = q3[tier][WORKLOADS[0]]["n_traces"]
        a = q3_tier_agg[tier]
        lines.append(f"| {tier} | {n} | {a['mean_of_means_c6']:.4f} | {a['mean_of_sds_c6']:.4f} |")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
