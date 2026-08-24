#!/usr/bin/env python3
"""E10 - naive-baseline VR under the exact S36 estimator.

Context for every VR number: what does this variance-ratio estimator
score for two trivial, non-generative "synthetic" constructions, so any
S36 VR can be read against how far even a "trivial-perfect" method
departs from 1.0 under the same estimator?

Same reduction as S36 (imported, not reimplemented):
    compute_vr_per_metric(real, syn) = mean_metric(syn.var(axis=(0,1))
                                        / (real.var(axis=(0,1)) + 1e-8))
imported directly from scripts/phase4/evaluation/eval_s36.py - identical
across all three tiers (confirmed byte-for-byte in the E2/D1 step-0
pass), so importing once from the a16 module is sufficient; the
reduction is tier-agnostic.

Composite_6 (exclude pod_memory_bytes BY NAME) is PRIMARY, matching
E2/E11. composite_7 is secondary.

Real traces: each tier's own combined_dataset.npz / combined_
normalization.json, matched r-range per tier (a16/tier3 r=1..10,
tier2 r=1..7 - this covers the full dataset for every tier, so the
r-range filter is a no-op in practice but is applied explicitly for
literal fidelity to spec and portability if the dataset ever gains
sparser r coverage). Correctly denormalized to raw physical units via
a straightforward per-metric affine transform (unlike eval_s36.py's own
denormalize(), which is a documented no-op due to a schema mismatch -
see D1 step-0 notes). This is deliberate and does NOT break comparability
with the S36 headline numbers: VR is a ratio of variances, and applying
any common per-metric affine transform to both the real and synthetic
side of the ratio leaves the ratio unchanged (var scales by the same
factor on both sides, so it cancels). Both baselines here apply real
denormalization to both real and "synthetic" identically, so their VR
values are numerically what they would be in eval_s36.py's own
(accidentally normalized-space) convention too - the choice of space is
immaterial to the reported ratio.

Two comparators, per (tier, workload):

  1. REAL-TRACE RESAMPLER (the noise floor). "Synthetic" = bootstrap
     resample (with replacement, same N) of the real traces themselves.
     Its VR is what "perfect" scores under this estimator - purely
     finite-N sampling noise around 1.0. K=30 distinct seeds, mean+SD.

  2. PER-METRIC GAUSSIAN. "Synthetic" = per metric, i.i.d. N(mean, std)
     draws matched to that metric's real per-workload pooled mean/std
     (same shape as real). Bounds what a method with correct marginals
     but no temporal/cross-pod structure achieves. K=30 seeds, mean+SD
     (cheap, no model loading - done even though optional).

Seeding: torch.manual_seed(42) + np.random.seed(42) set once at the top
of main(), documented as deliberate - these baselines have their own
sampling randomness (bootstrap resampling, Gaussian draws) independent
of any generator. Each of the K=30 draws per baseline additionally uses
its own deterministic np.random.default_rng(seed) for full
reproducibility regardless of call order.

Run from repo root:
    PYTHONPATH=$(pwd)/scripts/utils python3 scripts/analysis/naive_baselines.py
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
          REPO_ROOT / "scripts" / "phase4" / "evaluation"):
    sys.path.insert(0, str(p))

WORKLOADS = ["bert", "gpt2", "resnet152", "whisper", "yolo"]
EXCLUDED_METRIC = "pod_memory_bytes"
K = 30

TIER_CONFIG = {
    "a16": {"data_dir": "data/processed/phase4/unified", "replicas": list(range(1, 11))},
    "tier3": {"data_dir": "data/processed/tier3/unified", "replicas": list(range(1, 11))},
    "tier2": {"data_dir": "data/processed/tier2/unified", "replicas": list(range(1, 8))},
}
TIER_ORDER = ["a16", "tier3", "tier2"]

STORED_EVAL_RESULTS = {
    "a16": REPO_ROOT / "outputs/phase4/timegan_s36/s36_eval_results.json",
    "tier3": REPO_ROOT / "outputs/phase4/timegan_s36_tier3/s36_eval_results.json",
    "tier2": REPO_ROOT / "outputs/phase4/timegan_s36_tier2/s36_eval_results.json",
}

OUT_DIR = REPO_ROOT / "outputs" / "analysis" / "baselines"
OUT_JSON = OUT_DIR / "naive_baseline_vr.json"
OUT_MD = OUT_DIR / "naive_baseline_vr.md"


def composite(vr_per_metric, names, exclude=None):
    vals = vr_per_metric if exclude is None else [v for v, n in zip(vr_per_metric, names) if n != exclude]
    return float(np.mean(vals))


def stored_headline(tier, workload):
    d = json.loads(STORED_EVAL_RESULTS[tier].read_text())
    w = d["workloads"][workload]
    names = w["metric_names"]
    vr7 = w["vr_per_metric_smooth"]
    return composite(vr7, names, exclude=EXCLUDED_METRIC), composite(vr7, names)


def denorm_batch(traces_norm_7, norm_params, names):
    out = np.zeros_like(traces_norm_7)
    for i, name in enumerate(names):
        p = norm_params[name]
        out[:, :, i] = traces_norm_7[:, :, i] * (p["max"] - p["min"]) + p["min"]
    return out


def load_real(pp_module, workload, data_dir, replicas):
    data = pp_module.load_real_data(data_dir)
    norm_params = pp_module.load_normalization_params(workload, data_dir)
    wid = pp_module.WORKLOADS.index(workload)
    wl = data["workload_ids"]
    rc = data["replica_counts"]
    mask = (wl == wid) & np.isin(rc, replicas)
    traces_7_norm = data["traces"][mask][:, :, pp_module.TRAINED_INDICES]
    real = denorm_batch(traces_7_norm.astype(np.float64), norm_params, pp_module.TRAINED_NAMES)
    return real, pp_module.TRAINED_NAMES


def resampler_baseline(compute_vr, real, names, base_seed):
    n = real.shape[0]
    c6_draws, c7_draws = [], []
    for k in range(K):
        rng = np.random.default_rng(base_seed + k)
        idx = rng.integers(0, n, size=n)
        syn = real[idx]
        vr = compute_vr(real, syn)
        c6_draws.append(composite(vr, names, exclude=EXCLUDED_METRIC))
        c7_draws.append(composite(vr, names))
    c6_draws = np.array(c6_draws)
    c7_draws = np.array(c7_draws)
    return {
        "mean_c6": float(c6_draws.mean()), "sd_c6": float(c6_draws.std(ddof=1)),
        "mean_c7": float(c7_draws.mean()), "sd_c7": float(c7_draws.std(ddof=1)),
        "draws_c6": c6_draws.tolist(),
    }


def gaussian_baseline(compute_vr, real, names, base_seed):
    n, T, m = real.shape
    means = real.mean(axis=(0, 1))
    stds = real.std(axis=(0, 1))
    c6_draws, c7_draws = [], []
    for k in range(K):
        rng = np.random.default_rng(base_seed + 10_000 + k)
        syn = np.empty_like(real)
        for i in range(m):
            syn[:, :, i] = rng.normal(means[i], stds[i], size=(n, T))
        vr = compute_vr(real, syn)
        c6_draws.append(composite(vr, names, exclude=EXCLUDED_METRIC))
        c7_draws.append(composite(vr, names))
    c6_draws = np.array(c6_draws)
    c7_draws = np.array(c7_draws)
    return {
        "mean_c6": float(c6_draws.mean()), "sd_c6": float(c6_draws.std(ddof=1)),
        "mean_c7": float(c7_draws.mean()), "sd_c7": float(c7_draws.std(ddof=1)),
        "single_draw_c6": c6_draws[0], "single_draw_c7": c7_draws[0],
        "draws_c6": c6_draws.tolist(),
    }


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ev = importlib.import_module("eval_s36")  # tier-agnostic compute_vr_per_metric
    compute_vr = ev.compute_vr_per_metric

    cells = []
    seed_base = 0
    for tier in TIER_ORDER:
        pp = importlib.import_module({
            "a16": "postprocess_s36", "tier2": "postprocess_s36_tier2", "tier3": "postprocess_s36_tier3",
        }[tier])
        cfg = TIER_CONFIG[tier]
        for workload in WORKLOADS:
            real, names = load_real(pp, workload, cfg["data_dir"], cfg["replicas"])
            s36_c6, s36_c7 = stored_headline(tier, workload)

            resamp = resampler_baseline(compute_vr, real, names, seed_base)
            gauss = gaussian_baseline(compute_vr, real, names, seed_base)

            rec = {
                "tier": tier, "workload": workload, "n_traces": int(real.shape[0]),
                "s36_headline_c6": s36_c6, "s36_headline_c7": s36_c7,
                "resampler": resamp, "gaussian": gauss,
            }
            cells.append(rec)
            print(f"{tier:6s} {workload:10s} n={real.shape[0]:3d}  "
                  f"S36_c6={s36_c6:.4f}  resampler_c6={resamp['mean_c6']:.4f}+-{resamp['sd_c6']:.4f}  "
                  f"gaussian_c6={gauss['mean_c6']:.4f}+-{gauss['sd_c6']:.4f}", flush=True)
            seed_base += 1000

    cell_by_key = {(c["tier"], c["workload"]): c for c in cells}

    # ---- Per-tier resampler noise floor ----
    tier_noise_floor = {}
    for tier in TIER_ORDER:
        means = [cell_by_key[(tier, w)]["resampler"]["mean_c6"] for w in WORKLOADS]
        sds = [cell_by_key[(tier, w)]["resampler"]["sd_c6"] for w in WORKLOADS]
        tier_noise_floor[tier] = {
            "mean_of_resampler_means_c6": float(np.mean(means)),
            "mean_of_resampler_sds_c6": float(np.mean(sds)),
            "max_abs_departure_from_1_c6": float(np.max(np.abs(np.array(means) - 1.0))),
        }

    # ---- Batch A Task 2: Gaussian pass counts per tier, both S36 pass rules ----
    def pass_one_sided(vr):
        return vr > 0.8

    def pass_two_sided(vr):
        return 0.8 <= vr <= 1.25

    gaussian_pass_counts = {}
    s36_pass_counts_for_reference = {}
    for tier in TIER_ORDER:
        g_one = sum(pass_one_sided(cell_by_key[(tier, w)]["gaussian"]["mean_c6"]) for w in WORKLOADS)
        g_two = sum(pass_two_sided(cell_by_key[(tier, w)]["gaussian"]["mean_c6"]) for w in WORKLOADS)
        s_one = sum(pass_one_sided(cell_by_key[(tier, w)]["s36_headline_c6"]) for w in WORKLOADS)
        s_two = sum(pass_two_sided(cell_by_key[(tier, w)]["s36_headline_c6"]) for w in WORKLOADS)
        gaussian_pass_counts[tier] = {"one_sided": g_one, "two_sided": g_two}
        s36_pass_counts_for_reference[tier] = {"one_sided": s_one, "two_sided": s_two}

    tier3_gaussian_passes_5_5 = gaussian_pass_counts["tier3"]["one_sided"] == 5

    # Cells where Gaussian passes (one-sided) AND S36 also passes (one-sided) on Tier3 -
    # these are exactly the cells where VR cannot distinguish S36 from i.i.d. noise.
    tier3_indistinguishable_cells = [
        w for w in WORKLOADS
        if pass_one_sided(cell_by_key[("tier3", w)]["gaussian"]["mean_c6"])
        and pass_one_sided(cell_by_key[("tier3", w)]["s36_headline_c6"])
    ]

    if tier3_gaussian_passes_5_5:
        gaussian_verdict = (
            "Gaussian passes Tier3 5/5 (one-sided). It passes every cell S36 passes on "
            "Tier3 -> VR CANNOT DISTINGUISH S36 FROM I.I.D. NOISE ON TIER3 under the "
            "one-sided rule. VR passing on Tier3 is not evidence of learned temporal/"
            "cross-pod structure there; it is consistent with the estimator's own noise "
            "floor / correct-marginals-only baseline."
        )
    else:
        gaussian_verdict = (
            f"Gaussian does NOT pass Tier3 5/5 (one-sided: "
            f"{gaussian_pass_counts['tier3']['one_sided']}/5). VR retains at least some "
            f"discriminating power on Tier3 beyond matching marginals alone. Cells where "
            f"both Gaussian and S36 pass (one-sided) and are therefore indistinguishable "
            f"under this test: {tier3_indistinguishable_cells if tier3_indistinguishable_cells else 'none'}."
        )

    # ---- DECISION-CRITICAL: resnet152 Tier2 ----
    rn2 = cell_by_key[("tier2", "resnet152")]
    gauss_near_1 = abs(rn2["gaussian"]["mean_c6"] - 1.0) <= 2 * rn2["gaussian"]["sd_c6"] or \
        abs(rn2["gaussian"]["mean_c6"] - 1.0) < 0.15
    if gauss_near_1:
        rn2_finding = (
            "Gaussian baseline scores NEAR 1.0 by construction on resnet152 Tier2 "
            "-> S36's 0.369/0.410 is a HARDER failure than 'MIG is different': a method "
            "with only correct marginals (no temporal/cross-pod structure) already "
            "matches the target under this estimator, so S36 undershooting variance this "
            "much needs the tougher explanation (real model/mode-collapse deficiency), "
            "not just an estimator/data quirk of this cell."
        )
    else:
        rn2_finding = (
            "Gaussian baseline ALSO scores low on resnet152 Tier2 -> the low VR is at "
            "least PARTLY a property of the estimator/data on this specific cell (small "
            "n, or a real-data variance structure that even a matched-marginal i.i.d. "
            "Gaussian cannot reproduce under this estimator), not attributable to S36 "
            "alone."
        )

    rn2_readout = {
        "s36_headline_c6": rn2["s36_headline_c6"], "s36_headline_c7": rn2["s36_headline_c7"],
        "n_traces": rn2["n_traces"],
        "resampler": {"mean_c6": rn2["resampler"]["mean_c6"], "sd_c6": rn2["resampler"]["sd_c6"]},
        "gaussian": {"mean_c6": rn2["gaussian"]["mean_c6"], "sd_c6": rn2["gaussian"]["sd_c6"]},
        "finding": rn2_finding,
    }

    # ---- Write JSON ----
    out = {
        "config": {"K": K, "excluded_metric": EXCLUDED_METRIC,
                   "seed": {"torch_manual_seed": 42, "numpy_random_seed": 42,
                             "note": "deliberate - these baselines have their own sampling "
                                     "randomness; each K-draw additionally uses its own "
                                     "np.random.default_rng(seed) for reproducibility"}},
        "cells": cells,
        "tier_resampler_noise_floor_c6": tier_noise_floor,
        "resnet152_tier2_readout": rn2_readout,
        "gaussian_pass_counts_c6": gaussian_pass_counts,
        "s36_pass_counts_c6_for_reference": s36_pass_counts_for_reference,
        "tier3_gaussian_passes_5_5": tier3_gaussian_passes_5_5,
        "tier3_indistinguishable_from_noise_cells": tier3_indistinguishable_cells,
        "gaussian_verdict": gaussian_verdict,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))

    # ---- Write MD ----
    lines = []
    lines.append("# E10 - naive-baseline VR under the exact S36 estimator\n")
    lines.append(f"K={K} seeded draws per baseline. Composite_6 (exclude `{EXCLUDED_METRIC}` by name) is "
                  "PRIMARY, composite_7 secondary. `compute_vr_per_metric` imported unmodified from "
                  "`eval_s36.py` (tier-agnostic, confirmed byte-identical across tiers).\n")

    lines.append("## Batch A Task 2: Gaussian pass counts per tier\n")
    lines.append("| tier | Gaussian pass one-sided (>0.8) | Gaussian pass two-sided [0.8,1.25] | "
                  "S36 pass one-sided (ref) | S36 pass two-sided (ref) |")
    lines.append("|---|---|---|---|---|")
    for tier in TIER_ORDER:
        g = gaussian_pass_counts[tier]
        s = s36_pass_counts_for_reference[tier]
        lines.append(f"| {tier} | {g['one_sided']}/5 | {g['two_sided']}/5 | "
                      f"{s['one_sided']}/5 | {s['two_sided']}/5 |")
    lines.append(f"\n**Verdict: {gaussian_verdict}**\n")

    lines.append("## DECISION-CRITICAL: resnet152 Tier2 under both baselines\n")
    lines.append(f"- S36 headline: composite_6={rn2['s36_headline_c6']:.4f}, "
                  f"composite_7={rn2['s36_headline_c7']:.4f} (n_traces={rn2['n_traces']})")
    lines.append(f"- Real-trace resampler (noise floor): mean_c6={rn2['resampler']['mean_c6']:.4f} "
                  f"SD={rn2['resampler']['sd_c6']:.4f}")
    lines.append(f"- Per-metric Gaussian: mean_c6={rn2['gaussian']['mean_c6']:.4f} "
                  f"SD={rn2['gaussian']['sd_c6']:.4f}\n")
    lines.append(f"**S4 finding: {rn2_finding}**\n")

    lines.append("## Per-tier resampler noise floor (composite_6)\n")
    lines.append("| tier | mean of per-workload resampler means | mean of per-workload resampler SDs | "
                  "max |mean-1| across workloads |")
    lines.append("|---|---|---|---|")
    for tier in TIER_ORDER:
        nf = tier_noise_floor[tier]
        lines.append(f"| {tier} | {nf['mean_of_resampler_means_c6']:.4f} | "
                      f"{nf['mean_of_resampler_sds_c6']:.4f} | {nf['max_abs_departure_from_1_c6']:.4f} |")

    lines.append("\n## Per-cell table\n")
    lines.append("| tier | workload | n | S36 headline c6 | S36 headline c7 | resampler mean c6 | "
                  "resampler SD c6 | resampler mean c7 | gaussian mean c6 | gaussian SD c6 | "
                  "gaussian mean c7 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for tier in TIER_ORDER:
        for workload in WORKLOADS:
            c = cell_by_key[(tier, workload)]
            lines.append(
                f"| {c['tier']} | {c['workload']} | {c['n_traces']} | {c['s36_headline_c6']:.4f} | "
                f"{c['s36_headline_c7']:.4f} | {c['resampler']['mean_c6']:.4f} | "
                f"{c['resampler']['sd_c6']:.4f} | {c['resampler']['mean_c7']:.4f} | "
                f"{c['gaussian']['mean_c6']:.4f} | {c['gaussian']['sd_c6']:.4f} | "
                f"{c['gaussian']['mean_c7']:.4f} |"
            )

    OUT_MD.write_text("\n".join(lines) + "\n")

    print(f"\nWrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print()
    print("Per-tier resampler noise floor (composite_6):")
    for tier in TIER_ORDER:
        nf = tier_noise_floor[tier]
        print(f"  {tier}: mean={nf['mean_of_resampler_means_c6']:.4f} "
              f"sd={nf['mean_of_resampler_sds_c6']:.4f} max|mean-1|={nf['max_abs_departure_from_1_c6']:.4f}")
    print()
    print("resnet152 Tier2:")
    print(f"  S36 headline c6={rn2['s36_headline_c6']:.4f} c7={rn2['s36_headline_c7']:.4f}")
    print(f"  resampler mean_c6={rn2['resampler']['mean_c6']:.4f} sd={rn2['resampler']['sd_c6']:.4f}")
    print(f"  gaussian mean_c6={rn2['gaussian']['mean_c6']:.4f} sd={rn2['gaussian']['sd_c6']:.4f}")
    print(f"  {rn2_finding}")
    print()
    print("Gaussian pass counts per tier (composite_6):")
    for tier in TIER_ORDER:
        g = gaussian_pass_counts[tier]
        s = s36_pass_counts_for_reference[tier]
        print(f"  {tier}: gaussian one-sided={g['one_sided']}/5 two-sided={g['two_sided']}/5 "
              f"| S36(ref) one-sided={s['one_sided']}/5 two-sided={s['two_sided']}/5")
    print()
    print(gaussian_verdict)


if __name__ == "__main__":
    main()
