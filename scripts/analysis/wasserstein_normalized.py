#!/usr/bin/env python3
"""E4 - model-vs-real Wasserstein, recomputed scale-free.

Targets the fidelity-looking ~5x number: A16's own model-vs-real raw
pooled W (~3.2, see outputs/phase4/validation/s36/validation_report.json
mean_wasserstein_all) vs Tier2/Tier3's raw pooled W (~16.9 / ~16.6, see
outputs/phase4/validation/s36_tier{2,3}/validation_report.json). That
~5x gap is computed entirely in each tier's own raw physical units
(with the legacy MB convention on pod_memory_bytes) -- units that are
NOT comparable across tiers, since A16 and the H100 tiers have
different absolute min/max ranges per metric (different hardware, see
tools/load_experiment.py docstring re: gpu_utilization 0-100 on A16 vs
0-700 on Tier 2 MIG slices). This script recomputes the same model-vs-
real per-metric Wasserstein distance in two scale-free spaces so the
~5x gap can be checked for a scale artifact:

  PRIMARY   : min-max training space -- each tier's own stored
              combined_normalization.json (the space the model is
              actually optimized in).
  SECONDARY : std-normalized -- raw_w[metric] / std(real[metric]).
              (IQR skipped per instructions.)

Does NOT recompute the real-vs-real dataset distance (a16_vs_tier2 /
a16_vs_tier3_wasserstein.py already own that number, ~4.5-4.6 in their
own convention) -- this script is model-vs-real only.

Reuses the Wasserstein primitive (scipy.stats.wasserstein_distance) the
same way a16_vs_tier2_wasserstein.py / a16_vs_tier3_wasserstein.py do;
does not reimplement the distance.

Real/synthetic trace sources are the same frozen assets the S36 eval and
validate_s36*.py suite use: frozen generators under
models/phase4/timegan_s36{,_tier2,_tier3}/, the byte-identical
postprocess_s36{,_tier2,_tier3}.py pipeline (generate_postprocessed_traces),
and each tier's own combined_dataset.npz / combined_normalization.json
real data.

Seeding: generation elsewhere in this codebase (eval_s36*.py,
postprocess_s36*.py) is deliberately UNSEEDED (see E2/E13 step-0 notes).
This script is the exception: torch.manual_seed(42) AND np.random.seed(42)
are set once, deliberately, purely so this specific analysis is
reproducible run-to-run (np.random also matters here because
postprocess_trace's pod_memory_bytes reconstruction and dropped-metric
reconstruction draw from np.random, and pod_memory_bytes is one of the
7 trained metrics we score). This is a deliberate deviation from the
rest of the codebase's unseeded-generation convention, not an oversight.

Run from repo root:
    PYTHONPATH=$(pwd)/scripts/utils python3 scripts/analysis/wasserstein_normalized.py
"""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wasserstein_distance

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "phase4"))

WORKLOADS = ["bert", "gpt2", "resnet152", "whisper", "yolo"]

# tier label -> (postprocess module name, data_dir, matched real r-range)
TIER_CONFIG = {
    "a16": {
        "module": "postprocess_s36",
        "data_dir": "data/processed/phase4/unified",
        "replicas": list(range(1, 11)),  # r=1..10
    },
    "tier3": {
        "module": "postprocess_s36_tier3",
        "data_dir": "data/processed/tier3/unified",
        "replicas": list(range(1, 11)),  # r=1..10
    },
    "tier2": {
        "module": "postprocess_s36_tier2",
        "data_dir": "data/processed/tier2/unified",
        "replicas": list(range(1, 8)),  # r=1..7
    },
}

# Legacy MB convention applied only to pod_memory_bytes for the retired
# raw_pooled_w figure, matching ANALYSIS_METRICS['scale'] in
# validate_s36*.py so raw_pooled_w reconciles with validation_report.json
# mean_wasserstein_all (~3.19 a16, ~16.93 tier2, ~16.56 tier3).
RAW_SCALE = {"pod_memory_bytes": 1e6}

OUT_DIR = REPO_ROOT / "outputs" / "analysis" / "wasserstein"
OUT_JSON = OUT_DIR / "wasserstein_normalized.json"
OUT_MD = OUT_DIR / "wasserstein_normalized.md"


def denormalize(traces_norm_7, norm_params, names):
    out = np.zeros_like(traces_norm_7)
    for i, name in enumerate(names):
        p = norm_params[name]
        out[:, :, i] = traces_norm_7[:, :, i] * (p["max"] - p["min"]) + p["min"]
    return out


def load_tier_module(tier_label):
    cfg = TIER_CONFIG[tier_label]
    mod = importlib.import_module(cfg["module"])
    return mod, cfg


def gather_pooled(mod, cfg, workload):
    """Pool real (denormalized) and synthetic (postprocessed) values for
    the 7 trained metrics across the tier's full matched r-range.

    Returns:
        real_pooled: dict[metric_name] -> 1D np.ndarray (all r, all pods, all timesteps)
        synth_pooled: dict[metric_name] -> 1D np.ndarray (all r, all samples, all timesteps)
    """
    trained_names = mod.TRAINED_NAMES
    trained_idx = mod.TRAINED_INDICES

    data = mod.load_real_data(cfg["data_dir"])
    norm_params = mod.load_normalization_params(workload, cfg["data_dir"])
    all_traces = data["traces"]  # (N, 715, 10) normalized
    all_wl = data["workload_ids"]
    all_rc = data["replica_counts"]
    wid = mod.WORKLOADS.index(workload)

    real_chunks = {m: [] for m in trained_names}
    synth_chunks = {m: [] for m in trained_names}

    for r in cfg["replicas"]:
        mask = (all_wl == wid) & (all_rc == r)
        n_real_pods = int(mask.sum())
        if n_real_pods == 0:
            raise RuntimeError(
                f"{cfg['module']}/{workload}: no real pods at r={r} in "
                f"{cfg['data_dir']}/combined_dataset.npz - matched r-range assumption violated"
            )

        real_norm_7 = all_traces[mask][:, :, trained_idx]  # (n_pods, 715, 7)
        real_denorm_7 = denormalize(real_norm_7, norm_params, trained_names)

        # n_samples matches real pod count at this r, for a pooled
        # comparison of like-sized empirical distributions per r.
        synth_full = mod.generate_postprocessed_traces(
            workload, r, n_samples=n_real_pods, device="cpu",
            data_dir=cfg["data_dir"],
        )  # (n_real_pods, 715, 10)
        synth_7 = synth_full[:, :, trained_idx]  # (n_real_pods, 715, 7)

        for i, name in enumerate(trained_names):
            real_chunks[name].append(real_denorm_7[:, :, i].flatten())
            synth_chunks[name].append(synth_7[:, :, i].flatten())

    real_pooled = {m: np.concatenate(v) for m, v in real_chunks.items()}
    synth_pooled = {m: np.concatenate(v) for m, v in synth_chunks.items()}
    return real_pooled, synth_pooled, trained_names, norm_params


def main():
    # Deliberate seeding exception - see module docstring.
    torch.manual_seed(42)
    np.random.seed(42)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cells = []  # flat list of per (tier, workload, metric) records
    pooled = {}  # (tier, workload) -> {normalized_pooled_w, normalized_pooled_w_std, raw_pooled_w}

    for tier_label, cfg in TIER_CONFIG.items():
        mod, _ = load_tier_module(tier_label)
        print(f"=== {tier_label} (r={cfg['replicas'][0]}..{cfg['replicas'][-1]}) ===")

        for workload in WORKLOADS:
            print(f"  {workload}...", end="", flush=True)
            real_pooled, synth_pooled, trained_names, norm_params = gather_pooled(mod, cfg, workload)

            minmax_ws, std_ws, raw_ws_native, raw_ws_mbscale = {}, {}, {}, {}

            for name in trained_names:
                r_raw = real_pooled[name]
                s_raw = synth_pooled[name]

                # Native-unit raw distance (bytes, watts, fraction, etc.)
                raw_w_native = float(wasserstein_distance(r_raw, s_raw))
                raw_ws_native[name] = raw_w_native

                # SECONDARY: std-normalized. Wasserstein-1 is translation
                # invariant and positively homogeneous under a common
                # positive scale, so raw_w / std(real) == the distance
                # computed on (x - mean)/std for both real and synth
                # (using the real stdev to scale both). Unit-invariant.
                real_std = float(r_raw.std())
                std_ws[name] = raw_w_native / real_std if real_std > 0 else float("nan")

                # PRIMARY: min-max training space, this tier's own
                # stored normalization params (the space S36 optimizes in).
                p = norm_params[name]
                span = p["max"] - p["min"]
                r_mm = (r_raw - p["min"]) / span
                s_mm = (s_raw - p["min"]) / span
                minmax_ws[name] = float(wasserstein_distance(r_mm, s_mm))

                # Retired raw-unit-only figure, MB convention for parity
                # with validation_report.json's mean_wasserstein_all.
                scale = RAW_SCALE.get(name, 1.0)
                raw_ws_mbscale[name] = raw_w_native / scale

                cells.append({
                    "tier": tier_label,
                    "workload": workload,
                    "metric": name,
                    "minmax_w": minmax_ws[name],
                    "std_w": std_ws[name],
                    "raw_w_native_units": raw_w_native,
                    "real_std_native_units": real_std,
                })

            normalized_pooled_w = float(np.mean(list(minmax_ws.values())))
            normalized_pooled_w_std = float(np.mean(list(std_ws.values())))
            raw_pooled_w = float(np.mean(list(raw_ws_mbscale.values())))

            pooled[(tier_label, workload)] = {
                "normalized_pooled_w": normalized_pooled_w,
                "normalized_pooled_w_std": normalized_pooled_w_std,
                "raw_pooled_w": raw_pooled_w,
            }
            print(f" minmax_pooled={normalized_pooled_w:.4f} std_pooled={normalized_pooled_w_std:.4f} "
                  f"raw_pooled(MB-conv)={raw_pooled_w:.4f}")

    # ---- Tier-level aggregates (mean across 5 workloads) ----
    tier_agg = {}
    for tier_label in TIER_CONFIG:
        rows = [pooled[(tier_label, w)] for w in WORKLOADS]
        tier_agg[tier_label] = {
            "mean_normalized_pooled_w": float(np.mean([r["normalized_pooled_w"] for r in rows])),
            "mean_normalized_pooled_w_std": float(np.mean([r["normalized_pooled_w_std"] for r in rows])),
            "mean_raw_pooled_w": float(np.mean([r["raw_pooled_w"] for r in rows])),
        }

    a16_raw = tier_agg["a16"]["mean_raw_pooled_w"]
    tier2_raw = tier_agg["tier2"]["mean_raw_pooled_w"]
    tier3_raw = tier_agg["tier3"]["mean_raw_pooled_w"]
    a16_mm = tier_agg["a16"]["mean_normalized_pooled_w"]
    tier2_mm = tier_agg["tier2"]["mean_normalized_pooled_w"]
    tier3_mm = tier_agg["tier3"]["mean_normalized_pooled_w"]
    a16_std = tier_agg["a16"]["mean_normalized_pooled_w_std"]
    tier2_std = tier_agg["tier2"]["mean_normalized_pooled_w_std"]
    tier3_std = tier_agg["tier3"]["mean_normalized_pooled_w_std"]

    ratio_raw_tier2 = tier2_raw / a16_raw if a16_raw else float("nan")
    ratio_raw_tier3 = tier3_raw / a16_raw if a16_raw else float("nan")
    ratio_mm_tier2 = tier2_mm / a16_mm if a16_mm else float("nan")
    ratio_mm_tier3 = tier3_mm / a16_mm if a16_mm else float("nan")
    ratio_std_tier2 = tier2_std / a16_std if a16_std else float("nan")
    ratio_std_tier3 = tier3_std / a16_std if a16_std else float("nan")

    def classify(ratio):
        if ratio <= 1.5:
            return "collapses toward 1x"
        if ratio >= 3.5:
            return "stays high"
        return "intermediate"

    outcome_minmax = classify(max(ratio_mm_tier2, ratio_mm_tier3))
    outcome_std = classify(max(ratio_std_tier2, ratio_std_tier3))

    tier2_vs_tier3_raw_gap = abs(tier2_raw - tier3_raw)
    tier2_vs_tier3_mm_gap = abs(tier2_mm - tier3_mm)
    tier2_vs_tier3_std_gap = abs(tier2_std - tier3_std)
    # "near-identical" in raw was ~2% apart (16.93 vs 16.56); flag separation
    # if the normalized relative gap is much larger than the raw relative gap.
    raw_rel_gap = tier2_vs_tier3_raw_gap / ((tier2_raw + tier3_raw) / 2) if (tier2_raw + tier3_raw) else 0
    mm_rel_gap = tier2_vs_tier3_mm_gap / ((tier2_mm + tier3_mm) / 2) if (tier2_mm + tier3_mm) else 0
    std_rel_gap = tier2_vs_tier3_std_gap / ((tier2_std + tier3_std) / 2) if (tier2_std + tier3_std) else 0

    # ---- Write JSON ----
    out = {
        "description": "E4 model-vs-real Wasserstein, recomputed scale-free "
                        "(minmax primary, std-normalized secondary), plus "
                        "retired raw_pooled_w for reconciliation with "
                        "validation_report.json mean_wasserstein_all.",
        "seed": {"torch_manual_seed": 42, "numpy_random_seed": 42,
                 "note": "deliberate exception - generation elsewhere in this "
                         "codebase is unseeded; see module docstring"},
        "tier_replica_ranges": {t: c["replicas"] for t, c in TIER_CONFIG.items()},
        "cells": cells,  # 5 workloads x 3 tiers x 7 metrics = 105 rows
        "pooled_per_tier_workload": [
            {"tier": t, "workload": w, **pooled[(t, w)]}
            for t in TIER_CONFIG for w in WORKLOADS
        ],
        "tier_aggregates": tier_agg,
        "c6_ratios_vs_a16": {
            "raw_pooled_w": {"tier2_over_a16": ratio_raw_tier2, "tier3_over_a16": ratio_raw_tier3},
            "normalized_pooled_w_minmax": {"tier2_over_a16": ratio_mm_tier2, "tier3_over_a16": ratio_mm_tier3},
            "normalized_pooled_w_std": {"tier2_over_a16": ratio_std_tier2, "tier3_over_a16": ratio_std_tier3},
        },
        "c6_outcome": {
            "minmax_space": outcome_minmax,
            "std_space": outcome_std,
        },
        "tier2_vs_tier3": {
            "raw_pooled_w": {"tier2": tier2_raw, "tier3": tier3_raw, "relative_gap": raw_rel_gap},
            "normalized_pooled_w_minmax": {"tier2": tier2_mm, "tier3": tier3_mm, "relative_gap": mm_rel_gap},
            "normalized_pooled_w_std": {"tier2": tier2_std, "tier3": tier3_std, "relative_gap": std_rel_gap},
            "separation_note": "raw relative gap ~%.1f%%; minmax relative gap ~%.1f%%; "
                                "std relative gap ~%.1f%%. A markedly larger normalized "
                                "gap than the raw gap would support outcome (c)." % (
                                    raw_rel_gap * 100, mm_rel_gap * 100, std_rel_gap * 100),
        },
        "reconciliation_note": "raw_pooled_w is expected to reconcile with "
                                "outputs/phase4/validation/s36{,_tier2,_tier3}/"
                                "validation_report.json mean_wasserstein_all "
                                "(~3.19 a16, ~16.93 tier2, ~16.56 tier3) only up "
                                "to generation noise (unseeded elsewhere; this "
                                "script seeds torch+numpy at 42 but n_samples "
                                "and exact real-pod pairing differ from the "
                                "original validation run) - small mismatches are "
                                "expected, not errors.",
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))

    # ---- Write MD ----
    lines = []
    lines.append("# E4 - model-vs-real Wasserstein, recomputed scale-free\n")
    lines.append("PRIMARY = min-max training space (tier's own combined_normalization.json). "
                  "SECONDARY = std-normalized (raw_w / real stdev). raw_pooled_w is raw-unit-only "
                  "(MB convention on pod_memory_bytes), retired from paper tables, kept only for "
                  "reconciliation with legacy validation_report.json figures.\n")
    lines.append(f"Seed: torch.manual_seed(42) + np.random.seed(42), set once, deliberately - "
                 f"generation elsewhere in this codebase is unseeded.\n")

    lines.append("## Per-metric table (5 workloads x 3 tiers x 7 metrics)\n")
    lines.append("| tier | workload | metric | minmax_w (primary) | std_w (secondary) | raw_w (native units) |")
    lines.append("|---|---|---|---|---|---|")
    for c in cells:
        lines.append(
            f"| {c['tier']} | {c['workload']} | {c['metric']} | {c['minmax_w']:.5f} | "
            f"{c['std_w']:.5f} | {c['raw_w_native_units']:.6g} |"
        )

    lines.append("\n## Per (tier, workload) pooled summary\n")
    lines.append("| tier | workload | normalized_pooled_w (minmax) | normalized_pooled_w_std | raw_pooled_w (MB-conv) |")
    lines.append("|---|---|---|---|---|")
    for t in TIER_CONFIG:
        for w in WORKLOADS:
            p = pooled[(t, w)]
            lines.append(
                f"| {t} | {w} | {p['normalized_pooled_w']:.4f} | "
                f"{p['normalized_pooled_w_std']:.4f} | {p['raw_pooled_w']:.4f} |"
            )

    lines.append("\n## Per-tier aggregate (mean across 5 workloads)\n")
    lines.append("| tier | mean normalized_pooled_w (minmax) | mean normalized_pooled_w_std | mean raw_pooled_w (MB-conv) |")
    lines.append("|---|---|---|---|")
    for t in TIER_CONFIG:
        a = tier_agg[t]
        lines.append(f"| {t} | {a['mean_normalized_pooled_w']:.4f} | "
                      f"{a['mean_normalized_pooled_w_std']:.4f} | {a['mean_raw_pooled_w']:.4f} |")

    lines.append("\n## C6 ratios vs A16 (the ~5x number, recomputed per space)\n")
    lines.append("| space | tier2/a16 | tier3/a16 |")
    lines.append("|---|---|---|")
    lines.append(f"| raw_pooled_w | {ratio_raw_tier2:.2f}x | {ratio_raw_tier3:.2f}x |")
    lines.append(f"| normalized_pooled_w (minmax, PRIMARY) | {ratio_mm_tier2:.2f}x | {ratio_mm_tier3:.2f}x |")
    lines.append(f"| normalized_pooled_w_std (SECONDARY) | {ratio_std_tier2:.2f}x | {ratio_std_tier3:.2f}x |")
    lines.append(f"\n**C6 outcome, minmax (primary) space: {outcome_minmax}**")
    lines.append(f"\n**C6 outcome, std (secondary) space: {outcome_std}**\n")

    lines.append("## Tier2 vs Tier3 explicit comparison\n")
    lines.append(f"- raw_pooled_w: tier2={tier2_raw:.4f}, tier3={tier3_raw:.4f}, "
                  f"relative gap={raw_rel_gap*100:.1f}%")
    lines.append(f"- normalized_pooled_w (minmax): tier2={tier2_mm:.4f}, tier3={tier3_mm:.4f}, "
                  f"relative gap={mm_rel_gap*100:.1f}%")
    lines.append(f"- normalized_pooled_w_std: tier2={tier2_std:.4f}, tier3={tier3_std:.4f}, "
                  f"relative gap={std_rel_gap*100:.1f}%")

    lines.append(f"\n{out['reconciliation_note']}\n")

    OUT_MD.write_text("\n".join(lines) + "\n")

    print(f"\nWrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print()
    print(json.dumps(tier_agg, indent=2))
    print()
    print("C6 ratios vs A16:")
    print(f"  raw_pooled_w:              tier2/a16={ratio_raw_tier2:.3f}x  tier3/a16={ratio_raw_tier3:.3f}x")
    print(f"  normalized_pooled_w (mm):  tier2/a16={ratio_mm_tier2:.3f}x  tier3/a16={ratio_mm_tier3:.3f}x")
    print(f"  normalized_pooled_w_std:   tier2/a16={ratio_std_tier2:.3f}x  tier3/a16={ratio_std_tier3:.3f}x")
    print(f"C6 outcome (minmax primary): {outcome_minmax}")
    print(f"C6 outcome (std secondary):  {outcome_std}")
    print()
    print(f"Tier2 vs Tier3: raw rel gap={raw_rel_gap*100:.1f}%  minmax rel gap={mm_rel_gap*100:.1f}%  "
          f"std rel gap={std_rel_gap*100:.1f}%")


GAUSSIAN_SEED = 42
K_DRAWS_S36 = 30
EXCLUDED_METRIC = "pod_memory_bytes"

GAUSSIAN_OUT_JSON = OUT_DIR / "wasserstein_gaussian_check.json"
GAUSSIAN_OUT_MD = OUT_DIR / "wasserstein_gaussian_check.md"


def _wr_helpers():
    """Import the fast (precompute-once) generation helpers from
    wasserstein_reconciliation.py - reused, not reimplemented. That
    script already builds: TIER_ORDER, TIER_CONFIG, load_pp(),
    gather_real() (real traces per r, raw units), synth_by_r() (frozen
    generator + byte-identical postprocess, per-r synthetic, seeded),
    and minmax_pooled_w() (PRIMARY min-max space Wasserstein, same
    tier-normalization convention as this file's main()).
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    return importlib.import_module("wasserstein_reconciliation")


def gaussian_minmax_per_metric(real_by_r, names, norm_params, seed):
    """Per-metric Gaussian baseline: i.i.d. N(mean, std) matched to the
    real pooled (all-r) per-metric mean/std, same total sample count as
    real, one fixed seed. Same construction as naive_baselines.py's
    gaussian_baseline(). Wasserstein computed in the tier's own min-max
    training space (PRIMARY), matching this file's own convention.
    """
    rng = np.random.default_rng(seed)
    real_concat = {m: np.concatenate([real_by_r[r][:, :, i].flatten() for r in real_by_r])
                   for i, m in enumerate(names)}
    ws = {}
    for m in names:
        vals = real_concat[m]
        gauss = rng.normal(vals.mean(), vals.std(), size=vals.shape)
        p = norm_params[m]
        span = p["max"] - p["min"]
        real_mm = (vals - p["min"]) / span
        gauss_mm = (gauss - p["min"]) / span
        ws[m] = float(wasserstein_distance(real_mm, gauss_mm))
    return ws


def run_gaussian_discriminative_check():
    """Gaussian-Wasserstein discriminative check (extension).

    Q1: is normalized W saturated by the i.i.d. Gaussian the way VR is
        (naive_baselines.py, E10)?
    Q2: is the Tier2/Tier3 normalized-W separation (~1.95x, E4 Task 4b)
        a property of S36, or of the data/tier (does the Gaussian show
        the same separation)?
    Q3: per-metric, where does S36 beat the Gaussian and where not?

    S36 side: K_DRAWS_S36=30 independently seeded draws (torch.manual_seed
    0..29, deliberate per the established convention), mean+SD.
    Gaussian side: one fixed seed (GAUSSIAN_SEED=42), matching the
    instruction ("fixed seed for the Gaussian") - the Gaussian has no
    generator-checkpoint variance to characterize, just its own sampling
    noise, which naive_baselines.py already showed is tiny (SD ~0.002-0.004
    on VR); a single draw is representative.
    """
    wr = _wr_helpers()

    per_cell = {}
    for tier in wr.TIER_ORDER:
        pp = wr.load_pp(tier)
        cfg = wr.TIER_CONFIG[tier]
        print(f"=== {tier} ===", flush=True)
        memory_stats = pp.compute_memory_stats(cfg["data_dir"])
        dropped_stats = pp.compute_dropped_metric_stats(cfg["data_dir"])
        for workload in WORKLOADS:
            generator = pp.load_generator(workload, device="cpu")
            real_by_r, norm_params = wr.gather_real(pp, cfg, workload)
            names = pp.TRAINED_NAMES

            gaussian_ws = gaussian_minmax_per_metric(real_by_r, names, norm_params, seed=GAUSSIAN_SEED)
            gaussian_pooled6 = float(np.mean([v for m, v in gaussian_ws.items() if m != EXCLUDED_METRIC]))
            gaussian_pooled7 = float(np.mean(list(gaussian_ws.values())))

            per_metric_draws = {m: [] for m in names}
            pooled6_draws, pooled7_draws = [], []
            for k in range(K_DRAWS_S36):
                syn = wr.synth_by_r(pp, generator, workload, cfg, real_by_r, memory_stats, dropped_stats, seed=k)
                _, per_metric = wr.minmax_pooled_w(real_by_r, syn, names, norm_params)
                for m in names:
                    per_metric_draws[m].append(per_metric[m])
                pooled6_draws.append(float(np.mean([per_metric[m] for m in names if m != EXCLUDED_METRIC])))
                pooled7_draws.append(float(np.mean(list(per_metric.values()))))

            per_cell[(tier, workload)] = {
                "gaussian_per_metric": gaussian_ws,
                "gaussian_pooled6": gaussian_pooled6,
                "gaussian_pooled7": gaussian_pooled7,
                "s36_per_metric_mean": {m: float(np.mean(v)) for m, v in per_metric_draws.items()},
                "s36_per_metric_sd": {m: float(np.std(v, ddof=1)) for m, v in per_metric_draws.items()},
                "s36_pooled6_mean": float(np.mean(pooled6_draws)),
                "s36_pooled6_sd": float(np.std(pooled6_draws, ddof=1)),
                "s36_pooled7_mean": float(np.mean(pooled7_draws)),
                "s36_pooled7_sd": float(np.std(pooled7_draws, ddof=1)),
            }
            c = per_cell[(tier, workload)]
            print(f"  {workload}: gaussian6={gaussian_pooled6:.4f}  "
                  f"s36_6={c['s36_pooled6_mean']:.4f}+-{c['s36_pooled6_sd']:.4f}", flush=True)

    # ---- Tier aggregates (mean over 5 workloads) ----
    tier_agg = {}
    for tier in wr.TIER_ORDER:
        rows = [per_cell[(tier, w)] for w in WORKLOADS]
        tier_agg[tier] = {
            "gaussian_pooled6": float(np.mean([r["gaussian_pooled6"] for r in rows])),
            "s36_pooled6_mean": float(np.mean([r["s36_pooled6_mean"] for r in rows])),
            "s36_pooled6_sd_combined": float(np.sqrt(np.sum([r["s36_pooled6_sd"] ** 2 for r in rows]))) / len(rows),
        }

    # Q1: is S36 clearly below (better than) Gaussian, per tier?
    q1 = {}
    for tier in wr.TIER_ORDER:
        a = tier_agg[tier]
        below = a["s36_pooled6_mean"] < a["gaussian_pooled6"]
        margin = a["gaussian_pooled6"] - a["s36_pooled6_mean"]
        rel_margin = margin / a["gaussian_pooled6"] if a["gaussian_pooled6"] else float("nan")
        q1[tier] = {"s36_below_gaussian": below, "absolute_margin": margin, "relative_margin": rel_margin}

    # Q2: Gaussian Tier2/Tier3 separation vs S36's
    gaussian_ratio_tier2_tier3 = tier_agg["tier2"]["gaussian_pooled6"] / tier_agg["tier3"]["gaussian_pooled6"]
    s36_ratio_tier2_tier3 = tier_agg["tier2"]["s36_pooled6_mean"] / tier_agg["tier3"]["s36_pooled6_mean"]

    # Q3: per-metric, tier-aggregated S36 vs Gaussian (mean over 5 workloads)
    per_metric_names = pp.TRAINED_NAMES  # same 7 names, tier-agnostic
    per_metric_compare = {}
    for tier in wr.TIER_ORDER:
        per_metric_compare[tier] = {}
        for m in per_metric_names:
            g = float(np.mean([per_cell[(tier, w)]["gaussian_per_metric"][m] for w in WORKLOADS]))
            s = float(np.mean([per_cell[(tier, w)]["s36_per_metric_mean"][m] for w in WORKLOADS]))
            per_metric_compare[tier][m] = {
                "gaussian": g, "s36_mean": s, "s36_beats_gaussian": s < g,
            }

    out = {
        "config": {"gaussian_seed": GAUSSIAN_SEED, "k_draws_s36": K_DRAWS_S36,
                   "excluded_metric": EXCLUDED_METRIC},
        "per_cell": {f"{t}/{w}": per_cell[(t, w)] for t in wr.TIER_ORDER for w in WORKLOADS},
        "tier_aggregates": tier_agg,
        "q1_s36_vs_gaussian_per_tier": q1,
        "q2_tier2_tier3_separation": {
            "gaussian_ratio": gaussian_ratio_tier2_tier3,
            "s36_ratio": s36_ratio_tier2_tier3,
        },
        "q3_per_metric_tier_aggregated": per_metric_compare,
    }
    GAUSSIAN_OUT_JSON.write_text(json.dumps(out, indent=2))

    lines = ["# Gaussian-Wasserstein discriminative check\n"]
    lines.append(f"S36: {K_DRAWS_S36} independently seeded draws (mean+SD). Gaussian: one fixed seed "
                  f"({GAUSSIAN_SEED}). Both in PRIMARY min-max space, composite_6 (exclude "
                  f"`{EXCLUDED_METRIC}`).\n")

    lines.append("## Q1 - is normalized W saturated by the Gaussian the way VR is?\n")
    lines.append("| tier | Gaussian pooled_6 | S36 pooled_6 (mean+-SD) | S36 below Gaussian? | relative margin |")
    lines.append("|---|---|---|---|---|")
    for tier in wr.TIER_ORDER:
        a = tier_agg[tier]
        qq = q1[tier]
        lines.append(f"| {tier} | {a['gaussian_pooled6']:.4f} | "
                      f"{a['s36_pooled6_mean']:.4f} +/- {a['s36_pooled6_sd_combined']:.4f} | "
                      f"{qq['s36_below_gaussian']} | {qq['relative_margin']*100:+.1f}% |")

    lines.append("\n## Q2 - Tier2/Tier3 separation: S36-specific, or data property?\n")
    lines.append(f"- Gaussian Tier2/Tier3 ratio: {gaussian_ratio_tier2_tier3:.4f}x")
    lines.append(f"- S36 Tier2/Tier3 ratio: {s36_ratio_tier2_tier3:.4f}x\n")

    lines.append("## Q3 - per-metric: where S36 beats the Gaussian\n")
    lines.append("| tier | metric | Gaussian | S36 (mean) | S36 beats Gaussian? |")
    lines.append("|---|---|---|---|---|")
    for tier in wr.TIER_ORDER:
        for m in per_metric_names:
            c = per_metric_compare[tier][m]
            lines.append(f"| {tier} | {m} | {c['gaussian']:.5f} | {c['s36_mean']:.5f} | "
                          f"{c['s36_beats_gaussian']} |")

    GAUSSIAN_OUT_MD.write_text("\n".join(lines) + "\n")

    print(f"\nWrote {GAUSSIAN_OUT_JSON}")
    print(f"Wrote {GAUSSIAN_OUT_MD}")
    print()
    print("Q1 (S36 vs Gaussian, pooled_6, per tier):")
    for tier in wr.TIER_ORDER:
        a = tier_agg[tier]
        qq = q1[tier]
        print(f"  {tier}: gaussian={a['gaussian_pooled6']:.4f} s36={a['s36_pooled6_mean']:.4f}+-"
              f"{a['s36_pooled6_sd_combined']:.4f} below_gaussian={qq['s36_below_gaussian']} "
              f"rel_margin={qq['relative_margin']*100:+.1f}%")
    print()
    print(f"Q2: gaussian tier2/tier3 ratio={gaussian_ratio_tier2_tier3:.4f}x  "
          f"s36 tier2/tier3 ratio={s36_ratio_tier2_tier3:.4f}x")
    print()
    print("Q3 per-metric (tier-aggregated):")
    print(json.dumps(per_metric_compare, indent=2))


if __name__ == "__main__":
    if "--gaussian-check" in sys.argv:
        run_gaussian_discriminative_check()
    else:
        main()
