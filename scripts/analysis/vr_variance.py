#!/usr/bin/env python3
"""E11 - two-source VR variance characterization.

Composite = 6-metric composite (exclude pod_memory_bytes BY NAME),
matching the E2 headline (scripts/analysis/vr_per_metric.py). composite_7
(all 7 trained metrics) is carried as a secondary column throughout.

SOURCE 1 (PRIMARY) - generation variance. For each (tier, workload), the
frozen generator is re-run K=30 times, each draw with torch.manual_seed(s)
for s in 0..29. This deliberately overrides the otherwise-unseeded z in
GeneratorSeg.forward() (torch.randn when z=None) so the K draws are
reproducible and distinct. Everything else - data loading, denormalize(),
compute_vr_per_metric() (mean of per-metric syn_var/(real_var+1e-8),
axis=(0,1)), smooth_phase_boundaries() on the synthetic side only - is
imported directly from the frozen eval_s36{,_tier2,_tier3}.py /
postprocess_s36{,_tier2,_tier3}.py modules, not reimplemented. This is
the same reduction that produced outputs/phase4/timegan_s36*/s36_eval_
results.json (including that pipeline's known denormalize() no-op - see
E2/D1 notes - which applies identically to every draw here, so it does
not affect draw-to-draw variance).

SOURCE 2 (SECONDARY) - between-trace variance. The real pooled variance
in compute_vr_per_metric (var(axis=(0,1))) pools over BOTH the trace
axis (0) and the 715-timestep axis (1). This decomposes exactly (equal
trace length T=715 for every trace):

    pooled_var = mean_i(within_trace_i)  +  var(trace_means)
               =   within component      +   between component

The within component is well-estimated even at small val n (715 samples
per trace). The between component is estimated from only n_val trace-
level means and is fragile when n_val is small. Bootstrap (B=2000,
resample trace indices with replacement, axis 0 only) holding the
synthetic side FIXED (K-draw seed=0's synthetic pool) isolates this
between-trace uncertainty in the composite VR. Where n_val<=3 no CI is
emitted (flagged "insufficient between-trace support" - a bootstrap with
<=3 underlying trace-level means cannot support a meaningful CI); the
point estimate and n are still printed. Where n_val>=4 a 95% percentile
bootstrap CI is emitted.

FROZEN-ONLY all-traces column: for these frozen (already-selected, not
retuned here) S36 models, val was never used for model selection on
this branch, so recomputing the real variance over ALL traces (train+
val) is defensible and is added as an extra large-n column with its own
bootstrap CI (always emitted - n is large for every tier). This is NOT
extended to any hypothetical best-tuned/reselected model.

DECISION-CRITICAL:
  - Tier-step deltas in mean|VR-1| (composite_6): A16->Tier3, Tier3->Tier2,
    each with a generation-variance interval propagated from the 5
    per-workload K-draw SDs within each tier (independent-variance sum).
  - resnet152 Tier2: K-draw mean+SD, val n, within/between decomposition,
    and whether 0.369 [stored composite_7 headline] is distinguishable
    from 1.0 once between-trace variance is properly (pooled, large-n)
    estimated rather than read off a fragile small-n val split.

Run from repo root:
    PYTHONPATH=$(pwd)/scripts/utils python3 scripts/analysis/vr_variance.py
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
B_BOOT = 2000
WINDOW_SIZE = 5
N_GEN = 5
FROZEN_ALL_SEED = 9999
DEVICE = "cpu"

EVAL_MODULES = {"a16": "eval_s36", "tier2": "eval_s36_tier2", "tier3": "eval_s36_tier3"}
POSTPROCESS_MODULES = {"a16": "postprocess_s36", "tier2": "postprocess_s36_tier2", "tier3": "postprocess_s36_tier3"}
STORED_EVAL_RESULTS = {
    "a16": REPO_ROOT / "outputs/phase4/timegan_s36/s36_eval_results.json",
    "tier3": REPO_ROOT / "outputs/phase4/timegan_s36_tier3/s36_eval_results.json",
    "tier2": REPO_ROOT / "outputs/phase4/timegan_s36_tier2/s36_eval_results.json",
}
TIER_ORDER = ["a16", "tier3", "tier2"]  # hardware step, then sharing step

OUT_DIR = REPO_ROOT / "outputs" / "analysis" / "vr"
OUT_JSON = OUT_DIR / "vr_variance.json"
OUT_MD = OUT_DIR / "vr_variance.md"


def composite(vr_per_metric, kept_names, exclude=None):
    if exclude is None:
        vals = vr_per_metric
    else:
        vals = [v for v, n in zip(vr_per_metric, kept_names) if n != exclude]
    return float(np.mean(vals))


def stored_headline(tier, workload):
    d = json.loads(STORED_EVAL_RESULTS[tier].read_text())
    w = d["workloads"][workload]
    names = w["metric_names"]
    vr7 = w["vr_per_metric_smooth"]
    assert EXCLUDED_METRIC in names
    c7 = composite(vr7, names)
    c6 = composite(vr7, names, exclude=EXCLUDED_METRIC)
    assert round(c7, 5) == round(w["vr_smooth"], 5)
    return c6, c7


def prepare_cell(ev, pp, workload):
    data = np.load(ev.DATA_COMBINED, allow_pickle=True)
    norm = json.loads(Path(ev.NORM_COMBINED).read_text())
    workload_id = ev.WORKLOADS_ALL.index(workload)
    all_traces = data["traces"]
    all_rc = data["replica_counts"]
    all_wl = data["workload_ids"]
    val_idx = data["val_idx"]

    wl_mask = (all_wl == workload_id)
    traces_all = all_traces[wl_mask]
    rc_all = all_rc[wl_mask]

    drop = ev.DROP_PER_WORKLOAD.get(workload, set())
    kept_idx = [i for i, m in enumerate(ev.ALL_METRICS) if m not in drop]
    kept_names = [ev.ALL_METRICS[i] for i in kept_idx]
    n_metrics = len(kept_names)
    traces_all = traces_all[:, :, kept_idx].astype(np.float32)

    orig_to_new = {}
    new_idx = 0
    for old_idx in range(len(all_traces)):
        if wl_mask[old_idx]:
            orig_to_new[old_idx] = new_idx
            new_idx += 1
    new_val_idx = np.array([orig_to_new[i] for i in val_idx if i in orig_to_new])

    real_rc_val = rc_all[new_val_idx]

    model_path = Path(pp.MODEL_PATHS[workload])
    generator = ev.GeneratorSeg(ev.SEGMENT_LEN, n_metrics, ev.GEN_CFG).to(DEVICE)
    generator.load_state_dict(torch.load(model_path, map_location=DEVICE))
    generator.eval()

    norm_params = norm[workload]
    real_denorm_val = ev.denormalize(traces_all[new_val_idx], norm_params, kept_names)
    real_denorm_all = ev.denormalize(traces_all, norm_params, kept_names)

    return {
        "generator": generator, "kept_names": kept_names, "norm_params": norm_params,
        "real_denorm_val": real_denorm_val, "real_rc_val": real_rc_val,
        "real_denorm_all": real_denorm_all, "rc_all": rc_all,
        "n_val": int(real_denorm_val.shape[0]), "n_all": int(real_denorm_all.shape[0]),
    }


def generate_synth(ev, generator, unique_r, seed, kept_names, norm_params):
    torch.manual_seed(seed)
    syn_traces_smooth = []
    for r_val in unique_r:
        r_norm = (float(r_val) - 1.0) / 9.0
        traces_gen = generator.generate_trace(r_norm, DEVICE, n_samples=N_GEN)
        for trace in traces_gen:
            smooth_trace = ev.smooth_phase_boundaries(trace, ev.ACTUAL_BOUNDARIES, WINDOW_SIZE)
            syn_traces_smooth.append(smooth_trace)
    syn_traces_smooth = np.array(syn_traces_smooth)
    return ev.denormalize(syn_traces_smooth, norm_params, kept_names)


def decompose_within_between(real_denorm, metric_i):
    col = real_denorm[:, :, metric_i]  # (n_traces, 715)
    within = float(col.var(axis=1).mean())     # mean of per-trace temporal variance (ddof=0)
    between = float(col.mean(axis=1).var())    # variance of per-trace means (ddof=0)
    pooled = float(col.var())                  # var(axis=(0,1)) flattened == pooled
    return within, between, pooled


def bootstrap_composite6(real_denorm, syn_var_fixed, kept_names, n_boot, rng):
    n = real_denorm.shape[0]
    excl_idx = kept_names.index(EXCLUDED_METRIC)
    keep_mask = [i for i in range(len(kept_names)) if i != excl_idx]
    boot_vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        real_boot = real_denorm[idx]
        real_var_boot = real_boot.var(axis=(0, 1))
        vr_boot = syn_var_fixed / (real_var_boot + 1e-8)
        boot_vals[b] = float(np.mean(vr_boot[keep_mask]))
    return boot_vals


def process_cell(tier, workload, cell_seed_base):
    ev = importlib.import_module(EVAL_MODULES[tier])
    pp = importlib.import_module(POSTPROCESS_MODULES[tier])

    prep = prepare_cell(ev, pp, workload)
    kept_names = prep["kept_names"]
    real_denorm_val = prep["real_denorm_val"]
    unique_r_val = np.unique(prep["real_rc_val"])

    stored_c6, stored_c7 = stored_headline(tier, workload)

    # ---- SOURCE 1: K seeded generation draws ----
    c6_draws, c7_draws = [], []
    syn_draw0 = None
    for k in range(K):
        syn = generate_synth(ev, prep["generator"], unique_r_val, seed=k,
                              kept_names=kept_names, norm_params=prep["norm_params"])
        if k == 0:
            syn_draw0 = syn
        vr = ev.compute_vr_per_metric(real_denorm_val, syn)
        c6_draws.append(composite(vr, kept_names, exclude=EXCLUDED_METRIC))
        c7_draws.append(composite(vr, kept_names))
    c6_draws = np.array(c6_draws)
    c7_draws = np.array(c7_draws)
    k_mean_c6, k_sd_c6 = float(c6_draws.mean()), float(c6_draws.std(ddof=1))
    k_mean_c7, k_sd_c7 = float(c7_draws.mean()), float(c7_draws.std(ddof=1))
    abs_vr1_draws_c6 = np.abs(c6_draws - 1.0)
    sd_abs_vr1_c6 = float(abs_vr1_draws_c6.std(ddof=1))

    # ---- SOURCE 2: between-trace bootstrap, synthetic fixed at draw k=0 ----
    syn_var_fixed_val = syn_draw0.var(axis=(0, 1))
    within_between = {}
    for i, name in enumerate(kept_names):
        w, b, pooled = decompose_within_between(real_denorm_val, i)
        within_between[name] = {"within": w, "between": b, "pooled_check": pooled}

    n_val = prep["n_val"]
    rng = np.random.default_rng(cell_seed_base)
    if n_val >= 4:
        boot = bootstrap_composite6(real_denorm_val, syn_var_fixed_val, kept_names, B_BOOT, rng)
        ci_lo, ci_hi = [float(x) for x in np.percentile(boot, [2.5, 97.5])]
        val_ci = {"n": n_val, "sufficient": True, "ci95_lo": ci_lo, "ci95_hi": ci_hi,
                   "point_estimate": c6_draws[0]}
    else:
        val_ci = {"n": n_val, "sufficient": False,
                   "flag": "insufficient between-trace support",
                   "point_estimate": c6_draws[0]}

    # ---- FROZEN-ONLY all-traces column ----
    unique_r_all = np.unique(prep["rc_all"])
    syn_all = generate_synth(ev, prep["generator"], unique_r_all, seed=FROZEN_ALL_SEED,
                              kept_names=kept_names, norm_params=prep["norm_params"])
    vr_all = ev.compute_vr_per_metric(prep["real_denorm_all"], syn_all)
    c6_all = composite(vr_all, kept_names, exclude=EXCLUDED_METRIC)
    c7_all = composite(vr_all, kept_names)
    syn_var_fixed_all = syn_all.var(axis=(0, 1))
    rng_all = np.random.default_rng(cell_seed_base + 500000)
    boot_all = bootstrap_composite6(prep["real_denorm_all"], syn_var_fixed_all, kept_names, B_BOOT, rng_all)
    ci_all_lo, ci_all_hi = [float(x) for x in np.percentile(boot_all, [2.5, 97.5])]

    return {
        "tier": tier, "workload": workload,
        "stored_headline_c6": stored_c6, "stored_headline_c7": stored_c7,
        "k_draw_mean_c6": k_mean_c6, "k_draw_sd_c6": k_sd_c6,
        "k_draw_mean_c7": k_mean_c7, "k_draw_sd_c7": k_sd_c7,
        "sd_abs_vr_minus_1_c6": sd_abs_vr1_c6,
        "within_between_per_metric": within_between,
        "val_between_trace": val_ci,
        "frozen_all_traces": {
            "n": prep["n_all"], "composite_6": c6_all, "composite_7": c7_all,
            "ci95_lo": ci_all_lo, "ci95_hi": ci_all_hi,
        },
        "c6_draws": c6_draws.tolist(),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cells = {}
    seed_base = 0
    for tier in TIER_ORDER:
        for workload in WORKLOADS:
            print(f"=== {tier} / {workload} ===", flush=True)
            cells[(tier, workload)] = process_cell(tier, workload, seed_base)
            seed_base += 1000
            c = cells[(tier, workload)]
            print(f"  stored_c6={c['stored_headline_c6']:.4f} k_mean_c6={c['k_draw_mean_c6']:.4f} "
                  f"k_sd_c6={c['k_draw_sd_c6']:.4f} n_val={c['val_between_trace']['n']} "
                  f"n_all={c['frozen_all_traces']['n']}", flush=True)

    # ---- DECISION-CRITICAL: tier-step deltas in mean|VR-1| (composite_6) ----
    tier_stats = {}
    for tier in TIER_ORDER:
        headline_abs = [abs(cells[(tier, w)]["stored_headline_c6"] - 1.0) for w in WORKLOADS]
        sd_per_wl = [cells[(tier, w)]["sd_abs_vr_minus_1_c6"] for w in WORKLOADS]
        mean_abs = float(np.mean(headline_abs))
        propagated_sd = float(np.sqrt(sum(s ** 2 for s in sd_per_wl))) / 5.0
        tier_stats[tier] = {"mean_abs_vr_minus_1_c6": mean_abs, "propagated_gen_sd": propagated_sd}

    delta_a16_tier3 = tier_stats["tier3"]["mean_abs_vr_minus_1_c6"] - tier_stats["a16"]["mean_abs_vr_minus_1_c6"]
    delta_sd_a16_tier3 = float(np.sqrt(
        tier_stats["a16"]["propagated_gen_sd"] ** 2 + tier_stats["tier3"]["propagated_gen_sd"] ** 2))
    delta_tier3_tier2 = tier_stats["tier2"]["mean_abs_vr_minus_1_c6"] - tier_stats["tier3"]["mean_abs_vr_minus_1_c6"]
    delta_sd_tier3_tier2 = float(np.sqrt(
        tier_stats["tier3"]["propagated_gen_sd"] ** 2 + tier_stats["tier2"]["propagated_gen_sd"] ** 2))

    outside_a16_tier3 = abs(delta_a16_tier3) > delta_sd_a16_tier3
    outside_tier3_tier2 = abs(delta_tier3_tier2) > delta_sd_tier3_tier2

    # ---- DECISION-CRITICAL: resnet152 Tier2 ----
    rn2 = cells[("tier2", "resnet152")]
    rn2_k_mean_c6, rn2_k_sd_c6 = rn2["k_draw_mean_c6"], rn2["k_draw_sd_c6"]
    rn2_k_mean_c7, rn2_k_sd_c7 = rn2["k_draw_mean_c7"], rn2["k_draw_sd_c7"]
    rn2_n_val = rn2["val_between_trace"]["n"]
    rn2_val_sufficient = rn2["val_between_trace"]["sufficient"]
    rn2_frozen_all = rn2["frozen_all_traces"]

    gen_outside_1 = not (rn2_k_mean_c6 - rn2_k_sd_c6 <= 1.0 <= rn2_k_mean_c6 + rn2_k_sd_c6)
    frozen_all_outside_1 = not (rn2_frozen_all["ci95_lo"] <= 1.0 <= rn2_frozen_all["ci95_hi"])
    if rn2_val_sufficient:
        val_ci = rn2["val_between_trace"]
        val_outside_1 = not (val_ci["ci95_lo"] <= 1.0 <= val_ci["ci95_hi"])
    else:
        val_outside_1 = None

    rn2_verdict = (
        f"resnet152 Tier2 (stored composite_7 headline 0.369; composite_6={rn2['stored_headline_c6']:.4f}): "
        f"K-draw composite_6 mean={rn2_k_mean_c6:.4f} SD={rn2_k_sd_c6:.4f} "
        f"({'excludes' if gen_outside_1 else 'does NOT exclude'} 1.0 at +-1 generation SD). "
        f"val n={rn2_n_val} -> {'95% bootstrap CI computed' if rn2_val_sufficient else 'INSUFFICIENT between-trace support, no CI'}"
        + (f", CI=[{rn2['val_between_trace']['ci95_lo']:.4f}, {rn2['val_between_trace']['ci95_hi']:.4f}] "
           f"({'excludes' if val_outside_1 else 'does NOT exclude'} 1.0)" if rn2_val_sufficient else "")
        + f". Frozen all-traces (n={rn2_frozen_all['n']}) composite_6={rn2_frozen_all['composite_6']:.4f}, "
        f"95% CI=[{rn2_frozen_all['ci95_lo']:.4f}, {rn2_frozen_all['ci95_hi']:.4f}] "
        f"({'excludes' if frozen_all_outside_1 else 'does NOT exclude'} 1.0). "
        f"VERDICT: once the correct pooled (large-n, n={rn2_frozen_all['n']}) between-trace variance is used "
        f"instead of the fragile val-only (n={rn2_n_val}) estimate, VR=0.369-ish is "
        f"{'DISTINGUISHABLE' if frozen_all_outside_1 and gen_outside_1 else 'NOT cleanly distinguishable'} "
        f"from 1.0."
    )

    # ---- Write JSON ----
    out = {
        "config": {"K": K, "B_BOOT": B_BOOT, "window_size": WINDOW_SIZE, "n_gen": N_GEN,
                   "excluded_metric": EXCLUDED_METRIC, "frozen_all_seed": FROZEN_ALL_SEED,
                   "generation_seeds": "torch.manual_seed(s) for s in 0..29, deliberate "
                                       "override of otherwise-unseeded z"},
        "cells": [
            {**{k: v for k, v in cells[(t, w)].items() if k != "c6_draws"}}
            for t in TIER_ORDER for w in WORKLOADS
        ],
        "tier_stats_mean_abs_vr_minus_1_c6": tier_stats,
        "tier_step_deltas": {
            "A16_to_Tier3_hardware_step": {
                "delta": delta_a16_tier3, "propagated_gen_sd": delta_sd_a16_tier3,
                "outside_generation_interval": outside_a16_tier3,
            },
            "Tier3_to_Tier2_sharing_step": {
                "delta": delta_tier3_tier2, "propagated_gen_sd": delta_sd_tier3_tier2,
                "outside_generation_interval": outside_tier3_tier2,
            },
        },
        "resnet152_tier2_decision": {
            "stored_composite_7_headline": 0.369,
            "stored_composite_6": rn2["stored_headline_c6"],
            "k_draw_mean_c6": rn2_k_mean_c6, "k_draw_sd_c6": rn2_k_sd_c6,
            "k_draw_mean_c7": rn2_k_mean_c7, "k_draw_sd_c7": rn2_k_sd_c7,
            "val_n": rn2_n_val, "val_sufficient": rn2_val_sufficient,
            "val_between_trace": rn2["val_between_trace"],
            "frozen_all_traces": rn2_frozen_all,
            "within_between_per_metric": rn2["within_between_per_metric"],
            "verdict": rn2_verdict,
        },
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))

    # ---- Write MD ----
    lines = []
    lines.append("# E11 - two-source VR variance characterization\n")
    lines.append(f"K={K} seeded generation draws (torch.manual_seed 0..{K-1}), B={B_BOOT} between-trace "
                  f"bootstrap resamples. Composite = 6-metric (exclude `{EXCLUDED_METRIC}` by name), "
                  f"composite_7 carried as secondary.\n")

    lines.append("## DECISION-CRITICAL: tier-step deltas in mean|VR-1| (composite_6)\n")
    lines.append(f"- A16 -> Tier3 (hardware step): delta={delta_a16_tier3:+.4f}, "
                  f"propagated generation SD={delta_sd_a16_tier3:.4f} -> "
                  f"**{'OUTSIDE' if outside_a16_tier3 else 'INSIDE'}** the propagated generation interval "
                  f"(|delta| {'>' if outside_a16_tier3 else '<='} SD)")
    lines.append(f"- Tier3 -> Tier2 (sharing step): delta={delta_tier3_tier2:+.4f}, "
                  f"propagated generation SD={delta_sd_tier3_tier2:.4f} -> "
                  f"**{'OUTSIDE' if outside_tier3_tier2 else 'INSIDE'}** the propagated generation interval "
                  f"(|delta| {'>' if outside_tier3_tier2 else '<='} SD)\n")

    lines.append("## DECISION-CRITICAL: resnet152 Tier2\n")
    lines.append(rn2_verdict + "\n")

    lines.append("## Per-cell table\n")
    lines.append("| tier | workload | stored c6 (headline) | K-mean c6 | gen-SD c6 | stored c7 | "
                  "K-mean c7 | gen-SD c7 | val n | val 95% CI (c6) | frozen all-traces n | "
                  "frozen all-traces c6 | frozen all-traces 95% CI |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in TIER_ORDER:
        for w in WORKLOADS:
            c = cells[(t, w)]
            vbt = c["val_between_trace"]
            if vbt["sufficient"]:
                val_ci_str = f"[{vbt['ci95_lo']:.4f}, {vbt['ci95_hi']:.4f}]"
            else:
                val_ci_str = "insufficient between-trace support"
            fa = c["frozen_all_traces"]
            lines.append(
                f"| {t} | {w} | {c['stored_headline_c6']:.4f} | {c['k_draw_mean_c6']:.4f} | "
                f"{c['k_draw_sd_c6']:.4f} | {c['stored_headline_c7']:.4f} | {c['k_draw_mean_c7']:.4f} | "
                f"{c['k_draw_sd_c7']:.4f} | {vbt['n']} | {val_ci_str} | {fa['n']} | "
                f"{fa['composite_6']:.4f} | [{fa['ci95_lo']:.4f}, {fa['ci95_hi']:.4f}] |"
            )

    lines.append("\n## Per-metric within/between decomposition (val traces)\n")
    lines.append("Exact identity: pooled_var == within + between (equal trace length T=715).\n")
    lines.append("| tier | workload | metric | within (temporal) | between (trace-level) | pooled (check) |")
    lines.append("|---|---|---|---|---|---|")
    for t in TIER_ORDER:
        for w in WORKLOADS:
            c = cells[(t, w)]
            for name, wb in c["within_between_per_metric"].items():
                lines.append(
                    f"| {t} | {w} | {name} | {wb['within']:.6g} | {wb['between']:.6g} | {wb['pooled_check']:.6g} |"
                )

    lines.append("\n## Tier-level mean|VR-1| (composite_6) with propagated generation SD\n")
    lines.append("| tier | mean\\|VR-1\\| (headline, c6) | propagated generation SD |")
    lines.append("|---|---|---|")
    for t in TIER_ORDER:
        s = tier_stats[t]
        lines.append(f"| {t} | {s['mean_abs_vr_minus_1_c6']:.4f} | {s['propagated_gen_sd']:.4f} |")

    OUT_MD.write_text("\n".join(lines) + "\n")

    print(f"\nWrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print()
    print(f"A16->Tier3 delta={delta_a16_tier3:+.4f} sd={delta_sd_a16_tier3:.4f} outside={outside_a16_tier3}")
    print(f"Tier3->Tier2 delta={delta_tier3_tier2:+.4f} sd={delta_sd_tier3_tier2:.4f} outside={outside_tier3_tier2}")
    print()
    print(rn2_verdict)


if __name__ == "__main__":
    main()
