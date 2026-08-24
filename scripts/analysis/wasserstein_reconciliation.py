#!/usr/bin/env python3
"""Batch A Task 4 - E4 reconciliation, generation variance, exclusion.

4a. Reconcile E4's raw pooled Wasserstein against the committed reference
    (validation_report.json mean_wasserstein_all: A16~3.19, Tier3~16.56,
    Tier2~16.93). Two aggregation methods, generation setup held IDENTICAL
    between them (same r-range, same n_samples=n_real_pods, same seed,
    no clamp_batch) so only the aggregation order differs:
      (i)  pool all r into one distribution per metric, one Wasserstein  (E4's method)
      (ii) per-r Wasserstein per metric, then average over r             (validate_s36's method)
    If (ii) [under this held-constant setup] reproduces the committed
    figures, the gap is purely pool-vs-average. If it does NOT, we
    additionally call the ACTUAL committed compute_wasserstein_distances()
    from validate_s36{,_tier2,_tier3}.py (imported, not reimplemented) -
    which differs from our (ii) in three more ways: a restricted per-
    workload r subset (REPLICA_COUNTS_PER_WORKLOAD), a fixed n_samples=10
    instead of n_samples=n_real_pods, and a clamp_batch() call on the
    synthetic traces before the distance - to see whether THAT reproduces,
    isolating which of those extra factors matter. Only if neither
    reproduces do we call it a deeper bug.

4b. 30-seeded-draw generation variance on the normalized (minmax, PRIMARY)
    Tier2 vs Tier3 pooled separation, so the ~1.95x separation reported in
    E4 is checked against draw-to-draw noise, not asserted from one draw.
    Uses a precompute-once (generator, memory_stats, dropped_stats loaded
    once per tier/workload, reused across all 30 draws) pattern for
    tractable runtime - generate_postprocessed_traces() itself reloads
    the whole real dataset from disk on every call, which is far too slow
    for a 30-draw sweep; the lower-level generate_raw_trace() +
    postprocess_trace() primitives (imported unmodified from
    postprocess_s36{,_tier2,_tier3}.py) are composed here instead. No
    reimplementation of the actual generation/post-processing math.

4c. Normalized pooled W (minmax) with and without pod_memory_bytes, using
    the same single seed=42 generation gathered for 4a, for consistency
    with the composite_6 VR convention.

Run from repo root:
    PYTHONPATH=$(pwd)/scripts/utils python3 scripts/analysis/wasserstein_reconciliation.py
"""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wasserstein_distance

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "scripts" / "utils",
          REPO_ROOT / "scripts" / "phase4",
          REPO_ROOT / "scripts" / "phase4" / "evaluation"):
    sys.path.insert(0, str(p))

WORKLOADS = ["bert", "gpt2", "resnet152", "whisper", "yolo"]
SEED = 42
K_GEN = 30

TIER_CONFIG = {
    "a16": {"pp": "postprocess_s36", "data_dir": "data/processed/phase4/unified", "replicas": list(range(1, 11))},
    "tier3": {"pp": "postprocess_s36_tier3", "data_dir": "data/processed/tier3/unified", "replicas": list(range(1, 11))},
    "tier2": {"pp": "postprocess_s36_tier2", "data_dir": "data/processed/tier2/unified", "replicas": list(range(1, 8))},
}
TIER_ORDER = ["a16", "tier3", "tier2"]

COMMITTED_REFERENCE = {  # outputs/phase4/validation/s36{,_tier2,_tier3}/validation_report.json, tier mean of mean_wasserstein_all
    "a16": 3.1928, "tier3": 16.5622, "tier2": 16.9319,
}

RAW_SCALE = {"pod_memory_bytes": 1e6}

OUT_DIR = REPO_ROOT / "outputs" / "analysis" / "wasserstein"
OUT_JSON = OUT_DIR / "wasserstein_reconciliation.json"
OUT_MD = OUT_DIR / "wasserstein_reconciliation.md"


def load_pp(tier):
    return importlib.import_module(TIER_CONFIG[tier]["pp"])


def gen_postprocessed(pp, generator, workload, replica_count, n_samples, norm_params, memory_stats, dropped_stats):
    traces = []
    for _ in range(n_samples):
        raw = pp.generate_raw_trace(generator, replica_count, device="cpu")
        processed = pp.postprocess_trace(raw, workload, replica_count, norm_params, memory_stats, dropped_stats, blend_window=20)
        traces.append(processed)
    return np.array(traces)  # (n_samples, 715, 10)


def gather_real(pp, cfg, workload):
    data = pp.load_real_data(cfg["data_dir"])
    wid = pp.WORKLOADS.index(workload)
    wl = data["workload_ids"]
    rc = data["replica_counts"]
    norm_params = pp.load_normalization_params(workload, cfg["data_dir"])
    real_by_r = {}
    for r in cfg["replicas"]:
        mask = (wl == wid) & (rc == r)
        n = int(mask.sum())
        if n == 0:
            raise RuntimeError(f"{workload}: no real pods at r={r}")
        norm7 = data["traces"][mask][:, :, pp.TRAINED_INDICES]
        denorm7 = np.zeros_like(norm7, dtype=np.float64)
        for i, name in enumerate(pp.TRAINED_NAMES):
            p = norm_params[name]
            denorm7[:, :, i] = norm7[:, :, i] * (p["max"] - p["min"]) + p["min"]
        real_by_r[r] = denorm7
    return real_by_r, norm_params


def synth_by_r(pp, generator, workload, cfg, real_by_r, memory_stats, dropped_stats, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    out = {}
    norm_params = pp.load_normalization_params(workload, cfg["data_dir"])
    for r in cfg["replicas"]:
        n = real_by_r[r].shape[0]
        full = gen_postprocessed(pp, generator, workload, r, n, norm_params, memory_stats, dropped_stats)
        out[r] = full[:, :, pp.TRAINED_INDICES]  # (n, 715, 7) raw units
    return out


def method_i_pooled(real_by_r, syn_by_r, names):
    """Pool all r into one distribution per metric, one Wasserstein each."""
    real_pool = {m: [] for m in names}
    syn_pool = {m: [] for m in names}
    for r in real_by_r:
        for i, m in enumerate(names):
            real_pool[m].append(real_by_r[r][:, :, i].flatten())
            syn_pool[m].append(syn_by_r[r][:, :, i].flatten())
    raw_w = {}
    for m in names:
        rp = np.concatenate(real_pool[m])
        sp = np.concatenate(syn_pool[m])
        raw_w[m] = float(wasserstein_distance(rp, sp))
    return raw_w


def method_ii_per_r_avg(real_by_r, syn_by_r, names):
    """Per-r Wasserstein per metric, then average over r (validate_s36 style)."""
    per_r_per_metric = {m: [] for m in names}
    for r in real_by_r:
        for i, m in enumerate(names):
            rp = real_by_r[r][:, :, i].flatten()
            sp = syn_by_r[r][:, :, i].flatten()
            per_r_per_metric[m].append(float(wasserstein_distance(rp, sp)))
    raw_w = {m: float(np.mean(vals)) for m, vals in per_r_per_metric.items()}
    return raw_w


def mb_scaled_mean(raw_w, names):
    scaled = [raw_w[m] / RAW_SCALE.get(m, 1.0) for m in names]
    return float(np.mean(scaled))


def minmax_pooled_w(real_by_r, syn_by_r, names, norm_params, exclude=None):
    ws = {}
    for i, m in enumerate(names):
        p = norm_params[m]
        span = p["max"] - p["min"]
        real_all = np.concatenate([real_by_r[r][:, :, i].flatten() for r in real_by_r])
        syn_all = np.concatenate([syn_by_r[r][:, :, i].flatten() for r in syn_by_r])
        real_mm = (real_all - p["min"]) / span
        syn_mm = (syn_all - p["min"]) / span
        ws[m] = float(wasserstein_distance(real_mm, syn_mm))
    vals = [v for m, v in ws.items() if m != exclude]
    return float(np.mean(vals)), ws


def try_committed_reproduction(tier):
    """Import and call the ACTUAL committed compute_wasserstein_distances()
    from validate_s36{,_tier2,_tier3}.py - true reproduction attempt,
    seeded for determinism (original run was unseeded)."""
    mod_name = {"a16": "validate_s36", "tier2": "validate_s36_tier2", "tier3": "validate_s36_tier3"}[tier]
    try:
        vs = importlib.import_module(mod_name)
    except ImportError as e:
        return None, f"could not import {mod_name}: {e}"
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    wd_results = vs.compute_wasserstein_distances(WORKLOADS, n_samples=10, device="cpu")
    all_wd = []
    per_workload = {}
    for wl in WORKLOADS:
        wd_this = [v for r in wd_results[wl] for v in wd_results[wl][r].values()]
        per_workload[wl] = float(np.mean(wd_this))
        all_wd.extend(wd_this)
    tier_mean = float(np.mean(all_wd))
    return {"tier_mean": tier_mean, "per_workload": per_workload}, None


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 4a + 4c: single seed=42 generation, gathered per (tier, workload) ----
    per_cell = {}
    for tier in TIER_ORDER:
        pp = load_pp(tier)
        cfg = TIER_CONFIG[tier]
        print(f"=== {tier} ===", flush=True)
        memory_stats = pp.compute_memory_stats(cfg["data_dir"])
        dropped_stats = pp.compute_dropped_metric_stats(cfg["data_dir"])
        for workload in WORKLOADS:
            generator = pp.load_generator(workload, device="cpu")
            real_by_r, norm_params = gather_real(pp, cfg, workload)
            syn = synth_by_r(pp, generator, workload, cfg, real_by_r, memory_stats, dropped_stats, seed=SEED)

            raw_w_i = method_i_pooled(real_by_r, syn, pp.TRAINED_NAMES)
            raw_w_ii = method_ii_per_r_avg(real_by_r, syn, pp.TRAINED_NAMES)
            mb_i = mb_scaled_mean(raw_w_i, pp.TRAINED_NAMES)
            mb_ii = mb_scaled_mean(raw_w_ii, pp.TRAINED_NAMES)

            mm_pooled_7, mm_per_metric = minmax_pooled_w(real_by_r, syn, pp.TRAINED_NAMES, norm_params)
            mm_pooled_6, _ = minmax_pooled_w(real_by_r, syn, pp.TRAINED_NAMES, norm_params, exclude="pod_memory_bytes")

            per_cell[(tier, workload)] = {
                "method_i_pooled_raw_MBconv": mb_i,
                "method_ii_per_r_avg_raw_MBconv": mb_ii,
                "minmax_pooled_w_7": mm_pooled_7,
                "minmax_pooled_w_6_excl_memory": mm_pooled_6,
            }
            print(f"  {workload}: method_i={mb_i:.4f} method_ii={mb_ii:.4f} "
                  f"minmax7={mm_pooled_7:.4f} minmax6={mm_pooled_6:.4f}", flush=True)

    tier_method_i = {t: float(np.mean([per_cell[(t, w)]["method_i_pooled_raw_MBconv"] for w in WORKLOADS])) for t in TIER_ORDER}
    tier_method_ii = {t: float(np.mean([per_cell[(t, w)]["method_ii_per_r_avg_raw_MBconv"] for w in WORKLOADS])) for t in TIER_ORDER}
    tier_mm7 = {t: float(np.mean([per_cell[(t, w)]["minmax_pooled_w_7"] for w in WORKLOADS])) for t in TIER_ORDER}
    tier_mm6 = {t: float(np.mean([per_cell[(t, w)]["minmax_pooled_w_6_excl_memory"] for w in WORKLOADS])) for t in TIER_ORDER}

    def pct_gap(x, ref):
        return (x - ref) / ref * 100 if ref else float("nan")

    reconciliation = {}
    for t in TIER_ORDER:
        reconciliation[t] = {
            "committed_reference": COMMITTED_REFERENCE[t],
            "method_i_pooled": tier_method_i[t],
            "method_i_pct_gap": pct_gap(tier_method_i[t], COMMITTED_REFERENCE[t]),
            "method_ii_per_r_avg": tier_method_ii[t],
            "method_ii_pct_gap": pct_gap(tier_method_ii[t], COMMITTED_REFERENCE[t]),
        }
    print("\n4a reconciliation (held-constant setup, aggregation-only difference):")
    print(json.dumps(reconciliation, indent=2))

    # Decide if method (ii) [held-constant] reproduces (within 15% treated as "reproduces")
    TOL = 0.15
    ii_reproduces = all(abs(reconciliation[t]["method_ii_pct_gap"]) / 100 <= TOL for t in TIER_ORDER)

    committed_repro = {}
    if not ii_reproduces:
        print("\nmethod (ii) with held-constant setup does NOT reproduce within 15% - "
              "calling the ACTUAL committed compute_wasserstein_distances() to isolate "
              "whether r-subsetting/n_samples/clamping explain the remainder...")
        for t in TIER_ORDER:
            res, err = try_committed_reproduction(t)
            if err:
                committed_repro[t] = {"error": err}
            else:
                committed_repro[t] = res
                committed_repro[t]["pct_gap_vs_committed"] = pct_gap(res["tier_mean"], COMMITTED_REFERENCE[t])
            print(f"  {t}: {committed_repro[t]}")

    # ---- 4b: 30-draw generation variance on the minmax Tier2/Tier3 separation ----
    print("\n=== 4b: 30-draw generation variance (Tier2 vs Tier3, minmax pooled_7) ===")
    draw_results = {"tier2": [], "tier3": []}
    for tier in ["tier3", "tier2"]:
        pp = load_pp(tier)
        cfg = TIER_CONFIG[tier]
        memory_stats = pp.compute_memory_stats(cfg["data_dir"])
        dropped_stats = pp.compute_dropped_metric_stats(cfg["data_dir"])
        # Precompute per-workload: generator, real_by_r, norm_params (reused across 30 draws)
        precomp = {}
        for workload in WORKLOADS:
            generator = pp.load_generator(workload, device="cpu")
            real_by_r, norm_params = gather_real(pp, cfg, workload)
            precomp[workload] = (generator, real_by_r, norm_params)

        for k in range(K_GEN):
            wl_vals = []
            for workload in WORKLOADS:
                generator, real_by_r, norm_params = precomp[workload]
                syn = synth_by_r(pp, generator, workload, cfg, real_by_r, memory_stats, dropped_stats, seed=k)
                mm7, _ = minmax_pooled_w(real_by_r, syn, pp.TRAINED_NAMES, norm_params)
                wl_vals.append(mm7)
            draw_results[tier].append(float(np.mean(wl_vals)))
        print(f"  {tier}: {K_GEN} draws done, mean={np.mean(draw_results[tier]):.4f} "
              f"sd={np.std(draw_results[tier], ddof=1):.4f}", flush=True)

    tier3_draws = np.array(draw_results["tier3"])
    tier2_draws = np.array(draw_results["tier2"])
    sep_diff = tier2_draws - tier3_draws
    sep_ratio = tier2_draws / tier3_draws
    generation_variance_result = {
        "tier3_mean": float(tier3_draws.mean()), "tier3_sd": float(tier3_draws.std(ddof=1)),
        "tier2_mean": float(tier2_draws.mean()), "tier2_sd": float(tier2_draws.std(ddof=1)),
        "separation_diff_mean": float(sep_diff.mean()), "separation_diff_sd": float(sep_diff.std(ddof=1)),
        "separation_ratio_mean": float(sep_ratio.mean()), "separation_ratio_sd": float(sep_ratio.std(ddof=1)),
    }
    print(json.dumps(generation_variance_result, indent=2))

    # ---- Write outputs ----
    out = {
        "task_4a_reconciliation": reconciliation,
        "task_4a_ii_reproduces_within_15pct": ii_reproduces,
        "task_4a_committed_function_reproduction": committed_repro,
        "task_4b_generation_variance_minmax7_separation": generation_variance_result,
        "task_4c_per_cell_minmax_with_without_memory": {
            f"{t}/{w}": per_cell[(t, w)] for t in TIER_ORDER for w in WORKLOADS
        },
        "task_4c_tier_means": {"minmax7_with_memory": tier_mm7, "minmax6_excl_memory": tier_mm6},
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))

    lines = ["# Batch A Task 4 - E4 reconciliation, generation variance, exclusion\n"]
    lines.append("## 4a - raw pooled Wasserstein reconciliation (held-constant setup)\n")
    lines.append("| tier | committed ref | method (i) pooled | i % gap | method (ii) per-r avg | ii % gap |")
    lines.append("|---|---|---|---|---|---|")
    for t in TIER_ORDER:
        r = reconciliation[t]
        lines.append(f"| {t} | {r['committed_reference']:.4f} | {r['method_i_pooled']:.4f} | "
                      f"{r['method_i_pct_gap']:+.1f}% | {r['method_ii_per_r_avg']:.4f} | "
                      f"{r['method_ii_pct_gap']:+.1f}% |")
    lines.append(f"\nmethod (ii) reproduces committed figures within 15%: **{ii_reproduces}**\n")
    if not ii_reproduces:
        lines.append("### True committed-function reproduction attempt\n")
        for t in TIER_ORDER:
            cr = committed_repro.get(t, {})
            lines.append(f"- {t}: {json.dumps(cr)}")
        lines.append("")

    lines.append("\n## 4b - 30-draw generation variance, minmax pooled_7 separation (Tier2 vs Tier3)\n")
    lines.append(f"- Tier3: mean={generation_variance_result['tier3_mean']:.4f} "
                  f"sd={generation_variance_result['tier3_sd']:.4f}")
    lines.append(f"- Tier2: mean={generation_variance_result['tier2_mean']:.4f} "
                  f"sd={generation_variance_result['tier2_sd']:.4f}")
    lines.append(f"- Separation (diff, tier2-tier3): mean={generation_variance_result['separation_diff_mean']:.4f} "
                  f"sd={generation_variance_result['separation_diff_sd']:.4f}")
    lines.append(f"- Separation (ratio, tier2/tier3): mean={generation_variance_result['separation_ratio_mean']:.4f} "
                  f"sd={generation_variance_result['separation_ratio_sd']:.4f}\n")

    lines.append("## 4c - normalized pooled W (minmax) with vs without pod_memory_bytes\n")
    lines.append("| tier | mean minmax7 (with memory) | mean minmax6 (excl memory) |")
    lines.append("|---|---|---|")
    for t in TIER_ORDER:
        lines.append(f"| {t} | {tier_mm7[t]:.4f} | {tier_mm6[t]:.4f} |")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
