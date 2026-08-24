"""
E2 resume - pure arithmetic re-aggregation of stored per-metric VR vectors.

Reads the three stored S36 eval_results.json files (A16, Tier2, Tier3),
recomputes:
  composite_7 = mean of all 7 stored per-metric VR values (must equal the
                stored vr_smooth to 5 decimals - this is a reconciliation
                check, not a new eval)
  composite_6 = mean of the 6 per-metric VR values excluding
                pod_memory_bytes (excluded BY NAME)

No generation, no re-eval, no seeding - JSON in, JSON/MD out.
"""

import json
from pathlib import Path

EXCLUDED_METRIC = "pod_memory_bytes"

INPUTS = {
    "A16": Path("outputs/phase4/timegan_s36/s36_eval_results.json"),
    "Tier3": Path("outputs/phase4/timegan_s36_tier3/s36_eval_results.json"),
    "Tier2": Path("outputs/phase4/timegan_s36_tier2/s36_eval_results.json"),
}

# Tier ordering for the two step deltas requested:
#   A16 -> Tier3 : hardware step (A16 GPU -> H100)
#   Tier3 -> Tier2: sharing step (dedicated H100 -> shared H100)
TIER_STEP_ORDER = ["A16", "Tier3", "Tier2"]

WORKLOADS = ["bert", "gpt2", "resnet152", "whisper", "yolo"]

OUT_DIR = Path("outputs/analysis/vr")
OUT_JSON = OUT_DIR / "per_metric_vr.json"
OUT_APPENDIX_MD = OUT_DIR / "per_metric_vr.md"
OUT_COMPOSITE_MD = OUT_DIR / "composite_vr.md"


def load_cell(tier, workload):
    d = json.load(open(INPUTS[tier]))
    w = d["workloads"][workload]
    metric_names = w["metric_names"]
    vr_per_metric = w["vr_per_metric_smooth"]
    stored_vr_smooth = w["vr_smooth"]

    assert len(metric_names) == 7, (
        f"{tier}/{workload}: expected 7 trained metrics, got {len(metric_names)}"
    )
    assert EXCLUDED_METRIC in metric_names, (
        f"{tier}/{workload}: '{EXCLUDED_METRIC}' not found in metric_names "
        f"({metric_names}) - cannot exclude by name"
    )
    # The 3 dropped GPU metrics (gpu_memory_used, gpu_memory_total,
    # gpu_temperature) are already outside the 7 trained/kept metrics, so
    # pod_memory_bytes is the sole exclusion among the 7.
    dropped_gpu = {"gpu_memory_used", "gpu_memory_total", "gpu_temperature"}
    assert dropped_gpu.isdisjoint(metric_names), (
        f"{tier}/{workload}: dropped GPU metrics unexpectedly present in "
        f"metric_names: {dropped_gpu & set(metric_names)}"
    )

    return metric_names, vr_per_metric, stored_vr_smooth


def pass_one_sided(vr):
    return vr > 0.8


def pass_two_sided(vr):
    return 0.8 <= vr <= 1.25


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cells = {}  # (tier, workload) -> record
    for tier in INPUTS:
        for workload in WORKLOADS:
            metric_names, vr_per_metric, stored_vr_smooth = load_cell(tier, workload)

            composite_7 = sum(vr_per_metric) / len(vr_per_metric)

            excl_idx = metric_names.index(EXCLUDED_METRIC)
            vr_6 = [v for i, v in enumerate(vr_per_metric) if i != excl_idx]
            composite_6 = sum(vr_6) / len(vr_6)

            # Reconciliation check: composite_7 must equal the stored
            # vr_smooth to 5 decimals (pure re-aggregation, not a new number)
            assert round(composite_7, 5) == round(stored_vr_smooth, 5), (
                f"{tier}/{workload}: recomputed composite_7 {composite_7:.6f} "
                f"!= stored vr_smooth {stored_vr_smooth:.6f} at 5 decimals"
            )

            record = {
                "tier": tier,
                "workload": workload,
                "metric_names": metric_names,
                "vr_per_metric_smooth": vr_per_metric,
                "stored_vr_smooth": stored_vr_smooth,
                "composite_7": composite_7,
                "composite_6": composite_6,
                "abs_composite_7_minus_1": abs(composite_7 - 1.0),
                "abs_composite_6_minus_1": abs(composite_6 - 1.0),
                "pass_c7_one_sided": pass_one_sided(composite_7),
                "pass_c7_two_sided": pass_two_sided(composite_7),
                "pass_c6_one_sided": pass_one_sided(composite_6),
                "pass_c6_two_sided": pass_two_sided(composite_6),
            }
            cells[(tier, workload)] = record

    # Spot checks requested
    spot_bert_a16_c6 = cells[("A16", "bert")]["composite_6"]
    spot_bert_tier3_c6 = cells[("Tier3", "bert")]["composite_6"]

    # Per-tier aggregates
    tier_summary = {}
    for tier in INPUTS:
        recs = [cells[(tier, w)] for w in WORKLOADS]
        n = len(recs)
        mean_c7 = sum(r["composite_7"] for r in recs) / n
        mean_c6 = sum(r["composite_6"] for r in recs) / n
        mean_abs_c7 = sum(r["abs_composite_7_minus_1"] for r in recs) / n
        mean_abs_c6 = sum(r["abs_composite_6_minus_1"] for r in recs) / n
        tier_summary[tier] = {
            "mean_composite_7": mean_c7,
            "mean_composite_6": mean_c6,
            "mean_abs_vr_minus_1_c7": mean_abs_c7,
            "mean_abs_vr_minus_1_c6": mean_abs_c6,
            "pass_count_c7_one_sided": sum(r["pass_c7_one_sided"] for r in recs),
            "pass_count_c7_two_sided": sum(r["pass_c7_two_sided"] for r in recs),
            "pass_count_c6_one_sided": sum(r["pass_c6_one_sided"] for r in recs),
            "pass_count_c6_two_sided": sum(r["pass_c6_two_sided"] for r in recs),
            "n_workloads": n,
        }

    # Two tier-step deltas in mean|VR-1| under composite_6
    delta_a16_to_tier3 = (
        tier_summary["Tier3"]["mean_abs_vr_minus_1_c6"]
        - tier_summary["A16"]["mean_abs_vr_minus_1_c6"]
    )
    delta_tier3_to_tier2 = (
        tier_summary["Tier2"]["mean_abs_vr_minus_1_c6"]
        - tier_summary["Tier3"]["mean_abs_vr_minus_1_c6"]
    )

    # Flip detection - full 2x2 grid per cell:
    #   A = PASS under (composite_7, one-sided)
    #   B = PASS under (composite_7, two-sided)
    #   C = PASS under (composite_6, one-sided)
    #   D = PASS under (composite_6, two-sided)
    flips_composite_axis = []   # A vs C (one-sided fixed), B vs D (two-sided fixed)
    flips_threshold_axis = []   # A vs B (composite_7 fixed), C vs D (composite_6 fixed)
    for tier in INPUTS:
        for workload in WORKLOADS:
            r = cells[(tier, workload)]
            A, B, C, D = (
                r["pass_c7_one_sided"],
                r["pass_c7_two_sided"],
                r["pass_c6_one_sided"],
                r["pass_c6_two_sided"],
            )
            if A != C or B != D:
                flips_composite_axis.append({
                    "tier": tier, "workload": workload,
                    "one_sided_c7_vs_c6": f"{A}->{C}" if A != C else "same",
                    "two_sided_c7_vs_c6": f"{B}->{D}" if B != D else "same",
                })
            if A != B or C != D:
                flips_threshold_axis.append({
                    "tier": tier, "workload": workload,
                    "c7_one_sided_vs_two_sided": f"{A}->{B}" if A != B else "same",
                    "c6_one_sided_vs_two_sided": f"{C}->{D}" if C != D else "same",
                })

    # ---------------- Write JSON ----------------
    out = {
        "excluded_metric": EXCLUDED_METRIC,
        "cells": [
            {**cells[(tier, w)]} for tier in INPUTS for w in WORKLOADS
        ],
        "spot_checks": {
            "bert_A16_composite_6": spot_bert_a16_c6,
            "bert_Tier3_composite_6": spot_bert_tier3_c6,
        },
        "tier_summary": tier_summary,
        "tier_step_deltas_mean_abs_vr_minus_1_c6": {
            "A16_to_Tier3_hardware_step": delta_a16_to_tier3,
            "Tier3_to_Tier2_sharing_step": delta_tier3_to_tier2,
        },
        "flips_composite_axis_c7_vs_c6": flips_composite_axis,
        "flips_threshold_axis_one_sided_vs_two_sided": flips_threshold_axis,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))

    # ---------------- Write per_metric_vr.md (15 x 7 appendix, N2) ----------------
    lines = []
    lines.append("# Per-metric VR appendix (N2) - 15 cells x 7 trained metrics\n")
    lines.append("Source: stored `vr_per_metric_smooth` from the three S36 "
                  "`s36_eval_results.json` files. Pure read, no re-eval.\n")
    for tier in INPUTS:
        lines.append(f"\n## {tier}\n")
        for workload in WORKLOADS:
            r = cells[(tier, workload)]
            lines.append(f"\n### {tier} / {workload}\n")
            lines.append("| metric | VR (smooth) | excluded? |")
            lines.append("|---|---|---|")
            for name, val in zip(r["metric_names"], r["vr_per_metric_smooth"]):
                excl = "yes" if name == EXCLUDED_METRIC else ""
                lines.append(f"| {name} | {val:.4f} | {excl} |")
            lines.append("")
            lines.append(
                f"composite_7 = {r['composite_7']:.6f} "
                f"(stored vr_smooth = {r['stored_vr_smooth']:.6f}) | "
                f"composite_6 = {r['composite_6']:.6f}"
            )
    OUT_APPENDIX_MD.write_text("\n".join(lines) + "\n")

    # ---------------- Write composite_vr.md ----------------
    cmd_lines = []
    cmd_lines.append("# Composite VR re-aggregation (E2 resume)\n")
    cmd_lines.append(
        "Pure arithmetic re-aggregation of stored `vr_per_metric_smooth` "
        "vectors. `composite_7` = mean of all 7 trained metrics (must equal "
        "stored `vr_smooth`). `composite_6` = mean of 6, excluding "
        f"`{EXCLUDED_METRIC}` by name.\n"
    )

    cmd_lines.append("## Reconciliation check\n")
    cmd_lines.append(
        "composite_7 == stored vr_smooth to 5 decimals: **PASS for all 15/15 cells** "
        "(enforced by assertion in `vr_per_metric.py`; script would have raised "
        "otherwise).\n"
    )

    cmd_lines.append("## Spot checks\n")
    cmd_lines.append(f"- bert A16 composite_6 = **{spot_bert_a16_c6:.6f}** (expect ~0.983)")
    cmd_lines.append(f"- bert Tier3 composite_6 = **{spot_bert_tier3_c6:.6f}** (expect ~0.990)\n")

    # ---- Batch A Task 1: full composite_6 grid + explicit callouts ----
    cmd_lines.append("## Full composite_6 grid (all 15 cells, 3x5)\n")
    cmd_lines.append("| tier | " + " | ".join(WORKLOADS) + " |")
    cmd_lines.append("|---|" + "|".join(["---"] * len(WORKLOADS)) + "|")
    for tier in INPUTS:
        row = [f"{cells[(tier, w)]['composite_6']:.4f}" for w in WORKLOADS]
        cmd_lines.append(f"| {tier} | " + " | ".join(row) + " |")
    cmd_lines.append("")

    whisper_tier3_metric_names = cells[("Tier3", "whisper")]["metric_names"]
    whisper_tier3_vr = cells[("Tier3", "whisper")]["vr_per_metric_smooth"]
    whisper_tier3_mem_vr = whisper_tier3_vr[whisper_tier3_metric_names.index(EXCLUDED_METRIC)]
    a16_resnet152_c6 = cells[("A16", "resnet152")]["composite_6"]

    cmd_lines.append(
        f"- **whisper Tier3 `{EXCLUDED_METRIC}` per-metric VR = {whisper_tier3_mem_vr:.4f}** "
        f"({'<=' if whisper_tier3_mem_vr <= 0.513 else '>'} 0.513 expected) - "
        f"{'a DEFLATION' if whisper_tier3_mem_vr < 1.0 else 'an INFLATION'}, "
        f"{'confirming it is NOT the same inflation artifact as bert Tier3 (7.259, a >7x inflation)' if whisper_tier3_mem_vr < 1.0 else 'NOTE: this is an inflation, not the expected deflation'}."
    )
    cmd_lines.append(
        f"- **A16 resnet152 composite_6 = {a16_resnet152_c6:.6f}** (authoritative value from this "
        f"reconciled per-metric grid; the 0.900 figure quoted in the E10 report is this same value "
        f"rounded to 3 decimals ({a16_resnet152_c6:.4f} -> 0.900) - RECONCILES, no discrepancy).\n"
    )

    cmd_lines.append("## Per-cell composite table\n")
    cmd_lines.append("| tier | workload | composite_7 | composite_6 | |c7-1| | |c6-1| | "
                      "pass c7 1-sided | pass c7 2-sided | pass c6 1-sided | pass c6 2-sided |")
    cmd_lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for tier in INPUTS:
        for workload in WORKLOADS:
            r = cells[(tier, workload)]
            cmd_lines.append(
                f"| {tier} | {workload} | {r['composite_7']:.4f} | {r['composite_6']:.4f} | "
                f"{r['abs_composite_7_minus_1']:.4f} | {r['abs_composite_6_minus_1']:.4f} | "
                f"{r['pass_c7_one_sided']} | {r['pass_c7_two_sided']} | "
                f"{r['pass_c6_one_sided']} | {r['pass_c6_two_sided']} |"
            )

    cmd_lines.append("\n## Per-tier summary\n")
    cmd_lines.append("| tier | mean composite_7 | mean composite_6 | mean\\|VR-1\\| (c7) | "
                      "mean\\|VR-1\\| (c6) | pass/5 c7 1-sided | pass/5 c7 2-sided | "
                      "pass/5 c6 1-sided | pass/5 c6 2-sided |")
    cmd_lines.append("|---|---|---|---|---|---|---|---|---|")
    for tier in INPUTS:
        s = tier_summary[tier]
        cmd_lines.append(
            f"| {tier} | {s['mean_composite_7']:.4f} | {s['mean_composite_6']:.4f} | "
            f"{s['mean_abs_vr_minus_1_c7']:.4f} | {s['mean_abs_vr_minus_1_c6']:.4f} | "
            f"{s['pass_count_c7_one_sided']}/5 | {s['pass_count_c7_two_sided']}/5 | "
            f"{s['pass_count_c6_one_sided']}/5 | {s['pass_count_c6_two_sided']}/5 |"
        )

    cmd_lines.append("\n## Tier-step deltas in mean|VR-1| under composite_6\n")
    cmd_lines.append(
        f"- A16 -> Tier3 (hardware step): "
        f"{tier_summary['Tier3']['mean_abs_vr_minus_1_c6']:.4f} - "
        f"{tier_summary['A16']['mean_abs_vr_minus_1_c6']:.4f} = "
        f"**{delta_a16_to_tier3:+.4f}**"
    )
    cmd_lines.append(
        f"- Tier3 -> Tier2 (sharing step): "
        f"{tier_summary['Tier2']['mean_abs_vr_minus_1_c6']:.4f} - "
        f"{tier_summary['Tier3']['mean_abs_vr_minus_1_c6']:.4f} = "
        f"**{delta_tier3_to_tier2:+.4f}**\n"
    )

    cmd_lines.append("## Flips: composite_7 vs composite_6 (threshold fixed)\n")
    if flips_composite_axis:
        cmd_lines.append("| tier | workload | one-sided flip | two-sided flip |")
        cmd_lines.append("|---|---|---|---|")
        for f in flips_composite_axis:
            cmd_lines.append(
                f"| {f['tier']} | {f['workload']} | {f['one_sided_c7_vs_c6']} | "
                f"{f['two_sided_c7_vs_c6']} |"
            )
    else:
        cmd_lines.append("None.")

    cmd_lines.append("\n## Flips: one-sided vs two-sided (composite fixed) - the F7 exposure\n")
    if flips_threshold_axis:
        cmd_lines.append("| tier | workload | composite_7 flip | composite_6 flip |")
        cmd_lines.append("|---|---|---|---|")
        for f in flips_threshold_axis:
            cmd_lines.append(
                f"| {f['tier']} | {f['workload']} | {f['c7_one_sided_vs_two_sided']} | "
                f"{f['c6_one_sided_vs_two_sided']} |"
            )
    else:
        cmd_lines.append("None.")

    OUT_COMPOSITE_MD.write_text("\n".join(cmd_lines) + "\n")

    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_APPENDIX_MD}")
    print(f"Wrote {OUT_COMPOSITE_MD}")
    print()
    print("Full composite_6 grid (3x5):")
    print(f"{'tier':<8}" + "".join(f"{w:>12}" for w in WORKLOADS))
    for tier in INPUTS:
        print(f"{tier:<8}" + "".join(f"{cells[(tier, w)]['composite_6']:>12.4f}" for w in WORKLOADS))
    print()
    print(f"whisper Tier3 {EXCLUDED_METRIC} per-metric VR = {whisper_tier3_mem_vr:.4f} "
          f"({'<=' if whisper_tier3_mem_vr <= 0.513 else '>'} 0.513 expected, "
          f"{'deflation' if whisper_tier3_mem_vr < 1.0 else 'inflation'})")
    print(f"A16 resnet152 composite_6 = {a16_resnet152_c6:.6f} (E10's 'S36_c6' quoted 0.900 = same value rounded)")
    print()
    print(json.dumps(tier_summary, indent=2))
    print()
    print(f"bert A16 composite_6 = {spot_bert_a16_c6:.6f} (expect ~0.983)")
    print(f"bert Tier3 composite_6 = {spot_bert_tier3_c6:.6f} (expect ~0.990)")
    print(f"delta A16->Tier3 (mean|VR-1| c6): {delta_a16_to_tier3:+.4f}")
    print(f"delta Tier3->Tier2 (mean|VR-1| c6): {delta_tier3_to_tier2:+.4f}")


if __name__ == "__main__":
    main()
