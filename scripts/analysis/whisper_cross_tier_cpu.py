#!/usr/bin/env python3
"""E13 - whisper cross-tier CPU pressure / latency descriptive.

Pure raw-CSV descriptive statistics. No model, no VR, no Wasserstein.
Reads through tools/load_experiment.py's canonical schema aliasing only -
never a bare pd.read_csv on a metric CSV - because A16 (phase1_v3) and
Tier 3 raw CSVs do not share a column schema (A16 aggregated GPU CSV is
11-column DCGM_FI_DEV_GPU_UTIL-style, Tier 3's is 14-column; pod_psi_cpu
and pod_latency_avg are the metrics actually used here and are schema-
identical timestamp,value,pod after canonicalisation, but we still go
through the loader for consistency and to get the HALT check for free).

For workload=whisper, r=1..7, tiers {a16, tier2, tier3}:
  - pod_psi_cpu:      mean-over-trace (primary), peak-over-trace (secondary),
                       averaged across pods.
  - pod_latency_avg:  mean-over-trace (primary), peak-over-trace (secondary),
                       averaged across pods.
  - latency ratio:    mean_latency(r) / mean_latency(1)   [mean-ratio]
                       peak_latency(r) / peak_latency(1)  [peak-ratio]
                       both computed within-tier (tier's own r=1 baseline).

HALT CONDITION: if any A16 (phase1_v3) whisper r in 1..7 is missing or
empty, this script stops and reports - no workaround, no silent drop.

Run from repo root:
    python scripts/analysis/whisper_cross_tier_cpu.py
"""

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.load_experiment import load_experiment  # noqa: E402

WORKLOAD = "whisper"
REPLICAS = list(range(1, 8))  # r=1..7 (matched across all three tiers)

# Loader tier literal <- our label
TIERS = {
    "a16": "phase1_v3",
    "tier2": "tier2",
    "tier3": "tier3",
}

METRICS = ["pod_psi_cpu", "pod_latency_avg"]

OUT_DIR = REPO_ROOT / "outputs" / "analysis" / "whisper_cpu"
OUT_JSON = OUT_DIR / "whisper_cross_tier_cpu.json"
OUT_MD = OUT_DIR / "whisper_cross_tier_cpu.md"
OUT_PNG = OUT_DIR / "whisper_cross_tier_cpu.png"
OUT_NOTES_PATCH = OUT_DIR / "tier3_notes_section7_whisper.md"


def per_pod_mean_and_peak(df: pd.DataFrame) -> tuple[float, float]:
    """Mean-over-trace and peak-over-trace, averaged across pods.

    df has columns timestamp, value, pod (canonical pod-family schema).
    """
    by_pod = df.groupby("pod")["value"]
    pod_means = by_pod.mean()
    pod_peaks = by_pod.max()
    return float(pod_means.mean()), float(pod_peaks.mean())


def load_cell(tier_label: str, tier_arg: str, replicas: int):
    """Load one (tier, r) whisper experiment. Returns dict of stats, or
    raises FileNotFoundError / RuntimeError describing what's missing.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data = load_experiment(tier_arg, WORKLOAD, replicas, root="data/raw")

    missing = [m for m in METRICS if m not in data]
    if missing:
        raise RuntimeError(
            f"{tier_label} whisper r={replicas}: metrics missing from loader "
            f"result: {missing} (warnings: {[str(w.message) for w in caught]})"
        )

    stats = {}
    for m in METRICS:
        df = data[m]
        if df.empty:
            raise RuntimeError(
                f"{tier_label} whisper r={replicas}: metric {m!r} loaded but empty"
            )
        mean_v, peak_v = per_pod_mean_and_peak(df)
        stats[m] = {"mean": mean_v, "peak": peak_v, "n_pods": df["pod"].nunique(), "n_rows": len(df)}
    return stats


def halt_check_a16():
    """HALT CONDITION: any A16 whisper r in 1..7 missing or empty -> stop."""
    problems = []
    for r in REPLICAS:
        try:
            load_cell("a16", TIERS["a16"], r)
        except (FileNotFoundError, RuntimeError) as e:
            problems.append(str(e))
    if problems:
        print("=" * 80)
        print("HALT: A16 whisper precondition failed for E13.")
        print("E13 requires matched r=1..7 across all three tiers for the primary claim.")
        print("Problems found:")
        for p in problems:
            print(f"  - {p}")
        print("Not improvising a workaround. Not writing outputs. Stopping.")
        print("=" * 80)
        sys.exit(1)


def main():
    halt_check_a16()  # exits(1) and prints if it fails; falls through if OK

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # cells[tier_label][r] = {pod_psi_cpu: {mean,peak,...}, pod_latency_avg: {...}}
    cells: dict[str, dict[int, dict]] = {t: {} for t in TIERS}
    load_errors: list[str] = []

    for tier_label, tier_arg in TIERS.items():
        for r in REPLICAS:
            try:
                cells[tier_label][r] = load_cell(tier_label, tier_arg, r)
            except (FileNotFoundError, RuntimeError) as e:
                load_errors.append(str(e))

    if load_errors:
        print("=" * 80)
        print("Non-A16 tier load errors (Tier2/Tier3 are NOT covered by the HALT "
              "condition per the task spec, but incomplete matched-r data breaks "
              "the primary cross-tier claim). Reporting, not improvising:")
        for e in load_errors:
            print(f"  - {e}")
        print("=" * 80)

    # ---- Build per-(tier, r) records, and within-tier ratios vs r=1 ----
    records = []
    ratios = {t: {} for t in TIERS}  # tier -> r -> {mean_ratio, peak_ratio}

    for tier_label in TIERS:
        if 1 not in cells[tier_label]:
            continue  # r=1 baseline missing for this tier; ratios undefined
        base_mean = cells[tier_label][1]["pod_latency_avg"]["mean"]
        base_peak = cells[tier_label][1]["pod_latency_avg"]["peak"]
        for r in REPLICAS:
            if r not in cells[tier_label]:
                continue
            c = cells[tier_label][r]
            mean_ratio = c["pod_latency_avg"]["mean"] / base_mean if base_mean else float("nan")
            peak_ratio = c["pod_latency_avg"]["peak"] / base_peak if base_peak else float("nan")
            ratios[tier_label][r] = {"mean_ratio": mean_ratio, "peak_ratio": peak_ratio}
            records.append({
                "tier": tier_label,
                "r": r,
                "pod_psi_cpu_mean": c["pod_psi_cpu"]["mean"],
                "pod_psi_cpu_peak": c["pod_psi_cpu"]["peak"],
                "pod_latency_avg_mean": c["pod_latency_avg"]["mean"],
                "pod_latency_avg_peak": c["pod_latency_avg"]["peak"],
                "latency_mean_ratio_vs_r1": mean_ratio,
                "latency_peak_ratio_vs_r1": peak_ratio,
                "n_pods_psi_cpu": c["pod_psi_cpu"]["n_pods"],
                "n_pods_latency": c["pod_latency_avg"]["n_pods"],
            })

    # ---- COLLAPSE-OR-SURVIVE named fields ----
    def get_ratio(tier_label, r, kind):
        return ratios.get(tier_label, {}).get(r, {}).get(kind)

    tier3_r7_mean_ratio = get_ratio("tier3", 7, "mean_ratio")
    tier2_r7_mean_ratio = get_ratio("tier2", 7, "mean_ratio")
    a16_r7_mean_ratio = get_ratio("a16", 7, "mean_ratio")

    tier3_r7_peak_ratio = get_ratio("tier3", 7, "peak_ratio")
    tier2_r7_peak_ratio = get_ratio("tier2", 7, "peak_ratio")
    a16_r7_peak_ratio = get_ratio("a16", 7, "peak_ratio")

    def fmt(x):
        return "MISSING" if x is None else f"{x:.4f}"

    # ---- Verdicts ----
    def verdict(tier3_val, tier2_val, label):
        if tier3_val is None or tier2_val is None:
            return f"UNDETERMINED ({label}: tier3={fmt(tier3_val)}, tier2={fmt(tier2_val)} - one or both missing)"
        if tier2_val == 0:
            return f"UNDETERMINED ({label}: tier2 ratio is 0)"
        near = abs(tier3_val - tier2_val) <= 0.3 * abs(tier2_val)  # within ~30% of tier2's ratio treated as "near"
        if tier3_val < 2.0 and tier2_val >= 2.5:
            return (f"SURVIVES ({label}): tier3_r7={tier3_val:.3f} < 2.0 while "
                    f"tier2_r7={tier2_val:.3f} (~2.86 expected) - MIG-vs-whole-GPU "
                    f"distinction shows up in {label} ratio; needs a subtler mechanism "
                    f"than raw same-vCPU contention.")
        if near:
            return (f"COLLAPSES ({label}): tier3_r7={tier3_val:.3f} is close to "
                    f"tier2_r7={tier2_val:.3f} - Tier2 and Tier3 share the same 16 "
                    f"devlab vCPUs and produce similar {label} ratios at r=7, so "
                    f"'MIG preserves CPU contention' cannot explain a Tier2-vs-Tier3 "
                    f"difference in {label}.")
        return (f"AMBIGUOUS ({label}): tier3_r7={tier3_val:.3f} vs tier2_r7={tier2_val:.3f} "
                f"- neither clearly near nor clearly < 2.0; does not cleanly match either "
                f"stated collapse or survive criterion.")

    verdict_mean = verdict(tier3_r7_mean_ratio, tier2_r7_mean_ratio, "mean-ratio")
    verdict_peak = verdict(tier3_r7_peak_ratio, tier2_r7_peak_ratio, "peak-ratio")

    # ---- Write JSON ----
    out = {
        "workload": WORKLOAD,
        "replicas": REPLICAS,
        "tiers": TIERS,
        "records": records,
        "collapse_or_survive": {
            "tier3_r7_mean_ratio": tier3_r7_mean_ratio,
            "tier2_r7_mean_ratio": tier2_r7_mean_ratio,
            "a16_r7_mean_ratio": a16_r7_mean_ratio,
            "tier3_r7_peak_ratio": tier3_r7_peak_ratio,
            "tier2_r7_peak_ratio": tier2_r7_peak_ratio,
            "a16_r7_peak_ratio": a16_r7_peak_ratio,
            "verdict_mean_ratio": verdict_mean,
            "verdict_peak_ratio": verdict_peak,
        },
        "load_errors_nonA16": load_errors,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))

    # ---- Write PNG: PSI(r) and latency-ratio(r), one line per tier, mean+peak panels ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    colors = {"a16": "tab:blue", "tier2": "tab:orange", "tier3": "tab:green"}

    ax = axes[0, 0]
    for tier_label in TIERS:
        rs = sorted(cells[tier_label].keys())
        ys = [cells[tier_label][r]["pod_psi_cpu"]["mean"] for r in rs]
        ax.plot(rs, ys, marker="o", label=tier_label, color=colors[tier_label])
    ax.set_title("pod_psi_cpu - mean-over-trace")
    ax.set_xlabel("replicas (r)")
    ax.set_ylabel("PSI CPU (mean)")
    ax.legend()

    ax = axes[0, 1]
    for tier_label in TIERS:
        rs = sorted(cells[tier_label].keys())
        ys = [cells[tier_label][r]["pod_psi_cpu"]["peak"] for r in rs]
        ax.plot(rs, ys, marker="o", label=tier_label, color=colors[tier_label])
    ax.set_title("pod_psi_cpu - peak-over-trace")
    ax.set_xlabel("replicas (r)")
    ax.set_ylabel("PSI CPU (peak)")
    ax.legend()

    ax = axes[1, 0]
    for tier_label in TIERS:
        rs = sorted(ratios[tier_label].keys())
        ys = [ratios[tier_label][r]["mean_ratio"] for r in rs]
        ax.plot(rs, ys, marker="o", label=tier_label, color=colors[tier_label])
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("latency mean-ratio(r) vs r=1")
    ax.set_xlabel("replicas (r)")
    ax.set_ylabel("mean_latency(r) / mean_latency(1)")
    ax.legend()

    ax = axes[1, 1]
    for tier_label in TIERS:
        rs = sorted(ratios[tier_label].keys())
        ys = [ratios[tier_label][r]["peak_ratio"] for r in rs]
        ax.plot(rs, ys, marker="o", label=tier_label, color=colors[tier_label])
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("latency peak-ratio(r) vs r=1")
    ax.set_xlabel("replicas (r)")
    ax.set_ylabel("peak_latency(r) / peak_latency(1)")
    ax.legend()

    fig.suptitle("E13: whisper cross-tier pod_psi_cpu and latency ratio, r=1..7")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)

    # ---- Write MD ----
    lines = []
    lines.append("# E13 - whisper cross-tier CPU pressure / latency ratio (raw-CSV descriptive)\n")
    lines.append("No model, no VR, no Wasserstein. Read via `tools/load_experiment.py` "
                  "canonical schema aliasing (never a bare CSV read).\n")

    lines.append("## HALT condition\n")
    lines.append("A16 (phase1_v3) whisper r=1..7: all present and non-empty for "
                  "`pod_psi_cpu` and `pod_latency_avg`. HALT condition not triggered.\n")

    if load_errors:
        lines.append("## Non-A16 load errors\n")
        for e in load_errors:
            lines.append(f"- {e}")
        lines.append("")

    lines.append("## Per-(tier, r) table\n")
    lines.append("| tier | r | psi_cpu mean | psi_cpu peak | latency mean | latency peak | "
                  "mean-ratio vs r1 | peak-ratio vs r1 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for rec in records:
        lines.append(
            f"| {rec['tier']} | {rec['r']} | {rec['pod_psi_cpu_mean']:.5f} | "
            f"{rec['pod_psi_cpu_peak']:.5f} | {rec['pod_latency_avg_mean']:.5f} | "
            f"{rec['pod_latency_avg_peak']:.5f} | {rec['latency_mean_ratio_vs_r1']:.4f} | "
            f"{rec['latency_peak_ratio_vs_r1']:.4f} |"
        )

    lines.append("\n## Collapse-or-survive test (r=7)\n")
    lines.append(f"- tier3_r7_mean_ratio = **{fmt(tier3_r7_mean_ratio)}**")
    lines.append(f"- tier2_r7_mean_ratio = **{fmt(tier2_r7_mean_ratio)}**")
    lines.append(f"- a16_r7_mean_ratio = **{fmt(a16_r7_mean_ratio)}**")
    lines.append(f"- tier3_r7_peak_ratio = **{fmt(tier3_r7_peak_ratio)}**")
    lines.append(f"- tier2_r7_peak_ratio = **{fmt(tier2_r7_peak_ratio)}**")
    lines.append(f"- a16_r7_peak_ratio = **{fmt(a16_r7_peak_ratio)}**\n")

    lines.append(
        "Tier 2 and Tier 3 both ran on devlab with the same 16 vCPUs. If Tier 3's "
        "r=7 ratio is near Tier 2's (~2.86), the 'MIG preserves CPU contention' "
        "mechanism cannot hold and the story collapses. If Tier 3's ratio is "
        "meaningfully lower (say < 2.0 while Tier 2 ~2.86), the story survives but "
        "needs a subtler mechanism than raw same-vCPU contention.\n"
    )
    lines.append(f"**Mean-ratio verdict:** {verdict_mean}\n")
    lines.append(f"**Peak-ratio verdict:** {verdict_peak}\n")
    lines.append(
        "If the effect shows in mean but not peak (or vice versa), that is itself "
        "diagnostic - it would suggest the mechanism is about sustained contention "
        "(mean) rather than transient spikes (peak), or the reverse.\n"
    )

    lines.append("![whisper cross-tier CPU/latency](whisper_cross_tier_cpu.png)\n")

    OUT_MD.write_text("\n".join(lines) + "\n")

    # ---- Local draft patch for TIER3_NOTES.md section 7 (NOT applied in place) ----
    notes_lines = []
    notes_lines.append("<!-- DRAFT PATCH - local only, NOT applied to TIER3_NOTES.md -->")
    notes_lines.append("<!-- Proposed addition to TIER3_NOTES.md, Section 7, whisper row -->\n")
    notes_lines.append("| workload | r=7 latency mean-ratio (A16) | r=7 latency mean-ratio (Tier2) | "
                        "r=7 latency mean-ratio (Tier3) | r=7 latency peak-ratio (Tier2) | "
                        "r=7 latency peak-ratio (Tier3) | verdict (mean) | verdict (peak) |")
    notes_lines.append("|---|---|---|---|---|---|---|---|")
    notes_lines.append(
        f"| whisper | {fmt(a16_r7_mean_ratio)} | {fmt(tier2_r7_mean_ratio)} | "
        f"{fmt(tier3_r7_mean_ratio)} | {fmt(tier2_r7_peak_ratio)} | {fmt(tier3_r7_peak_ratio)} | "
        f"{verdict_mean.split(':')[0]} | {verdict_peak.split(':')[0]} |"
    )
    notes_lines.append("")
    notes_lines.append(f"Full detail: {verdict_mean}")
    notes_lines.append("")
    notes_lines.append(f"Full detail: {verdict_peak}")
    notes_lines.append("")
    notes_lines.append(
        "Source: `scripts/analysis/whisper_cross_tier_cpu.py` -> "
        "`outputs/analysis/whisper_cpu/whisper_cross_tier_cpu.{json,md,png}` (E13)."
    )
    OUT_NOTES_PATCH.write_text("\n".join(notes_lines) + "\n")

    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_NOTES_PATCH}")
    print()
    print("tier3_r7_mean_ratio =", fmt(tier3_r7_mean_ratio))
    print("tier2_r7_mean_ratio =", fmt(tier2_r7_mean_ratio))
    print("a16_r7_mean_ratio   =", fmt(a16_r7_mean_ratio))
    print("tier3_r7_peak_ratio =", fmt(tier3_r7_peak_ratio))
    print("tier2_r7_peak_ratio =", fmt(tier2_r7_peak_ratio))
    print("a16_r7_peak_ratio   =", fmt(a16_r7_peak_ratio))
    print()
    print("MEAN-RATIO VERDICT:", verdict_mean)
    print("PEAK-RATIO VERDICT:", verdict_peak)


if __name__ == "__main__":
    main()
