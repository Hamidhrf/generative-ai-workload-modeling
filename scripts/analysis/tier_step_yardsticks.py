#!/usr/bin/env python3
"""Batch A Task 3 - per-workload floor + floor-corrected tier-step deltas,
read against both yardsticks (E11 generation SD, E10 resampler floor).

Pure arithmetic over already-computed outputs:
  - outputs/analysis/vr/per_metric_vr.json          (E2: S36 headline composite_6 per cell)
  - outputs/analysis/vr/vr_variance.json            (E11: generation-variance SD per tier)
  - outputs/analysis/baselines/naive_baseline_vr.json (E10: real-trace resampler per cell)

No model calls, no re-eval. Nothing committed.
"""
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKLOADS = ["bert", "gpt2", "resnet152", "whisper", "yolo"]
TIER_MAP = {"a16": "A16", "tier3": "Tier3", "tier2": "Tier2"}  # naive_baselines label -> vr_per_metric label
TIER_ORDER = ["a16", "tier3", "tier2"]

OUT_DIR = REPO_ROOT / "outputs" / "analysis" / "vr"
OUT_JSON = OUT_DIR / "tier_step_floor_corrected.json"
OUT_MD = OUT_DIR / "tier_step_floor_corrected.md"


def main():
    vr = json.loads((REPO_ROOT / "outputs/analysis/vr/per_metric_vr.json").read_text())
    gen = json.loads((REPO_ROOT / "outputs/analysis/vr/vr_variance.json").read_text())
    base = json.loads((REPO_ROOT / "outputs/analysis/baselines/naive_baseline_vr.json").read_text())

    s36_abs = {(c["tier"], c["workload"]): c["abs_composite_6_minus_1"] for c in vr["cells"]}
    floor_abs = {}
    n_traces = {}
    for c in base["cells"]:
        floor_abs[(c["tier"], c["workload"])] = abs(c["resampler"]["mean_c6"] - 1.0)
        n_traces[(c["tier"], c["workload"])] = c["n_traces"]

    # per (tier, workload) table
    per_cell = []
    for tier in TIER_ORDER:
        tvr = TIER_MAP[tier]
        for w in WORKLOADS:
            per_cell.append({
                "tier": tier, "workload": w,
                "s36_abs_vr_minus_1_c6": s36_abs[(tvr, w)],
                "resampler_floor_abs_vr_minus_1_c6": floor_abs[(tier, w)],
                "n_traces": n_traces[(tier, w)],
                "floor_corrected": s36_abs[(tvr, w)] - floor_abs[(tier, w)],
            })

    # per-tier floor (mean over workloads of |resampler-1|)
    tier_floor = {}
    for tier in TIER_ORDER:
        vals = [floor_abs[(tier, w)] for w in WORKLOADS]
        tier_floor[tier] = float(np.mean(vals))

    n_by_tier = {"a16": 55, "tier3": 55, "tier2": 28}

    # per-tier S36 headline mean|VR-1| (c6) - from E2 tier_summary
    tier_s36_mean = {t: gen["tier_stats_mean_abs_vr_minus_1_c6"][t]["mean_abs_vr_minus_1_c6"] for t in TIER_ORDER}
    tier_gen_sd = {t: gen["tier_stats_mean_abs_vr_minus_1_c6"][t]["propagated_gen_sd"] for t in TIER_ORDER}

    # Floor-corrected per-tier mean|VR-1|
    tier_corrected = {t: tier_s36_mean[t] - tier_floor[t] for t in TIER_ORDER}

    # Raw (uncorrected) tier-step deltas, both yardsticks
    delta_hw_raw = tier_s36_mean["tier3"] - tier_s36_mean["a16"]
    delta_sh_raw = tier_s36_mean["tier2"] - tier_s36_mean["tier3"]
    delta_hw_gen_sd = float(np.sqrt(tier_gen_sd["a16"] ** 2 + tier_gen_sd["tier3"] ** 2))
    delta_sh_gen_sd = float(np.sqrt(tier_gen_sd["tier3"] ** 2 + tier_gen_sd["tier2"] ** 2))
    delta_hw_floor_yardstick = float(np.sqrt(tier_floor["a16"] ** 2 + tier_floor["tier3"] ** 2))
    delta_sh_floor_yardstick = float(np.sqrt(tier_floor["tier3"] ** 2 + tier_floor["tier2"] ** 2))

    # Floor-corrected tier-step deltas
    delta_hw_corrected = tier_corrected["tier3"] - tier_corrected["a16"]
    delta_sh_corrected = tier_corrected["tier2"] - tier_corrected["tier3"]

    # Does tier2's floor track trace count?
    floor_tracks_n = tier_floor["tier2"] > tier_floor["tier3"] and tier_floor["tier3"] > 0
    floor_ratio = tier_floor["tier2"] / tier_floor["tier3"] if tier_floor["tier3"] else float("nan")
    n_ratio = n_by_tier["tier3"] / n_by_tier["tier2"]  # 55/28 ~ 1.96, i.e. tier2 has ~half the traces

    out = {
        "per_cell": per_cell,
        "tier_floor_mean_abs_vr_minus_1_c6": tier_floor,
        "n_traces_by_tier": n_by_tier,
        "tier_s36_headline_mean_abs_vr_minus_1_c6": tier_s36_mean,
        "tier_generation_sd": tier_gen_sd,
        "tier_floor_corrected_mean_abs_vr_minus_1_c6": tier_corrected,
        "deltas": {
            "hardware_step_A16_to_Tier3": {
                "raw_delta": delta_hw_raw,
                "generation_sd_yardstick": delta_hw_gen_sd,
                "resampler_floor_yardstick": delta_hw_floor_yardstick,
                "outside_generation_yardstick": abs(delta_hw_raw) > delta_hw_gen_sd,
                "outside_floor_yardstick": abs(delta_hw_raw) > delta_hw_floor_yardstick,
                "floor_corrected_delta": delta_hw_corrected,
            },
            "sharing_step_Tier3_to_Tier2": {
                "raw_delta": delta_sh_raw,
                "generation_sd_yardstick": delta_sh_gen_sd,
                "resampler_floor_yardstick": delta_sh_floor_yardstick,
                "outside_generation_yardstick": abs(delta_sh_raw) > delta_sh_gen_sd,
                "outside_floor_yardstick": abs(delta_sh_raw) > delta_sh_floor_yardstick,
                "floor_corrected_delta": delta_sh_corrected,
            },
        },
        "floor_vs_trace_count": {
            "tier2_floor": tier_floor["tier2"], "tier3_floor": tier_floor["tier3"],
            "tier2_worse_and_tracks_n": floor_tracks_n,
            "floor_ratio_tier2_over_tier3": floor_ratio,
            "trace_count_ratio_tier3_over_tier2": n_ratio,
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))

    lines = []
    lines.append("# Batch A Task 3 - per-workload floor + floor-corrected tier-step deltas\n")
    lines.append("## Per (tier, workload): S36|VR-1| vs resampler floor|VR-1|\n")
    lines.append("| tier | workload | n | S36 \\|VR-1\\| (c6) | resampler floor \\|VR-1\\| | floor-corrected |")
    lines.append("|---|---|---|---|---|---|")
    for r in per_cell:
        lines.append(f"| {r['tier']} | {r['workload']} | {r['n_traces']} | "
                      f"{r['s36_abs_vr_minus_1_c6']:.4f} | {r['resampler_floor_abs_vr_minus_1_c6']:.4f} | "
                      f"{r['floor_corrected']:+.4f} |")

    lines.append("\n## Per-tier floor vs trace count\n")
    lines.append(f"- Tier3 floor = {tier_floor['tier3']:.4f} (n=55)")
    lines.append(f"- Tier2 floor = {tier_floor['tier2']:.4f} (n=28)")
    lines.append(f"- Tier2/Tier3 floor ratio = {floor_ratio:.2f}x; trace-count ratio (Tier3/Tier2) = {n_ratio:.2f}x")
    lines.append(f"- Tier2 floor worse and tracks lower trace count: **{floor_tracks_n}**\n")

    lines.append("## Tier-step deltas against both yardsticks\n")
    lines.append("| step | raw delta | generation-SD yardstick | outside gen? | resampler-floor yardstick | outside floor? |")
    lines.append("|---|---|---|---|---|---|")
    hw = out["deltas"]["hardware_step_A16_to_Tier3"]
    sh = out["deltas"]["sharing_step_Tier3_to_Tier2"]
    lines.append(f"| hardware (A16->Tier3) | {hw['raw_delta']:+.4f} | {hw['generation_sd_yardstick']:.4f} | "
                  f"{hw['outside_generation_yardstick']} | {hw['resampler_floor_yardstick']:.4f} | "
                  f"{hw['outside_floor_yardstick']} |")
    lines.append(f"| sharing (Tier3->Tier2) | {sh['raw_delta']:+.4f} | {sh['generation_sd_yardstick']:.4f} | "
                  f"{sh['outside_generation_yardstick']} | {sh['resampler_floor_yardstick']:.4f} | "
                  f"{sh['outside_floor_yardstick']} |")

    lines.append("\n## Floor-corrected tier-step deltas\n")
    lines.append(f"- Floor-corrected mean|VR-1| per tier: A16={tier_corrected['a16']:+.4f}, "
                  f"Tier3={tier_corrected['tier3']:+.4f}, Tier2={tier_corrected['tier2']:+.4f}")
    lines.append(f"- **Corrected hardware step (A16->Tier3): {delta_hw_corrected:+.4f}** "
                  f"(raw was {delta_hw_raw:+.4f})")
    lines.append(f"- **Corrected sharing step (Tier3->Tier2): {delta_sh_corrected:+.4f}** "
                  f"(raw was {delta_sh_raw:+.4f})")

    OUT_MD.write_text("\n".join(lines) + "\n")

    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print()
    print("Per-tier floor (mean|resampler-1|, c6):")
    for t in TIER_ORDER:
        print(f"  {t}: floor={tier_floor[t]:.4f} n={n_by_tier[t]}")
    print(f"Tier2/Tier3 floor ratio={floor_ratio:.2f}x  trace-count ratio (Tier3/Tier2)={n_ratio:.2f}x  "
          f"tracks_n={floor_tracks_n}")
    print()
    print("Raw deltas vs both yardsticks:")
    print(f"  hardware: delta={hw['raw_delta']:+.4f} gen_sd={hw['generation_sd_yardstick']:.4f} "
          f"(outside={hw['outside_generation_yardstick']})  floor={hw['resampler_floor_yardstick']:.4f} "
          f"(outside={hw['outside_floor_yardstick']})")
    print(f"  sharing:  delta={sh['raw_delta']:+.4f} gen_sd={sh['generation_sd_yardstick']:.4f} "
          f"(outside={sh['outside_generation_yardstick']})  floor={sh['resampler_floor_yardstick']:.4f} "
          f"(outside={sh['outside_floor_yardstick']})")
    print()
    print(f"Floor-corrected: hardware={delta_hw_corrected:+.4f} (raw {delta_hw_raw:+.4f}), "
          f"sharing={delta_sh_corrected:+.4f} (raw {delta_sh_raw:+.4f})")


if __name__ == "__main__":
    main()
