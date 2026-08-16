"""
load_experiment.py — canonical loader for Phase 1 v3, Tier 1, Tier 2, and Tier 3 data.

Normalises four data collection tiers into one canonical DataFrame schema so
downstream analysis (S36 retrain, fidelity scoring, thesis figures, extension
paper plots) reads a single interface regardless of which tier collected the
data.

Tiers:
    phase1_v3 : A16 whole-GPU, time-sliced. 5 workloads x r=1..10. 50 experiments.
    tier1     : H100 whole-GPU. 5 workloads x r=1. 5 experiments.
    tier2     : H100 MIG 1g.12gb, 7 slices. 5 workloads x r=1..7. 35 experiments.
    tier3     : H100 whole-GPU, time-sliced 10 replicas. 5 workloads x r=1..10.
                50 experiments. Same 22-metric surface as tier1 (no per-slice
                or GPM extras) — collected with run_experiment_v3.py, not v4.

Canonical schema per metric family:
    app_* / node_*       : timestamp, value
    pod_*                : timestamp, value, pod
    aggregated gpu_*     : timestamp, value
    per-slice gpu_*      : timestamp, value, gpu_i_id, pod, namespace  (tier2 only)

Cross-tier aliasing rules (see TIER2_NOTES.md section 8):
  1. `application` (phase1_v3) and `app` (tier1/tier2) are both dropped; workload
     identity is passed in explicitly by the caller.
  2. Extra GPU CSV columns present in tier1/tier2 (`container`, `namespace`,
     `pci_bus_id`, `pod` in the aggregated GPU header) are dropped in the
     canonical view. Raw labels remain accessible via df.attrs['raw_labels'].
  3. `gpu_utilization` numeric range is preserved as-collected. Phase 1 v3 and
     Tier 1 read 0-100 (percent). Tier 2 reads 0-700 at r=7 fully busy (sum of
     per-slice GR_ENGINE_ACTIVE * 100). No auto-rescaling. Caller is expected
     to be aware of the semantic difference; see TIER2_NOTES.md section 7.

Author: Hamidreza Fathollahzadeh
"""

from __future__ import annotations

import glob
import os
import warnings
from pathlib import Path
from typing import Literal

import pandas as pd

Tier = Literal["phase1_v3", "tier1", "tier2", "tier3"]

# Metric families and where they live per tier.
# Values are the metric-name stems used in the CSV filenames (between
# "<workload>_r<n>_" and "_<timestamp>.csv").
_TIER1_METRICS = {
    "app": [
        "app_latency_p50", "app_latency_p95", "app_latency_p99", "app_throughput",
    ],
    "node": [
        "node_cpu_usage",
        "node_memory_available_bytes", "node_memory_used_percent",
        "node_psi_cpu", "node_psi_io", "node_psi_memory",
    ],
    "pod": [
        "pod_cpu_usage", "pod_memory_bytes",
        "pod_latency_avg", "pod_throughput",
        "pod_psi_cpu", "pod_psi_io", "pod_psi_memory",
    ],
    "gpu_agg": [
        "gpu_utilization", "gpu_memory_used", "gpu_memory_total",
        "gpu_power_watts", "gpu_temperature",
    ],
    "gpu_slice": [],
    "gpu_agg_new": [],
    "gpu_slice_new": [],
}

# Phase 1 v3 has the same 22-metric surface as Tier 1 (no per-slice or GPM extras).
_PHASE1_V3_METRICS = {
    "app": list(_TIER1_METRICS["app"]),
    "node": list(_TIER1_METRICS["node"]),
    "pod": list(_TIER1_METRICS["pod"]),
    "gpu_agg": list(_TIER1_METRICS["gpu_agg"]),
    "gpu_slice": [],
    "gpu_agg_new": [],
    "gpu_slice_new": [],
}

_TIER2_METRICS = {
    "app": list(_TIER1_METRICS["app"]),
    "node": list(_TIER1_METRICS["node"]),
    "pod": list(_TIER1_METRICS["pod"]),
    "gpu_agg": list(_TIER1_METRICS["gpu_agg"]),
    "gpu_slice": [
        "gpu_utilization_per_slice", "gpu_memory_used_per_slice",
        "gpu_memory_total_per_slice", "gpu_power_watts_per_slice",
        "gpu_temperature_per_slice",
    ],
    "gpu_agg_new": [
        "gpu_dram_active", "gpu_pipe_tensor_active",
        "gpu_total_energy_consumption",
    ],
    "gpu_slice_new": [
        "gpu_dram_active_per_slice", "gpu_pipe_tensor_active_per_slice",
        "gpu_total_energy_consumption_per_slice",
    ],
}

# Tier 3 has the same 22-metric surface as Tier 1 (no per-slice or GPM
# extras) — whole-GPU, no MIG, collected with run_experiment_v3.py.
_TIER3_METRICS = {
    "app": list(_TIER1_METRICS["app"]),
    "node": list(_TIER1_METRICS["node"]),
    "pod": list(_TIER1_METRICS["pod"]),
    "gpu_agg": list(_TIER1_METRICS["gpu_agg"]),
    "gpu_slice": [],
    "gpu_agg_new": [],
    "gpu_slice_new": [],
}

_TIER_METRICS: dict[Tier, dict[str, list[str]]] = {
    "phase1_v3": _PHASE1_V3_METRICS,
    "tier1": _TIER1_METRICS,
    "tier2": _TIER2_METRICS,
    "tier3": _TIER3_METRICS,
}

# Default directory names under `root` for each tier.
_TIER_DIRS: dict[Tier, str] = {
    "phase1_v3": "phase1_v3",
    "tier1": "extension_tier1",
    "tier2": "extension_tier2",
    "tier3": "extension_tier3",
}

# Columns to drop from raw CSVs before returning canonical DataFrames.
# All values from those columns are preserved in df.attrs['raw_labels'] for
# anyone who needs to introspect (rare).
_AGG_GPU_DROP_COLS = [
    "DCGM_FI_DRIVER_VERSION", "Hostname", "UUID", "__name__",
    "container", "device", "gpu", "instance", "job", "modelName",
    "namespace", "pci_bus_id", "pod", "pci_bus_id",
]

_PER_SLICE_KEEP_COLS = ["timestamp", "value", "gpu_i_id", "pod", "namespace"]

_POD_METRIC_KEEP_COLS = ["timestamp", "value", "pod"]

_APP_NODE_KEEP_COLS = ["timestamp", "value"]

# The workload-label column on pod_latency_avg and pod_throughput went from
# `application` (phase1_v3) to `app` (tier1, tier2). Both dropped from canonical.
_WORKLOAD_LABEL_ALIASES = {"application", "app"}


def list_available_metrics(tier: Tier) -> dict[str, list[str]]:
    """Return the metric family -> metric list mapping for the given tier.

    Useful for callers that need to know what to expect before loading:
        list_available_metrics("tier2")["gpu_slice"]
        -> ["gpu_utilization_per_slice", ...]
    """
    if tier not in _TIER_METRICS:
        raise ValueError(f"unknown tier: {tier!r}. valid: {list(_TIER_METRICS)}")
    return {k: list(v) for k, v in _TIER_METRICS[tier].items()}


def _experiment_dir(root: str, tier: Tier, workload: str, replicas: int) -> Path:
    p = Path(root) / _TIER_DIRS[tier] / f"{workload}_r{replicas}"
    if not p.is_dir():
        raise FileNotFoundError(
            f"experiment directory not found: {p}. "
            f"expected layout: {root}/{_TIER_DIRS[tier]}/<workload>_r<n>/"
        )
    return p


def _find_metric_file(exp_dir: Path, metric_stem: str) -> Path | None:
    """Locate the CSV file matching a metric stem inside an experiment directory.

    Filename convention: <workload>_r<n>_<metric_stem>_<timestamp>.csv
    Returns None if no match, warns if multiple (takes newest by timestamp
    suffix embedded in filename).
    """
    # Match the stem with a trailing underscore then the timestamp suffix.
    # e.g. bert_r1_gpu_utilization_20260812_100844.csv
    #      but NOT bert_r1_gpu_utilization_per_slice_20260812_100844.csv
    pattern = f"*_{metric_stem}_*.csv"
    candidates = sorted(exp_dir.glob(pattern))
    # Filter: filename must contain _<metric_stem>_ followed by 8 digits (date).
    # This disambiguates gpu_utilization from gpu_utilization_per_slice.
    filtered: list[Path] = []
    marker = f"_{metric_stem}_"
    for c in candidates:
        name = c.name
        idx = name.find(marker)
        if idx < 0:
            continue
        # Character right after the marker should be a digit (timestamp start).
        after = name[idx + len(marker):]
        if after[:8].isdigit():
            filtered.append(c)
    if not filtered:
        return None
    if len(filtered) > 1:
        # Take the newest by lex sort of timestamp suffix (naming is
        # YYYYMMDD_HHMMSS so lex == chronological).
        warnings.warn(
            f"multiple files matched {metric_stem} in {exp_dir}: "
            f"{[f.name for f in filtered]}. using newest.",
            RuntimeWarning,
        )
        filtered.sort(key=lambda p: p.name)
    return filtered[-1]


def _canonicalise_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp", *[c for c in ("pod", "gpu_i_id") if c in df.columns]])
    df = df.reset_index(drop=True)
    return df


def _load_app_or_node(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns or "value" not in df.columns:
        raise ValueError(f"expected timestamp,value columns in {csv_path}, got {list(df.columns)}")
    keep = [c for c in _APP_NODE_KEEP_COLS if c in df.columns]
    df = df[keep]
    return _canonicalise_timestamp(df)


def _load_pod(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns or "value" not in df.columns:
        raise ValueError(f"expected timestamp,value in {csv_path}, got {list(df.columns)}")
    if "pod" not in df.columns:
        raise ValueError(f"expected pod column in {csv_path}, got {list(df.columns)}")
    # Drop workload label (application/app) — caller passes workload explicitly.
    drop_cols = [c for c in df.columns if c in _WORKLOAD_LABEL_ALIASES]
    df = df.drop(columns=drop_cols)
    keep = [c for c in _POD_METRIC_KEEP_COLS if c in df.columns]
    df = df[keep]
    return _canonicalise_timestamp(df)


def _load_gpu_agg(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns or "value" not in df.columns:
        raise ValueError(f"expected timestamp,value in {csv_path}, got {list(df.columns)}")
    # Preserve everything else in df.attrs for introspection, then drop.
    raw_labels: dict[str, list] = {}
    for c in df.columns:
        if c in ("timestamp", "value"):
            continue
        # Only keep if the column has any non-null unique value worth remembering.
        vals = df[c].dropna().unique()
        if len(vals) > 0:
            raw_labels[c] = list(vals[:5])  # cap to 5 unique values
    canonical = df[["timestamp", "value"]].copy()
    canonical = _canonicalise_timestamp(canonical)
    canonical.attrs["raw_labels"] = raw_labels
    return canonical


def _load_gpu_slice(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"timestamp", "value"}
    if not required.issubset(df.columns):
        raise ValueError(f"expected {required} in {csv_path}, got {list(df.columns)}")
    # DCGM column name in per-slice files is GPU_I_ID (uppercase). Normalise
    # to lowercase gpu_i_id for the canonical schema.
    if "GPU_I_ID" in df.columns:
        df = df.rename(columns={"GPU_I_ID": "gpu_i_id"})
    if "gpu_i_id" not in df.columns:
        raise ValueError(f"expected GPU_I_ID / gpu_i_id in {csv_path}, got {list(df.columns)}")
    for c in ("pod", "namespace"):
        if c not in df.columns:
            df[c] = ""
    keep = [c for c in _PER_SLICE_KEEP_COLS if c in df.columns]
    df = df[keep]
    return _canonicalise_timestamp(df)


def _family_for_metric(metric_stem: str, tier_metrics: dict[str, list[str]]) -> str | None:
    for family, metrics in tier_metrics.items():
        if metric_stem in metrics:
            return family
    return None


def load_experiment(
    tier: Tier,
    workload: str,
    replicas: int,
    root: str = "data/raw",
    include_per_slice: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load a single experiment (one workload at one replica count) from disk.

    Returns a dict keyed by canonical metric name. Value schema depends on
    the metric family (see module docstring).

    Metrics missing from a given tier are simply absent from the returned
    dict — no empty DataFrame placeholders, no exceptions. Use
    list_available_metrics(tier) to know what to expect.

    Parameters
    ----------
    tier : "phase1_v3" | "tier1" | "tier2" | "tier3"
    workload : e.g. "bert", "gpt2", "resnet152", "whisper", "yolo"
    replicas : integer replica count (must match a directory on disk)
    root : root under which the tier directory lives (default "data/raw")
    include_per_slice : if True and tier == "tier2", also load the 8 per-slice
        GPU CSVs. Ignored (with a warning) for other tiers.

    Returns
    -------
    dict[str, pd.DataFrame]

    Raises
    ------
    ValueError : unknown tier or missing timestamp/value columns
    FileNotFoundError : experiment directory does not exist
    """
    if tier not in _TIER_METRICS:
        raise ValueError(f"unknown tier: {tier!r}. valid: {list(_TIER_METRICS)}")

    tier_metrics = _TIER_METRICS[tier]
    exp_dir = _experiment_dir(root, tier, workload, replicas)

    if include_per_slice and tier != "tier2":
        warnings.warn(
            f"include_per_slice=True is only meaningful for tier2, ignoring for tier={tier!r}",
            UserWarning,
        )

    # Build the list of metric stems to load for this tier and options.
    stems: list[str] = []
    stems.extend(tier_metrics["app"])
    stems.extend(tier_metrics["node"])
    stems.extend(tier_metrics["pod"])
    stems.extend(tier_metrics["gpu_agg"])
    stems.extend(tier_metrics["gpu_agg_new"])
    if include_per_slice and tier == "tier2":
        stems.extend(tier_metrics["gpu_slice"])
        stems.extend(tier_metrics["gpu_slice_new"])

    result: dict[str, pd.DataFrame] = {}
    for stem in stems:
        csv = _find_metric_file(exp_dir, stem)
        if csv is None:
            warnings.warn(f"metric {stem!r} not found in {exp_dir}", RuntimeWarning)
            continue
        family = _family_for_metric(stem, tier_metrics)
        if family in ("app", "node"):
            result[stem] = _load_app_or_node(csv)
        elif family == "pod":
            result[stem] = _load_pod(csv)
        elif family in ("gpu_agg", "gpu_agg_new"):
            result[stem] = _load_gpu_agg(csv)
        elif family in ("gpu_slice", "gpu_slice_new"):
            result[stem] = _load_gpu_slice(csv)
        else:
            raise RuntimeError(f"internal: unmapped metric family for {stem}")
    return result


def _self_test() -> None:
    """Minimal smoke test — verifies loader against real data if present.

    Not a full pytest suite. Prints per-tier metric counts and one sample
    DataFrame head so a human can eyeball the output.
    """
    import sys

    for tier, workload, replicas in [
        ("tier1", "bert", 1),
        ("tier2", "bert", 1),
        ("tier2", "whisper", 7),
    ]:
        print(f"\n=== {tier} / {workload} r={replicas} ===")
        try:
            data = load_experiment(tier, workload, replicas, include_per_slice=(tier == "tier2"))
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue
        print(f"  loaded {len(data)} metrics:")
        for name, df in data.items():
            cols = list(df.columns)
            print(f"    {name:<40} rows={len(df):>5} cols={cols}")
        # Print one sample head
        if data:
            first_name, first_df = next(iter(data.items()))
            print(f"\n  head of {first_name}:")
            print(first_df.head(2).to_string(index=False))


if __name__ == "__main__":
    _self_test()