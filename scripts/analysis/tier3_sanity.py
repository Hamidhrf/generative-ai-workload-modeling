#!/usr/bin/env python3
"""Tier 3 sanity plots and summary tables.

Loads Tier 3 experiments across all 5 workloads and r=1..10, produces:
  - figures/tier3_sanity/<workload>.png  (2x2 subplot per workload)
  - stdout: compact 5-row summary + full 50-row per-(workload, r) table

Run from repo root:
    python scripts/analysis/tier3_sanity.py

Assumes:
  - Repo layout with tools/load_experiment.py at repo root
  - This script placed at scripts/analysis/tier3_sanity.py
  - Tier 3 raw data present under data/raw/extension_tier3/
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # headless; script produces PNGs, not an interactive window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Repo root = two levels up from this file: scripts/analysis/ -> scripts/ -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.load_experiment import load_experiment  # noqa: E402

try:
    import tabulate  # noqa: F401
    _HAVE_TABULATE = True
except ImportError:
    _HAVE_TABULATE = False

WORKLOADS = ['whisper', 'gpt2', 'resnet152', 'bert', 'yolo']
REPLICAS = list(range(1, 11))
TIER = 'tier3'
FIG_DIR = REPO_ROOT / 'figures' / 'tier3_sanity'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# (canonical_key, plot_label, family)
#   family='system' -> one row per timestamp already
#   family='pod'    -> multi-row per timestamp, aggregate as mean across pods
PLOTS = [
    ('gpu_utilization', 'GPU utilization (%)',    'system'),
    ('pod_cpu_usage',   'Pod CPU usage (cores)',  'pod'),
    ('pod_psi_cpu',     'Pod PSI CPU (avg)',      'pod'),
    ('pod_latency_avg', 'Pod latency avg (ms)',   'pod'),
]

# Short names used in the summary tables
SHORT = {
    'gpu_utilization': 'gpu',
    'pod_cpu_usage':   'cpu',
    'pod_psi_cpu':     'psi_cpu',
    'pod_latency_avg': 'latency',
}


def _minute_axis(ts):
    """Elapsed minutes from t0. Robust to string, datetime, or numeric seconds."""
    s = pd.to_datetime(ts, errors='coerce', utc=True)
    if s.isna().all():
        s_num = pd.to_numeric(ts, errors='coerce')
        return (s_num - s_num.iloc[0]) / 60.0
    return (s - s.iloc[0]).dt.total_seconds() / 60.0


def _series_for_metric(data, metric_key, family):
    """Return (minutes, values) aggregated to one point per timestamp, or (None, None)."""
    if metric_key not in data:
        return None, None
    df = data[metric_key]
    if df is None or len(df) == 0:
        return None, None
    if family == 'system':
        g = df.sort_values('timestamp')
        return _minute_axis(g['timestamp']), g['value'].to_numpy()
    # pod family: mean across pods per timestamp
    g = df.groupby('timestamp', as_index=False)['value'].mean().sort_values('timestamp')
    return _minute_axis(g['timestamp']), g['value'].to_numpy()


def load_all(workload):
    """Load all replicas for a workload. Returns {r: data_dict_or_None}."""
    out = {}
    for r in REPLICAS:
        try:
            out[r] = load_experiment(TIER, workload, r)
        except FileNotFoundError:
            warnings.warn(f'{TIER} {workload} r={r}: not found -- skipping')
            out[r] = None
        except Exception as e:
            warnings.warn(f'{TIER} {workload} r={r}: load failed ({e!r}) -- skipping')
            out[r] = None
    return out


def plot_workload(workload, loaded):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes = axes.flatten()
    cmap = plt.get_cmap('viridis')

    for ax, (key, label, family) in zip(axes, PLOTS):
        plotted_any = False
        for r in REPLICAS:
            data = loaded.get(r)
            if data is None:
                continue
            mins, vals = _series_for_metric(data, key, family)
            if mins is None or len(mins) == 0:
                continue
            color = cmap((r - 1) / (len(REPLICAS) - 1))
            ax.plot(mins, vals, color=color, lw=1.1, label=f'r={r}')
            plotted_any = True
        ax.set_title(label)
        ax.set_xlabel('Elapsed time (min)')
        ax.grid(alpha=0.3)
        if not plotted_any:
            ax.text(0.5, 0.5, f'no data for {key}', ha='center', va='center',
                    transform=ax.transAxes, color='gray')

    # Single legend for the whole figure, on the right
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='center right',
                   bbox_to_anchor=(1.0, 0.5), fontsize=9, title='Replicas')
    fig.suptitle(f'Tier 3 sanity -- {workload}', fontsize=14, y=1.0)
    fig.tight_layout(rect=[0, 0, 0.94, 0.98])

    out = FIG_DIR / f'{workload}.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return out


def per_replica_stats(workload, loaded):
    """Row per (workload, r) with mean and peak for each of the 4 metrics."""
    rows = []
    for r in REPLICAS:
        data = loaded.get(r)
        row = {'workload': workload, 'replicas': r}
        if data is None:
            for short in SHORT.values():
                row[f'mean_{short}'] = np.nan
                row[f'peak_{short}'] = np.nan
            rows.append(row)
            continue
        for key, _label, family in PLOTS:
            short = SHORT[key]
            _, vals = _series_for_metric(data, key, family)
            if vals is None or len(vals) == 0:
                row[f'mean_{short}'] = np.nan
                row[f'peak_{short}'] = np.nan
            else:
                row[f'mean_{short}'] = float(np.nanmean(vals))
                row[f'peak_{short}'] = float(np.nanmax(vals))
        rows.append(row)
    return rows


def compact_summary(full_df):
    """One row per workload: peak-of-peaks and the r at which each peak occurred."""
    out = []
    for w, g in full_df.groupby('workload', sort=False):
        row = {'workload': w}
        for short in ('gpu', 'cpu', 'psi_cpu', 'latency'):
            col = f'peak_{short}'
            if col not in g or g[col].isna().all():
                row[f'peak_{short}'] = np.nan
                row[f'peak_{short}_at_r'] = np.nan
                continue
            idx = g[col].idxmax()
            row[f'peak_{short}'] = g.loc[idx, col]
            row[f'peak_{short}_at_r'] = int(g.loc[idx, 'replicas'])
        out.append(row)
    return pd.DataFrame(out)


def _print_table(df, title):
    print(f'\n## {title}\n', flush=True)
    if _HAVE_TABULATE:
        print(df.to_markdown(index=False), flush=True)
    else:
        # Fallback if tabulate is not installed
        print(df.to_string(index=False), flush=True)


def main():
    all_rows = []
    for w in WORKLOADS:
        print(f'Loading {w}...', flush=True)
        loaded = load_all(w)
        loaded_ok = sum(1 for v in loaded.values() if v is not None)
        print(f'  {loaded_ok}/{len(REPLICAS)} replicas loaded', flush=True)
        path = plot_workload(w, loaded)
        print(f'  saved {path.relative_to(REPO_ROOT)}', flush=True)
        all_rows.extend(per_replica_stats(w, loaded))

    full = pd.DataFrame(all_rows)
    compact = compact_summary(full)

    # Round for readability; keep NaN as NaN
    for c in full.select_dtypes(include='float').columns:
        full[c] = full[c].round(3)
    for c in compact.select_dtypes(include='float').columns:
        compact[c] = compact[c].round(3)

    _print_table(compact, 'Compact summary (peak-of-peaks per workload)')
    _print_table(full, 'Full per-(workload, r) table')


if __name__ == '__main__':
    main()