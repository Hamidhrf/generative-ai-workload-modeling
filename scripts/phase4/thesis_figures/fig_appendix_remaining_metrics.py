"""
Appendix: Remaining Metrics Scaling Curves
===========================================
Generates figures/appendix_remaining_metrics.pdf

4 rows x 5 cols:
  Row 0: CPU PSI (pod_psi_cpu)
  Row 1: Throughput (req/s)
  Row 2: Pod Memory (MB)
  Row 3: GPU Power (W)

  Cols: BERT, GPT-2, ResNet-152, Whisper, YOLO

Matches the visual style of fig_scaling_curves.py exactly.

Run from project root:
    conda activate tracegen
    cd ~/generative-ai-workload-modeling
    python scripts/phase4/thesis_figures/fig_appendix_remaining_metrics.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

os.makedirs('figures', exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

# === Constants (identical to fig_scaling_curves.py) ===

WORKLOADS  = ['bert', 'gpt2', 'resnet152', 'whisper', 'yolo']
WL_LABELS  = ['BERT', 'GPT-2', 'ResNet-152', 'Whisper', 'YOLO']
WL_ID      = {wl: i for i, wl in enumerate(WORKLOADS)}

TRAINED_NAMES   = [
    'pod_cpu_usage', 'pod_memory_bytes', 'pod_psi_cpu',
    'pod_latency_avg', 'pod_throughput',
    'gpu_utilization', 'gpu_power_watts',
]
TRAINED_INDICES = [0, 1, 2, 3, 4, 5, 8]

# 10-metric array indices
IDX_CPU  = 0
IDX_MEM  = 1
IDX_PSI  = 2
IDX_LAT  = 3
IDX_TPUT = 4
IDX_GPU  = 5
IDX_PWR  = 8

REPLICA_COUNTS_PER_WL = {
    'bert':      [1, 2, 3, 5, 6, 8, 10],
    'gpt2':      [1, 2, 3, 5, 6, 8, 10],
    'resnet152': [1, 2, 3, 5, 6, 8, 10],
    'whisper':   [1, 2, 3, 5, 6, 8, 10],
    'yolo':      [1, 2, 3, 5, 6, 8, 10],
}

C_REAL  = '#1565C0'
C_SYNTH = '#E65100'


# === Data loading (identical helpers to fig_scaling_curves.py) ===

def load_combined_data(data_dir='data/processed/phase4/unified'):
    return np.load(os.path.join(data_dir, 'combined_dataset.npz'), allow_pickle=True)


def load_norm_params(workload, data_dir='data/processed/phase4/unified'):
    with open(os.path.join(data_dir, 'combined_normalization.json')) as f:
        norm = json.load(f)
    return norm[workload]['params']


def get_real_traces_all_r(workload, metric_idx):
    """Return (replicas, means, stds) for a workload/metric across all r."""
    data = load_combined_data()
    norm_params = load_norm_params(workload)
    wid = WL_ID[workload]

    replicas = REPLICA_COUNTS_PER_WL[workload]
    means, stds = [], []

    for r in replicas:
        mask = (data['workload_ids'] == wid) & (data['replica_counts'] == r)
        matched = data['traces'][mask]

        if metric_idx in TRAINED_INDICES:
            mname = TRAINED_NAMES[TRAINED_INDICES.index(metric_idx)]
            p = norm_params[mname]
            vals = matched[:, :, metric_idx] * (p['max'] - p['min']) + p['min']
        else:
            vals = matched[:, :, metric_idx]

        pod_means = vals.mean(axis=1)
        means.append(float(pod_means.mean()))
        stds.append(float(pod_means.std()))

    return replicas, means, stds


def load_postprocessed(workload, replica_count):
    """Load S36 postprocessed trace. Returns (715, 10) in real units."""
    path = (f'outputs/phase4/postprocessed/s36/'
            f'{workload}_r{replica_count}_s36_postprocessed.npy')
    return np.load(path)[0]


def synth_mean_at_r(wl, midx, r, available=(1, 5, 10)):
    """Get synthetic mean at replica count r, interpolating if needed."""
    if r in available:
        return float(load_postprocessed(wl, r)[:, midx].mean())
    avail_sorted = sorted(available)
    lo = max((rv for rv in avail_sorted if rv <= r), default=avail_sorted[0])
    hi = min((rv for rv in avail_sorted if rv >= r), default=avail_sorted[-1])
    if lo == hi:
        return float(load_postprocessed(wl, lo)[:, midx].mean())
    frac = (r - lo) / (hi - lo)
    v_lo = float(load_postprocessed(wl, lo)[:, midx].mean())
    v_hi = float(load_postprocessed(wl, hi)[:, midx].mean())
    return v_lo + frac * (v_hi - v_lo)


# === Figure ===

def fig_appendix_remaining_metrics():
    # (metric_idx, ylabel, convert_to_mb)
    # pod_memory_bytes is stored in bytes; divide by 1e6 for MB display
    # (metric_idx, ylabel, convert_to_mb, y_formatter)
    # PSI is a dimensionless fraction [0, 1]. Most workloads sit near zero
    # so we use scientific notation to avoid crowded tick labels.
    # Whisper correctly reaches ~0.6 at r=10 (PSI=0.63 documented).
    metrics = [
        (IDX_PSI,  'CPU PSI (fraction)', False, 'sci'),
        (IDX_TPUT, 'Throughput (req/s)', False, None),
        (IDX_MEM,  'Pod Memory (MB)',    True,  None),
        (IDX_PWR,  'GPU Power (W)',      False, None),
    ]

    fig, axes = plt.subplots(4, 5, figsize=(14, 9), sharex=False,
                             gridspec_kw={'hspace': 0.6, 'wspace': 0.4})

    for col, (wl, wl_label) in enumerate(zip(WORKLOADS, WL_LABELS)):
        replicas = REPLICA_COUNTS_PER_WL[wl]

        for row, (midx, ylabel, to_mb, y_fmt) in enumerate(metrics):
            _, rm, rs = get_real_traces_all_r(wl, midx)
            sm = [synth_mean_at_r(wl, midx, r) for r in replicas]

            if to_mb:
                rm = [m / 1e6 for m in rm]
                rs = [s / 1e6 for s in rs]
                sm = [m / 1e6 for m in sm]

            ax = axes[row, col]

            ax.plot(replicas, rm, 'o-', color=C_REAL, markersize=3.5,
                    linewidth=1.4, zorder=3)
            ax.fill_between(replicas,
                            [m - s for m, s in zip(rm, rs)],
                            [m + s for m, s in zip(rm, rs)],
                            alpha=0.12, color=C_REAL)
            ax.plot(replicas, sm, 's--', color=C_SYNTH, markersize=3.5,
                    linewidth=1.2, zorder=2)

            ax.set_xticks([r for r in replicas if r in [1, 3, 5, 8, 10]])
            ax.tick_params(labelsize=7)

            # Apply y-axis formatter
            if y_fmt == 'sci':
                ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
                ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            elif y_fmt is not None:
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(y_fmt))
            else:
                ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
                ax.ticklabel_format(axis='y', style='plain', useOffset=False)

            if row == 0:
                ax.set_title(wl_label, fontsize=9, weight='bold')
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=8)
            if row == 3:
                ax.set_xlabel('Replicas', fontsize=8)

    handles = [
        plt.Line2D([0], [0], color=C_REAL, marker='o', linewidth=1.4,
                   markersize=4, label='Real (mean)'),
        mpatches.Patch(color=C_REAL, alpha=0.3, label='Real ($\\pm$1 std)'),
        plt.Line2D([0], [0], color=C_SYNTH, marker='s', linewidth=1.2,
                   linestyle='--', markersize=3.5, label='S36 Synthetic'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('figures/appendix_remaining_metrics.pdf', bbox_inches='tight')
    plt.close()
    print("Saved figures/appendix_remaining_metrics.pdf")


if __name__ == '__main__':
    fig_appendix_remaining_metrics()