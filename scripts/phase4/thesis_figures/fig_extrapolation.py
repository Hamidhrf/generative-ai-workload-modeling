"""
Extrapolation Curves Figure (standalone)
=========================================
Generates figures/extrapolation_curves.pdf

2x2 grid:
  GPT-2: System GPU Utilisation, Average Latency (ms)
  Whisper: CPU Pressure (PSI), Average Latency (ms)

Shows real data (r=1-10), S36 synthetic points (r=1,5,10),
and bounded trend extrapolation to r=50.

Fixes:
  - Trend line visible and prominent (thicker, darker)
  - Trend fitted to real data r=1-10, shown as solid fit + dashed extrapolation
  - "measured/extrapolated" labels repositioned to avoid overlap
  - Latency in ms for consistency with other figures
  - S36 points larger and more visible

Run from project root:
    conda activate tracegen
    cd ~/generative-ai-workload-modeling
    python scripts/phase4/thesis_figures/fig_extrapolation.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.makedirs('figures', exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

# === Constants ===
WORKLOADS = ['bert', 'gpt2', 'resnet152', 'whisper', 'yolo']
WL_ID = {wl: i for i, wl in enumerate(WORKLOADS)}

TRAINED_NAMES = [
    'pod_cpu_usage', 'pod_memory_bytes', 'pod_psi_cpu',
    'pod_latency_avg', 'pod_throughput',
    'gpu_utilization', 'gpu_power_watts',
]
TRAINED_INDICES = [0, 1, 2, 3, 4, 5, 8]

IDX_CPU, IDX_LAT, IDX_GPU, IDX_PSI = 0, 3, 5, 2

REPLICA_COUNTS_PER_WL = {
    'bert':      [1, 2, 3, 5, 6, 8, 10],
    'gpt2':      [1, 2, 3, 5, 6, 8, 10],
    'resnet152': [1, 2, 3, 5, 6, 8, 10],
    'whisper':   [1, 2, 3, 5, 6, 10],
    'yolo':      [1, 2, 3, 5, 6, 8, 10],
}

C_REAL = '#1565C0'
C_SYNTH = '#E65100'
C_TREND = '#546E7A'


# === Data loading ===

def load_combined_data(data_dir='data/processed/phase4/unified'):
    return np.load(os.path.join(data_dir, 'combined_dataset.npz'), allow_pickle=True)


def load_norm_params(workload, data_dir='data/processed/phase4/unified'):
    with open(os.path.join(data_dir, 'combined_normalization.json')) as f:
        norm = json.load(f)
    return norm[workload]['params']


def get_real_traces_all_r(workload, metric_idx):
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
    path = (f'outputs/phase4/postprocessed/s36/'
            f'{workload}_r{replica_count}_s36_postprocessed.npy')
    return np.load(path)[0]


def real_means_for_wl(wl, midx, system_scale=False, to_ms=False):
    """Get per-replica means and stds for r=1..10."""
    rc = REPLICA_COUNTS_PER_WL[wl]
    _, means, stds = get_real_traces_all_r(wl, midx)
    r_all = np.arange(1, 11)
    mean_map = dict(zip(rc, means))
    std_map = dict(zip(rc, stds))

    rm = np.array([mean_map.get(r, np.nan) for r in r_all])
    rs = np.array([std_map.get(r, 0.0) for r in r_all])

    if system_scale:
        rm = rm * r_all
        rs = rs * r_all
    if to_ms:
        rm = rm * 1000
        rs = rs * 1000
    return r_all, rm, rs


def synth_points(wl, midx, system_scale=False, to_ms=False):
    """Get synthetic values at r=1, 5, 10."""
    result = {}
    for r in (1, 5, 10):
        val = float(load_postprocessed(wl, r)[:, midx].mean())
        if system_scale:
            val = val * r
        if to_ms:
            val = val * 1000
        result[r] = val
    return result


def bounded_trend(r_obs, y_obs, r_extrap, kind='saturating', ymin=None, ymax=None):
    """Fit trend to observed data and extrapolate."""
    valid = ~np.isnan(y_obs)
    r_v = r_obs[valid].astype(float)
    y_v = y_obs[valid]

    if kind == 'saturating':
        # Log fit with saturation cap
        y_cap = float(y_v.max())
        slope = (y_v[-1] - y_v[0]) / (np.log(r_v[-1] + 1)
                                       - np.log(r_v[0] + 1) + 1e-9)
        intercept = y_v[0] - slope * np.log(r_v[0] + 1)
        trend = intercept + slope * np.log(r_extrap.astype(float) + 1)
        cap = y_cap * 1.1
        trend = np.minimum(trend, cap)
    else:  # 'growth'
        # Power-law fit
        log_y = np.log(np.maximum(y_v, 1e-6))
        coeffs = np.polyfit(np.log(r_v), log_y, 1)
        trend = np.exp(np.polyval(coeffs, np.log(r_extrap.astype(float))))

    if ymin is not None:
        trend = np.maximum(trend, ymin)
    if ymax is not None:
        trend = np.minimum(trend, ymax)
    return trend


# === Figure ===

def fig_extrapolation():
    r_measured = np.arange(1, 11)
    r_extrap = np.arange(11, 51)
    r_full = np.arange(1, 51)

    # Prepare data for all 4 panels
    # GPT-2 GPU: system-level
    _, gpt2_gpu_rm, gpt2_gpu_rs = real_means_for_wl('gpt2', IDX_GPU, system_scale=True)
    gpt2_gpu_synth = synth_points('gpt2', IDX_GPU, system_scale=True)
    gpt2_gpu_trend = bounded_trend(r_measured, gpt2_gpu_rm, r_extrap,
                                   kind='saturating', ymin=0, ymax=100)

    # GPT-2 Latency: ms
    _, gpt2_lat_rm, gpt2_lat_rs = real_means_for_wl('gpt2', IDX_LAT, to_ms=True)
    gpt2_lat_synth = synth_points('gpt2', IDX_LAT, to_ms=True)
    gpt2_lat_trend = bounded_trend(r_measured, gpt2_lat_rm, r_extrap,
                                   kind='growth', ymin=0)

    # Whisper PSI
    _, wh_psi_rm, wh_psi_rs = real_means_for_wl('whisper', IDX_PSI)
    wh_psi_synth = synth_points('whisper', IDX_PSI)
    wh_psi_trend = bounded_trend(r_measured, wh_psi_rm, r_extrap,
                                 kind='saturating', ymin=0, ymax=1.0)

    # Whisper Latency: ms
    _, wh_lat_rm, wh_lat_rs = real_means_for_wl('whisper', IDX_LAT, to_ms=True)
    wh_lat_synth = synth_points('whisper', IDX_LAT, to_ms=True)
    wh_lat_trend = bounded_trend(r_measured, wh_lat_rm, r_extrap,
                                 kind='growth', ymin=0)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5),
                             gridspec_kw={'hspace': 0.55, 'wspace': 0.35})

    panels = [
        (axes[0, 0], gpt2_gpu_rm, gpt2_gpu_rs, gpt2_gpu_trend, gpt2_gpu_synth,
         'GPT-2: System GPU Utilisation', 'System GPU Util. (%)'),
        (axes[0, 1], gpt2_lat_rm, gpt2_lat_rs, gpt2_lat_trend, gpt2_lat_synth,
         'GPT-2: Average Latency', 'Latency (ms)'),
        (axes[1, 0], wh_psi_rm, wh_psi_rs, wh_psi_trend, wh_psi_synth,
         'Whisper: CPU Pressure (PSI)', 'CPU PSI'),
        (axes[1, 1], wh_lat_rm, wh_lat_rs, wh_lat_trend, wh_lat_synth,
         'Whisper: Average Latency', 'Latency (ms)'),
    ]

    for ax, rm, rs, trend, synth_dict, title, ylabel in panels:
        # Real data (r=1-10) with error bars
        valid = ~np.isnan(rm)
        ax.errorbar(r_measured[valid], rm[valid], yerr=rs[valid],
                    fmt='o-', color=C_REAL, markersize=4, linewidth=1.5,
                    capsize=3, zorder=3)

        # S36 synthetic points (r=1, 5, 10) connected with dashed line
        sx = sorted(synth_dict.keys())
        sy = [synth_dict[r] for r in sx]
        ax.plot(sx, sy, 's--', color=C_SYNTH, markersize=7, linewidth=1.3,
                zorder=4, markeredgecolor='black', markeredgewidth=0.5)

        # Extrapolation trend line (r=11-50 only, dashed)
        ax.plot(r_extrap, trend, '--', color=C_TREND, linewidth=2.0, zorder=2)

        # Vertical separator at r=10
        ax.axvline(x=10, color='#BDBDBD', linewidth=1.2,
                   linestyle=':', zorder=1)

        # Labels on each side of the separator — positioned at TOP of plot
        ylo, yhi = ax.get_ylim()
        label_y = yhi - (yhi - ylo) * 0.05
        ax.text(5, label_y, 'measured',
                fontsize=7, color='#757575', style='italic',
                ha='center', va='top',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1))
        ax.text(30, label_y, 'extrapolated',
                fontsize=7, color='#757575', style='italic',
                ha='center', va='top',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1))

        ax.set_title(title, fontsize=10, weight='bold')
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel('Replica Count', fontsize=9)
        ax.set_xlim(0, 52)
        ax.tick_params(labelsize=7.5)

    # Legend below figure
    handles = [
        plt.Line2D([0], [0], color=C_REAL, marker='o', linewidth=1.5,
                   markersize=4, label='Real ($r=1$--$10$)'),
        plt.Line2D([0], [0], color=C_SYNTH, marker='s', linewidth=1.3,
                   linestyle='--', markersize=6, label='S36 synthetic'),
        plt.Line2D([0], [0], color=C_TREND, linestyle='--', linewidth=2.0,
                   label='Bounded trend (extrapolated)'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig('figures/extrapolation_curves.pdf', bbox_inches='tight')
    plt.close()
    print("Saved figures/extrapolation_curves.pdf")


if __name__ == '__main__':
    fig_extrapolation()