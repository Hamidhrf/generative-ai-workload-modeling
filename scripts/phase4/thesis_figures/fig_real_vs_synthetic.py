"""
Real vs Synthetic at r=5 (standalone)
======================================
Generates figures/real_vs_synthetic_r5.pdf

5 workloads x 3 metrics (GPU Util, Latency, CPU Usage)
Each metric has Real (blue) and S36 (orange) side by side.

Fix: Latency converted to ms to avoid ugly x10^-3 / x10^-2 notation.

Run from project root:
    conda activate tracegen
    cd ~/generative-ai-workload-modeling
    python scripts/phase4/thesis_figures/fig_real_vs_synthetic.py
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
WL_LABELS = ['BERT', 'GPT-2', 'ResNet-152', 'Whisper', 'YOLO']
WL_ID = {wl: i for i, wl in enumerate(WORKLOADS)}

TRAINED_NAMES = [
    'pod_cpu_usage', 'pod_memory_bytes', 'pod_psi_cpu',
    'pod_latency_avg', 'pod_throughput',
    'gpu_utilization', 'gpu_power_watts',
]
TRAINED_INDICES = [0, 1, 2, 3, 4, 5, 8]

IDX_CPU = 0
IDX_LAT = 3
IDX_GPU = 5

T = 715
PHASE_BOUNDARIES = [120, 240, 360, 480, 600]

C_REAL = '#1565C0'
C_SYNTH = '#E65100'


# === Data loading ===

def load_norm_params(workload, data_dir='data/processed/phase4/unified'):
    with open(os.path.join(data_dir, 'combined_normalization.json')) as f:
        norm = json.load(f)
    return norm[workload]['params']


def load_combined_data(data_dir='data/processed/phase4/unified'):
    return np.load(os.path.join(data_dir, 'combined_dataset.npz'), allow_pickle=True)


def denormalize_7(trace_norm_7, norm_params):
    out = np.zeros_like(trace_norm_7, dtype=np.float64)
    for i, name in enumerate(TRAINED_NAMES):
        p = norm_params[name]
        out[:, i] = trace_norm_7[:, i] * (p['max'] - p['min']) + p['min']
    return out


def get_real_trace(workload, replica_count, sample_idx=0):
    data = load_combined_data()
    norm_params = load_norm_params(workload)
    wid = WL_ID[workload]

    mask = (data['workload_ids'] == wid) & (data['replica_counts'] == replica_count)
    matching = data['traces'][mask]
    idx = min(sample_idx, len(matching) - 1)
    trace_norm = matching[idx]

    trace_norm_7 = trace_norm[:, TRAINED_INDICES]
    trace_denorm_7 = denormalize_7(trace_norm_7, norm_params)

    out = np.zeros((T, 10), dtype=np.float64)
    for i, midx in enumerate(TRAINED_INDICES):
        out[:, midx] = trace_denorm_7[:, i]
    return out


def load_postprocessed(workload, replica_count):
    path = (f'outputs/phase4/postprocessed/s36/'
            f'{workload}_r{replica_count}_s36_postprocessed.npy')
    return np.load(path)[0]


def add_phase_lines(ax, alpha=0.2):
    for b in PHASE_BOUNDARIES:
        ax.axvline(x=b, color='#BDBDBD', linewidth=0.6,
                   linestyle='--', alpha=alpha)


# === Figure ===

def fig_real_vs_synthetic():
    t = np.arange(T)

    # (metric_index, label_real, label_synth, to_ms)
    show = [
        (IDX_GPU, 'GPU Util. (%)',      False),
        (IDX_LAT, 'Latency (ms)',       True),
        (IDX_CPU, 'CPU Usage (cores)',  False),
    ]

    fig, axes = plt.subplots(5, 6, figsize=(16, 9), sharex=True,
                             gridspec_kw={'wspace': 0.35, 'hspace': 0.55})

    for row, (wl, wl_label) in enumerate(zip(WORKLOADS, WL_LABELS)):
        real = get_real_trace(wl, 5)
        synth = load_postprocessed(wl, 5)

        for col, (midx, mlabel, to_ms) in enumerate(show):
            ax_r = axes[row, col * 2]
            ax_s = axes[row, col * 2 + 1]

            real_vals = real[:, midx].copy()
            synth_vals = synth[:, midx].copy()

            if to_ms:
                real_vals = real_vals * 1000
                synth_vals = synth_vals * 1000

            ax_r.plot(t, real_vals, color=C_REAL, linewidth=0.6, alpha=0.9)
            ax_s.plot(t, synth_vals, color=C_SYNTH, linewidth=0.6, alpha=0.9)

            for ax in (ax_r, ax_s):
                add_phase_lines(ax)
                ax.set_xlim(0, T)
                ax.tick_params(labelsize=6)
                ax.locator_params(axis='y', nbins=4)

            if col == 0:
                ax_r.set_ylabel(wl_label, fontsize=9, weight='bold',
                                labelpad=12)

            if row == 0:
                ax_r.set_title(f'{mlabel}\n(Real)', fontsize=7.5, weight='bold')
                ax_s.set_title(f'{mlabel}\n(S36)', fontsize=7.5,
                               weight='bold', color=C_SYNTH)
            if row == 4:
                ax_r.set_xlabel('Time Step', fontsize=7)
                ax_s.set_xlabel('Time Step', fontsize=7)

    handles = [
        mpatches.Patch(color=C_REAL, label='Real'),
        mpatches.Patch(color=C_SYNTH, label='S36 Synthetic (post-processed)'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig('figures/real_vs_synthetic_r5.pdf', bbox_inches='tight')
    plt.close()
    print("Saved figures/real_vs_synthetic_r5.pdf")


if __name__ == '__main__':
    fig_real_vs_synthetic()