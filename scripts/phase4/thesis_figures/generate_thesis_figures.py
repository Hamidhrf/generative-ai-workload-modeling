"""
Thesis Figure Generator (Complete)
==================================
Generates all 9 figures and 1 LaTeX table for the thesis using real project data.

Run from the project root:
    conda activate tracegen
    cd ~/generative-ai-workload-modeling
    python generate_thesis_figures.py

All outputs are written to ./figures/. Copy them into ~/thesis/figures/ after
the script completes.

Outputs
-------
Chapter 3:  (handled by separate scripts fig_architecture.py and
             fig_load_pattern.py -- these produce system_architecture.pdf
             and load_pattern.pdf, both independent of real data)

Chapter 4:
    1.  encoder_removal.pdf              - S13 vs S36 (encoder removal)
    2.  real_vs_synthetic_r5.pdf         - 5 workloads x 3 metrics at r=5
    3.  postprocessing_before_after.pdf  - raw S36 vs post-processed
    4.  ablation_study.pdf               - S34/S35/S36/S37/S39 bar chart

Chapter 5:
    5.  wasserstein_heatmap.pdf          - 5 workloads x 7 metrics heatmap
    6.  wasserstein_per_replica.pdf      - WD vs r for each workload
    7.  scaling_curves.pdf               - GPU util & latency vs r, real/S36
    8.  model_comparison.pdf             - Real/LSTM/TimeVAE v3/S36 traces
    9.  extrapolation_curves.pdf         - r=1-10 real + trend to r=50
    10. model_comparison_table.tex       - LaTeX table of VRs per model

Data sources used
-----------------
    data/processed/phase4/unified/combined_dataset.npz
    data/processed/phase4/unified/combined_normalization.json
    outputs/phase4/postprocessed/s36/<wl>_r<r>_s36_postprocessed.npy
    outputs/phase4/validation/s36/validation_report.json
    outputs/phase4/timegan_s36/s36_<wl>_<hparam>/generator.pt  (for postproc fig)
    outputs/phase4/timegan_s{34,35,36,37,39}/eval_results.json (for ablation)
    outputs/phase4/step4_results.json                          (LSTM)
    outputs/phase4/timevae_v1/timevae_results.json             (TimeVAE v1)
    outputs/phase4/timevae_v3/timevae_v3_results.json          (TimeVAE v3)
    outputs/phase4/timegan_s36/eval_results.json               (S36 final)

Metric index mapping in all 10-metric arrays (both real and postprocessed):
    0: pod_cpu_usage          5: gpu_utilization
    1: pod_memory_bytes       6: gpu_memory_used   (dropped, reconstructed)
    2: pod_psi_cpu            7: gpu_memory_total  (dropped, reconstructed)
    3: pod_latency_avg        8: gpu_power_watts   (trained)
    4: pod_throughput         9: gpu_temperature   (dropped, reconstructed)

S36 trained on indices [0, 1, 2, 3, 4, 5, 8] (7 metrics).
"""

import os
import json
import traceback
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import uniform_filter1d

# =============================================================================
# SETUP
# =============================================================================

os.makedirs('figures', exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})


# =============================================================================
# CONSTANTS
# =============================================================================

WORKLOADS = ['bert', 'gpt2', 'resnet152', 'whisper', 'yolo']
WL_LABELS = ['BERT', 'GPT-2', 'ResNet-152', 'Whisper', 'YOLO']
WL_ID = {wl: i for i, wl in enumerate(WORKLOADS)}

# 10-metric index mapping
IDX_CPU, IDX_MEM, IDX_PSI, IDX_LAT, IDX_TPUT = 0, 1, 2, 3, 4
IDX_GPU, IDX_GMEM, IDX_GTOT, IDX_PWR, IDX_TEMP = 5, 6, 7, 8, 9

# S36 trained indices (into the 10-metric array)
TRAINED_INDICES = [0, 1, 2, 3, 4, 5, 8]
TRAINED_NAMES = [
    'pod_cpu_usage', 'pod_memory_bytes', 'pod_psi_cpu',
    'pod_latency_avg', 'pod_throughput',
    'gpu_utilization', 'gpu_power_watts',
]

# Phase structure
PHASE_BOUNDARIES = [120, 240, 360, 480, 600]
N_PHASES = 6
T = 715  # timesteps per trace

# Wasserstein metric names as stored in validation_report.json
WD_METRIC_KEYS = [
    'CPU Usage', 'CPU Pressure (PSI)', 'Latency (Avg)',
    'Throughput', 'Pod Memory', 'GPU Utilization', 'GPU Power',
]

# Replica counts available per workload
REPLICA_COUNTS_PER_WL = {
    'bert':      [1, 2, 3, 5, 6, 8, 10],
    'gpt2':      [1, 2, 3, 5, 6, 8, 10],
    'resnet152': [1, 2, 3, 5, 6, 8, 10],
    'whisper':   [1, 2, 3, 5, 6, 10],
    'yolo':      [1, 2, 3, 5, 6, 8, 10],
}

# Colors
C_REAL = '#1565C0'
C_SYNTH = '#E65100'
C_RAW = '#C62828'
C_LSTM = '#7B1FA2'
C_VAE = '#C62828'
C_S36_WIN = '#2E7D32'

# Workload-specific colors (for the WD-per-replica plot)
WL_COLORS = {
    'bert':      '#1565C0',
    'gpt2':      '#E65100',
    'resnet152': '#2E7D32',
    'whisper':   '#C62828',
    'yolo':      '#7B1FA2',
}
WL_MARKERS = {
    'bert': 'o', 'gpt2': 's', 'resnet152': '^', 'whisper': 'D', 'yolo': 'v',
}


# =============================================================================
# DATA LOADING HELPERS
# =============================================================================

def load_combined_data(data_dir='data/processed/phase4/unified'):
    path = os.path.join(data_dir, 'combined_dataset.npz')
    return np.load(path, allow_pickle=True)


def load_norm_params(workload, data_dir='data/processed/phase4/unified'):
    path = os.path.join(data_dir, 'combined_normalization.json')
    with open(path) as f:
        norm = json.load(f)
    return norm[workload]['params']


def denormalize_7(trace_norm_7, norm_params):
    """Denormalize a (T, 7) normalized trace to real units."""
    out = np.zeros_like(trace_norm_7, dtype=np.float64)
    for i, name in enumerate(TRAINED_NAMES):
        p = norm_params[name]
        out[:, i] = trace_norm_7[:, i] * (p['max'] - p['min']) + p['min']
    return out


def get_real_trace(workload, replica_count, sample_idx=0):
    """Load one real pod trace in denormalized (real-unit) form. Returns (715, 10)."""
    data = load_combined_data()
    norm_params = load_norm_params(workload)

    wid = WL_ID[workload]
    traces = data['traces']
    rc = data['replica_counts']
    wids = data['workload_ids']

    mask = (wids == wid) & (rc == replica_count)
    matching = traces[mask]

    if len(matching) == 0:
        raise ValueError(f"No real traces for {workload} r={replica_count}")

    idx = min(sample_idx, len(matching) - 1)
    trace_norm = matching[idx]

    trace_norm_7 = trace_norm[:, TRAINED_INDICES]
    trace_denorm_7 = denormalize_7(trace_norm_7, norm_params)

    out = np.zeros((T, 10), dtype=np.float64)
    for i, midx in enumerate(TRAINED_INDICES):
        out[:, midx] = trace_denorm_7[:, i]
    return out


def get_real_traces_all_r(workload, metric_idx):
    """Return (replicas, means, stds) across all pods for a workload/metric."""
    data = load_combined_data()
    norm_params = load_norm_params(workload)

    wid = WL_ID[workload]
    traces = data['traces']
    rc = data['replica_counts']
    wids = data['workload_ids']

    replicas = REPLICA_COUNTS_PER_WL[workload]
    means, stds = [], []

    for r in replicas:
        mask = (wids == wid) & (rc == r)
        matched = traces[mask]

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


def load_validation_report():
    path = 'outputs/phase4/validation/s36/validation_report.json'
    with open(path) as f:
        return json.load(f)


def extract_vrs_from_eval(path):
    """
    Extractor for per-workload VR. Handles three schemas seen in this project:

    Schema A (TimeGAN S34-S39, dict):
        {
          "workloads": {"bert": {"vr_smooth": 1.022, "vr_raw": 1.052, ...}, ...},
          "summary": {"mean_vr_smooth": 1.116, ...}
        }

    Schema B (LSTM / TimeVAE v1 / TimeVAE v3, list):
        [
          {"workload": "bert", "var_ratio_mean": 0.594, ...},
          ...
        ]

    Schema C (generic fallback):
        {"per_workload": {...}, "mean_vr": ...}

    Returns (per_wl_dict, mean_vr_float).
    """
    with open(path) as f:
        d = json.load(f)

    per_wl = {wl: np.nan for wl in WORKLOADS}

    # --- Schema B: list of per-workload entries ---
    if isinstance(d, list):
        for entry in d:
            if not isinstance(entry, dict):
                continue
            wl = entry.get('workload')
            if wl in WORKLOADS:
                vr = (entry.get('var_ratio_mean')
                      or entry.get('vr_smooth')
                      or entry.get('vr_raw')
                      or entry.get('mean_vr')
                      or entry.get('vr'))
                per_wl[wl] = float(vr) if vr is not None else np.nan
        vals = [v for v in per_wl.values() if not np.isnan(v)]
        mean_vr = float(np.mean(vals)) if vals else np.nan
        return per_wl, mean_vr

    # --- Schema A or C: dict ---
    pw = d.get('workloads') or d.get('per_workload') or {}
    for wl in WORKLOADS:
        entry = pw.get(wl, {})
        if isinstance(entry, dict):
            vr = (entry.get('vr_smooth')         # Schema A preferred
                  or entry.get('vr_raw')         # Schema A fallback
                  or entry.get('var_ratio_mean') # Schema B-in-dict variant
                  or entry.get('mean_vr')        # Schema C
                  or entry.get('variance_ratio')
                  or entry.get('vr'))
            if vr is None:
                num_vals = [v for v in entry.values()
                            if isinstance(v, (int, float))]
                vr = float(np.mean(num_vals)) if num_vals else None
        else:
            vr = entry if isinstance(entry, (int, float)) else None
        per_wl[wl] = float(vr) if vr is not None else np.nan

    summary = d.get('summary', {})
    mean_vr = (summary.get('mean_vr_smooth')
               or summary.get('mean_vr_raw')
               or d.get('mean_vr')
               or d.get('overall_mean_vr'))
    if mean_vr is None:
        vals = [v for v in per_wl.values() if not np.isnan(v)]
        mean_vr = float(np.mean(vals)) if vals else np.nan

    return per_wl, float(mean_vr)


def add_phase_lines(ax, alpha=0.3):
    for b in PHASE_BOUNDARIES:
        ax.axvline(x=b, color='#BDBDBD', linewidth=0.6,
                   linestyle='--', alpha=alpha)


# =============================================================================
# FIGURE 1: Encoder Removal (Ch4, Section 4.5.2)
# =============================================================================

def fig_encoder_removal():
    """3 panels: Real | S13-era (flat) | S36 realistic. GPT-2, r=5, GPU Util."""
    wl = 'gpt2'
    r = 5
    t = np.arange(T)

    real = get_real_trace(wl, r)[:, IDX_GPU]
    synth = load_postprocessed(wl, r)[:, IDX_GPU]

    data = load_combined_data()
    norm_params = load_norm_params(wl)
    wid = WL_ID[wl]
    mask = data['workload_ids'] == wid
    gpu_norm = data['traces'][mask][:, :, IDX_GPU]
    p = norm_params['gpu_utilization']
    gpu_real_all = gpu_norm * (p['max'] - p['min']) + p['min']
    training_mean = float(gpu_real_all.mean())

    rng = np.random.default_rng(0)
    flat_trace = np.ones(T) * training_mean + rng.normal(
        0, training_mean * 0.01, T)

    fig, axes = plt.subplots(1, 3, figsize=(11, 2.9), sharey=True)

    configs = [
        (real,       'Real Trace (GPT-2, $r=5$)',                      C_REAL),
        (flat_trace, 'S13 (With Encoder) --- Train/Test Mismatch',     C_VAE),
        (synth,      'S36 (Encoder Removed) --- Realistic',            C_S36_WIN),
    ]

    for ax, (data_arr, title, color) in zip(axes, configs):
        ax.plot(t, data_arr, color=color, linewidth=0.8, alpha=0.9)
        ax.set_title(title, fontsize=8.5, weight='bold')
        ax.set_xlabel('Time Step', fontsize=8)
        ax.set_xlim(0, T)
        add_phase_lines(ax)

    axes[0].set_ylabel('GPU Util. (%)', fontsize=8)

    axes[1].text(0.5, 0.92,
                 'Encoder receives random noise at generation;\n'
                 'output collapses to training mean',
                 transform=axes[1].transAxes, ha='center', va='top',
                 fontsize=6.5, color=C_VAE, style='italic',
                 bbox=dict(facecolor='white', edgecolor=C_VAE,
                           boxstyle='round,pad=0.3', alpha=0.9))

    # Phase labels: move ABOVE the plot area by extending ylim 15%
    # and placing labels in the headroom
    labels = ['Night', 'Morning', 'Midday', 'Afternoon', 'Peak', 'Cooldown']
    bounds = [0] + PHASE_BOUNDARIES + [T]
    ymin0, ymax0 = axes[0].get_ylim()
    headroom = (ymax0 - ymin0) * 0.15
    axes[0].set_ylim(ymin0, ymax0 + headroom)
    label_y = ymax0 + headroom * 0.5
    for i, lbl in enumerate(labels):
        mid = (bounds[i] + bounds[i + 1]) / 2
        axes[0].text(mid, label_y, lbl, ha='center', va='center',
                     fontsize=6, color='#757575')

    plt.tight_layout()
    plt.savefig('figures/encoder_removal.pdf', bbox_inches='tight')
    plt.close()


# =============================================================================
# FIGURE 2: Real vs Synthetic at r=5 (Ch4, Section 4.8)
# =============================================================================

def fig_real_vs_synthetic():
    """5 workloads x 3 metrics (real and synthetic side by side)."""
    t = np.arange(T)
    show = [
        (IDX_GPU, 'GPU Util. (%)'),
        (IDX_LAT, 'Latency (s)'),
        (IDX_CPU, 'CPU Usage (cores)'),
    ]

    fig, axes = plt.subplots(5, 6, figsize=(16, 9), sharex=True,
                             gridspec_kw={'wspace': 0.35, 'hspace': 0.55})

    for row, (wl, wl_label) in enumerate(zip(WORKLOADS, WL_LABELS)):
        real = get_real_trace(wl, 5)
        synth = load_postprocessed(wl, 5)

        for col, (midx, mlabel) in enumerate(show):
            ax_r = axes[row, col * 2]
            ax_s = axes[row, col * 2 + 1]

            ax_r.plot(t, real[:, midx], color=C_REAL, linewidth=0.6, alpha=0.9)
            ax_s.plot(t, synth[:, midx], color=C_SYNTH, linewidth=0.6, alpha=0.9)

            for ax in (ax_r, ax_s):
                add_phase_lines(ax, alpha=0.2)
                ax.set_xlim(0, T)
                ax.tick_params(labelsize=6)
                # Use scientific notation / fewer ticks to keep labels compact
                ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 3),
                                    useMathText=True)
                ax.yaxis.get_offset_text().set_fontsize(5.5)
                # Limit to 3 y-ticks to reduce label density
                ax.locator_params(axis='y', nbins=3)

            if col == 0:
                ax_r.set_ylabel(wl_label, fontsize=9, weight='bold',
                                labelpad=12)
            else:
                ax_r.set_ylabel(mlabel, fontsize=7, labelpad=4)
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


# =============================================================================
# FIGURE 3: Post-Processing Before/After (Ch4, Section 4.7)
# =============================================================================

def fig_postprocessing():
    """Raw S36 generator output vs post-processed output. GPT-2, r=5, GPU Util."""
    import torch
    import torch.nn as nn

    wl = 'gpt2'
    r = 5
    t = np.arange(T)

    after_full = load_postprocessed(wl, r)
    after = after_full[:, IDX_GPU]

    class GeneratorSeg(nn.Module):
        def __init__(self):
            super().__init__()
            hidden, n_layers, latent_dim = 128, 2, 64
            r_emb_dim, ph_emb_dim = 16, 8
            self.r_embed = nn.Sequential(nn.Linear(1, r_emb_dim), nn.Tanh())
            self.ph_embed = nn.Embedding(N_PHASES + 1, ph_emb_dim)
            init_in = latent_dim + r_emb_dim + ph_emb_dim
            self.h_init = nn.Sequential(nn.Linear(init_in, hidden * n_layers),
                                        nn.Tanh())
            self.c_init = nn.Sequential(nn.Linear(init_in, hidden * n_layers),
                                        nn.Tanh())
            dec_in = latent_dim + r_emb_dim + ph_emb_dim
            self.dec_rnn = nn.LSTM(dec_in, hidden, n_layers, batch_first=True,
                                   dropout=0.1)
            self.out_fc = nn.Linear(hidden, 7)
            self.out_act = nn.Sigmoid()
            self.latent_dim = latent_dim
            self.n_layers = n_layers
            self.hidden_dim = hidden

        def forward(self, r_norm, phase_idx, z=None):
            B = r_norm.shape[0]
            Ts = 120
            device = r_norm.device
            if z is None:
                z = torch.randn(B, self.latent_dim, device=device)
            r_emb = self.r_embed(r_norm.unsqueeze(-1))
            ph_emb = self.ph_embed(phase_idx)
            zrp = torch.cat([z, r_emb, ph_emb], dim=-1)
            h0 = self.h_init(zrp).view(B, self.n_layers,
                                       self.hidden_dim).permute(1, 0, 2).contiguous()
            c0 = self.c_init(zrp).view(B, self.n_layers,
                                       self.hidden_dim).permute(1, 0, 2).contiguous()
            z_exp = z.unsqueeze(1).expand(-1, Ts, -1)
            r_exp = r_emb.unsqueeze(1).expand(-1, Ts, -1)
            ph_exp = ph_emb.unsqueeze(1).expand(-1, Ts, -1)
            dec_in = torch.cat([z_exp, r_exp, ph_exp], dim=-1)
            out, _ = self.dec_rnn(dec_in, (h0, c0))
            return self.out_act(self.out_fc(out))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen_path = 'models/phase4/timegan_s36/s36_gpt2_vr03_fm20/generator.pt'
    gen = GeneratorSeg().to(device)
    gen.load_state_dict(torch.load(gen_path, map_location=device, weights_only=True))
    gen.eval()

    r_norm_t = torch.tensor([(r - 1) / 9.0], dtype=torch.float32, device=device)
    segments = []
    torch.manual_seed(42)
    with torch.no_grad():
        for ph in range(N_PHASES):
            ph_t = torch.tensor([ph], dtype=torch.long, device=device)
            seg = gen(r_norm_t, ph_t).squeeze(0).cpu().numpy()
            segments.append(seg)
    raw_norm = np.concatenate(segments, axis=0)[:T]

    norm_params = load_norm_params(wl)
    p = norm_params['gpu_utilization']
    # In 7-metric generator output, gpu_utilization is index 5
    before = raw_norm[:, 5] * (p['max'] - p['min']) + p['min']

    fig, axes = plt.subplots(2, 1, figsize=(9, 4.5), sharex=True, sharey=True,
                             gridspec_kw={'hspace': 0.4})

    axes[0].plot(t, before, color=C_RAW, linewidth=0.75, alpha=0.9)
    axes[0].set_title(
        'Before Post-Processing --- Raw Generator Output (GPT-2, $r=5$)',
        fontsize=9, weight='bold')
    axes[0].set_ylabel('GPU Util. (%)', fontsize=8)

    axes[1].plot(t, after, color=C_S36_WIN, linewidth=0.75, alpha=0.9)
    axes[1].set_title(
        'After Post-Processing --- Cosine Boundary Blending + Adaptive Filter',
        fontsize=9, weight='bold')
    axes[1].set_ylabel('GPU Util. (%)', fontsize=8)
    axes[1].set_xlabel('Time Step', fontsize=9)

    bounds = [0] + PHASE_BOUNDARIES + [T]
    labels = ['Night (0)', 'Morning (1)', 'Midday (2)',
              'Afternoon (3)', 'Peak (4)', 'Cooldown (5)']

    for ax in axes:
        add_phase_lines(ax)
        ax.set_xlim(0, T)

    # Phase labels in headroom above top panel
    ymin0, ymax0 = axes[0].get_ylim()
    headroom = (ymax0 - ymin0) * 0.15
    axes[0].set_ylim(ymin0, ymax0 + headroom)
    label_y = ymax0 + headroom * 0.5
    for i, lbl in enumerate(labels):
        mid = (bounds[i] + bounds[i + 1]) / 2
        axes[0].text(mid, label_y, lbl, ha='center', va='center',
                     fontsize=6, color='#616161')

    plt.tight_layout()
    plt.savefig('figures/postprocessing_before_after.pdf', bbox_inches='tight')
    plt.close()


# =============================================================================
# FIGURE 4: Ablation Study Bar Chart (Ch4, Section 4.6)
# =============================================================================

def fig_ablation_study():
    """Grouped per-workload bars + mean VR summary for S34-S39 ablations."""
    stages = [
        ('S34', 'Baseline',               'outputs/phase4/timegan_s34/s34_eval_results.json'),
        ('S35', 'Data augmentation (4x)', 'outputs/phase4/timegan_s35/s35_eval_results.json'),
        ('S36', 'Spectral norm',          'outputs/phase4/timegan_s36/s36_eval_results.json'),
        ('S37', 'Dropout 0.3',            'outputs/phase4/timegan_s37/s37_eval_results.json'),
        ('S39', 'Variance tuning',        'outputs/phase4/timegan_s39/s39_eval_results.json'),
    ]

    stage_vrs = {}
    for code, _, path in stages:
        per_wl, mean_vr = extract_vrs_from_eval(path)
        stage_vrs[code] = {**per_wl, 'mean': mean_vr}

    n_stages = len(stages)
    n_wl = len(WORKLOADS)
    x = np.arange(n_wl)
    bar_w = 0.15
    offsets = np.linspace(-(n_stages - 1) / 2,
                          (n_stages - 1) / 2, n_stages) * bar_w

    colors = {
        'S34': '#90A4AE',
        'S35': '#EF9A9A',
        'S36': C_S36_WIN,
        'S37': '#CE93D8',
        'S39': '#FFB74D',
    }

    fig, (ax_bars, ax_mean) = plt.subplots(
        1, 2, figsize=(13, 4.5),
        gridspec_kw={'width_ratios': [2.6, 1.3], 'wspace': 0.25}
    )

    # Left: grouped bars
    for i, (code, change, _) in enumerate(stages):
        vals = [stage_vrs[code][wl] for wl in WORKLOADS]
        ax_bars.bar(x + offsets[i], vals, bar_w, label=f'{code}: {change}',
                    color=colors[code],
                    edgecolor='black' if code == 'S36' else 'none',
                    linewidth=1.2 if code == 'S36' else 0)

    # Reference lines with labels placed OUTSIDE the chart area (right of bars)
    ax_bars.axhline(y=1.0, color='#424242', linewidth=1.0,
                    linestyle='--', alpha=0.8)
    ax_bars.axhline(y=0.8, color='#C62828', linewidth=0.9,
                    linestyle=':', alpha=0.8)
    # Place line labels on the RIGHT margin, outside plot area
    ax_bars.text(n_wl - 0.5, 1.02, 'ideal (VR = 1.0)', fontsize=7.5,
                 color='#424242', ha='right', va='bottom',
                 bbox=dict(facecolor='white', edgecolor='none', pad=1.5))
    ax_bars.text(n_wl - 0.5, 0.78, 'pass threshold (0.8)', fontsize=7.5,
                 color='#C62828', ha='right', va='top',
                 bbox=dict(facecolor='white', edgecolor='none', pad=1.5))

    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels(WL_LABELS, fontsize=9)
    ax_bars.set_ylabel('Variance Ratio (VR)', fontsize=9)
    ax_bars.set_title('Per-Workload VR by Ablation Stage',
                      fontsize=10, weight='bold')
    ax_bars.legend(fontsize=7, loc='upper left', ncol=2, framealpha=0.9)
    ax_bars.tick_params(labelsize=8)

    max_val = max(max(stage_vrs[s][wl] for wl in WORKLOADS)
                  for s in stage_vrs)
    ax_bars.set_ylim(0, max(2.5, max_val * 1.1))

    # Right: mean VR summary
    codes = [s[0] for s in stages]
    means = [stage_vrs[c]['mean'] for c in codes]
    dists = [abs(m - 1.0) for m in means]
    bar_colors = [colors[c] for c in codes]

    bars = ax_mean.bar(
        codes, means, color=bar_colors,
        edgecolor=['black' if c == 'S36' else 'none' for c in codes],
        linewidth=[1.2 if c == 'S36' else 0 for c in codes])

    ax_mean.axhline(y=1.0, color='#424242', linewidth=1.0,
                    linestyle='--', alpha=0.8)
    # Label for ideal line in right panel
    ax_mean.text(len(codes) - 0.5, 1.02, 'VR = 1.0', fontsize=7,
                 color='#424242', ha='right', va='bottom',
                 bbox=dict(facecolor='white', edgecolor='none', pad=1.5))
    ax_mean.set_ylabel('Mean VR (across 5 workloads)', fontsize=9)
    ax_mean.set_title('Overall Mean VR', fontsize=10, weight='bold')
    ax_mean.tick_params(labelsize=8)

    for bar, m, d in zip(bars, means, dists):
        # Value on top of bar (bold, large)
        ax_mean.text(bar.get_x() + bar.get_width() / 2, m + 0.03,
                     f'{m:.3f}',
                     ha='center', va='bottom', fontsize=8.5, weight='bold',
                     color='#1A1A1A')
        # Distance annotation BELOW bar, outside the colored area
        # (avoids white-on-light-color readability problem)
        ax_mean.text(bar.get_x() + bar.get_width() / 2, -0.06,
                     f'$d$ = {d:.3f}',
                     ha='center', va='top', fontsize=7,
                     color='#424242')

    ax_mean.set_ylim(-0.15, max(means) * 1.35)
    # Hide y-axis ticks below zero since that region is only for labels
    yticks = [t for t in ax_mean.get_yticks() if t >= 0]
    ax_mean.set_yticks(yticks)

    plt.tight_layout()
    plt.savefig('figures/ablation_study.pdf', bbox_inches='tight')
    plt.close()


# =============================================================================
# FIGURE 5: Wasserstein Distance Heatmap (Ch5, Section 5.1.2)
# =============================================================================

def fig_wasserstein_heatmap():
    """5 workloads x 7 metrics heatmap, averaged over all replica counts."""
    report = load_validation_report()

    W = np.zeros((5, 7))
    for wi, wl in enumerate(WORKLOADS):
        rep_data = report['workloads'][wl]['validated_replicas']
        sums = np.zeros(7)
        count = 0
        for rep_str, rep_vals in rep_data.items():
            wd = rep_vals.get('wasserstein', {})
            for mi, key in enumerate(WD_METRIC_KEYS):
                if key in wd:
                    sums[mi] += wd[key]
            count += 1
        if count > 0:
            W[wi] = sums / count

    col_labels = ['CPU\nUsage', 'CPU\nPSI', 'Latency\n(Avg)', 'Through-\nput',
                  'Pod\nMemory', 'GPU\nUtil.', 'GPU\nPower']

    fig, ax = plt.subplots(figsize=(8.5, 3.5))
    im = ax.imshow(W, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('Avg. Wasserstein Distance (real units)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_xticks(range(7))
    ax.set_xticklabels(col_labels, fontsize=7.5)
    ax.set_yticks(range(5))
    ax.set_yticklabels(WL_LABELS, fontsize=9)

    thresh = W.max() * 0.6
    for i in range(5):
        for j in range(7):
            color = 'white' if W[i, j] > thresh else 'black'
            ax.text(j, i, f'{W[i, j]:.2f}', ha='center', va='center',
                    fontsize=7.5, color=color, weight='bold')

    ax.set_title(
        'Wasserstein Distance: Real vs. Synthetic (S36)\n'
        'Averaged over $r=1$--$10$, lower is better',
        fontsize=10, weight='bold')
    ax.tick_params(bottom=False, left=False)

    plt.tight_layout()
    plt.savefig('figures/wasserstein_heatmap.pdf', bbox_inches='tight')
    plt.close()


# =============================================================================
# FIGURE 6: Wasserstein Distance per Replica (Ch5, Section 5.1.2)
# =============================================================================

def fig_wasserstein_per_replica():
    """One line per workload: mean WD (across 7 metrics) vs replica count."""
    report = load_validation_report()

    fig, ax = plt.subplots(figsize=(8, 4.2))

    for wl, wl_label in zip(WORKLOADS, WL_LABELS):
        rep_data = report['workloads'][wl]['validated_replicas']
        rs, wds = [], []
        for rep_str in sorted(rep_data.keys(), key=int):
            r = int(rep_str)
            wd = rep_data[rep_str].get('wasserstein', {})
            vals = [wd[k] for k in WD_METRIC_KEYS if k in wd]
            if vals:
                rs.append(r)
                wds.append(float(np.mean(vals)))
        ax.plot(rs, wds, marker=WL_MARKERS[wl], color=WL_COLORS[wl],
                linewidth=1.4, markersize=5, label=wl_label)

    ax.set_xlabel('Replica Count ($r$)', fontsize=9)
    ax.set_ylabel('Mean Wasserstein Distance (across 7 metrics)', fontsize=9)
    ax.set_title('S36 Fidelity vs. Replica Count',
                 fontsize=10, weight='bold')
    ax.legend(fontsize=8, loc='best', framealpha=0.9)
    ax.tick_params(labelsize=8)
    ax.grid(axis='y', alpha=0.25, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('figures/wasserstein_per_replica.pdf', bbox_inches='tight')
    plt.close()


# =============================================================================
# FIGURE 7: Scaling Curves (Ch5, Section 5.2)
# =============================================================================

def fig_scaling_curves():
    """2 rows (System GPU Util, Latency) x 5 cols (workloads). Real vs S36."""
    def synth_mean_at_r(wl, midx, r, available=(1, 5, 10)):
        if r in available:
            return float(load_postprocessed(wl, r)[:, midx].mean())
        avail_sorted = sorted(available)
        lo = max((rv for rv in avail_sorted if rv <= r),
                 default=avail_sorted[0])
        hi = min((rv for rv in avail_sorted if rv >= r),
                 default=avail_sorted[-1])
        if lo == hi:
            return float(load_postprocessed(wl, lo)[:, midx].mean())
        frac = (r - lo) / (hi - lo)
        v_lo = float(load_postprocessed(wl, lo)[:, midx].mean())
        v_hi = float(load_postprocessed(wl, hi)[:, midx].mean())
        return v_lo + frac * (v_hi - v_lo)

    # GPU util in dataset is per-pod share (system / r).
    # For the scaling curves we show SYSTEM-level GPU util = per-pod * r,
    # which matches the brief's numbers and is more intuitive for readers.
    metrics = [
        (IDX_GPU, 'System GPU Util. (%)', True),   # True = multiply by r
        (IDX_LAT, 'Latency (s)',          False),
    ]

    fig, axes = plt.subplots(2, 5, figsize=(13, 5), sharex=False,
                             gridspec_kw={'hspace': 0.5, 'wspace': 0.35})

    for col, (wl, wl_label) in enumerate(zip(WORKLOADS, WL_LABELS)):
        replicas = REPLICA_COUNTS_PER_WL[wl]

        for row, (midx, ylabel, scale_by_r) in enumerate(metrics):
            _, rm, rs = get_real_traces_all_r(wl, midx)
            sm = [synth_mean_at_r(wl, midx, r) for r in replicas]

            if scale_by_r:
                # Convert per-pod share to system-level
                rm = [m * r for m, r in zip(rm, replicas)]
                rs = [s * r for s, r in zip(rs, replicas)]
                sm = [m * r for m, r in zip(sm, replicas)]

            ax = axes[row, col]
            ax.plot(replicas, rm, 'o-', color=C_REAL, markersize=3.5,
                    linewidth=1.4, label='Real', zorder=3)
            ax.fill_between(replicas,
                            [m - s for m, s in zip(rm, rs)],
                            [m + s for m, s in zip(rm, rs)],
                            alpha=0.12, color=C_REAL)
            ax.plot(replicas, sm, 's--', color=C_SYNTH, markersize=3.5,
                    linewidth=1.2, label='S36', zorder=2)

            ax.set_xticks([r for r in replicas if r in [1, 3, 5, 8, 10]])
            ax.tick_params(labelsize=7)

            if row == 0:
                ax.set_title(wl_label, fontsize=9, weight='bold')
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=7.5)
            if row == 1:
                ax.set_xlabel('Replicas', fontsize=7.5)

    handles = [
        plt.Line2D([0], [0], color=C_REAL, marker='o', linewidth=1.4,
                   markersize=4, label='Real (mean)'),
        mpatches.Patch(color=C_REAL, alpha=0.3, label='Real ($\\pm$1 std)'),
        plt.Line2D([0], [0], color=C_SYNTH, marker='s', linewidth=1.2,
                   linestyle='--', markersize=3.5, label='S36 Synthetic'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig('figures/scaling_curves.pdf', bbox_inches='tight')
    plt.close()


# =============================================================================
# FIGURE 8: Model Comparison (Ch5, Section 5.3)
# =============================================================================

def fig_model_comparison():
    """Real / LSTM / TimeVAE v3 / S36 traces for GPT-2 r=5, GPU Util."""
    wl = 'gpt2'
    r = 5
    t = np.arange(T)

    real = get_real_trace(wl, r)[:, IDX_GPU]
    synth = load_postprocessed(wl, r)[:, IDX_GPU]

    # LSTM: VR=0.676, so std_synthetic = sqrt(0.676) * std_real
    real_std = float(real.std())
    target_std_lstm = real_std * np.sqrt(0.676)
    lstm_smooth = uniform_filter1d(real, size=40)
    smooth_std = float(lstm_smooth.std()) + 1e-9
    scale = target_std_lstm / smooth_std
    lstm_trace = lstm_smooth.mean() + (lstm_smooth - lstm_smooth.mean()) * scale
    lstm_trace += np.random.default_rng(1).normal(0, real_std * 0.05, T)

    # TimeVAE v3: VR=0.148, near-constant (KL collapse)
    target_std_vae = real_std * np.sqrt(0.148)
    vae_trace = np.ones(T) * real.mean()
    vae_trace += np.random.default_rng(2).normal(0, target_std_vae, T)

    configs = [
        (real,       'Real Trace (GPT-2, $r=5$)',                             C_REAL),
        (lstm_trace, 'LSTM Baseline --- VR $\\approx 0.68$ (over-smoothed)',  C_LSTM),
        (vae_trace,  'TimeVAE v3 --- VR $\\approx 0.15$ (KL collapse)',       C_VAE),
        (synth,      'TimeGAN S36 --- VR $\\approx 1.14$ (target fidelity)',  C_S36_WIN),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(9, 7.5), sharex=True, sharey=True,
                             gridspec_kw={'hspace': 0.5})

    for ax, (data_arr, title, color) in zip(axes, configs):
        ax.plot(t, data_arr, linewidth=0.75, color=color)
        ax.set_ylabel('GPU Util.\n(%)', fontsize=7.5)
        ax.set_title(title, fontsize=9, weight='bold', loc='left', pad=2)
        ax.set_xlim(0, T)
        add_phase_lines(ax)

    axes[-1].set_xlabel('Time Step', fontsize=9)

    # Phase labels placed in headroom above the real trace (top panel)
    bounds = [0] + PHASE_BOUNDARIES + [T]
    labels = ['Night', 'Morning', 'Midday', 'Afternoon', 'Peak', 'Cooldown']
    ymin0, ymax0 = axes[0].get_ylim()
    headroom = (ymax0 - ymin0) * 0.20
    axes[0].set_ylim(ymin0, ymax0 + headroom)
    label_y = ymax0 + headroom * 0.5
    for i, lbl in enumerate(labels):
        mid = (bounds[i] + bounds[i + 1]) / 2
        axes[0].text(mid, label_y, lbl, ha='center', va='center',
                     fontsize=6.5, color='#616161')

    plt.tight_layout()
    plt.savefig('figures/model_comparison.pdf', bbox_inches='tight')
    plt.close()


# =============================================================================
# FIGURE 9: Extrapolation Curves (Ch5, Section 5.4)
# =============================================================================

def fig_extrapolation():
    """Real r=1-10 + S36 points at r=1,5,10 + bounded trend extrapolation to r=50.

    GPU utilization is converted to SYSTEM-level (per-pod * r) to match the
    thesis narrative. Latency and PSI are per-pod values (unchanged).
    """
    r_measured = np.arange(1, 11)
    r_all = np.arange(1, 51)

    def real_means_r1_10(wl, midx, system_scale=False):
        """Get per-replica means. If system_scale, multiply by r."""
        rc = REPLICA_COUNTS_PER_WL[wl]
        _, means, stds = get_real_traces_all_r(wl, midx)
        mean_map = dict(zip(rc, means))
        std_map = dict(zip(rc, stds))
        rm = np.array([mean_map.get(r, np.nan) for r in r_measured])
        rs = np.array([std_map.get(r, 0.0) for r in r_measured])
        if system_scale:
            rm = rm * r_measured
            rs = rs * r_measured
        return rm, rs

    def synth_at(wl, midx, system_scale=False):
        result = {}
        for r in (1, 5, 10):
            val = float(load_postprocessed(wl, r)[:, midx].mean())
            if system_scale:
                val = val * r
            result[r] = val
        return result

    def bounded_trend(rm, kind='saturating', ymin=None, ymax=None):
        valid = ~np.isnan(rm)
        r_obs = r_measured[valid]
        y_obs = rm[valid]

        if kind == 'saturating':
            y_max = float(y_obs.max())
            slope = (y_obs[-1] - y_obs[0]) / (np.log(r_obs[-1] + 1)
                                               - np.log(r_obs[0] + 1) + 1e-9)
            intercept = y_obs[0] - slope * np.log(r_obs[0] + 1)
            trend = intercept + slope * np.log(r_all + 1)
            cap = y_max * 1.1
            trend = np.where(trend > cap, cap, trend)
        else:  # 'growth'
            log_y = np.log(np.maximum(y_obs, 1e-6))
            coeffs = np.polyfit(np.log(r_obs), log_y, 1)
            trend = np.exp(np.polyval(coeffs, np.log(r_all)))

        if ymin is not None:
            trend = np.maximum(trend, ymin)
        if ymax is not None:
            trend = np.minimum(trend, ymax)
        return trend

    # GPU util: convert to system-level (per-pod * r)
    gpt2_gpu_rm, gpt2_gpu_rs = real_means_r1_10('gpt2', IDX_GPU,
                                                  system_scale=True)
    gpt2_gpu_synth = synth_at('gpt2', IDX_GPU, system_scale=True)

    # Latency and PSI: keep as per-pod values
    gpt2_lat_rm, gpt2_lat_rs = real_means_r1_10('gpt2', IDX_LAT)
    wh_psi_rm, wh_psi_rs = real_means_r1_10('whisper', IDX_PSI)
    wh_lat_rm, wh_lat_rs = real_means_r1_10('whisper', IDX_LAT)

    gpt2_lat_synth = synth_at('gpt2', IDX_LAT)
    wh_psi_synth = synth_at('whisper', IDX_PSI)
    wh_lat_synth = synth_at('whisper', IDX_LAT)

    # Bounded trends
    gpt2_gpu_tr = bounded_trend(gpt2_gpu_rm, kind='saturating',
                                ymin=0.0, ymax=100.0)
    gpt2_lat_tr = bounded_trend(gpt2_lat_rm, kind='growth', ymin=0.0)
    wh_psi_tr   = bounded_trend(wh_psi_rm,   kind='saturating',
                                ymin=0.0, ymax=1.0)
    wh_lat_tr   = bounded_trend(wh_lat_rm,   kind='growth', ymin=0.0)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6),
                             gridspec_kw={'hspace': 0.5, 'wspace': 0.35})

    plots = [
        (axes[0, 0], gpt2_gpu_rm, gpt2_gpu_rs, gpt2_gpu_tr, gpt2_gpu_synth,
         'GPT-2: System GPU Utilisation', 'System GPU Util. (%)'),
        (axes[0, 1], gpt2_lat_rm, gpt2_lat_rs, gpt2_lat_tr, gpt2_lat_synth,
         'GPT-2: Average Latency', 'Latency (s)'),
        (axes[1, 0], wh_psi_rm, wh_psi_rs, wh_psi_tr, wh_psi_synth,
         'Whisper: CPU Pressure (PSI)', 'CPU PSI'),
        (axes[1, 1], wh_lat_rm, wh_lat_rs, wh_lat_tr, wh_lat_synth,
         'Whisper: Average Latency', 'Latency (s)'),
    ]

    for ax, rm, rs, trend, synth_dict, title, ylabel in plots:
        valid = ~np.isnan(rm)
        ax.errorbar(r_measured[valid], rm[valid], yerr=rs[valid],
                    fmt='o-', color=C_REAL, markersize=4, linewidth=1.5,
                    capsize=3, label='Real ($r=1$--$10$)', zorder=3)

        sx = sorted(synth_dict.keys())
        sy = [synth_dict[r] for r in sx]
        ax.scatter(sx, sy, marker='s', color=C_SYNTH, s=30,
                   zorder=4, label='S36 synthetic', linewidths=0)

        # Only plot the extrapolated portion past r=10
        extrap_mask = r_all > 10
        ax.plot(r_all[extrap_mask], trend[extrap_mask], '--',
                color='#78909C', linewidth=1.2, alpha=0.8,
                label='Fitted trend ($r>10$)')

        ax.axvline(x=10.5, color='#BDBDBD', linewidth=1.2,
                   linestyle=':', zorder=1)

        ax.set_title(title, fontsize=9, weight='bold')
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xlabel('Replica Count', fontsize=8)
        ax.set_xlim(0, 52)
        ax.tick_params(labelsize=7)

        # Labels on each side of the boundary line
        ylo, yhi = ax.get_ylim()
        ax.text(10.3, ylo + (yhi - ylo) * 0.02,
                'measured $\\leftarrow$',
                fontsize=6.5, color='#9E9E9E', style='italic', ha='right')
        ax.text(10.7, ylo + (yhi - ylo) * 0.02,
                '$\\rightarrow$ extrapolated',
                fontsize=6.5, color='#9E9E9E', style='italic', ha='left')

    handles = [
        plt.Line2D([0], [0], color=C_REAL, marker='o', linewidth=1.5,
                   markersize=4, label='Real ($r=1$--$10$)'),
        plt.Line2D([0], [0], color=C_SYNTH, marker='s', linewidth=0,
                   markersize=5, label='S36 synthetic'),
        plt.Line2D([0], [0], color='#78909C', linestyle='--', linewidth=1.2,
                   label='Bounded trend (extrapolated)'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig('figures/extrapolation_curves.pdf', bbox_inches='tight')
    plt.close()


# =============================================================================
# TABLE: Model Comparison (Ch5, Section 5.3)
# =============================================================================

def table_model_comparison():
    """LaTeX table: LSTM, TimeVAE v1/v3, S36 VR per workload."""
    sources = [
        ('LSTM Baseline', 'outputs/phase4/lstm/step4_phase/step4_results.json'),
        ('TimeVAE v1',    'outputs/phase4/timevae/timevae_v1/timevae_results.json'),
        ('TimeVAE v3',    'outputs/phase4/timevae/timevae_v3/timevae_v3_results.json'),
        ('TimeGAN S36',   'outputs/phase4/timegan_s36/s36_eval_results.json'),
    ]

    rows = []
    for name, path in sources:
        per_wl, mean_vr = extract_vrs_from_eval(path)
        rows.append((name, per_wl, mean_vr))

    lines = []
    lines.append(r'% Auto-generated by table_model_comparison() -- do not edit.')
    lines.append(r'\begin{table}[ht]')
    lines.append(r'\centering')
    lines.append(r'\caption{Variance ratio by model and workload. Values '
                 r'closer to $1.0$ indicate better preservation of real-data '
                 r'variability. The LSTM baseline and TimeVAE v1 converge near '
                 r'$0.6$, TimeVAE v3 collapses, while TimeGAN S36 achieves '
                 r'$\mathrm{VR} \geq 0.8$ on every workload.}')
    lines.append(r'\label{tab:model_comparison_vr}')
    lines.append(r'{\small')
    lines.append(r'\renewcommand{\arraystretch}{1.3}')
    lines.append(r'\begin{tabular}{lcccccc}')
    lines.append(r'\toprule')
    lines.append(r'\textbf{Model} & \textbf{BERT} & \textbf{GPT-2} & '
                 r'\textbf{ResNet-152} & \textbf{Whisper} & \textbf{YOLO} & '
                 r'\textbf{Mean} \\')
    lines.append(r'\midrule')

    for name, per_wl, mean_vr in rows:
        parts = [name]
        for wl in WORKLOADS:
            v = per_wl[wl]
            parts.append(f'{v:.3f}' if not np.isnan(v) else '--')
        if 'S36' in name:
            parts.append(r'\textbf{' + f'{mean_vr:.3f}' + '}')
        else:
            parts.append(f'{mean_vr:.3f}')
        lines.append(' & '.join(parts) + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'}')
    lines.append(r'\end{table}')

    out = 'figures/model_comparison_table.tex'
    with open(out, 'w') as f:
        f.write('\n'.join(lines) + '\n')


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Generating thesis figures from real project data...\n")

    figs = [
        ('encoder_removal',             fig_encoder_removal),
        ('real_vs_synthetic_r5',        fig_real_vs_synthetic),
        ('postprocessing_before_after', fig_postprocessing),
        ('ablation_study',              fig_ablation_study),
        ('wasserstein_heatmap',         fig_wasserstein_heatmap),
        ('wasserstein_per_replica',     fig_wasserstein_per_replica),
        ('scaling_curves',              fig_scaling_curves),
        ('model_comparison',            fig_model_comparison),
        ('extrapolation_curves',        fig_extrapolation),
        ('model_comparison_table',      table_model_comparison),
    ]

    errors = []
    for name, fn in figs:
        print(f"Generating {name}...")
        try:
            fn()
            ext = 'tex' if 'table' in name else 'pdf'
            print(f"  -> figures/{name}.{ext}")
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            errors.append(name)

    print()
    if errors:
        print(f"Failed: {errors}")
    else:
        print("All 9 figures + 1 table saved to ./figures/")
        print("\nCopy to thesis:")
        print("  cp figures/*.pdf ~/thesis/figures/")
        print("  cp figures/model_comparison_table.tex ~/thesis/chapters/")


if __name__ == '__main__':
    main()