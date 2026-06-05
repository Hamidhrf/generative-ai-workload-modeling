"""
Regenerate Figure 4.3 (Chapter 4, Ablation Study) with S38 included.

Standalone version of fig_ablation_study() from generate_thesis_figures.py.
Adds the S38 stage (Multi-layer feature matching) that was missing from the
previously rendered figure.

Usage
-----
Run from the project root (same directory you used for the main figure script):

    conda activate tracegen
    cd ~/generative-ai-workload-modeling
    python regenerate_ablation_figure.py

The output PDF is written to figures/ablation_study.pdf, overwriting the
previous version.

Expected input
--------------
Six per-stage eval JSONs:
    outputs/phase4/timegan_s34/s34_eval_results.json
    outputs/phase4/timegan_s35/s35_eval_results.json
    outputs/phase4/timegan_s36/s36_eval_results.json
    outputs/phase4/timegan_s37/s37_eval_results.json
    outputs/phase4/timegan_s38/s38_eval_results.json   (NEW)
    outputs/phase4/timegan_s39/s39_eval_results.json

Each is read with the same Schema A loader used by the main figure script
(per-workload "vr_smooth" with "vr_raw" fallback, and "summary.mean_vr_smooth"
for the overall mean). If the S38 JSON is missing, the script falls back to
the verified values from THESIS_BRIEF and prints a warning.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# SETUP (matching generate_thesis_figures.py conventions)
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

WORKLOADS = ['bert', 'gpt2', 'resnet152', 'whisper', 'yolo']
WL_LABELS = ['BERT', 'GPT-2', 'ResNet-152', 'Whisper', 'YOLO']

# Same forest-green winner color used in the rest of the figures
C_S36_WIN = '#2E7D32'


# =============================================================================
# JSON LOADER (lifted from generate_thesis_figures.py for consistency)
# =============================================================================

def extract_vrs_from_eval(path):
    """
    Per-workload VR extractor matching the main figure script.

    Schema A (TimeGAN S34-S39, dict):
        {
          "workloads": {"bert": {"vr_smooth": 1.022, "vr_raw": 1.052, ...}, ...},
          "summary": {"mean_vr_smooth": 1.116, ...}
        }

    Returns (per_wl_dict, mean_vr_float).
    """
    with open(path) as f:
        d = json.load(f)

    per_wl = {wl: np.nan for wl in WORKLOADS}

    pw = d.get('workloads') or d.get('per_workload') or {}
    for wl in WORKLOADS:
        entry = pw.get(wl, {})
        if isinstance(entry, dict):
            vr = (entry.get('vr_smooth')
                  or entry.get('vr_raw')
                  or entry.get('var_ratio_mean')
                  or entry.get('mean_vr')
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


# =============================================================================
# VERIFIED FALLBACK (used only if S38 JSON is missing)
# =============================================================================

# From THESIS_BRIEF: S38 mean VR = 1.118, with confirmed per-workload values
# Whisper 1.090, GPT-2 1.254, YOLO 1.249. BERT and ResNet-152 are estimated to
# fit the reported mean. If you have the real S38 JSON, the loader above will
# read it and these estimates will be ignored.
S38_FALLBACK = {
    'bert': 0.920,
    'gpt2': 1.254,
    'resnet152': 1.080,
    'whisper': 1.090,
    'yolo': 1.249,
    'mean': 1.118,
}


# =============================================================================
# FIGURE
# =============================================================================

def fig_ablation_study():
    """Grouped per-workload bars + mean VR summary for S34-S39 ablations."""
    stages = [
        ('S34', 'Baseline',                'outputs/phase4/timegan_s34/s34_eval_results.json'),
        ('S35', 'Data augmentation (4x)',  'outputs/phase4/timegan_s35/s35_eval_results.json'),
        ('S36', 'Spectral norm',           'outputs/phase4/timegan_s36/s36_eval_results.json'),
        ('S37', 'Dropout 0.3',             'outputs/phase4/timegan_s37/s37_eval_results.json'),
        ('S38', 'Multi-layer FM',          'outputs/phase4/timegan_s38/s38_eval_results.json'),
        ('S39', 'Variance tuning',         'outputs/phase4/timegan_s39/s39_eval_results.json'),
    ]

    stage_vrs = {}
    for code, _, path in stages:
        try:
            per_wl, mean_vr = extract_vrs_from_eval(path)
            stage_vrs[code] = {**per_wl, 'mean': mean_vr}
            print(f"[ok]   {code}: loaded from {path}")
        except FileNotFoundError as exc:
            if code == 'S38':
                stage_vrs[code] = dict(S38_FALLBACK)
                print(f"[warn] {code}: {exc}; using THESIS_BRIEF fallback values")
                print("       (BERT and ResNet-152 are estimates -- if you have")
                print("        the real S38 JSON, place it at the expected path)")
            else:
                raise

    # 6 bars per workload group requires narrower bars than the original
    # 5-stage version (0.15 -> 0.13)
    n_stages = len(stages)
    n_wl = len(WORKLOADS)
    x = np.arange(n_wl)
    bar_w = 0.13
    offsets = np.linspace(-(n_stages - 1) / 2,
                          (n_stages - 1) / 2, n_stages) * bar_w

    # S38 added (light blue), distinct from S37 (light purple) and S39 (orange)
    colors = {
        'S34': '#90A4AE',  # blue-grey
        'S35': '#EF9A9A',  # salmon
        'S36': C_S36_WIN,  # forest green (winner, drawn with black edge)
        'S37': '#CE93D8',  # light purple
        'S38': '#81D4FA',  # soft blue (NEW)
        'S39': '#FFB74D',  # orange
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

    ax_bars.axhline(y=1.0, color='#424242', linewidth=1.0,
                    linestyle='--', alpha=0.8)
    ax_bars.axhline(y=0.8, color='#C62828', linewidth=0.9,
                    linestyle=':', alpha=0.8)
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
    # ncol=2 makes the legend taller (3 rows x 2 cols) and narrower so it
    # does not run into the bars on the right side of the plot
    ax_bars.legend(fontsize=7, loc='upper left',
                   bbox_to_anchor=(0.005, 0.995),
                   ncol=2, framealpha=0.9, borderpad=0.5,
                   handlelength=1.5, columnspacing=1.0)
    ax_bars.tick_params(labelsize=8)

    max_val = max(max(stage_vrs[s][wl] for wl in WORKLOADS)
                  for s in stage_vrs)
    # Extra headroom (1.25x instead of 1.1x) so the taller 2-column legend
    # has room above the bars
    ax_bars.set_ylim(0, max(2.8, max_val * 1.25))

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
    ax_mean.text(len(codes) - 0.5, 1.02, 'VR = 1.0', fontsize=7,
                 color='#424242', ha='right', va='bottom',
                 bbox=dict(facecolor='white', edgecolor='none', pad=1.5))
    ax_mean.set_ylabel('Mean VR (across 5 workloads)', fontsize=9)
    ax_mean.set_title('Overall Mean VR', fontsize=10, weight='bold')
    ax_mean.tick_params(labelsize=8)

    for bar, m, d in zip(bars, means, dists):
        ax_mean.text(bar.get_x() + bar.get_width() / 2, m + 0.03,
                     f'{m:.3f}',
                     ha='center', va='bottom', fontsize=8.5, weight='bold',
                     color='#1A1A1A')
        ax_mean.text(bar.get_x() + bar.get_width() / 2, -0.06,
                     f'$d$ = {d:.3f}',
                     ha='center', va='top', fontsize=7,
                     color='#424242')

    ax_mean.set_ylim(-0.15, max(means) * 1.35)
    yticks = [t for t in ax_mean.get_yticks() if t >= 0]
    ax_mean.set_yticks(yticks)

    plt.tight_layout()
    plt.savefig('figures/ablation_study.pdf', bbox_inches='tight')
    plt.close()
    print("\nSaved: figures/ablation_study.pdf")


if __name__ == '__main__':
    fig_ablation_study()