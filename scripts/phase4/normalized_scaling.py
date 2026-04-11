import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

report_path = Path("outputs/phase4/validation/s36/validation_report.json")
with open(report_path) as f:
    report = json.load(f)

WORKLOADS = ['gpt2', 'whisper']
METRICS = ['CPU Usage', 'CPU Pressure (PSI)', 'Latency (Avg)', 
           'Throughput', 'GPU Utilization', 'GPU Power']
UNITS = ['cores', 'fraction', 'seconds', 'req/s', '%', 'W']

colors = {'gpt2': '#ff7f0e', 'whisper': '#d62728'}
labels = {'gpt2': 'GPT-2', 'whisper': 'Whisper'}

fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

for pi, (metric_name, unit) in enumerate(zip(METRICS, UNITS)):
    row, col = pi // 3, pi % 3
    ax = fig.add_subplot(gs[row, col])

    # Extrapolation zone
    ax.axvspan(10.5, 55, alpha=0.08, color='red')
    ax.axvline(10.5, color='red', ls=':', alpha=0.4, lw=1)

    for wname in WORKLOADS:
        c = colors[wname]
        lbl = labels[wname]
        w_data = report['workloads'][wname]

        real_rs, real_vals = [], []
        synth_rs, synth_vals = [], []

        for r_str, r_data in w_data.get('validated_replicas', {}).items():
            r = int(r_str)
            if metric_name in r_data and 'real_mean' in r_data[metric_name]:
                real_rs.append(r)
                real_vals.append(r_data[metric_name]['real_mean'])
                synth_rs.append(r)
                synth_vals.append(r_data[metric_name]['synth_mean'])

        for r_str, r_data in w_data.get('extrapolation_replicas', {}).items():
            r = int(r_str)
            if metric_name in r_data and 'synth_mean' in r_data[metric_name]:
                synth_rs.append(r)
                synth_vals.append(r_data[metric_name]['synth_mean'])

        real_order = np.argsort(real_rs)
        real_rs = [real_rs[i] for i in real_order]
        real_vals = [real_vals[i] for i in real_order]
        synth_order = np.argsort(synth_rs)
        synth_rs = [synth_rs[i] for i in synth_order]
        synth_vals = [synth_vals[i] for i in synth_order]

        ax.errorbar(real_rs, real_vals, fmt='o-', color=c, lw=2, 
                     markersize=6, alpha=0.85, label=f'{lbl} (real)', zorder=3)
        ax.plot(synth_rs, synth_vals, 's--', color=c, lw=1.5, 
                markersize=5, alpha=0.6, label=f'{lbl} (synth)', zorder=2)

    ax.set_title(metric_name, fontsize=12, fontweight='bold')
    ax.set_xlabel('Replica Count', fontsize=10)
    ax.set_ylabel(unit, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)

    if pi == 0:
        ax.legend(fontsize=8, loc='best')

# Hide unused subplots
for i in [6, 7, 8]:
    ax = fig.add_subplot(gs[i // 3, i % 3])
    ax.axis('off')

fig.suptitle(
    "Scaling Curves: GPT-2 (GPU-bound) vs Whisper (CPU-bound)\n"
    "Solid = Real (r=1-10), Dashed = Synthetic, Red zone = Extrapolation (r>10)",
    fontsize=13, fontweight='bold', y=0.995)

out_path = "outputs/phase4/validation/s36/gpt2_whisper_scaling.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {out_path}")