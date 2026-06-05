"""
System Architecture Diagram
============================
Generates figures/system_architecture.pdf

Layout:
    Top row:     [Workload Pods]  ----uses GPU--->  [NVIDIA A16 GPU]
    Middle row:  [cAdvisor] [App Exporter] [DCGM Exporter] [Node Exporter]
    Bottom row:  [Prometheus] ---query--> [Grafana]

Arrow directions (data flow):
    Pods --> GPU                  (pods use GPU for inference)
    Pods ..> cAdvisor             (cAdvisor monitors container CPU/mem)
    Pods ..> App Exporter         (pods expose /metrics endpoint)
    GPU  --> DCGM Exporter        (DCGM reads hardware counters)
    All 4 exporters --> Prometheus (Prometheus scrapes all exporters)
    Prometheus --> Grafana         (Grafana queries Prometheus)

Run from project root:
    python scripts/phase4/thesis_figures/fig_system_architecture.py
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

os.makedirs('figures', exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

fig, ax = plt.subplots(figsize=(11, 7.5))
ax.set_xlim(0, 11)
ax.set_ylim(0, 7.5)
ax.axis('off')


def add_box(ax, x, y, w, h, title, subtitle=None, fc='#E3F2FD', ec='#1565C0'):
    box = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.1',
                         facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=2)
    ax.add_patch(box)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.62, title, ha='center', va='center',
                fontsize=9, weight='bold', zorder=3)
        ax.text(x + w / 2, y + h * 0.3, subtitle, ha='center', va='center',
                fontsize=7.5, color='#555', style='italic', zorder=3)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha='center', va='center',
                fontsize=9, weight='bold', zorder=3)


def add_arrow(ax, x1, y1, x2, y2, label='', color='#424242',
              label_offset=(0, 0.15), style='arc3,rad=0.0'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.8,
                                connectionstyle=style),
                zorder=4)
    if label:
        mx = (x1 + x2) / 2 + label_offset[0]
        my = (y1 + y2) / 2 + label_offset[1]
        ax.text(mx, my, label, ha='center', va='center',
                fontsize=7, color=color, style='italic', zorder=5)


# === TOP ROW: Workload Pods + GPU ===

# Workload Pods
add_box(ax, 0.5, 5.5, 3.5, 1.5,
        'Workload Pods (r = 1...10)',
        'BERT | GPT-2 | ResNet-152 | Whisper | YOLO',
        fc='#E8F5E9', ec='#2E7D32')

# GPU
add_box(ax, 6.5, 5.5, 3.5, 1.5,
        'NVIDIA A16 GPU',
        '16 GB GDDR6 | 10 Time-Slices',
        fc='#FFF3E0', ec='#E65100')

# Pods --> GPU
add_arrow(ax, 4.0, 6.25, 6.5, 6.25,
          'uses GPU\n(time-sliced)', color='#2E7D32')

# === MIDDLE ROW: Exporters ===

add_box(ax, 0.3, 3.2, 2.0, 1.2,
        'cAdvisor', 'CPU / Memory\nper container',
        fc='#F3E5F5', ec='#7B1FA2')

add_box(ax, 2.8, 3.2, 2.0, 1.2,
        'App Exporter', 'Latency / Throughput\n/ PSI per pod',
        fc='#E1F5FE', ec='#0277BD')

add_box(ax, 5.6, 3.2, 2.2, 1.2,
        'DCGM Exporter', 'GPU Util / Power\n/ Memory / Temp',
        fc='#FBE9E7', ec='#BF360C')

add_box(ax, 8.3, 3.2, 2.2, 1.2,
        'Node Exporter', 'System-level\nCPU / Mem / Disk',
        fc='#ECEFF1', ec='#546E7A')

# Pods --> cAdvisor (cAdvisor monitors pods)
add_arrow(ax, 1.3, 5.5, 1.3, 4.4,
          'container\nmetrics', color='#7B1FA2', label_offset=(0.65, 0))

# Pods --> App Exporter (pods expose /metrics)
add_arrow(ax, 2.8, 5.5, 3.8, 4.4,
          'expose\n/metrics', color='#0277BD', label_offset=(0.55, 0))

# GPU --> DCGM Exporter (DCGM reads HW counters)
add_arrow(ax, 6.7, 5.5, 6.7, 4.4,
          'hardware\ncounters', color='#BF360C', label_offset=(0.6, 0))

# === BOTTOM ROW: Prometheus + Grafana ===

add_box(ax, 2.5, 0.8, 3.0, 1.4,
        'Prometheus', 'Scrape & Store (5s interval)',
        fc='#FFCCBC', ec='#E64A19')

add_box(ax, 7.0, 0.8, 2.5, 1.4,
        'Grafana', 'Dashboards',
        fc='#FCE4EC', ec='#C62828')

# All exporters --> Prometheus
add_arrow(ax, 1.3, 3.2, 3.5, 2.2,
          '', color='#E64A19', style='arc3,rad=0.1')
add_arrow(ax, 3.8, 3.2, 4.0, 2.2,
          '', color='#E64A19')
add_arrow(ax, 6.7, 3.2, 5.0, 2.2,
          '', color='#E64A19', style='arc3,rad=-0.1')
add_arrow(ax, 9.4, 3.2, 5.3, 2.2,
          '', color='#E64A19', style='arc3,rad=-0.2')

# Label "scrape" once in the middle of the arrow cluster
ax.text(4.0, 2.75, 'scrape /metrics', ha='center', fontsize=7.5,
        color='#E64A19', style='italic')

# Prometheus --> Grafana
add_arrow(ax, 5.5, 1.5, 7.0, 1.5,
          'query', color='#C62828')

# === Kubernetes cluster bounding box ===
cluster = FancyBboxPatch((0.1, 0.4), 10.7, 7.0,
                          boxstyle='round,pad=0.1',
                          facecolor='none', edgecolor='#90A4AE',
                          linewidth=1.5, linestyle='--', zorder=1)
ax.add_patch(cluster)
ax.text(5.5, 7.2, 'Single-Node Kubernetes Cluster (v1.34.0)',
        ha='center', fontsize=11, weight='bold', color='#37474F')

ax.text(0.3, 0.55, 'Ubuntu 24.04 LTS | 16 vCPU | 62.5 GB RAM',
        fontsize=7, color='#78909C')

plt.tight_layout()
plt.savefig('figures/system_architecture.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figures/system_architecture.pdf")