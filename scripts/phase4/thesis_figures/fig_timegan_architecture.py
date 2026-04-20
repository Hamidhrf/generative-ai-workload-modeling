"""
TimeGAN S36 Architecture Diagram
==================================
Generates figures/timegan_architecture.pdf

Three-zone layout:
    GENERATOR (top):
        [z] [r] [w] [p] --> [Concat + Proj] --> [LSTM] --> [Linear+Sigmoid] --> [Segment]

    DISCRIMINATOR (middle):
        [Segment] --fake--> [BiLSTM + Spectral Norm FC] <--real-- [Real Segments]

    LOSSES (bottom):
        [Adversarial]   [Feature Matching]   [Variance Regression]

Run from project root:
    python scripts/phase4/thesis_figures/fig_timegan_architecture.py
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

fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis('off')


def box(ax, x, y, w, h, text, fc='#E3F2FD', ec='#1565C0', fs=8.5, bold=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.08',
                       facecolor=fc, edgecolor=ec, linewidth=1.4, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fs, weight='bold' if bold else 'normal', zorder=3)


def arr(ax, x0, y0, x1, y1, label='', color='#424242', lw=1.5,
        cs='arc3,rad=0.0', lo=(0, 0.12)):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                connectionstyle=cs), zorder=4)
    if label:
        mx = (x0 + x1) / 2 + lo[0]
        my = (y0 + y1) / 2 + lo[1]
        ax.text(mx, my, label, ha='center', va='center',
                fontsize=7, color=color, style='italic', zorder=5)


# ===================== ZONE 1: GENERATOR (top) =====================

# Zone background
gen_bg = FancyBboxPatch((0.2, 4.5), 11.6, 2.3, boxstyle='round,pad=0.1',
                         facecolor='#F5F9FF', edgecolor='#1565C0',
                         linewidth=1.2, linestyle='--', zorder=0)
ax.add_patch(gen_bg)
ax.text(0.5, 6.55, 'Generator', fontsize=10, weight='bold', color='#1565C0')

# Input nodes
box(ax, 0.4, 5.2, 1.0, 0.65, '$z \\sim \\mathcal{N}(0,I)$\ndim=64',
    fc='#E8F5E9', ec='#2E7D32', fs=7)
box(ax, 0.4, 4.6, 1.0, 0.5, '$r$ (replica)',
    fc='#E8F5E9', ec='#2E7D32', fs=7)

box(ax, 1.6, 5.2, 1.0, 0.65, '$w$ (workload)',
    fc='#E8F5E9', ec='#2E7D32', fs=7)
box(ax, 1.6, 4.6, 1.0, 0.5, '$p$ (phase)',
    fc='#E8F5E9', ec='#2E7D32', fs=7)

# Processing blocks
box(ax, 3.1, 4.75, 1.8, 0.95, 'Concat +\nLinear Proj.',
    fc='#BBDEFB', ec='#1565C0', fs=8)
box(ax, 5.3, 4.75, 1.8, 0.95, 'LSTM\n(2 layers, h=128)',
    fc='#BBDEFB', ec='#1565C0', fs=8, bold=True)
box(ax, 7.5, 4.75, 1.8, 0.95, 'Linear +\nSigmoid',
    fc='#BBDEFB', ec='#1565C0', fs=8)

# Output
box(ax, 9.7, 4.75, 2.0, 0.95, 'Synthetic\nSegment (120x7)',
    fc='#C8E6C9', ec='#2E7D32', fs=8, bold=True)

# Input arrows --> Concat
arr(ax, 1.4, 5.4, 3.1, 5.35, color='#2E7D32')
arr(ax, 1.4, 4.85, 3.1, 5.1, color='#2E7D32')
arr(ax, 2.6, 5.4, 3.1, 5.35, color='#2E7D32')
arr(ax, 2.6, 4.85, 3.1, 5.1, color='#2E7D32')

# Concat --> LSTM --> Sigmoid --> Output
arr(ax, 4.9, 5.25, 5.3, 5.25, color='#1565C0')
arr(ax, 7.1, 5.25, 7.5, 5.25, color='#1565C0')
arr(ax, 9.3, 5.25, 9.7, 5.25, color='#2E7D32')


# ===================== ZONE 2: DISCRIMINATOR (middle) =====================

disc_bg = FancyBboxPatch((0.2, 2.3), 11.6, 1.9, boxstyle='round,pad=0.1',
                          facecolor='#FFFAF0', edgecolor='#E65100',
                          linewidth=1.2, linestyle='--', zorder=0)
ax.add_patch(disc_bg)
ax.text(0.5, 4.0, 'Discriminator', fontsize=10, weight='bold', color='#E65100')

# Discriminator block
box(ax, 3.5, 2.6, 4.0, 1.2, 'BiLSTM (2x128) + Spectral Norm FC\n--> Real / Fake',
    fc='#FFE0B2', ec='#E65100', fs=8.5, bold=True)

# Real Segments input
box(ax, 9.2, 2.75, 2.2, 0.9, 'Real\nSegments',
    fc='#E8F5E9', ec='#1B5E20', fs=8)

# Synthetic --> Discriminator (fake path)
arr(ax, 10.7, 4.75, 6.5, 3.8,
    'fake', color='#E65100', cs='arc3,rad=-0.15', lo=(-0.8, 0.1))

# Real --> Discriminator (real path)
arr(ax, 9.2, 3.2, 7.5, 3.2,
    'real', color='#1B5E20')

# Conditioning: r, w, p also feed into discriminator
arr(ax, 2.1, 4.6, 4.5, 3.8,
    'conditions (r, w, p)', color='#E65100', cs='arc3,rad=0.15',
    lo=(0.5, 0.1))


# ===================== ZONE 3: LOSSES (bottom) =====================

loss_bg = FancyBboxPatch((0.2, 0.2), 11.6, 1.8, boxstyle='round,pad=0.1',
                          facecolor='#FFF5F5', edgecolor='#C62828',
                          linewidth=1.2, linestyle='--', zorder=0)
ax.add_patch(loss_bg)
ax.text(0.5, 1.8, 'Training Losses', fontsize=10, weight='bold',
        color='#C62828')

box(ax, 0.5, 0.4, 2.5, 0.9, 'Adversarial\nLoss ($\\mathcal{L}_{adv}$)',
    fc='#FFCDD2', ec='#C62828', fs=8)
box(ax, 4.0, 0.4, 2.8, 0.9, 'Feature Matching\nLoss ($\\mathcal{L}_{FM}$)',
    fc='#FFCDD2', ec='#C62828', fs=8)
box(ax, 8.2, 0.4, 3.0, 0.9, 'Variance Regression\nLoss ($\\mathcal{L}_{var}$)',
    fc='#FFCDD2', ec='#C62828', fs=8)

# D output --> Adversarial Loss
arr(ax, 4.5, 2.6, 1.75, 1.3,
    'D(x) output', color='#C62828', cs='arc3,rad=0.1', lo=(0.3, 0.1))

# D intermediate features --> Feature Matching Loss
arr(ax, 5.5, 2.6, 5.4, 1.3,
    'intermediate\nfeatures', color='#C62828', lo=(0, 0.05))

# Synthetic segment variance --> Variance Regression Loss
arr(ax, 10.7, 4.75, 9.7, 1.3,
    'output\nvariance', color='#C62828', cs='arc3,rad=0.3', lo=(0.5, 0))

# Title
ax.text(6.0, 6.85, 'TimeGAN S36 Architecture', ha='center',
        fontsize=13, weight='bold')

plt.tight_layout()
plt.savefig('figures/timegan_architecture.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figures/timegan_architecture.pdf")