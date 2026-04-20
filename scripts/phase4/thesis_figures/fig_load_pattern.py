"""
Load Pattern Diagram
====================
Generates figures/load_pattern.pdf

Shows the 6-phase business-day load pattern used in all experiments.
Y-axis is RELATIVE REQUEST RATE (0-1), NOT GPU utilization.
This is an abstract illustration of the traffic shape, not real data.

Run from project root:
    python scripts/phase4/thesis_figures/fig_load_pattern.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

os.makedirs('figures', exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

fig, ax = plt.subplots(1, 1, figsize=(8, 3.5))

# Phase boundaries in minutes (matching thesis Section 3.4)
boundaries = [0, 8, 15, 25, 35, 50, 60]

# Relative request rate per phase (abstract, not measured)
rates = [0.3, 0.6, 0.85, 0.8, 1.0, 0.4]

phase_names = [
    'Phase 0\nNight/\nWarmup',
    'Phase 1\nMorning\nRamp-up',
    'Phase 2\nMidday\nHigh',
    'Phase 3\nAfternoon\nSustained',
    'Phase 4\nPeak/\nEvening',
    'Phase 5\nCool-\ndown',
]
phase_colors = ['#E3F2FD', '#BBDEFB', '#90CAF9', '#64B5F6', '#42A5F5', '#E3F2FD']

# Build smooth curve with transitions
t = np.linspace(0, 60, 720)
rate_curve = np.zeros_like(t)
for i in range(len(rates)):
    mask = (t >= boundaries[i]) & (t < boundaries[i + 1])
    rate_curve[mask] = rates[i]

# Smooth with Gaussian filter for realistic ramp transitions
rate_smooth = gaussian_filter1d(rate_curve, sigma=8)

# Shade phases
for i in range(len(rates)):
    ax.axvspan(boundaries[i], boundaries[i + 1], alpha=0.15,
               color=phase_colors[i])

# Plot the smooth rate curve
ax.plot(t, rate_smooth, color='#1565C0', linewidth=2.2)
ax.fill_between(t, 0, rate_smooth, alpha=0.12, color='#1565C0')

# Phase boundary dashed lines
for b in boundaries[1:-1]:
    ax.axvline(x=b, color='#BDBDBD', linestyle='--', linewidth=0.8)

# Phase labels BELOW the plot (between x-axis ticks and xlabel)
# Use a separate row at the bottom of the figure
for i in range(len(rates)):
    mid = (boundaries[i] + boundaries[i + 1]) / 2
    ax.text(mid, -0.14, phase_names[i], ha='center', va='top', fontsize=6.5,
            family='sans-serif', color='#424242',
            transform=ax.get_xaxis_transform())

ax.set_xlabel('Time (minutes)', fontsize=10, family='sans-serif', labelpad=45)
ax.set_ylabel('Relative Request Rate', fontsize=10, family='sans-serif')
ax.set_xlim(0, 60)
ax.set_ylim(0, 1.15)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig('figures/load_pattern.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figures/load_pattern.pdf")