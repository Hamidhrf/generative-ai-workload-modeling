"""
eval_s21.py  --  S21 vs Real data evaluation plots
Self-contained: GeneratorSeg is defined inline, no imports from training script.

Run from repo root:
    python scripts/phase4/eval_s21.py
"""

import os
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn as nn

# ------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------
REPO_ROOT  = Path(__file__).resolve().parent.parent.parent
DATA_DIR   = REPO_ROOT / "data/processed/phase4/raw"
MODEL_DIR  = REPO_ROOT / "models/phase4/timegan_s21/s21_seg_vr03_fm10_ae150"
OUT_DIR    = REPO_ROOT / "outputs/phase4/timegan_s21/s21_seg_vr03_fm10_ae150/eval_plots"
WORKLOADS  = ["bert", "gpt2", "resnet152", "whisper", "yolo"]
N_SYNTH_PER_RC = 8
N_PHASES   = 6
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUT_DIR, exist_ok=True)

ALL_METRICS = [
    "pod_cpu_usage", "pod_memory_bytes", "pod_psi_cpu",
    "pod_latency_avg", "pod_throughput",
    "gpu_utilization", "gpu_memory_used", "gpu_memory_total",
    "gpu_power_watts", "gpu_temperature",
]

DROP_PER_WORKLOAD = {
    "bert":      {"gpu_memory_total", "gpu_memory_used", "pod_memory_bytes",
                  "gpu_temperature",  "gpu_power_watts"},
    "gpt2":      {"gpu_memory_total", "pod_memory_bytes",
                  "gpu_temperature",  "gpu_power_watts"},
    "resnet152": {"gpu_memory_total", "gpu_memory_used", "pod_memory_bytes",
                  "gpu_temperature",  "gpu_power_watts"},
    "whisper":   {"gpu_memory_total", "gpu_memory_used", "pod_memory_bytes",
                  "pod_throughput",   "gpu_temperature",  "gpu_power_watts"},
    "yolo":      {"gpu_memory_total", "gpu_memory_used", "pod_memory_bytes",
                  "gpu_temperature",  "gpu_power_watts"},
}

LSTM_VR = {"bert": 0.594, "gpt2": 0.676, "resnet152": 0.607,
           "whisper": 0.748, "yolo": 0.438}
S19_VR  = {"bert": 1.226, "gpt2": 1.011, "resnet152": 1.024,
           "whisper": 0.788, "yolo": 1.228}

PHASE_BOUNDARIES = [0, 96, 180, 300, 420, 600]


# ------------------------------------------------------------------
# GeneratorSeg (inlined -- identical architecture to timegan_s21.py)
# ------------------------------------------------------------------
class GeneratorSeg(nn.Module):
    def __init__(self, seg_len, n_metrics, cfg):
        super().__init__()
        self.seg_len    = seg_len
        self.n_metrics  = n_metrics
        hidden          = cfg["hidden_dim"]
        n_layers        = cfg["num_layers"]
        latent_dim      = cfg["latent_dim"]
        r_emb_dim       = cfg["replica_embed_dim"]
        ph_emb_dim      = cfg["phase_embed_dim"]
        dropout         = cfg["dropout"] if n_layers > 1 else 0.0

        self.r_embed  = nn.Sequential(nn.Linear(1, r_emb_dim), nn.Tanh())
        self.ph_embed = nn.Embedding(N_PHASES + 1, ph_emb_dim)

        init_in      = latent_dim + r_emb_dim + ph_emb_dim
        self.h_init  = nn.Sequential(nn.Linear(init_in, hidden * n_layers), nn.Tanh())
        self.c_init  = nn.Sequential(nn.Linear(init_in, hidden * n_layers), nn.Tanh())

        dec_in_dim   = latent_dim + r_emb_dim + ph_emb_dim
        self.dec_rnn = nn.LSTM(dec_in_dim, hidden, n_layers,
                               batch_first=True, dropout=dropout)
        self.out_fc  = nn.Linear(hidden, n_metrics)
        self.out_act = nn.Sigmoid()

        self.latent_dim = latent_dim
        self.n_layers   = n_layers
        self.hidden_dim = hidden

    def forward(self, r_norm, phase_idx, z=None):
        B, T, device = r_norm.shape[0], self.seg_len, r_norm.device
        if z is None:
            z = torch.randn(B, self.latent_dim, device=device)
        r_emb  = self.r_embed(r_norm.unsqueeze(-1))
        ph_emb = self.ph_embed(phase_idx)
        zrp    = torch.cat([z, r_emb, ph_emb], dim=-1)
        h0 = (self.h_init(zrp)
              .view(B, self.n_layers, self.hidden_dim)
              .permute(1, 0, 2).contiguous())
        c0 = (self.c_init(zrp)
              .view(B, self.n_layers, self.hidden_dim)
              .permute(1, 0, 2).contiguous())
        z_exp  = z.unsqueeze(1).expand(-1, T, -1)
        r_exp  = r_emb.unsqueeze(1).expand(-1, T, -1)
        ph_exp = ph_emb.unsqueeze(1).expand(-1, T, -1)
        dec_in = torch.cat([z_exp, r_exp, ph_exp], dim=-1)
        dec_out, _ = self.dec_rnn(dec_in, (h0, c0))
        return self.out_act(self.out_fc(dec_out))

    @torch.no_grad()
    def generate_segment(self, r_norm_val, phase_idx_val, device, n_samples=1):
        self.eval()
        r_t  = torch.full((n_samples,), r_norm_val, dtype=torch.float32, device=device)
        ph_t = torch.full((n_samples,), phase_idx_val, dtype=torch.long,  device=device)
        return self(r_t, ph_t).cpu().numpy()

    @torch.no_grad()
    def generate_trace(self, r_norm_val, device, n_samples=1):
        self.eval()
        segs = [self.generate_segment(r_norm_val, ph, device, n_samples)
                for ph in range(N_PHASES)]
        return np.concatenate(segs, axis=1)   # (n_samples, 720, M)


# ------------------------------------------------------------------
# DATA
# ------------------------------------------------------------------
def get_kept_indices(workload):
    drop = DROP_PER_WORKLOAD.get(workload, set())
    kept_idx, kept_names = [], []
    for i, m in enumerate(ALL_METRICS):
        if m not in drop:
            kept_idx.append(i)
            kept_names.append(m)
    return kept_idx, kept_names


def load_workload(workload):
    path = DATA_DIR / f"{workload}_traces.npz"
    d    = np.load(path, allow_pickle=True)
    kept_idx, kept_names = get_kept_indices(workload)
    traces_all     = d["traces"].astype(np.float32)
    traces         = traces_all[:, :, kept_idx]          # (N, T, M_kept)
    replica_counts = d["replica_counts"].astype(np.int32)
    train_idx      = d["train_idx"].astype(np.int32)
    val_idx        = d["val_idx"].astype(np.int32)
    return traces, replica_counts, train_idx, val_idx, kept_names


def load_generator(workload, n_metrics):
    cfg_path = MODEL_DIR / workload / "config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    gen = GeneratorSeg(seg_len=cfg["seg_len"], n_metrics=n_metrics, cfg=cfg)
    ckpt = MODEL_DIR / workload / "generator.pt"
    gen.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    gen.to(DEVICE).eval()
    return gen


def generate_all(gen, unique_rcs, n_per_rc, trace_len):
    synth = {}
    for rc in unique_rcs:
        r_norm_val = (rc - 1.0) / 9.0
        full       = gen.generate_trace(r_norm_val, DEVICE, n_samples=n_per_rc)
        synth[int(rc)] = full[:, :trace_len, :]
    return synth


def rc_colormap(unique_rcs):
    cmap = cm.get_cmap("tab10", max(len(unique_rcs), 1))
    return {int(rc): cmap(i) for i, rc in enumerate(sorted(unique_rcs))}


# ------------------------------------------------------------------
# METRICS
# ------------------------------------------------------------------
def variance_ratio_per_metric(real, synth, cap=5.0):
    var_r = real.reshape(-1, real.shape[-1]).var(axis=0)
    var_s = synth.reshape(-1, synth.shape[-1]).var(axis=0)
    return np.where(var_r > 1e-10,
                    np.clip(var_s / var_r, 0, cap),
                    np.ones_like(var_r))


def autocorr_mean(x, max_lag=60):
    x  = x - x.mean(axis=(0, 1), keepdims=True)
    ac = np.zeros(max_lag)
    ac[0] = 1.0
    for lag in range(1, max_lag):
        num    = (x[:, :x.shape[1] - lag, :] * x[:, lag:, :]).mean()
        den    = (x ** 2).mean() + 1e-10
        ac[lag] = float(num / den)
    return ac


def phase_means(traces):
    T    = traces.shape[1]
    ends = PHASE_BOUNDARIES[1:] + [T]
    return np.array([traces[:, t0:t1, :].mean(axis=(0, 1))
                     for t0, t1 in zip(PHASE_BOUNDARIES, ends)])


# ------------------------------------------------------------------
# PLOTS
# ------------------------------------------------------------------
def plot_overlay(real, rc_real, synth_by_rc, metric_names, workload, colors):
    M     = len(metric_names)
    ncols = min(4, M)
    nrows = (M + ncols - 1) // ncols
    T     = real.shape[1]
    t     = np.arange(T)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.2 * nrows))
    axes = np.array(axes).flatten()

    for mi, mname in enumerate(metric_names):
        ax = axes[mi]
        for rc, col in colors.items():
            mask = (rc_real == rc)
            if mask.sum() == 0:
                continue
            r_m = real[mask, :, mi].mean(axis=0)
            r_s = real[mask, :, mi].std(axis=0)
            ax.plot(t, r_m, color=col, lw=1.3, alpha=0.9)
            ax.fill_between(t, r_m - r_s, r_m + r_s, color=col, alpha=0.10)
            if rc in synth_by_rc:
                s_m = synth_by_rc[rc][:, :, mi].mean(axis=0)
                ax.plot(t, s_m, color=col, lw=1.3, linestyle="--", alpha=0.85)
        for t0 in PHASE_BOUNDARIES[1:]:
            ax.axvline(t0, color="gray", lw=0.6, linestyle=":")
        ax.set_title(mname, fontsize=8)
        ax.set_xlabel("timestep", fontsize=7)
        ax.tick_params(labelsize=7)

    handles = (
        [plt.Line2D([0], [0], color=col, lw=2, label=f"r={rc}")
         for rc, col in colors.items() if (rc_real == rc).any()]
        + [plt.Line2D([0], [0], color="k", lw=1.5, linestyle="-",  label="real"),
           plt.Line2D([0], [0], color="k", lw=1.5, linestyle="--", label="synthetic S21")]
    )
    fig.legend(handles=handles, loc="lower center",
               ncol=min(12, len(handles)), fontsize=7, bbox_to_anchor=(0.5, -0.01))
    for ax in axes[M:]:
        ax.set_visible(False)
    fig.suptitle(f"{workload.upper()}  --  Real (solid) vs Synthetic S21 (dashed)\n"
                 f"mean +/- std band per replica count", fontsize=11, y=1.01)
    plt.tight_layout()
    out = OUT_DIR / f"{workload}_1_overlay.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  saved {out.name}")


def plot_vr_bars(real, synth_flat, metric_names, workload):
    vr     = variance_ratio_per_metric(real, synth_flat)
    colors = ["#2ca02c" if 0.5 <= v <= 2.0 else "#d62728" for v in vr]
    y      = np.arange(len(metric_names))
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(metric_names) + 2.5))
    ax.barh(y, vr, color=colors, alpha=0.8)
    ax.axvline(1.0, color="black", lw=1.2, linestyle="--", label="VR=1.0")
    ax.axvline(0.5, color="gray",  lw=0.8, linestyle=":")
    ax.axvline(2.0, color="gray",  lw=0.8, linestyle=":", label="VR=0.5 / 2.0")
    ax.set_yticks(y)
    ax.set_yticklabels(metric_names, fontsize=9)
    ax.set_xlabel("Variance Ratio  (synthetic / real)", fontsize=9)
    ax.set_title(f"{workload.upper()}  --  Per-metric Variance Ratio\n"
                 f"mean VR = {vr.mean():.4f}   target: 0.5 - 2.0", fontsize=10)
    ax.legend(fontsize=8)
    for i, v in enumerate(vr):
        ax.text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    out = OUT_DIR / f"{workload}_2_vr_bars.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  saved {out.name}")


def plot_distributions(real, synth_flat, metric_names, workload):
    M   = len(metric_names)
    fig, axes = plt.subplots(1, M, figsize=(3.0 * M, 4.5))
    if M == 1:
        axes = [axes]
    rng = np.random.default_rng(0)
    for mi, mname in enumerate(metric_names):
        ax = axes[mi]
        rv = real[:, :, mi].flatten()
        sv = synth_flat[:, :, mi].flatten()
        if len(rv) > 40000:
            rv = rng.choice(rv, 40000, replace=False)
        if len(sv) > 40000:
            sv = rng.choice(sv, 40000, replace=False)
        parts = ax.violinplot([rv, sv], positions=[1, 2],
                              showmedians=True, showextrema=False)
        parts["bodies"][0].set_facecolor("#1f77b4")
        parts["bodies"][1].set_facecolor("#ff7f0e")
        for b in parts["bodies"]:
            b.set_alpha(0.65)
        vr = float(np.var(sv) / (np.var(rv) + 1e-12))
        ax.text(0.5, 0.97, f"VR={vr:.3f}", transform=ax.transAxes,
                ha="center", va="top", fontsize=7.5,
                color="#2ca02c" if 0.5 <= vr <= 2.0 else "#d62728")
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["real", "synth"], fontsize=8)
        ax.set_title(mname, fontsize=8, pad=2)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"{workload.upper()}  --  Value distributions\n"
                 f"blue=real   orange=synthetic S21", fontsize=10)
    plt.tight_layout()
    out = OUT_DIR / f"{workload}_3_distributions.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  saved {out.name}")


def plot_autocorr(real, synth_flat, metric_names, workload, max_lag=60):
    ac_real  = autocorr_mean(real,       max_lag)
    ac_synth = autocorr_mean(synth_flat, max_lag)
    diff     = float(np.abs(ac_real - ac_synth).mean())
    lags     = np.arange(max_lag)
    fig, ax  = plt.subplots(figsize=(7, 3.5))
    ax.plot(lags, ac_real,  color="#1f77b4", lw=1.8, label="real")
    ax.plot(lags, ac_synth, color="#ff7f0e", lw=1.8, linestyle="--",
            label="synthetic S21")
    ax.axhline(0, color="gray", lw=0.7)
    ax.set_xlabel("lag (timesteps)", fontsize=9)
    ax.set_ylabel("mean autocorrelation", fontsize=9)
    ax.set_title(f"{workload.upper()}  --  Autocorrelation structure\n"
                 f"mean |diff| across lags = {diff:.4f}", fontsize=10)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = OUT_DIR / f"{workload}_4_autocorr.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  saved {out.name}")


def plot_phase_heatmap(real, synth_flat, metric_names, workload):
    pm_real  = phase_means(real)
    pm_synth = phase_means(synth_flat)
    vmin = min(pm_real.min(), pm_synth.min())
    vmax = max(pm_real.max(), pm_synth.max())
    fig, axes = plt.subplots(1, 2, figsize=(max(8, len(metric_names) * 1.4), 4))
    for ax, data, label in zip(axes, [pm_real, pm_synth], ["Real", "Synthetic S21"]):
        im = ax.imshow(data, aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_xticks(range(len(metric_names)))
        ax.set_xticklabels(metric_names, rotation=45, ha="right", fontsize=7.5)
        ax.set_yticks(range(N_PHASES))
        ax.set_yticklabels([f"phase {i+1}" for i in range(N_PHASES)], fontsize=8)
        ax.set_title(label, fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"{workload.upper()}  --  Phase mean heatmap (normalized values)\n"
                 f"rows=phases 1-6   cols=metrics", fontsize=10)
    plt.tight_layout()
    out = OUT_DIR / f"{workload}_5_phase_heatmap.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  saved {out.name}")


def plot_summary(wl_vr_dict):
    wls     = list(wl_vr_dict.keys())
    vr_s21  = [wl_vr_dict[w]    for w in wls]
    vr_lstm = [LSTM_VR.get(w, 0) for w in wls]
    vr_s19  = [S19_VR.get(w, 0)  for w in wls]
    x = np.arange(len(wls))
    w = 0.24
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w, vr_lstm, w, label="LSTM baseline",  color="#aec7e8", alpha=0.9)
    ax.bar(x,     vr_s19,  w, label="S19",            color="#ffbb78", alpha=0.9)
    ax.bar(x + w, vr_s21,  w, label="S21 (selected)", color="#2ca02c", alpha=0.9)
    ax.axhline(1.0, color="black", lw=1.1, linestyle="--", label="VR=1.0 (perfect)")
    ax.axhline(0.8, color="gray",  lw=0.8, linestyle=":",  label="VR=0.8 (thesis target)")
    ax.set_xticks(x)
    ax.set_xticklabels([w.upper() for w in wls], fontsize=11)
    ax.set_ylabel("Mean Variance Ratio", fontsize=10)
    ax.set_title(
        "S21 vs LSTM vs S19  --  Mean Variance Ratio per Workload\n"
        f"S21 mean={np.mean(vr_s21):.3f}   LSTM mean={np.mean(vr_lstm):.3f}",
        fontsize=11)
    ax.legend(fontsize=9)
    for xi, v in zip(x + w, vr_s21):
        ax.text(xi, v + 0.01, f"{v:.3f}", ha="center",
                fontsize=8, color="#2ca02c", fontweight="bold")
    plt.tight_layout()
    out = OUT_DIR / "summary_vr_all_workloads.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  saved {out.name}")


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    print("=" * 60)
    print("S21 Evaluation Plots  --  Real vs Synthetic")
    print(f"Device : {DEVICE}")
    print(f"Output : {OUT_DIR}")
    print("=" * 60)

    wl_vr = {}

    for workload in WORKLOADS:
        print(f"\n--- {workload.upper()} ---")

        traces, rc, train_idx, val_idx, metric_names = load_workload(workload)
        T, M       = traces.shape[1], traces.shape[2]
        unique_rcs = sorted(set(rc.tolist()))
        print(f"  traces (kept): {traces.shape}  metrics: {metric_names}")
        print(f"  replica counts: {unique_rcs}")

        gen = load_generator(workload, M)
        print(f"  generator loaded  (n_metrics={M})")

        synth_by_rc = generate_all(gen, unique_rcs, N_SYNTH_PER_RC, T)
        synth_flat  = np.concatenate([synth_by_rc[r] for r in unique_rcs], axis=0)
        colors      = rc_colormap(unique_rcs)

        vr_per  = variance_ratio_per_metric(traces, synth_flat)
        vr_mean = float(vr_per.mean())
        wl_vr[workload] = vr_mean
        print(f"  mean VR = {vr_mean:.4f}")
        for mn, vr in zip(metric_names, vr_per):
            flag = "" if 0.5 <= vr <= 2.0 else "  <-- outside target"
            print(f"    {mn:30s}  VR={vr:.4f}{flag}")

        plot_overlay(traces, rc, synth_by_rc, metric_names, workload, colors)
        plot_vr_bars(traces, synth_flat, metric_names, workload)
        plot_distributions(traces, synth_flat, metric_names, workload)
        plot_autocorr(traces, synth_flat, metric_names, workload)
        plot_phase_heatmap(traces, synth_flat, metric_names, workload)

    print("\n--- SUMMARY ---")
    plot_summary(wl_vr)
    for wl, vr in wl_vr.items():
        print(f"  {wl:12s}  VR={vr:.4f}  LSTM={LSTM_VR[wl]:.3f}  "
              f"improvement={vr / LSTM_VR[wl]:.2f}x")
    print(f"\n  Overall S21 mean VR : {np.mean(list(wl_vr.values())):.4f}")
    print(f"  Overall LSTM mean VR: {np.mean(list(LSTM_VR.values())):.4f}")
    print(f"\nAll plots saved to: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()