#!/usr/bin/env python3
"""E6 report (NOT committed) - fm-weight search on the low-data-budget (n=28)
failures identified by E1.

E1 showed the r=1..7/n=28 budget breaks the frozen recipe on both Tier 2
(MIG) and Tier 3-matched (size-matched time-slicing): composite_6 VR drops
and the recipe stops beating the i.i.d. Gaussian on the 3 contention
metrics (gpu_utilization, pod_throughput, pod_cpu_usage). This script
evaluates the 22 E6 fm-override retrains (see run_e6_fmsearch.sh) against
those same two criteria to test whether the break is recoverable by tuning
lambda_fm_stat, or is a hard data-volume floor.

VR generation/smoothing logic is copied unmodified from eval_s36_tier3.py
/ eval_s36_tier2.py's evaluate_workload() (denormalize, n_gen=5 samples per
r, smooth_phase_boundaries, compute_vr_per_metric), parametrized by an
arbitrary checkpoint path instead of the frozen MODEL_PATHS dict (which
this script does not touch). Wasserstein-vs-Gaussian reuses (imports, does
not reimplement) wasserstein_reconciliation.gather_real/synth_by_r/
minmax_pooled_w and wasserstein_normalized.gaussian_minmax_per_metric, the
same primitives E1 used, with the generator swapped for the E6 checkpoint.

Run from repo root (after run_e6_fmsearch.sh has produced the checkpoints):
    PYTHONPATH=$(pwd)/scripts/utils python3 scripts/analysis/e6_fmsearch_report.py
"""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "scripts" / "utils",
          REPO_ROOT / "scripts" / "phase4",
          REPO_ROOT / "scripts" / "phase4" / "evaluation",
          REPO_ROOT / "scripts" / "analysis"):
    sys.path.insert(0, str(p))

from boundary_smoothing import smooth_phase_boundaries  # noqa: E402

WORKLOADS_ALL = ["bert", "gpt2", "resnet152", "whisper", "yolo"]
EXCLUDED_METRIC = "pod_memory_bytes"
CONTENTION_METRICS = ["gpu_utilization", "pod_throughput", "pod_cpu_usage"]
N_PHASES = 6
SEGMENT_LEN = 120
ACTUAL_BOUNDARIES = [120, 240, 360, 480, 600]
GEN_CFG = {
    "hidden_dim": 128, "num_layers": 2, "latent_dim": 64,
    "replica_embed_dim": 16, "phase_embed_dim": 8, "dropout": 0.1,
}
K = 30
GAUSSIAN_SEED = 42

TIER_CONFIG = {
    "tier2": {"pp": "postprocess_s36_tier2", "data_dir": "data/processed/tier2/unified",
              "replicas": list(range(1, 8))},
    "tier3_matched": {"pp": "postprocess_s36_tier3_matched", "data_dir": "data/processed/tier3_matched/unified",
                       "replicas": list(range(1, 8))},
}

# (tier, workload, [fm values searched])
E6_VARIANTS = [
    ("tier2", "gpt2", [0.125, 0.25, 0.5]),
    ("tier2", "resnet152", [0.125, 0.25, 0.5, 2.0, 4.0]),
    ("tier3_matched", "gpt2", [0.125, 0.25, 0.5]),
    ("tier3_matched", "resnet152", [0.125, 0.25, 0.5, 2.0, 4.0]),
    ("tier3_matched", "whisper", [0.125, 0.25, 0.5]),
    ("tier3_matched", "yolo", [0.125, 0.25, 0.5]),
]

E1_REPORT = REPO_ROOT / "outputs/analysis/e1_matched_tier3/e1_matched_tier3_report.json"

OUT_DIR = REPO_ROOT / "outputs" / "analysis" / "e6_fmsearch"
OUT_JSON = OUT_DIR / "e6_fmsearch_report.json"
OUT_MD = OUT_DIR / "e6_fmsearch_report.md"


def fmtag(fm):
    return f"fm{str(fm).replace('.', 'p')}"


class GeneratorSeg(nn.Module):
    """S27/S34/S36 generator - identical across all tiers/variants."""
    def __init__(self, seg_len, n_metrics, cfg):
        super().__init__()
        self.seg_len = seg_len
        self.n_metrics = n_metrics
        hidden = cfg["hidden_dim"]
        n_layers = cfg["num_layers"]
        latent_dim = cfg["latent_dim"]
        r_emb_dim = cfg["replica_embed_dim"]
        ph_emb_dim = cfg["phase_embed_dim"]
        dropout = cfg["dropout"] if n_layers > 1 else 0.0

        self.r_embed = nn.Sequential(nn.Linear(1, r_emb_dim), nn.Tanh())
        self.ph_embed = nn.Embedding(N_PHASES + 1, ph_emb_dim)

        init_in = latent_dim + r_emb_dim + ph_emb_dim
        self.h_init = nn.Sequential(nn.Linear(init_in, hidden * n_layers), nn.Tanh())
        self.c_init = nn.Sequential(nn.Linear(init_in, hidden * n_layers), nn.Tanh())

        dec_in_dim = latent_dim + r_emb_dim + ph_emb_dim
        self.dec_rnn = nn.LSTM(dec_in_dim, hidden, n_layers, batch_first=True, dropout=dropout)

        self.out_fc = nn.Linear(hidden, n_metrics)
        self.out_act = nn.Sigmoid()

        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden

    def forward(self, r_norm, phase_idx, z=None):
        B = r_norm.shape[0]
        T = self.seg_len
        device = r_norm.device
        if z is None:
            z = torch.randn(B, self.latent_dim, device=device)
        r_emb = self.r_embed(r_norm.unsqueeze(-1))
        ph_emb = self.ph_embed(phase_idx)
        zrp = torch.cat([z, r_emb, ph_emb], dim=-1)
        h0 = self.h_init(zrp)
        c0 = self.c_init(zrp)
        h0 = h0.view(B, self.n_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        c0 = c0.view(B, self.n_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        z_exp = z.unsqueeze(1).expand(-1, T, -1)
        r_exp = r_emb.unsqueeze(1).expand(-1, T, -1)
        ph_exp = ph_emb.unsqueeze(1).expand(-1, T, -1)
        dec_in = torch.cat([z_exp, r_exp, ph_exp], dim=-1)
        dec_out, _ = self.dec_rnn(dec_in, (h0, c0))
        return self.out_act(self.out_fc(dec_out))

    @torch.no_grad()
    def generate_trace(self, r_norm_val, device, n_samples=1):
        self.eval()
        segments = []
        for ph in range(N_PHASES):
            r_norm = torch.full((n_samples,), r_norm_val, dtype=torch.float32, device=device)
            ph_idx = torch.full((n_samples,), ph, dtype=torch.long, device=device)
            seg = self(r_norm, ph_idx)
            segments.append(seg.cpu().numpy())
        full_trace = np.concatenate(segments, axis=1)
        return full_trace[:, :715, :]


def load_e6_generator(tier, workload, fm, n_metrics, device="cpu"):
    ckpt = REPO_ROOT / f"models/phase4/timegan_s36_{tier}_e6_{fmtag(fm)}/{workload}/generator.pt"
    if not ckpt.exists():
        return None
    gen = GeneratorSeg(SEGMENT_LEN, n_metrics, GEN_CFG).to(device)
    gen.load_state_dict(torch.load(ckpt, map_location=device))
    gen.eval()
    return gen


def denormalize(traces_norm, norm_params, kept_names):
    traces = traces_norm.copy()
    for i, metric in enumerate(kept_names):
        if metric in norm_params:
            traces[:, :, i] = traces[:, :, i] * norm_params[metric]["std"] + norm_params[metric]["mean"]
    return traces


def compute_composite6_vr(pp, ev, tier, workload, fm, device="cpu", window_size=5, n_gen=5):
    """Mirrors eval_s36_tier3.py's evaluate_workload() exactly, but with
    an arbitrary E6 checkpoint instead of the frozen MODEL_PATHS entry."""
    cfg = TIER_CONFIG[tier]
    data = np.load(REPO_ROOT / cfg["data_dir"] / "combined_dataset.npz", allow_pickle=True)
    with open(REPO_ROOT / cfg["data_dir"] / "combined_normalization.json") as f:
        norm = json.load(f)

    workload_id = WORKLOADS_ALL.index(workload)
    all_traces = data["traces"]
    all_rc = data["replica_counts"]
    all_wl = data["workload_ids"]
    val_idx = data["val_idx"]

    wl_mask = (all_wl == workload_id)
    traces = all_traces[wl_mask]
    rc = all_rc[wl_mask]

    kept_idx = pp.TRAINED_INDICES
    kept_names = pp.TRAINED_NAMES
    n_metrics = len(kept_names)
    traces = traces[:, :, kept_idx].astype(np.float32)

    orig_to_new = {}
    new_idx = 0
    for old_idx in range(len(all_traces)):
        if wl_mask[old_idx]:
            orig_to_new[old_idx] = new_idx
            new_idx += 1
    new_val_idx = np.array([orig_to_new[i] for i in val_idx if i in orig_to_new])

    real_traces_val = traces[new_val_idx]
    real_rc_val = rc[new_val_idx]

    generator = load_e6_generator(tier, workload, fm, n_metrics, device)
    if generator is None:
        return None

    unique_r = np.unique(real_rc_val)
    syn_traces_smooth = []
    for r_val in unique_r:
        r_norm = (r_val - 1.0) / 9.0
        traces_gen = generator.generate_trace(r_norm, device, n_samples=n_gen)
        for trace in traces_gen:
            smooth_trace = smooth_phase_boundaries(trace, ACTUAL_BOUNDARIES, window_size)
            syn_traces_smooth.append(smooth_trace)
    syn_traces_smooth = np.array(syn_traces_smooth)

    real_denorm = denormalize(real_traces_val, norm[workload], kept_names)
    syn_smooth_denorm = denormalize(syn_traces_smooth, norm[workload], kept_names)

    vr_smooth = ev.compute_vr_per_metric(real_denorm, syn_smooth_denorm)
    c6 = float(np.mean([v for v, n in zip(vr_smooth, kept_names) if n != EXCLUDED_METRIC]))
    c7 = float(np.mean(vr_smooth))
    return {"c6": c6, "c7": c7, "per_metric": dict(zip(kept_names, [float(v) for v in vr_smooth]))}


def compute_contention_wasserstein(pp, wr, wn, tier, workload, fm, device="cpu"):
    """Mirrors E1's Q2: K=30-draw S36 vs fixed-seed Gaussian, minmax space,
    per metric, restricted to the 3 contention metrics for reporting."""
    cfg = TIER_CONFIG[tier]
    memory_stats = pp.compute_memory_stats(cfg["data_dir"])
    dropped_stats = pp.compute_dropped_metric_stats(cfg["data_dir"])
    names = pp.TRAINED_NAMES
    n_metrics = len(names)

    generator = load_e6_generator(tier, workload, fm, n_metrics, device)
    if generator is None:
        return None

    real_by_r, norm_params = wr.gather_real(pp, cfg, workload)
    gaussian_ws = wn.gaussian_minmax_per_metric(real_by_r, names, norm_params, seed=GAUSSIAN_SEED)

    # wr.synth_by_r expects pp.load_generator-compatible generator + pp's own
    # generate_raw_trace/postprocess_trace (module-level functions, generator-
    # agnostic) -- monkeypatch-free: it takes the generator object directly.
    per_metric_draws = {m: [] for m in names}
    for k in range(K):
        syn = wr.synth_by_r(pp, generator, workload, cfg, real_by_r, memory_stats, dropped_stats, seed=k)
        _, per_metric = wr.minmax_pooled_w(real_by_r, syn, names, norm_params)
        for m in names:
            per_metric_draws[m].append(per_metric[m])

    s36_mean = {m: float(np.mean(v)) for m, v in per_metric_draws.items()}
    beats = {m: bool(s36_mean[m] < gaussian_ws[m]) for m in names}
    return {"gaussian": gaussian_ws, "s36_mean": s36_mean, "beats_gaussian": beats}


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pp_mods = {t: importlib.import_module(cfg["pp"]) for t, cfg in TIER_CONFIG.items()}
    wr = importlib.import_module("wasserstein_reconciliation")
    wn = importlib.import_module("wasserstein_normalized")
    ev = importlib.import_module("eval_s36")

    frozen_c6 = {}
    if E1_REPORT.exists():
        e1 = json.loads(E1_REPORT.read_text())
        for tier in TIER_CONFIG:
            frozen_c6[tier] = {wl: v["c6"] for wl, v in e1["q1_composite6_vr_headline"].get(tier, {}).items()}

    results = {}  # (tier, workload) -> {fm: {...}}
    for tier, workload, fm_list in E6_VARIANTS:
        pp = pp_mods[tier]
        results[(tier, workload)] = {}
        print(f"\n{'='*70}\n{tier} / {workload}  (frozen c6={frozen_c6.get(tier, {}).get(workload, 'n/a')})\n{'='*70}")
        for fm in fm_list:
            print(f"  fm={fm} ...", flush=True)
            vr = compute_composite6_vr(pp, ev, tier, workload, fm)
            if vr is None:
                print(f"    MISSING checkpoint for fm={fm}, skipping")
                continue
            wass = compute_contention_wasserstein(pp, wr, wn, tier, workload, fm)
            n_beats = sum(1 for m in CONTENTION_METRICS if wass["beats_gaussian"].get(m))
            results[(tier, workload)][fm] = {
                "c6": vr["c6"], "c7": vr["c7"], "per_metric_vr": vr["per_metric"],
                "contention_beats_gaussian": {m: wass["beats_gaussian"][m] for m in CONTENTION_METRICS},
                "contention_gaussian": {m: wass["gaussian"][m] for m in CONTENTION_METRICS},
                "contention_s36": {m: wass["s36_mean"][m] for m in CONTENTION_METRICS},
                "n_contention_metrics_beating_gaussian": n_beats,
            }
            print(f"    c6={vr['c6']:.4f}  contention beats-Gaussian: {n_beats}/3 "
                  f"{results[(tier, workload)][fm]['contention_beats_gaussian']}", flush=True)

    # ---- Pick best fm per (tier, workload): most contention metrics beating
    # Gaussian first, then composite_6 closest to 1.0 as tiebreak. ----
    best = {}
    for (tier, workload), fm_results in results.items():
        if not fm_results:
            continue
        ranked = sorted(fm_results.items(),
                         key=lambda kv: (-kv[1]["n_contention_metrics_beating_gaussian"], abs(kv[1]["c6"] - 1.0)))
        best_fm, best_rec = ranked[0]
        best[(tier, workload)] = {
            "best_fm": best_fm, "c6": best_rec["c6"],
            "n_contention_metrics_beating_gaussian": best_rec["n_contention_metrics_beating_gaussian"],
            "contention_beats_gaussian": best_rec["contention_beats_gaussian"],
            "frozen_c6": frozen_c6.get(tier, {}).get(workload),
            "recovers_beats_gaussian_fully": best_rec["n_contention_metrics_beating_gaussian"] == 3,
        }

    # ---- Write JSON ----
    out = {
        "config": {"K": K, "gaussian_seed": GAUSSIAN_SEED, "contention_metrics": CONTENTION_METRICS,
                   "excluded_metric_c6": EXCLUDED_METRIC},
        "frozen_c6_reference": frozen_c6,
        "per_variant": {f"{t}/{w}": {str(fm): v for fm, v in fm_res.items()}
                         for (t, w), fm_res in results.items()},
        "best_per_cell": {f"{t}/{w}": v for (t, w), v in best.items()},
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))

    # ---- Write MD ----
    lines = ["# E6 report - fm-weight search on low-data-budget failures (NOT committed)\n"]
    lines.append("Frozen S27 hyperparameters held fixed except lambda_fm_stat, searched per (tier, workload). "
                  "'beats Gaussian' = S36's minmax-space Wasserstein distance to real is SMALLER than the "
                  "i.i.d. per-metric Gaussian baseline's, on that contention metric.\n")

    lines.append("## All variants\n")
    lines.append("| tier | workload | fm | composite_6 VR | gpu_util beats G? | throughput beats G? | cpu beats G? | #/3 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for (tier, workload, fm_list) in E6_VARIANTS:
        for fm in fm_list:
            rec = results.get((tier, workload), {}).get(fm)
            if rec is None:
                lines.append(f"| {tier} | {workload} | {fm} | MISSING | | | | |")
                continue
            b = rec["contention_beats_gaussian"]
            lines.append(f"| {tier} | {workload} | {fm} | {rec['c6']:.4f} | {b['gpu_utilization']} | "
                          f"{b['pod_throughput']} | {b['pod_cpu_usage']} | {rec['n_contention_metrics_beating_gaussian']}/3 |")

    lines.append("\n## Best fm per (tier, workload)\n")
    lines.append("| tier | workload | frozen c6 | best fm | best c6 | contention beats-Gaussian | recovers fully (3/3)? |")
    lines.append("|---|---|---|---|---|---|---|")
    for (tier, workload), b in best.items():
        frozen_str = f"{b['frozen_c6']:.4f}" if b["frozen_c6"] is not None else "n/a"
        lines.append(f"| {tier} | {workload} | {frozen_str} | {b['best_fm']} | {b['c6']:.4f} | "
                      f"{b['n_contention_metrics_beating_gaussian']}/3 | {b['recovers_beats_gaussian_fully']} |")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")

    print("\nSUMMARY:")
    for (tier, workload), b in best.items():
        verdict = "RECOVERS" if b["recovers_beats_gaussian_fully"] else f"partial ({b['n_contention_metrics_beating_gaussian']}/3)"
        print(f"  {tier:14s} {workload:10s} best_fm={b['best_fm']:<6} c6={b['c6']:.4f}  {verdict}")


if __name__ == "__main__":
    main()
