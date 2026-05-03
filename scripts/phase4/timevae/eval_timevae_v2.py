"""
TimeVAE v2 Evaluation
======================
Evaluates committed TimeVAE v2 checkpoints in models/phase4/timevae/timevae_v2/.
Mirrors the in-script evaluation protocol from timevae_v1.py: generates traces
on training-replica counts, denormalizes, computes per-metric variance ratio
in physical units. This produces a JSON in the same schema as
outputs/phase4/timevae/timevae_results.json so v2 numbers are directly
comparable to v1 and v3.

Why this script exists separately: timevae_v2.py runs end-to-end (train + eval)
but its output JSON was never committed (only plots/ exists in
outputs/phase4/timevae/timevae_v2/). This script just runs the eval portion
against existing checkpoints.

Usage:
    python eval_timevae_v2.py
    python eval_timevae_v2.py --workloads bert
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

DATA_RAW_DIR = Path("data/processed/phase4/raw")
OUTPUT_DIR = Path("outputs/phase4/timevae/timevae_v2")
MODEL_DIR = Path("models/phase4/timevae/timevae_v2")

WORKLOADS = ["bert", "gpt2", "resnet152", "whisper", "yolo"]

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

DEFAULT_PHASE_BOUNDARIES = [0, 96, 180, 300, 420, 600]
N_PHASES = 6


def build_phase_sequence(seq_len, boundaries):
    phase_seq = np.zeros(seq_len, dtype=np.int64)
    for phase_idx, start in enumerate(boundaries):
        end = boundaries[phase_idx + 1] if phase_idx + 1 < len(boundaries) else seq_len
        phase_seq[start:end] = phase_idx
    return phase_seq


def get_kept_metrics(workload):
    drop = DROP_PER_WORKLOAD[workload]
    kept_idx = [i for i, m in enumerate(ALL_METRICS) if m not in drop]
    kept_names = [ALL_METRICS[i] for i in kept_idx]
    return kept_idx, kept_names


def load_raw_data(workload):
    path = DATA_RAW_DIR / f"{workload}_traces.npz"
    data = np.load(path, allow_pickle=True)
    norm_path = DATA_RAW_DIR / f"{workload}_normalization.json"
    with open(norm_path) as f:
        norm = json.load(f)
    return data, norm


def denormalize(traces_norm, metric_names, norm_params):
    lookup = norm_params["params"] if "params" in norm_params else norm_params
    out = traces_norm.copy().astype(np.float64)
    for j, m in enumerate(metric_names):
        if m not in lookup:
            continue
        mn = lookup[m]["min"]
        mx = lookup[m]["max"]
        out[:, :, j] = traces_norm[:, :, j] * (mx - mn) + mn
    return out


def compute_variance_ratio(real, synthetic, cap=5.0):
    vr = []
    for i in range(real.shape[2]):
        var_r = np.var(real[:, :, i])
        var_s = np.var(synthetic[:, :, i])
        ratio = var_s / (var_r + 1e-10)
        vr.append(min(float(ratio), cap))
    return np.array(vr), float(np.mean(vr))


def compute_autocorr_similarity(real, synthetic, max_lag=20):
    diffs = []
    for i in range(real.shape[2]):
        r_flat = real[:, :, i].flatten()
        s_flat = synthetic[:, :, i].flatten()
        for lag in range(1, max_lag + 1):
            ac_r = float(np.corrcoef(r_flat[:-lag], r_flat[lag:])[0, 1])
            ac_s = float(np.corrcoef(s_flat[:-lag], s_flat[lag:])[0, 1])
            if not (np.isnan(ac_r) or np.isnan(ac_s)):
                diffs.append(abs(ac_r - ac_s))
    return float(np.mean(diffs)) if diffs else float("nan")


# ---- v2 model architecture (must match exactly to load checkpoints) ----

class TimeVAEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, latent_dim, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0)
        self.fc_mu = nn.Linear(2 * hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(2 * hidden_dim, latent_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_fwd = h_n[-2]
        h_bwd = h_n[-1]
        h = torch.cat([h_fwd, h_bwd], dim=1)
        return self.fc_mu(h), self.fc_logvar(h)


class TimeVAEDecoder(nn.Module):
    def __init__(self, latent_dim, r_embed_dim, phase_embed_dim,
                 hidden_dim, num_layers, output_dim, n_phases, dropout):
        super().__init__()
        self.phase_embed = nn.Embedding(n_phases, phase_embed_dim)
        self.r_proj = nn.Sequential(nn.Linear(1, r_embed_dim), nn.Tanh())
        cond_dim = latent_dim + r_embed_dim
        self.h0 = nn.Linear(cond_dim, num_layers * hidden_dim)
        self.c0 = nn.Linear(cond_dim, num_layers * hidden_dim)
        lstm_input = output_dim + r_embed_dim + phase_embed_dim
        self.lstm = nn.LSTM(
            lstm_input, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.r_embed_dim = r_embed_dim

    def _init_hidden(self, z, r_embed, device):
        cond = torch.cat([z, r_embed], dim=1)
        B = z.size(0)
        h0 = self.h0(cond).view(B, self.num_layers, self.hidden_dim)
        h0 = h0.permute(1, 0, 2).contiguous()
        c0 = self.c0(cond).view(B, self.num_layers, self.hidden_dim)
        c0 = c0.permute(1, 0, 2).contiguous()
        return h0, c0

    def forward_generate(self, z, r_norm, phase_seq, device):
        T = len(phase_seq)
        r_embed = self.r_proj(r_norm.unsqueeze(1))
        h, c = self._init_hidden(z, r_embed, device)

        ph_t = torch.tensor(phase_seq, dtype=torch.long, device=device)
        ph_emb = self.phase_embed(ph_t)

        x_t = torch.zeros(1, self.output_dim, device=device)
        outs = []
        for t in range(T):
            inp = torch.cat([x_t, r_embed, ph_emb[t:t+1]], dim=1)
            inp = inp.unsqueeze(1)
            out_h, (h, c) = self.lstm(inp, (h, c))
            x_t = torch.sigmoid(self.fc_out(out_h.squeeze(1)))
            outs.append(x_t)

        return torch.cat(outs, dim=0).cpu().numpy()


class TimeVAE(nn.Module):
    def __init__(self, n_metrics, cfg):
        super().__init__()
        self.encoder = TimeVAEEncoder(
            n_metrics, cfg["enc_hidden_dim"],
            cfg["enc_num_layers"], cfg["latent_dim"], cfg["dropout"])
        self.decoder = TimeVAEDecoder(
            cfg["latent_dim"], cfg["r_embed_dim"], cfg["phase_embed_dim"],
            cfg["dec_hidden_dim"], cfg["dec_num_layers"],
            n_metrics, N_PHASES, cfg["dropout"])
        self.latent_dim = cfg["latent_dim"]

    def generate(self, r_norm_val, n_pods, phase_seq, device):
        self.eval()
        with torch.no_grad():
            traces = []
            for _ in range(n_pods):
                z = torch.randn(1, self.latent_dim, device=device)
                r_norm = torch.tensor([r_norm_val], dtype=torch.float32, device=device)
                trace = self.decoder.forward_generate(z, r_norm, phase_seq, device)
                traces.append(trace)
        return np.stack(traces, axis=0)


def evaluate_workload(workload, device, n_gen=5):
    print(f"\n{'='*60}")
    print(f"  {workload.upper()}")
    print(f"{'='*60}")

    # Load saved config to know exact hyperparameters used at training
    config_path = MODEL_DIR / workload / "config.json"
    if not config_path.exists():
        print(f"  ERROR: config not found at {config_path}")
        return None
    with open(config_path) as f:
        cfg = json.load(f)

    # Load raw data
    data, norm = load_raw_data(workload)
    traces = data["traces"].astype(np.float64)
    replica_counts = data["replica_counts"]
    train_idx = data["train_idx"]

    kept_idx, kept_names = get_kept_metrics(workload)
    raw_kept = traces[:, :, kept_idx].astype(np.float32)
    n_metrics = len(kept_names)

    norm_params = norm["params"] if "params" in norm else norm

    print(f"  Metrics ({n_metrics}): {kept_names}")
    print(f"  Train pods: {len(train_idx)}")

    # Build model with the saved config
    model = TimeVAE(n_metrics, cfg).to(device)
    model_path = MODEL_DIR / workload / "model.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"  Loaded: {model_path}")

    # Build phase sequence
    boundaries = cfg.get("phase_boundaries", DEFAULT_PHASE_BOUNDARIES)
    seq_len = cfg.get("seq_len", 715)
    phase_seq = build_phase_sequence(seq_len, boundaries)

    # Generate (in-sample protocol matching v1/v3)
    real_norm = raw_kept[train_idx]
    real_orig = denormalize(real_norm, kept_names, norm_params)
    train_r = replica_counts[train_idx]

    unique_r = np.unique(train_r)
    syn_list = []
    syn_r_list = []

    for r_val in unique_r:
        r_norm_val = float((r_val - 1.0) / 9.0)
        for _ in range(n_gen):
            pkg = model.generate(r_norm_val, int(r_val), phase_seq, device)
            for pod_trace in pkg:
                syn_list.append(pod_trace)
                syn_r_list.append(int(r_val))

    syn_norm = np.stack(syn_list, axis=0)
    syn_orig = denormalize(syn_norm, kept_names, norm_params)

    vr, vr_mean = compute_variance_ratio(real_orig, syn_orig)
    ac_diff = compute_autocorr_similarity(real_orig, syn_orig)

    print(f"  var_ratio_mean = {vr_mean:.4f}  (target: close to 1.0)")
    print(f"  autocorr_diff  = {ac_diff:.4f}")
    print(f"  Per-metric VR:")
    for j, m in enumerate(kept_names):
        print(f"    {m:<22} {vr[j]:.4f}")

    return {
        "workload": workload,
        "model": "timevae_v2",
        "var_ratio_mean": vr_mean,
        "var_ratio_per_metric": vr.tolist(),
        "autocorr_diff": ac_diff,
        "metrics_generated": list(kept_names),
        "n_metrics_generated": n_metrics,
        "n_real_train_pods": int(len(train_idx)),
        "n_syn_pods": int(len(syn_list)),
    }


def main():
    parser = argparse.ArgumentParser(description="TimeVAE v2 evaluation")
    parser.add_argument("--workloads", nargs="+", default=WORKLOADS)
    parser.add_argument("--n-gen", type=int, default=5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    torch.manual_seed(42)

    all_results = []
    for wl in args.workloads:
        result = evaluate_workload(wl, device, args.n_gen)
        if result:
            all_results.append(result)

    print(f"\n{'='*60}")
    print("TIMEVAE v2 SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        print(f"  {r['workload']:<12}  VR = {r['var_ratio_mean']:.4f}")
    if all_results:
        mean_vr = float(np.mean([r['var_ratio_mean'] for r in all_results]))
        print(f"  {'MEAN':<12}  VR = {mean_vr:.4f}")

    out_path = OUTPUT_DIR / "timevae_v2_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()