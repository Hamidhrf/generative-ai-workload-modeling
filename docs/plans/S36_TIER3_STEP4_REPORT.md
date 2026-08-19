# S36 Tier 3 Retrain — Step 4 Report

Five sequential S36 retrains on Tier 3 (H100) data, one per workload,
each in its own tmux session with output tee'd to a log file. No
commits made. `eval_s36.py` was not run (Step 5, separate task).

## Setup adaptations (from `timegan_s36.py`)

The plan assumed a `--vr`/`--fm` CLI and a single `OUTPUT_DIR`
constant; neither exists in the real script. Confirmed and agreed
before running (see prior turn): `timegan_s36.py` takes only
`--workloads` (no per-run hyperparam override — hyperparams are
hardcoded per-workload in `S27_HYPERPARAMS`, which already matches the
frozen list exactly, verified below). Two data-path constants
(`DATA_COMBINED`, `NORM_COMBINED`) and two hardcoded output-path
prefixes (`models/phase4/timegan_s36/`, `outputs/phase4/timegan_s36/`)
needed retargeting instead of one `DATA_PATH`/`OUTPUT_DIR` each.

`data/processed/tier3/unified/combined_normalization.json` did not
exist (Step 3's `unify_tier3.py` never produced it) but is required —
`load_workload_data()` opens it and indexes `norm[workload_name]`.
Traced the `norm` value through `train_workload()`: captured but never
referenced again — pure plumbing, not consumed by any training math.
Built it by combining the 5 existing
`data/processed/tier3/{workload}_normalization.json` files, same shape
as A16's. Verified against
`data/processed/phase4/unified/combined_normalization.json` (which
does exist, as expected): identical top-level keys (5 workloads),
identical sub-keys (`method`, `metric_names`, `params`) and identical
`metric_names` per workload.

Full diff, `timegan_s36.py` → `timegan_s36_tier3.py` (header comment +
4 substantive line changes, everything else byte-identical):

```diff
1a2,7
> Adapted from timegan_s36.py for H100 Tier 3 data. Only
> DATA_COMBINED, NORM_COMBINED, and the two output path prefixes
> (models/phase4/timegan_s36_tier3/, outputs/phase4/timegan_s36_tier3/)
> changed. S27_HYPERPARAMS, architecture, seeds, and epoch counts
> are untouched -- frozen per the S36 Tier 3 retrain plan.
47,49c53,55
< # Data paths (same as S34 - NO augmentation)
< DATA_COMBINED = Path("data/processed/phase4/unified/combined_dataset.npz")
< NORM_COMBINED = Path("data/processed/phase4/unified/combined_normalization.json")
---
> # Data paths (Tier 3 H100)
> DATA_COMBINED = Path("data/processed/tier3/unified/combined_dataset.npz")
> NORM_COMBINED = Path("data/processed/tier3/unified/combined_normalization.json")
572c578
<         model_dir = Path(f"models/phase4/timegan_s36/{tag}")
---
>         model_dir = Path(f"models/phase4/timegan_s36_tier3/{tag}")
587c593
<     out_dir = Path("outputs/phase4/timegan_s36")
---
>     out_dir = Path("outputs/phase4/timegan_s36_tier3")
```

## Hyperparameter receipt — S27_HYPERPARAMS in `timegan_s36_tier3.py`

Verified programmatically against the frozen list before any training
ran:

| workload | lambda_var_reg | lambda_fm_stat | short_tag | match frozen list |
|---|---|---|---|---|
| bert | 0.5 | 1.5 | vr05/fm15 | True |
| gpt2 | 0.3 | 2.0 | vr03/fm20 | True |
| resnet152 | 0.4 | 1.0 | vr04/fm10 | True |
| whisper | 0.5 | 0.5 | vr05/fm05 | True |
| yolo | 0.3 | 1.2 | vr03/fm12 | True |

**All 5 match the frozen list exactly.** No hyperparameters were
tuned, adjusted, or overridden at any point — the script has no
mechanism to override them per-run even if someone tried (no CLI flag
exists for it). `n_disc_steps` and `lambda_smooth` (not part of the
frozen vr/fm shorthand but part of the same frozen `S27_HYPERPARAMS`
dict) are also untouched: `{bert:2, gpt2:2, resnet152:2, whisper:1,
yolo:2}` disc steps, `{bert:0.05, gpt2:0.1, resnet152:0.1,
whisper:0.05, yolo:0.1}` smooth — identical to `timegan_s36.py`.
Architecture (`GEN_CFG`, `DISC_CFG`, spectral norm), seed (42), warmup
epochs (20), adversarial epochs (150), phase boundaries, and LR
schedule are all byte-identical between the two scripts (confirmed by
the diff above touching nothing else).

## Run order and results

Order: bert, resnet152, yolo, gpt2, whisper (fast/light workloads
first, GPU-heavier ones last), strictly sequential — each run waited
on to completion before the next started. All 5 exited `EXIT_0`.

| workload | wall-clock | epoch 1 val | best (final) val | checkpoint | size | params |
|---|---|---|---|---|---|---|
| bert | 868.6s (~14.5 min) | 14.02183 | 0.05282 | `models/phase4/timegan_s36_tier3/s36_bert_vr05_fm15/generator.pt` | 1,166,117 B | 290,271 |
| resnet152 | 869.4s (~14.5 min) | 29.23445 | 0.14673 | `models/phase4/timegan_s36_tier3/s36_resnet152_vr04_fm10/generator.pt` | 1,166,117 B | 290,271 |
| yolo | 887.1s (~14.8 min) | 1149.29808 | 0.17650 | `models/phase4/timegan_s36_tier3/s36_yolo_vr03_fm12/generator.pt` | 1,166,117 B | 290,271 |
| gpt2 | 862.5s (~14.4 min) | 9.36366 | 0.33574 | `models/phase4/timegan_s36_tier3/s36_gpt2_vr03_fm20/generator.pt` | 1,166,117 B | 290,271 |
| whisper | 457.5s (~7.6 min) | 20.60372 | 0.09135 | `models/phase4/timegan_s36_tier3/s36_whisper_vr05_fm05/generator.pt` | 1,166,117 B | 290,271 |

All 5 runs finished well inside the 30-45 min estimate (whisper
notably faster at ~7.6 min, consistent with `n_disc_steps=1` for
whisper vs. 2 for the other four — half the discriminator updates per
epoch). All 5 checkpoints are byte-identical in size/param count, as
expected — same generator architecture and `n_metrics=7` for every
workload (`DROP_PER_WORKLOAD` drops the same 3 GPU metrics for all 5).

Val-loss trajectories all show the expected shape: large epoch-1 spike
(9–1149, driven by the FM-stat loss before the generator has learned
anything), then monotonic-ish decline through the 170 epochs with the
usual GAN-training noise (occasional up-ticks mid-run, e.g. gpt2 epoch
111 val=1.51 after epoch 101 val=0.75) but no divergence or NaN at any
point.

**No warnings surfaced.** `grep -iE 'warning|error|traceback|collapse'`
across all 5 logs returns only the one benign PyTorch TF32-deprecation
`UserWarning` at interpreter startup, once per log — not
training-related, no action needed.

`data: 55 pods` and `Segments: train=X*6, val=Y*6` per workload (6
phases per pod, `SEGMENT_LEN=120`). Train/val pod counts per workload
(inherited from the 245/30 unified split, filtered per-workload, so
not uniform): bert 48/7, resnet152 48/7, yolo 49/6, gpt2 48/7,
whisper 52/3. Whisper's 3-pod validation set is the smallest — worth
keeping in mind if Step 5's val-based numbers for whisper look noisy,
though `best_val` selection during training already used this same
3-pod val set consistently across the whole 170-epoch run.

## Raw grep of "VR" from training logs — important scope note

```
$ grep -E 'VR|variance.ratio' logs/s36_tier3_*.log
logs/s36_tier3_bert.log:Saved: models/phase4/timegan_s36_tier3/s36_bert_vr05_fm15
logs/s36_tier3_gpt2.log:Saved: models/phase4/timegan_s36_tier3/s36_gpt2_vr03_fm20
logs/s36_tier3_resnet152.log:Saved: models/phase4/timegan_s36_tier3/s36_resnet152_vr04_fm10
logs/s36_tier3_whisper.log:Saved: models/phase4/timegan_s36_tier3/s36_whisper_vr05_fm05
logs/s36_tier3_yolo.log:Saved: models/phase4/timegan_s36_tier3/s36_yolo_vr03_fm12`
```

**Every match is the `vrNN` substring in a checkpoint save path, not a
Variance Ratio metric.** Training never computes or logs VR at all —
each log's own final line is literally "Next: Evaluate with
eval_s36.py to compare vs S34," confirming VR is an eval-time metric
computed in Step 5, not during training. What training *does* log
every 10th epoch is `val=<FM-stat validation loss>` (see the
wall-clock table above for each workload's trajectory) — that's the
closest thing to a "training metric" available from these logs, and
it's not the same quantity as VR. Flagging this explicitly so Step 5's
report isn't compared apples-to-oranges against this one.

## Unexpected items

None. Specifically:
- **NaN/Inf losses:** zero occurrences across all 5 logs
  (`grep -iE 'nan|inf'` empty).
- **Early terminations:** none — all 5 ran the full 170 epochs
  (20 warmup + 150 adversarial).
- **Mode collapse warnings:** none in logs; nothing structurally
  suggestive of collapse either (val losses vary run-to-run in the
  expected noisy-GAN way, not flatlined).
- **Hangs:** none — every run finished well under the 90-minute
  tripwire (max 887.1s ≈ 14.8 min).
- **s36_summary.json caveat (not an error, just a note):** the script
  overwrites `outputs/phase4/timegan_s36_tier3/s36_summary.json` fresh
  on every single-workload invocation rather than merging across runs
  — after 5 runs it only contains whisper's entry. Not used as a
  source for this report; all numbers above were pulled from each
  workload's own log and checkpoint directly.

## Status

**All 5 clean, ready for Step 5.**

Five `generator.pt` checkpoints exist at
`models/phase4/timegan_s36_tier3/s36_{workload}_vr{XX}_fm{YY}/`, all
290,271 params, all trained with hyperparameters verified identical to
the frozen thesis list, all completed the full 170-epoch schedule with
no NaN, no crashes, no early termination, no hung runs. Nothing has
been committed.
