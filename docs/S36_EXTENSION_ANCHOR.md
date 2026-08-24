# S36 Extension - Anchor (read this first)

The fixed reference for the extension paper. If any other doc drifts from
this, this wins. One page on purpose. Detail lives in the reserve doc
(S36_EXTENSION_RESERVE.md) and the working draft
(S36_EXTENSION_WORKING_DRAFT.md); this is the spine.

Supersedes the earlier sharing-mode framing. E1 (matched-size control)
falsified the sharing-mode claim; this is the corrected, supported story.

## The story, one paragraph

S36 is the TimeGAN recipe that won model selection on A16 (the thesis).
The extension asks whether the frozen S36 recipe still works when ported
to H100, across hardware generation and GPU sharing mode. Answer: at full
training data, S36 transfers cleanly across the hardware generation under
time-slicing (A16 -> H100, both r=1..10). It breaks on H100 MIG. The
natural question is whether MIG breaks it because of the sharing mode or
because of something else - and here the hardware settles it: a MIG H100
exposes at most 7 compute instances, so MIG data is necessarily capped at
r=1..7 and is roughly half the size of a full time-slicing dataset. We
cannot collect MIG at full range; it is physically impossible. So we ran
the control we can run - time-slicing restricted to MIG's r=1..7 budget -
and it broke the same way. The break is gated by training-data volume,
not by sharing mode or hardware generation. The recipe is data-hungry:
below a data threshold it fails to reach fidelity regardless of
configuration.

## The tiers

| Tier | Hardware | Sharing mode | Max replicas | Role |
|---|---|---|---|---|
| A16 (thesis) | A16 | time-slicing | r=1..10 | baseline, S36 selected here |
| Tier 3 | H100 | time-slicing | r=1..10 | hardware change, full data |
| Tier 3-matched (E1) | H100 | time-slicing | r=1..7 | the control: full-mode, MIG-size |
| Tier 2 | H100 | MIG partitioning | r=1..7 (hard cap) | the case that surfaced the finding |
| Tier 4 (Kostya) | H100 | custom time-slicing | TBD | additive, not blocking |

MIG's 7-instance cap is a hardware constraint, not a collection choice.
It is the reason the sharing-mode effect is intrinsically confounded with
data volume on MIG, and the reason E1 (matched-size time-slicing) is the
only available control. Present tiers in logical order in the paper:
A16 -> H100 time-slicing (full) -> H100 time-slicing (matched) -> MIG.

## The finding that matters

Frozen S36 recipe, per-workload variance ratio over the 6 generated
metrics (pod_memory_bytes excluded - reconstructed in post-processing).

1. Transfer holds across hardware at full data: A16 and full H100
   time-slicing (both r=1..10) fit S36 about equally well. The hardware
   step is within the estimator's own noise. Not confounded - both are
   full datasets.

2. Breaks below a data threshold, regardless of sharing mode: at the
   r=1..7 / 28-trace budget, BOTH MIG (Tier 2) and size-matched
   time-slicing (Tier 3-matched, E1) break. Matched time-slicing loses to
   an i.i.d. Gaussian on all three contention metrics, exactly as MIG
   does - and its VR headline lands at or below MIG. The break is data
   volume, not sharing mode.

3. The estimator floor confirms it: the real-vs-real resampling floor
   tracks trace count, not sharing mode (n=28 floors cluster: matched
   0.891, MIG 0.907; n=55 full Tier 3 is 0.956). Part of what looked like
   a sharing-mode gap was dataset size leaking through the estimator.

4. Per-pod attribution (independent methodological contribution): MIG
   per-instance DCGM labels give native per-pod GPU measurement, which
   time-slicing structurally cannot provide (dividing a whole-GPU counter
   by replica count yields exactly zero cross-pod variance by
   construction). This contribution does not depend on the fidelity story
   and stands on its own.

## The two contributions

1. Frozen-recipe transfer across GPU configurations is gated by
   training-data volume, not by sharing mode or hardware generation.
   Hardware transfer holds at full data; the MIG break is reproduced by
   size-matched time-slicing, isolating data volume as the cause. The
   7-instance MIG cap makes this an intrinsic property of MIG-based
   collection, not a fixable gap.

2. MIG per-instance labels enable native per-pod GPU attribution;
   time-sliced measurement cannot observe per-pod GPU heterogeneity at
   all. A methodological point about what each sharing mode can and
   cannot measure, independent of contribution 1.

## In scope vs reserve

IN SCOPE (front of paper):
- The transfer-holds-at-full-data / breaks-below-threshold story.
- E1 as the load-bearing control, with the MIG 7-instance cap as its
  design justification.
- The per-metric contention-metric collapse (S36 -> Gaussian level at
  matched size).
- Per-pod attribution as the second, independent contribution.
- Honest corrections: exclude reconstructed memory metric; VR pass-counts
  shown saturable; Whisper CPU-regime story cut.

RESERVE (appendix / answer-if-asked):
- Normalized-Wasserstein Tier2/Tier3 separation and its 1.21/1.71
  decomposition (now framed as: the model-side component is not
  separable from data volume, per E1 - so this is supporting texture,
  not a sharing-mode claim).
- Floor-corrected VR deltas, both-yardstick tables.
- Bootstrap CIs, generation-variance SDs, all-traces VR column.
- Raw pooled Wasserstein (retired).
- Full 15-cell composite grids.

## What is required before writing

E1 is done and it decided the framing. The remaining experiments are
done too; nothing experimental blocks drafting.

- E6 (done): wider feature-matching search on the low-data failures, run
  on both r=1..7 tiers. Result: GPT-2 is a hard floor - no fm value
  recovers it at n=28 on either MIG or matched time-slicing. ResNet-152
  recovers on MIG at fm=4.0 (stable across 30 draws on all three
  contention metrics), in the opposite direction from the other
  workloads. This is the data-threshold answer the discussion needs: the
  budget is a hard floor for some workloads and tunable for others -
  workload-dependent, not a blanket failure.
- E3 (dropped): the MIG gpu_utilization counter control. Structurally
  unnecessary given E1's design; do not run.

Deferred / skip: A16 re-baseline, OOD replica generalization, seed
sweeps, stratified resplit. Post-submission or skip.

MIG at full range: impossible (7-instance cap). State as an intrinsic
limitation, not future work.

## The deliverables

- Paper = this extension (Paper A framing). Writeable now; E6's result sharpens
  the limitations section. Most sections drafted, need reframing off the
  old sharing-mode spine.
- Research thesis = foundation (S36 on A16). Done, submitted. Paper cites.
- Tier 4 (Kostya) = additive. If it lands, it is another sharing mode to
  test the data-volume finding against; does not block.
- Future master thesis = extends the tier comparison; the data-volume
  gating question generalizes and the infrastructure is built.

## Stop condition

E1 resolved the central fork. The claim is now supported and controlled.
Stop stress-testing. From here: reframe the draft off
this anchor, then write. Do not reopen the sharing-mode claim - it is
falsified and the confound is permanent by hardware constraint.
