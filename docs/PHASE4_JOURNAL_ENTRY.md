# Journal Entry: Phase 4 Complete — Model Training and Validation

**Date:** March 24, 2026  
**Phase:** 4 (Model Implementation and Evaluation)  
**Status:** Complete

---

## Summary

Phase 4 is complete. The final model (S36) has been trained, post-processed, and validated. The work spanned 39 model stages (S1-S39), culminating in a systematic ablation study (S34-S39) that selected spectral normalization as the winning technique.

---

## Key Achievements

**Model quality:** S36 achieves mean VR=1.116 with all 5 workloads passing the VR >= 0.8 thesis target. Per-workload scores: BERT=1.022, GPT2=1.137, ResNet152=0.985, Whisper=1.397, YOLO=1.039.

**Post-processing pipeline:** Three-stage post-processing (cosine boundary blending, adaptive per-metric filtering, pod memory reconstruction) significantly improves visual trace quality without changing the statistical properties.

**Validation suite:** Comprehensive validation including scaling curves (r=1-50), Wasserstein distance heatmaps, and physical constraint verification. Zero clamping violations across all workloads.

**Extrapolation assessment:** GPT2 extrapolates plausibly to r=50. Whisper to r=20-30. BERT, ResNet152, YOLO saturate by r=15-20. This bounds the useful generation range per workload.

---

## Ablation Study Results (S34-S39)

| Stage | Change | Mean VR | Distance from 1.0 | Result |
|-------|--------|---------|-------------------|--------|
| S34 | Baseline (5 workloads, 7 metrics) | 1.205 | 0.205 | Starting point |
| S35 | Data augmentation (4x) | 1.359 | 0.359 | Failed (+75%) |
| S36 | Spectral normalization | 1.116 | 0.116 | Winner (-43%) |
| S37 | Discriminator dropout (0.3) | 1.089 | 0.089 | Mixed (3/5 degraded) |
| S38 | Multi-layer feature matching | 1.147 | 0.147 | Failed (+26%) |
| S39 | Targeted variance tuning | 1.328 | 0.328 | Disaster (+183%) |

---

## Major Architectural Insights

Three critical discoveries shaped the entire Phase 4 trajectory:

1. **S14 — Encoder removal.** The encoder-based generator had a train/test mismatch (encoder sees real traces during training, random noise during generation). Removing the encoder and mapping (z, r_norm, phase_idx) directly to traces eliminated all flat-trace failures. This was the single most impactful change.

2. **S20 — Segment-based generation.** Generating 6 independent 120-step segments instead of 720 steps end-to-end solved the phase boundary discontinuity problem. Jump ratios dropped from 8-42x to below 0.2x.

3. **S21 — Precomputed FM targets.** With 176 traces and 42 phase-replica groups, per-batch feature matching statistics were too noisy. Precomputing population-level targets provided stable gradients and enabled the model to learn correct per-phase per-replica statistics.

---

## Negative Results (Defensible for Thesis)

Several approaches were tested and failed, providing valuable negative results:

- **Data augmentation (S35):** Augmenting 176 to 704 traces introduced artifacts. Conclusion: the dataset is sufficient for the architecture; more data does not help when the signal-to-noise ratio of augmented samples is low.

- **Autocorrelation loss (S24-S26):** Adding temporal autocorrelation matching caused regressions in GPT2 and level collapse in Whisper. The loss interacted poorly with the segment-based architecture.

- **Unified cross-workload model (S29):** A single generator with workload embedding achieved good mean VR but could not match per-workload tuned models. The diversity across workloads (CPU-bound Whisper vs GPU-bound GPT2) is too large for a single generator.

---

## Git Status

All Phase 4 work is committed on `phase4-model-training` branch:
- 16 training scripts (S21-S39)
- 27 evaluation/comparison/post-processing scripts
- Milestone model checkpoints (S21, S27, S34, S36 + all S10-S39)
- Processed dataset (176 traces, normalization parameters)
- Validation outputs (scaling curves, Wasserstein heatmaps, JSON report)

---

## Next Steps (Phase 5)

1. **Kwok trace generation script:** Generate post-processed traces formatted for Kwok import.
2. **Kwok integration:** Import synthetic traces into Kwok simulation framework.
3. **Thesis writing:** Document methodology, results, and limitations.
4. **Framing for 10x-100x claim:** The model generates N pod-level traces at observed contention levels (r=1-10). For scaled deployments, the contention level is set to match per-node density.

---

**Time Spent:** ~4 weeks (Feb 24 - Mar 24, 2026)  
**Total Model Stages:** 39 (S1-S39)  
**Final Model:** S36 (spectral normalization on S34 baseline)  
**Thesis Target Met:** Yes (mean VR=1.116, 5/5 workloads >= 0.8)