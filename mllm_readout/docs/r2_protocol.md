# R2 protocol — pre-registered 2026-10-15, before any of the runs below

Two weaknesses left open by R1 (docs/r1_report.md): B1 used synthetic pairs only and missed its bar (0.65 vs 0.70)
with 4 thin cells; H1 used one fixed steering strength (alpha=16, chosen post hoc on the 9 development models) and
met its bar on 2/4 held-out models.

## E. Real-image counterfactual partners (What'sUp Controlled_Images A+B)
Each object pair (split, object, reference) is photographed in 4 relations (A: on/under/left/right; B:
left/right/front/behind), and the prompt is identical within a group ("Where is the {a} relative to the {b}? ..."
with the same option list). An item's partners are the other images of its group (different gold relation).
- Cells: the 6 models with What'sUp states (Qwen3-VL-4B/8B/32B, Gemma-3-4B/27B, InternVL3.5-8B) plus 7 added now,
  before running them: LLaVA-1.5-7B, Qwen2-VL-7B, Qwen3-VL-2B, Qwen3.5-2B/4B/9B, Qwen2-VL-2B (states only, n=820).
- Cross-fitting: 5 folds grouped by object pair, so an item and all its partners are in the same test fold and the
  probe/calibrator never saw the object pair. Calibrator: affine on the 6 candidate logits; probe: linear, last
  position, layer by inner CV on the training folds (as cf_analysis.py). Error labels bias/loss/perc as usual.
- E1 (= B1 on real images): over cells with >= 20 loss errors, median fraction of (loss error, partner) pairs on
  which the probe is right on the partner >= 0.70.
- E2 (= B2): the model is right on those partners less often than the probe in >= 2/3 of those cells.
- E3 (no bar, reported): the same two rates on correctly answered items and on bias errors.
If fewer than 3 cells reach 20 loss errors, E1/E2 are reported as underpowered, not as holds/fails.

## F. Steering strength selected per model on held-out images
For every model with G3 states (9 development + 4 held-out) and both sets (illusions, CountBench), steering exactly as
steer_g3.py (layer at 75% depth, class-mean difference, 5 folds by image, edit only where probe != answer), for
alpha in {0, 1, 2, 4, 8, 16, 32, 64} (alpha=0 = unsteered). The prefix is generated once per item; only the
decision-step forward pass is repeated per alpha.
Selection: images (keys) are split 50/50 into validation/test, 20 random splits (seeds 0..19). alpha* = argmax of
illusions accuracy on the validation images (ties -> smaller alpha). Statistics are means over the 20 splits, on
test images only; calibration accuracy is the cross-fitted calibrator of steer_g3.analyze restricted to the same test
images.
- F1 (primary, held-out models): illusions test accuracy at alpha* >= max(raw, calib) + 10 pts in >= 3 of 4.
- F2 (all 13 models): same bar in >= 2/3 of the valid illusion cells (G3 validity filter).
- F3 (dissociation): the illusions-selected alpha* applied to CountBench gives test accuracy below calibration in >= 2/3
  of the 13 models.
- F4 (no bar, reported): alpha* per model; alpha selected on CountBench validation images (expected: 0 or small).
Note that alpha=16 was chosen after looking at the 9 development models, so F2 is not fully confirmatory; F1 is.
Deviations will be logged at the end of this file.
