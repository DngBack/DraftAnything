# R1 report (protocol: docs/r1_protocol.md; g3_protocol.md addendum 4) — 2026-10-15

## A. Readout-capacity ladder (ladder.py, results/r1_ladder.csv; 24 closed-form + 40 G3 cells)
| prediction | closed-form | G3 | verdict |
|---|---|---|---|
| A1 per-class offset (D) recovers >= 80% of the affine calibrator's gain | 0/18 | 5/14 | **fails** |
| A2 MLP on candidate logits adds <= 2 pts over affine | 24/24 | 36/40 | holds |
| A3 MLP probe adds <= 2 pts over linear probe | 20/24 | 34/40 | holds |
| A4 Rec>=50% count, best-layer -> nested layer selection changes <= 2 | 16 -> 19 | 13 -> 16 | **fails** (towards calibration) |
Descriptive: D recovers a median 31% (closed-form) / 60% (G3) of C's gain -> "bias" is a linear re-weighting of the
candidate logits, NOT a threshold shift; paper wording "threshold" must go. Loss errors already readable from the
final-layer state: 84% / 77% (median); final-layer probe = nested probe (median diff 0.2 / -0.3 pts) -> the
information reaches the last layer and is lost in the projection to the candidate logits. Bootstrap 95% CI width of
loss share: median 16 pts. Fold seeds: Rec>=50% cells 16,16,16,16,17 (closed-form).

## B. Counterfactual partners (cf_pairs.py, cf_analysis.py, results/r1_cf.csv; 6 models x count/spatial/attr)
- B1 loss errors, probe right on the partner (median over 4 cells with >= 20 loss errors): 0.65 -> **fails** (bar 0.70).
  Per cell: Gemma-27B count .68, Gemma-4B count .48, Gemma-4B spatial .87, InternVL count .62. For comparison, on
  correctly answered items the probe is right on the partner in 75-99%; its answer changes with the image in 73-100%.
- B2 model right on the partner less often than the probe: 3/4 -> holds (Gemma-27B count ties .68/.68).
Most cells have too few loss errors (attr: 0-3) — synthetic tasks are mostly solved.

## C. Calibration transfer
- C1 prompt paraphrase: transferred calibrator keeps >= 50% of in-template gain in 5/8 cells -> **fails** (bar 2/3).
- C2 skewed training prior (one class x3), class-balanced test: gain 6.8 -> 5.3 pts, 92% retained (median).

## D. Steering controls (results/r1_steer_ctrl.csv; alpha=16)
| model | set | raw | real | random dir. | shuffled labels |
|---|---|---|---|---|---|
| Qwen3-VL-4B | illusions | 54.2 | 86.6 | 57.1 | 54.2 |
| Gemma-3-4B | illusions | 52.7 | 61.9 | 53.5 | 51.8 |
| InternVL3.5-8B | illusions | 52.1 | 85.5 | 54.7 | 52.7 |
D1 (controls within 5 pts of raw on illusions, all 3): **holds**. On CountBench the shuffled control collapses accuracy
(31-53%), the random one does not (64-73%).

## Addendum 4: held-out models, alpha fixed at 16 (results/g3_heldout.csv)
| model | illusions raw / calib / steer16 | CountBench raw / calib / steer16 |
|---|---|---|
| Qwen3.5-2B | 55.7 / 42.5 / **76.0** | 75.7 / 82.8 / 67.3 |
| Qwen3.5-4B | 57.2 / 46.9 / 65.8 | 74.4 / 81.0 / 73.5 |
| Qwen3.5-9B | 59.7 / 38.8 / **71.9** | 70.0 / 81.9 / 62.2 |
| Qwen2-VL-2B | 50.0 / 33.8 / 50.0 | 74.1 / 81.1 / 53.8 |
- H1 (illusions steer >= max(raw, calib) + 10): 2/4 -> **fails** (Qwen3.5-4B +8.6; Qwen2-VL-2B: always answers one
  class; the alpha=16 edit raises both candidate logits about equally, checked by printing scores — not a bug).
- H2 (CountBench steer < calib): 4/4 holds. H3a illusions loss > bias 4/4 holds; H3b CountBench bias > loss 4/4 holds.
Reading: the decomposition replicates on a new family; the double dissociation replicates in direction (steering
helps illusions in 3/4, hurts CountBench in 4/4) but the pre-registered strength bar is met in only 2/4 with a fixed
alpha — the edit strength needed differs across models.

## Verdict summary
holds: A2, A3, D1, B2, H2, H3a, H3b. fails: A1, A4 (favourable direction), B1 (0.65 vs 0.70), C1 (5/8), H1 (2/4).
