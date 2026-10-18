# G4 protocol: bracketed diagnostic and repair matrix

Written 2026-10-16, before any repair-matrix run (R2 cross-fitted selection, R3x, R4, R5) and before the grid-pooled
input probes are analysed. Exploratory results already seen and NOT counted as confirmation: mean-pooled projector
probe (explore_flow.py), SigLIP2 probe on illusions/CountBench/MMVP (explore_extenc.py), test-selected hidden guidance
on Qwen3-VL-4B / Gemma-3-4B (explore_hcfg.py). Qwen3.5 models were used as held-out models in R1; they are not unseen.

## Items and models

- Sets: illusions (VLMBias, 6 label-symmetric subsets), CountBench (generation prompt), MMVP. Same items, prompts and
  decision step as G3 (diag_gen.py). Repair columns are evaluated on one view per (image, wording) for illusions
  (first copy in item order; 264 items) and on all items for CountBench (437) and MMVP (300).
- Models: Qwen3.5-2B/4B/9B (thinking mode available), Qwen3-VL-4B, Qwen3-VL-8B. Cells where a model fails to reach a
  decision step on >10% of items are excluded (G3 rule).

## Diagnostic (per error, cross-fitted, 5 folds grouped by base image)

- cal: affine calibrator on candidate logits (as G3).
- ans: linear probe on the decision-token state at 75% depth (the steering layer; fixed, no layer selection).
- ext: linear probe on SigLIP2-base 4x4+mean pooled patch features (primary input control). Secondary: projector-output
  grid of the same MLLM (layer 0), reported, not used for the labels.
- Labels for errors: BIAS = cal right. UNSPOKEN = cal wrong, ans right, ext wrong. INHERITED = cal wrong, ans right,
  ext right. DROPPED = cal wrong, ans wrong, ext right. UNSEEN = cal, ans, ext all wrong.
- MMVP: the answer depends on the question, so ext is undefined; MMVP errors get BIAS / LOSS-unbracketed / UNSEEN
  (LOSS-unbracketed rows are reported but excluded from the four-row matrix test).

## Input-side checks (grid features)

- I1: illusions, ext accuracy >= ans accuracy - 5 points in >= 7 of 9 models with G3 states.
- I2: CountBench, ans accuracy >= ext accuracy + 20 points in >= 7 of 9 models.
- I3 (chance-corrected inheritance): on illusions, P(ext right | LOSS error) exceeds P(ext right | shuffled-label ext
  probe) by >= 25 points, pooled over models.

## Repairs (columns); fixed = wrong -> right, broken = right -> wrong

- R1 calibration: cross-fitted calibrator (as above).
- R2 label-free hidden guidance (explore_hcfg.py): replace layer-l outputs at text positions with h_grey + g (h_real -
  h_grey). (l, g, scope) chosen by 2-fold cross-fitting over base images within each cell from the fixed grid
  {25,50,75,90% depth} x {1.5,2,3,5} x {last, post}; the choice on half A is applied to half B and vice versa.
- R3 label steering: class-mean difference at 75% depth, alpha = 16, target = ans probe class (steer_g3.py).
- R3x external-target steering: as R3, but the target class is the ext (SigLIP2) probe's prediction for the item;
  directions are the same MLLM class means. Only illusions and CountBench.
- R4 compute: Qwen3.5 with enable_thinking=True (max 1536 new tokens, greedy), answer parsed from the text after the
  reasoning block with the G3 parser. Qwen3-VL: not run.
- R5 see more: the image upscaled 2x (bicubic; long side capped at 1536 instead of 768), plain G3 decoding.

## Predictions

- P1 (main): pooled over models and sets, for each of the four rows BIAS, UNSPOKEN, INHERITED, UNSEEN the
  pre-assigned repair (R1, R2, R4, R5) has the highest fix rate among the columns available for that row, excluding R3
  and R3x, in >= 3 of 4 rows; and a permutation test that shuffles row labels within cells gives p < 0.05 for the sum
  of diagonal minus mean off-diagonal fix rates.
- P2: BIAS errors: R1 fix rate >= 2x R2 fix rate.
- P3: INHERITED errors: R2 fix rate <= 15% and lower than R2's fix rate on UNSPOKEN errors.
- P4 (probe-in-disguise): on illusions, R3x accuracy >= R3 accuracy - 3 points in >= 4 of 5 models.
- P5: UNSEEN errors: R5 fix rate exceeds both R1's and R2's.
- P6: INHERITED errors: R4 fix rate exceeds R2's (Qwen3.5 only).

Every repair column also reports overall accuracy and broken rate. Rows with < 15 errors pooled are reported but do not
count. Any deviation is logged in an addendum with its date before results are read.

## Addendum 1 (2026-10-16, before any repair-matrix result is read)

- R4: in a 16-item smoke test (CountBench, Qwen3.5-2B; not counted) 6/16 reasoning traces hit the 1536-token cap.
  Truncated traces are closed with budget forcing (append "\n</think>\n\n", then up to 24 greedy tokens) instead of
  being scored as no answer. Cap unchanged. The smoke-test items are part of the full run like all others.
- flash-linear-attention 0.5.2 was installed for Qwen3.5 speed (kernel replaces the reference implementation; numerics
  may differ slightly from the G3 extraction). Baselines for fixed/broken are the stored G3 decisions.
- R3 on MMVP needs MMVP states for Qwen3.5 (results/g3_heldout, extracted now with the G3 code) and alpha-16 steering
  for all five models; R3x is not defined on MMVP (no ext probe).

## Addendum 2 (2026-10-18): results of the pre-registered tests and exploratory additions

Pre-registered (g4_matrix.py, 5 models x 3 sets): I1, I2, I3 PASS. P1 FAIL (diagonal best in 2 of 4 rows; permutation
p = 0.0005 for the diagonal statistic, but the rule needs both). P2 PASS. P3 FAIL (R2 fixes 18% of INHERITED vs 13% of
UNSPOKEN). P4 PASS (5/5). P5 PASS. P6 PASS (25% vs 15%). R4 counts unparsed answers as wrong as specified; 21-33% of
Qwen3.5-4B/9B illusions answers are free text. The MMVP R3 numbers come from a re-run after fixing steer_g3.py (the
letter gate was missing; the first run edited no item, outputs kept in */invalid_mmvp_steer/).

Exploratory, not counted for any prediction:
- R6x question-before-image (explore_repair.py --mode qfirst) with a free-generation control (--mode plain): CountBench
  +0 to +7.1 vs plain (mean +3.4); MMVP breaks the answer format; illusions mixed. Pilot on errors + 40 correct
  overestimated the gain.
- Answer extraction for R4 (explore_think_fix.py): unparsed thinking outputs are re-read with "Final answer: {".
- Probe-in-disguise: steering with target = true class ("oracle") and = a wrong class ("anti"); R3x on 7 more models.
- Value added by the LLM (explore_valueadd.py): probe on [SigLIP2 mean; decision state] vs each alone.
