# G3 protocol: the decomposition on the generation benchmarks (pre-registered 2026-10-14, before any G3 run)

Motivation: the paper says MMVP errors are "perceptual, as the diagnostic predicts", but the probe decomposition was
only run on the closed-form diagnostic tasks (G1.5/G2), never on MMVP, POPE, VLMBias or the CountBench generation
setting. The large models (Qwen3-VL-32B, Gemma-3-27B) also contribute few diagnostic cells (too few errors). G3 runs the
same per-error decomposition, unchanged, at the decision step of free-form generation on those benchmarks.

## Cells
Models (9): qwen3vl2b, qwen3vl4b, qwen3vl8b, qwen3vl32b, internvl35_8b, gemma3_4b, gemma3_27b, qwen2vl_7b, llava15_7b.
Sets (7), same items/prompts/candidates/gates as gen_calib.py:
- illusions: the six VLMBias illusion subsets (792 items, Yes/No);
- Car Logos (270), Dice Pattern (168), Tally Pattern (168): VLMBias, numeric candidates;
- MMVP (300, a/b);
- POPE: 3000 items, 1000 per setting, drawn once with np.random.default_rng(0) (POPE has 9000; subsampled for disk/probe cost);
- CountBench: the El-Shangiti generation setting (437 items, 1-9).

## Measurement
Greedy generation with the gen_calib decision rule (first step whose arg-max is a trigger token while the gate is open),
no intervention. At that step: candidate scores (as gen_calib) and the hidden states of the prefix's last position
(all layers) and the mean over image tokens (all layers), as in pilot.py. Items with no decision step are dropped and
counted; cells with >10% such items are excluded (the same rule as the paper's generation tables).
The model's answer = arg-max of the decision-step candidate scores; items whose gold is not a candidate are dropped.

## Decomposition (identical to analyze_g15.cell)
5-fold cross-fitted: calibrator = multinomial LR on the candidate scores; probe = best layer x {last, img}.
bias / loss / perc as in G1.5. Only change: for illusions the calibrator is fitted per subset (as in generation), the
probe is pooled over the six subsets. Cells count if >= 30 errors.

## Pre-registered predictions
1. Confirmatory replication of the G1.5 rule on these new cells: P-cal if Rec >= 50% in >= 2/3 of cells, P-read if
   loss >= 30% and loss > bias in >= 2/3, else mixed. (G1.5 gave mixed, 14/22.)
2. MMVP is perception-dominated: in >= 2/3 of MMVP cells, perc > bias and perc > loss.
   If this fails, the sentence "whose errors are perceptual---as the diagnostic predicts" is removed from the paper and
   the MMVP result is reported as what it is.
3. POPE has little bias: bias < 30% of errors in >= 2/3 of POPE cells (generation calibration gains are <= 2.6 points).
4. Descriptive only (partly mechanical, both use the decision-step scores): rank correlation across cells between the
   bias share and the 20-split offline generation gain (gencal_stats off_cal - off_raw).

All results are reported whichever way they come out; deviations are logged in docs/g3_report.md.

## Addendum (2026-10-14, after a first partial run, before any valid G3 number was computed)
The first partial run (qwen3vl2b complete; 3 other models partly) gave probes at 98-100% on illusions/Car Logos/Dice,
which exposed two flaws in the design, not in the hypotheses. Those outputs are discarded (results/old/g3_nokey/):
- Leakage: every VLMBias image appears 6 times (2 question wordings x 3 resolutions, 384/768/1152 px; after the 768 px
  thumbnail two are near-identical) with the same label; ungrouped 5-fold CV puts copies on both sides. Fix: all CV
  (calibrator, probe, text-only control) is grouped by base image (VLMBias image path without the _px suffix; MMVP pair;
  POPE image; CountBench item).
- Label readable from the prompt: on Car Logos a layer-1 last-token probe was 100% correct, i.e. the gold is a function
  of the question (which logo), not of the image. A decomposition is meaningless there. New rule, fixed before any
  text-only accuracy was computed: a cell is excluded if a text-only control (majority train-fold label for the same
  prompt string, same grouped folds) reaches >= 75% accuracy.
Disclosure: the per-cell numbers of that flawed partial run were seen (e.g. qwen3vl2b MMVP perc 0.70, POPE bias 0.20,
11 cells / 5 with Rec >= 50%). Predictions 1-4 were not changed.

## Addendum 2 (2026-10-14, before running it): blank-image text probe
The exact-prompt control is weak (unique prompts fall back to the majority). Stronger control: the same prompts with the
image replaced by a uniform grey image of the same size; hidden states at the last prompt position (all layers); a
probe with the same grouped folds (best layer) = what the text alone makes linearly decodable. loss_txt = loss errors
on which this text probe is wrong. Criteria fixed now: the claim "edited illusions are genuine loss" stands if the
median loss_txt over illusion cells is >= 30%; the claim "POPE's apparent loss is a prompt prior" stands if the median
loss_txt over POPE cells is <= 10%. Otherwise the paper is changed accordingly. Sets: illusions, tally, mmvp, pope,
countbench (Car Logos / Dice already excluded).

## Addendum 3 (2026-10-14, before running it): causal test — probe-guided steering at the decision step
If illusion errors are readout loss, editing the answer-position state toward the probe's class should fix errors that
no calibrator fixes; where errors are bias (CountBench), it should add little over calibration.
- Sets: illusions (9 models), CountBench (8 models; LLaVA excluded as before).
- 5-fold cross-fitting grouped by image. Layer fixed in advance at 75% depth (as the closed-form gated repair).
  On training folds: linear probe and class means of the decision-step last-position state at that layer.
- Test item: regenerate the greedy prefix up to the decision step; if the probe's class p differs from the model's
  answer m, add (mean_p - mean_m) to that layer's output at the last position and re-read the candidate scores;
  steered answer = their arg-max (no edit if p == m).
- Prediction: on illusions, steered accuracy exceeds calibrated accuracy by >= 10 points in >= 2/3 of cells.
  CountBench: descriptive (steered vs calibrated). Also reported: correct answers broken by the edit.

## Addendum 4 (2026-10-15, before any run on these models): confirmatory steering on held-out models
alpha=16 was chosen after seeing alpha=1 on the 9 main models (Addendum 3 result). Confirmation on models never used
in this project: Qwen3.5-2B, Qwen3.5-4B, Qwen3.5-9B (new family, hybrid attention; chat template with
enable_thinking=False, the counterpart of the Instruct models) and Qwen2-VL-2B. Everything else unchanged: diag_gen
(illusions, CountBench only), steer_g3 (layer at 75% depth, folds grouped by image, alpha in {1, 16}).
Cell validity as before: >= 30 errors, <= 10% items with no decision step (decided before seeing any accuracy).
Predictions (alpha fixed at 16):
  H1 illusions: steered >= max(raw, calibrated) + 10 points in >= 2/3 of valid cells.
  H2 CountBench: steered < calibrated in >= 2/3 of valid cells.
  H3 the decomposition: on illusions loss > bias in >= 2/3 of valid cells; on CountBench bias > loss in >= 2/3.
Reported whatever the outcome (also broken correct answers). If fewer than 2 valid cells exist for a set, that
prediction is reported as untestable, not as holding.
Disclosure: before the full runs, a pipeline smoke test on Qwen3.5-2B (first 40 CountBench items, alpha=16; plus 6
items per set for extraction) was run in a scratch directory and discarded; its output was seen (raw 0.775, steered
0.500 on 40 items, prefix reproduced 100%). Predictions were not changed.
