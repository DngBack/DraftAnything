# Pilot G1 protocol (pre-registered 2026-10-13, before any model output was inspected)

Goal: decide go/no-go for the "route-then-read" readout-bottleneck paper.

## Setup
- Models: Qwen3-VL-4B-Instruct (Qwen3 LLM, SigLIP2 encoder, DeepStack) and Gemma-3-4B-it (Gemma LLM, SigLIP encoder). Different LLM families.
- Families (synthetic, parametric, seeded; `stimuli.py`):
  - COUNT: 2–9 shapes, answer = digit. Counterfactual pair = same layout with one shape removed.
  - SPATIAL: red vs blue target among gray distractors, answer in {left, right, above, below}. Counterfactual pair = targets swapped.
- Answer = argmax of next-token logits restricted to the candidate answer tokens (forced choice, greedy).

## Measurements
1. Model accuracy (forced choice).
2. Linear probes (logistic regression, 5-fold CV) on each LLM layer at (a) last prompt token, (b) mean of image-token states. Control: same probe with shuffled labels.
3. Interchange patching on counterfactual pairs (both runs correct): replace layer-L states at {image tokens} or {last token} with the source run's; metric = fraction whose answer becomes the source answer.
4. Repair: on model errors, add the class-mean difference (probe-predicted class minus model-answer class) to the last-token state at one late layer (all fitted on held-out folds, no test labels used). Primary config fixed a priori: steer layer = round(0.75 * n_layers), alpha = 1, probe = last-token probe at the layer with best train-fold CV accuracy. Other layers/alphas are reported as a sweep only.

## Decision criteria (per model x family, need >= 3 of 4 cells)
- G1a (encoded-but-unused): best-layer probe accuracy minus model accuracy >= 10 points, AND probe accuracy on model-error items >= 50% (chance: 12.5% count, 25% spatial).
- G1b (causal, position-correct): repair flips >= 30% of model errors to correct while breaking <= 5% of correct items.
- Route-then-read signature (reported, not gating): image-token patching effective only up to mid depth; last-token patching effective only from mid depth on.

## If it fails
- Probes also fail on errors -> perception failure; pivot to RL-ladder chapter as headline.
- Probes succeed but no repair -> pivot to metacognition gate as headline.

## Deviations log
- 2026-10-13, after seeing qwen3vl4b_count (best probe = final layer, last token): added a control not in the original plan —
  affine recalibration of the model's own candidate logits (5-fold CV logistic regression on the 9/4 candidate scores).
  If calibration closes most of the probe-model gap, the gap is a readout *bias*, not unused hidden information; G1a is then
  re-evaluated against calib_acc instead of model_acc. Gates themselves are unchanged.
