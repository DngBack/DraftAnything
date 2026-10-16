# R1 protocol: revision experiments answering the simulated review (pre-registered 2026-10-15, before any R1 analysis)

Review points addressed: W1 (operational labels vs mechanism; readout capacity), W2 (visual dependence), W3 (post hoc
steering strength; controls), W4 (calibration transfer), P5 (uncertainty). Cells: the 24 closed-form cells with >= 30
errors (G1.5/G2, results/*_states.npz) and the 40 valid G3 cells (results/g3/). Same 5-fold cross-fitting as in the
paper (grouped by image for G3). Nothing below changes any number already in the paper.

## A. Readout-capacity ladder (W1)  -- ladder.py, offline
Readouts trained on the same folds, same labels:
  D   diagonal: candidate scores + per-class offset (a pure threshold shift; W = I).
  C   affine: W s + b (the paper's calibrator).
  M   MLP on the candidate scores (1 hidden layer, 64 ReLU units, weight decay, 300 steps).
  F   linear probe on the final-layer state at the answer position (every linear function of the full-vocabulary
      logits is a linear function of this state, so F upper-bounds any linear readout of all output logits).
  P   linear probe, layer chosen by nested CV on the training folds (the paper picks the best layer on all data).
  PM  MLP probe (same architecture as M) on the layer chosen for P.
Predictions:
  A1  bias is mostly a threshold shift: in >= 2/3 of cells where C beats the model by >= 2 points, D recovers >= 80%
      of C's gain.
  A2  loss is not a calibrator-capacity artefact: in >= 2/3 of cells, M exceeds C by <= 2 points.
  A3  perc is not a probe-capacity artefact: in >= 2/3 of cells, PM exceeds P by <= 2 points.
  A4  the paper's verdicts survive nested layer selection: Rec >= 50% count with P instead of best-layer probe changes
      by <= 2 cells in each of the closed-form and G3 sets.
Descriptive: where the loss sits -- share of loss errors that F already gets right (lost between the final state and
the candidate logits) vs. that only an earlier layer gets right (lost inside the network).

## B. Counterfactual image pairs (W2)  -- cf_pairs.py, GPU
Synthetic count / spatial / attr (stimuli.py partners: one object removed / the two objects swapped / the target
recoloured; same prompt, same candidates) for the 6 closed-form models. Partner forward pass, same answer position.
Probe and calibrator are trained on the original items (folds grouped by pair) and applied to the partner.
Measure on each error type of the original item: does the probe follow the image change (correct on the partner)?
Predictions:
  B1  on loss errors, the probe is correct on the partner in >= 70% of pairs (median over cells with >= 20 loss
      errors) -- the decodable signal follows controlled visual evidence, not a prompt shortcut.
  B2  on loss errors, the model's own answer is correct on the partner less often than the probe's (in >= 2/3 cells).

## C. Calibration transfer (W4)  -- cf_pairs.py (paraphrase pass) + offline
  C1  prompt transfer: a paraphrased prompt per synthetic family (same candidates); calibrator trained on the original
      template, tested on the paraphrase. Prediction: transferred calibration keeps >= 50% of the in-template gain
      (C - model) in >= 2/3 of cells with an in-template gain >= 2 points.
  C2  label-prior shift (offline): train on a skewed resample (one class x3), test on the class-balanced complement.
      Descriptive (no prediction): gain retained.

## D. Steering controls (W3)  -- steer_g3.py STEER_CTRL, GPU
Illusions and CountBench, alpha = 16, models Qwen3-VL-4B, Gemma-3-4B, InternVL3.5-8B (fixed now):
  random: a random direction of the same norm as alpha (mu_p - mu_m), same gating;
  shuffled: probe and class means fitted on permuted training labels.
Prediction D1: on illusions, both controls stay within 5 points of raw accuracy in every one of the 3 models, while
the real direction (already measured) does not. Prospective alpha: addendum 4 of g3_protocol.md (held-out models).

## E. Uncertainty (P5)  -- ladder.py
95% bootstrap intervals (1000 resamples of images/pairs) on bias / loss / perc shares per cell; 5 fold seeds for the
Rec >= 50% counts. Descriptive.

All outcomes are reported whichever way they come out, in docs/r1_report.md and the paper/supplement.
