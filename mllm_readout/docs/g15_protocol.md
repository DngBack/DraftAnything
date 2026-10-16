# G1.5 protocol: is "seeing but not saying" mostly miscalibration? (pre-registered 2026-10-13)

Motivation: pilot G1 (docs/pilot_g1_report.md) failed; recalibrating the model's own answer logits closed the probe gap
in 2/4 cells. This experiment decides between two papers, both viable:
- P-cal: "most of 'encoded but unused' is miscalibration; genuine readout loss is rare" (+ 4-way diagnostic).
- P-read: original readout-bottleneck paper, now with a calibration control competitors lack.

## Cells
Models: qwen3vl4b, gemma3_4b (from G1), qwen3vl8b, internvl35_8b, gemma3_27b.
Families: count, spatial (synthetic, as G1), count_real (CountBenchQA, counts 2-9).

## Per-error decomposition (5-fold CV predictions, no test labels used for fitting)
For each item the model gets wrong:
- bias: affine recalibration of the model's own candidate logits gets it right;
- loss: calibration wrong, best-layer linear probe right;
- perc: both wrong (information not linearly available anywhere tested).

## Decision rule
Over all cells with >= 30 errors:
- P-cal if, in >= 2/3 of cells, calibration recovers >= 50% of the probe-model gap
  (calib_acc - model_acc >= 0.5 * (probe_acc - model_acc)).
- P-read if, in >= 2/3 of cells, loss share of errors >= 30% AND loss > bias.
- Otherwise: mixed -> the paper is the 4-way diagnostic itself ("where does each MLLM fail"), with the decomposition as headline.
