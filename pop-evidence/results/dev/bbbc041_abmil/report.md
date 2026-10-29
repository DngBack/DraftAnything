# Development, pooled dev folds [0, 1, 2, 3, 4] — bbbc041_abmil, units=cls_bbbc041

Intervals: 95 % bootstrap over combinations; point estimates and intervals averaged over training seeds. Primary endpoint: macro AP. `F1 margin` is per-sample F1 minus the all-present predictor's. Thresholds: per-species argmax-F1 on in-distribution calibration bags.

## n_fields = 1

| method | macro AP | mean AUROC | F1 margin | macro F1 | mean FPR | #pred / #true | params |
|---|---|---|---|---|---|---|---|
| abmil | 0.459 [0.438, 0.493] | 0.800 [0.781, 0.820] | 0.244 [0.219, 0.270] | 0.479 [0.452, 0.506] | 0.166 | 1.34 / 1.04 | 166282 |
| abmil_tile1 | 0.485 [0.461, 0.521] | 0.824 [0.805, 0.843] | 0.272 [0.245, 0.300] | 0.510 [0.480, 0.538] | 0.158 | 1.32 / 1.04 | 166282 |

Per-species AP (AUROC):

| method | gametocyte | leukocyte | ring | schizont | trophozoite |
|---|---|---|---|---|---|
| abmil | 0.25 (0.76) | 0.21 (0.76) | 0.62 (0.79) | 0.32 (0.80) | 0.90 (0.89) |
| abmil_tile1 | 0.26 (0.80) | 0.23 (0.78) | 0.63 (0.81) | 0.40 (0.84) | 0.90 (0.90) |
| pos/neg combos | 91/775 | 70/796 | 200/666 | 112/754 | 429/437 |

Paired difference against `abmil` (same bags, same combination resamples):

| method | Δ macro AP | P(Δ>0) | Δ mean AUROC | Δ mean FPR |
|---|---|---|---|---|
| abmil_tile1 | 0.025 [0.008, 0.046] | 1.00 | 0.024 [0.010, 0.037] | -0.008 [-0.019, 0.002] |

Context sensitivity (per-combination mean score: spread over positive combos S_pos, positive-negative gap, combination-level AUROC):

| method | gametocyte | leukocyte | ring | schizont | trophozoite |
|---|---|---|---|---|---|
| abmil | 0.16 / +0.12 / 0.77 | 0.14 / +0.11 / 0.76 | 0.36 / +0.33 / 0.79 | 0.23 / +0.21 / 0.80 | 0.30 / +0.58 / 0.90 |
| abmil_tile1 | 0.15 / +0.13 / 0.80 | 0.11 / +0.11 / 0.78 | 0.29 / +0.29 / 0.81 | 0.18 / +0.21 / 0.84 | 0.28 / +0.49 / 0.90 |

Own/context attribution (logit of per-combination mean score regressed on the presence vector; own = diagonal [95 % CI over combinations], context = mean |off-diagonal|, inverted = own < 0 (resolved: CI below 0), positive resolved = CI above 0):

| method | own gametocyte | own leukocyte | own ring | own schizont | own trophozoite | context | inverted (resolved) | positive resolved |
|---|---|---|---|---|---|---|---|---|
| abmil | +1.57 [+1.3, +1.9] | +0.93 [+0.6, +1.2] | +2.26 [+1.9, +2.6] | +1.91 [+1.5, +2.3] | +5.31 [+4.9, +5.7] | 0.74 | 0 (0) | 5 |
| abmil_tile1 | +1.66 [+1.4, +1.9] | +0.92 [+0.7, +1.1] | +1.89 [+1.7, +2.1] | +1.75 [+1.5, +2.0] | +3.25 [+3.0, +3.5] | 0.64 | 0 (0) | 5 |
