# Development, pooled dev folds [0, 1] — pannuke_abmil, units=cls_pannuke

Intervals: 95 % bootstrap over combinations; point estimates and intervals averaged over training seeds. Primary endpoint: macro AP. `F1 margin` is per-sample F1 minus the all-present predictor's. Thresholds: per-species argmax-F1 on in-distribution calibration bags.

## n_fields = 1

| method | macro AP | mean AUROC | F1 margin | macro F1 | mean FPR | #pred / #true | params |
|---|---|---|---|---|---|---|---|
| abmil | 0.779 [0.758, 0.805] | 0.880 [0.867, 0.892] | 0.274 [0.264, 0.283] | 0.741 [0.719, 0.761] | 0.223 | 2.13 / 1.94 | 166282 |
| abmil_tile1 | 0.769 [0.747, 0.796] | 0.876 [0.863, 0.888] | 0.271 [0.260, 0.281] | 0.728 [0.705, 0.750] | 0.216 | 2.11 / 1.94 | 166282 |

Per-species AP (AUROC):

| method | neoplastic | inflammatory | connective | dead | epithelial |
|---|---|---|---|---|---|
| abmil | 0.90 (0.93) | 0.89 (0.88) | 0.94 (0.87) | 0.41 (0.84) | 0.75 (0.89) |
| abmil_tile1 | 0.88 (0.92) | 0.87 (0.87) | 0.94 (0.86) | 0.40 (0.86) | 0.75 (0.88) |
| pos/neg combos | 1111/1479 | 1368/1222 | 1906/684 | 81/2509 | 557/2033 |

Paired difference against `abmil` (same bags, same combination resamples):

| method | Δ macro AP | P(Δ>0) | Δ mean AUROC | Δ mean FPR |
|---|---|---|---|---|
| abmil_tile1 | -0.010 [-0.023, 0.004] | 0.10 | -0.005 [-0.010, 0.001] | -0.007 [-0.013, -0.000] |

Context sensitivity (per-combination mean score: spread over positive combos S_pos, positive-negative gap, combination-level AUROC):

| method | neoplastic | inflammatory | connective | dead | epithelial |
|---|---|---|---|---|---|
| abmil | 0.26 / +0.61 / 0.93 | 0.25 / +0.43 / 0.88 | 0.18 / +0.38 / 0.87 | 0.17 / +0.17 / 0.84 | 0.33 / +0.48 / 0.89 |
| abmil_tile1 | 0.24 / +0.56 / 0.92 | 0.22 / +0.40 / 0.87 | 0.17 / +0.35 / 0.86 | 0.19 / +0.17 / 0.86 | 0.32 / +0.44 / 0.88 |

Own/context attribution (logit of per-combination mean score regressed on the presence vector; own = diagonal [95 % CI over combinations], context = mean |off-diagonal|, inverted = own < 0 (resolved: CI below 0), positive resolved = CI above 0):

| method | own neoplastic | own inflammatory | own connective | own dead | own epithelial | context | inverted (resolved) | positive resolved |
|---|---|---|---|---|---|---|---|---|
| abmil | +4.76 [+4.6, +5.0] | +2.37 [+2.3, +2.5] | +2.60 [+2.4, +2.8] | +1.79 [+1.4, +2.2] | +3.09 [+2.9, +3.3] | 0.53 | 0 (0) | 5 |
| abmil_tile1 | +4.00 [+3.8, +4.2] | +2.16 [+2.0, +2.3] | +2.27 [+2.1, +2.4] | +2.00 [+1.6, +2.4] | +2.98 [+2.8, +3.2] | 0.48 | 0 (0) | 5 |
