# Development, pooled dev folds [0, 1] — valid_tile_pannuke, units=cls_pannuke

Intervals: 95 % bootstrap over combinations; point estimates and intervals averaged over training seeds. Primary endpoint: macro AP. `F1 margin` is per-sample F1 minus the all-present predictor's. Thresholds: per-species argmax-F1 on in-distribution calibration bags.

## n_fields = 1

| method | macro AP | mean AUROC | F1 margin | macro F1 | mean FPR | #pred / #true | params |
|---|---|---|---|---|---|---|---|
| abmil | 0.779 [0.759, 0.805] | 0.880 [0.867, 0.892] | 0.274 [0.264, 0.283] | 0.741 [0.720, 0.760] | 0.223 | 2.13 / 1.94 | 166282 | split-rank AUROC 0.880 [0.867, 0.892]
| abmil_tileV | 0.747 [0.727, 0.772] | 0.866 [0.855, 0.878] | 0.264 [0.253, 0.275] | 0.717 [0.697, 0.737] | 0.233 | 2.15 / 1.94 | 166282 | split-rank AUROC 0.866 [0.855, 0.878]
| abmil_tileV_cla1 | 0.724 [0.707, 0.748] | 0.852 [0.840, 0.864] | 0.251 [0.241, 0.261] | 0.692 [0.671, 0.712] | 0.269 | 2.23 / 1.94 | 166282 | split-rank AUROC 0.852 [0.840, 0.864]
| pe_quantile_m1_tileV | 0.749 [0.734, 0.769] | 0.865 [0.853, 0.876] | 0.266 [0.255, 0.276] | 0.703 [0.687, 0.719] | 0.226 | 2.13 / 1.94 | 165677 | split-rank AUROC 0.864 [0.853, 0.876]
| tileonlyV_mean | 0.733 [0.714, 0.757] | 0.850 [0.838, 0.862] | 0.222 [0.213, 0.230] | 0.668 [0.648, 0.689] | 0.352 | 2.54 / 1.94 | 165677 | split-rank AUROC 0.850 [0.839, 0.863]

Per-species AP (AUROC):

| method | neoplastic | inflammatory | connective | dead | epithelial |
|---|---|---|---|---|---|
| abmil | 0.90 (0.93) | 0.89 (0.88) | 0.94 (0.87) | 0.41 (0.84) | 0.75 (0.89) |
| abmil_tileV | 0.88 (0.91) | 0.87 (0.86) | 0.94 (0.85) | 0.32 (0.83) | 0.73 (0.87) |
| abmil_tileV_cla1 | 0.86 (0.90) | 0.86 (0.86) | 0.93 (0.84) | 0.28 (0.82) | 0.68 (0.85) |
| pe_quantile_m1_tileV | 0.92 (0.93) | 0.87 (0.86) | 0.94 (0.86) | 0.24 (0.77) | 0.78 (0.90) |
| tileonlyV_mean | 0.91 (0.92) | 0.85 (0.82) | 0.93 (0.83) | 0.27 (0.81) | 0.72 (0.87) |
| pos/neg combos | 1111/1479 | 1368/1222 | 1906/684 | 81/2509 | 557/2033 |

Paired difference against `abmil` (same bags, same combination resamples):

| method | Δ macro AP | P(Δ>0) | Δ mean AUROC | Δ mean FPR |
|---|---|---|---|---|
| abmil_tileV | -0.033 [-0.046, -0.016] | 0.00 | -0.015 [-0.020, -0.009] | 0.011 [0.003, 0.018] |
| abmil_tileV_cla1 | -0.056 [-0.068, -0.041] | 0.00 | -0.028 [-0.034, -0.023] | 0.047 [0.040, 0.053] |
| pe_quantile_m1_tileV | -0.030 [-0.044, -0.016] | 0.00 | -0.016 [-0.022, -0.010] | 0.004 [-0.003, 0.010] |
| tileonlyV_mean | -0.047 [-0.062, -0.028] | 0.00 | -0.031 [-0.037, -0.025] | 0.129 [0.121, 0.138] |

Context sensitivity (per-combination mean score: spread over positive combos S_pos, positive-negative gap, combination-level AUROC):

| method | neoplastic | inflammatory | connective | dead | epithelial |
|---|---|---|---|---|---|
| abmil | 0.26 / +0.61 / 0.93 | 0.25 / +0.43 / 0.88 | 0.18 / +0.38 / 0.87 | 0.17 / +0.17 / 0.84 | 0.33 / +0.48 / 0.89 |
| abmil_tileV | 0.23 / +0.55 / 0.91 | 0.23 / +0.39 / 0.86 | 0.17 / +0.33 / 0.85 | 0.16 / +0.14 / 0.83 | 0.31 / +0.41 / 0.87 |
| abmil_tileV_cla1 | 0.24 / +0.52 / 0.90 | 0.23 / +0.38 / 0.86 | 0.24 / +0.34 / 0.84 | 0.30 / +0.37 / 0.82 | 0.26 / +0.39 / 0.85 |
| pe_quantile_m1_tileV | 0.26 / +0.62 / 0.93 | 0.24 / +0.40 / 0.87 | 0.18 / +0.37 / 0.87 | 0.09 / +0.07 / 0.79 | 0.33 / +0.49 / 0.90 |
| tileonlyV_mean | 0.29 / +0.28 / 0.92 | 0.22 / +0.13 / 0.83 | 0.33 / +0.30 / 0.83 | 0.05 / +0.02 / 0.81 | 0.16 / +0.10 / 0.87 |

Own/context attribution (logit of per-combination mean score regressed on the presence vector; own = diagonal [95 % CI over combinations], context = mean |off-diagonal|, inverted = own < 0 (resolved: CI below 0), positive resolved = CI above 0):

| method | own neoplastic | own inflammatory | own connective | own dead | own epithelial | context | inverted (resolved) | positive resolved |
|---|---|---|---|---|---|---|---|---|
| abmil | +4.76 [+4.6, +5.0] | +2.37 [+2.3, +2.5] | +2.60 [+2.4, +2.8] | +1.79 [+1.4, +2.2] | +3.09 [+2.9, +3.3] | 0.53 | 0 (0) | 5 |
| abmil_tileV | +3.92 [+3.8, +4.1] | +2.11 [+2.0, +2.2] | +2.16 [+2.0, +2.3] | +1.67 [+1.3, +2.0] | +2.77 [+2.6, +2.9] | 0.47 | 0 (0) | 5 |
| abmil_tileV_cla1 | +4.43 [+4.2, +4.6] | +2.09 [+2.0, +2.2] | +2.12 [+2.0, +2.3] | +1.69 [+1.3, +2.1] | +3.31 [+3.1, +3.5] | 0.61 | 0 (0) | 5 |
| pe_quantile_m1_tileV | +4.82 [+4.6, +5.0] | +2.09 [+2.0, +2.2] | +2.35 [+2.2, +2.5] | +0.92 [+0.7, +1.2] | +3.27 [+3.1, +3.5] | 0.48 | 0 (0) | 5 |
| tileonlyV_mean | +4.80 [+4.6, +5.0] | +2.70 [+2.5, +2.9] | +2.94 [+2.7, +3.2] | +1.99 [+1.5, +2.5] | +3.41 [+3.2, +3.6] | 0.61 | 0 (0) | 5 |
