# Development, pooled dev folds [0, 1] — pannuke_nobce, units=cls_pannuke

Intervals: 95 % bootstrap over combinations; point estimates and intervals averaged over training seeds. Primary endpoint: macro AP. `F1 margin` is per-sample F1 minus the all-present predictor's. Thresholds: per-species argmax-F1 on in-distribution calibration bags.

## n_fields = 1

| method | macro AP | mean AUROC | F1 margin | macro F1 | mean FPR | #pred / #true | params |
|---|---|---|---|---|---|---|---|
| pe_max_tile1_m1 | 0.761 [0.745, 0.781] | 0.873 [0.861, 0.884] | 0.262 [0.253, 0.271] | 0.704 [0.688, 0.720] | 0.244 | 2.21 / 1.94 | 165652 |
| pe_max_tileonly_m1 | 0.765 [0.744, 0.791] | 0.874 [0.861, 0.886] | 0.262 [0.253, 0.272] | 0.723 [0.703, 0.743] | 0.249 | 2.16 / 1.94 | 165652 |
| pe_q90_tileonly_m1 | 0.793 [0.771, 0.815] | 0.882 [0.870, 0.894] | 0.267 [0.256, 0.278] | 0.734 [0.713, 0.753] | 0.219 | 2.14 / 1.94 | 165677 |

Per-species AP (AUROC):

| method | neoplastic | inflammatory | connective | dead | epithelial |
|---|---|---|---|---|---|
| pe_max_tile1_m1 | 0.88 (0.91) | 0.89 (0.87) | 0.95 (0.87) | 0.33 (0.83) | 0.76 (0.88) |
| pe_max_tileonly_m1 | 0.89 (0.92) | 0.86 (0.85) | 0.94 (0.85) | 0.38 (0.86) | 0.75 (0.89) |
| pe_q90_tileonly_m1 | 0.91 (0.93) | 0.87 (0.86) | 0.94 (0.86) | 0.44 (0.85) | 0.79 (0.90) |
| pos/neg combos | 1111/1479 | 1368/1222 | 1906/684 | 81/2509 | 557/2033 |

Paired difference against `pe_max_tile1_m1` (same bags, same combination resamples):

| method | Δ macro AP | P(Δ>0) | Δ mean AUROC | Δ mean FPR |
|---|---|---|---|---|
| pe_max_tileonly_m1 | 0.004 [-0.009, 0.019] | 0.70 | 0.000 [-0.005, 0.006] | 0.006 [-0.001, 0.013] |
| pe_q90_tileonly_m1 | 0.031 [0.020, 0.040] | 1.00 | 0.009 [0.004, 0.014] | -0.025 [-0.031, -0.018] |

Context sensitivity (per-combination mean score: spread over positive combos S_pos, positive-negative gap, combination-level AUROC):

| method | neoplastic | inflammatory | connective | dead | epithelial |
|---|---|---|---|---|---|
| pe_max_tile1_m1 | 0.20 / +0.47 / 0.93 | 0.24 / +0.42 / 0.88 | 0.18 / +0.37 / 0.87 | 0.17 / +0.18 / 0.86 | 0.30 / +0.44 / 0.90 |
| pe_max_tileonly_m1 | 0.14 / +0.46 / 0.92 | 0.12 / +0.29 / 0.85 | 0.08 / +0.19 / 0.85 | 0.34 / +0.43 / 0.86 | 0.25 / +0.47 / 0.89 |
| pe_q90_tileonly_m1 | 0.19 / +0.56 / 0.93 | 0.21 / +0.40 / 0.86 | 0.15 / +0.29 / 0.87 | 0.31 / +0.33 / 0.85 | 0.31 / +0.52 / 0.90 |

Own/context attribution (logit of per-combination mean score regressed on the presence vector; own = diagonal [95 % CI over combinations], context = mean |off-diagonal|, inverted = own < 0 (resolved: CI below 0), positive resolved = CI above 0):

| method | own neoplastic | own inflammatory | own connective | own dead | own epithelial | context | inverted (resolved) | positive resolved |
|---|---|---|---|---|---|---|---|---|
| pe_max_tile1_m1 | +2.41 [+2.3, +2.5] | +2.31 [+2.2, +2.4] | +2.41 [+2.3, +2.6] | +1.68 [+1.3, +2.0] | +2.39 [+2.2, +2.5] | 0.40 | 0 (0) | 5 |
| pe_max_tileonly_m1 | +4.33 [+4.1, +4.5] | +2.40 [+2.3, +2.5] | +2.70 [+2.5, +2.9] | +2.58 [+2.0, +3.1] | +3.29 [+3.1, +3.5] | 0.50 | 0 (0) | 5 |
| pe_q90_tileonly_m1 | +4.19 [+4.0, +4.4] | +2.42 [+2.3, +2.6] | +2.70 [+2.5, +2.9] | +2.56 [+2.0, +3.1] | +3.16 [+3.0, +3.3] | 0.55 | 0 (0) | 5 |
