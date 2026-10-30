# Development, pooled dev folds [0, 1, 2, 3, 4] — objproto_phoebi, units=cls

Intervals: 95 % bootstrap over combinations; point estimates and intervals averaged over training seeds. Primary endpoint: macro AP. `F1 margin` is per-sample F1 minus the all-present predictor's. Thresholds: per-species argmax-F1 on in-distribution calibration bags.

## n_fields = 1

| method | macro AP | mean AUROC | F1 margin | macro F1 | mean FPR | #pred / #true | params |
|---|---|---|---|---|---|---|---|
| proto_mean | 0.519 [0.456, 0.606] | 0.544 [0.477, 0.614] | -0.009 [-0.047, 0.027] | 0.633 [0.576, 0.674] | 0.796 | 5.04 / 2.77 | 2304 | split-rank AUROC 0.531 [0.473, 0.592]
| proto_vote | 0.503 [0.443, 0.576] | 0.550 [0.472, 0.627] | -0.045 [-0.104, 0.013] | 0.576 [0.518, 0.615] | 0.588 | 3.84 / 2.77 | 2304 | split-rank AUROC 0.571 [0.498, 0.647]
| proto_knn | 0.551 [0.494, 0.630] | 0.608 [0.544, 0.671] | -0.000 [-0.037, 0.036] | 0.634 [0.584, 0.673] | 0.683 | 4.53 / 2.77 | 2304 | split-rank AUROC 0.617 [0.554, 0.679]
| mean_lin | 0.508 [0.460, 0.569] | 0.554 [0.486, 0.615] | -0.137 [-0.203, -0.078] | 0.500 [0.432, 0.548] | 0.383 | 2.63 / 2.77 | 2310 | split-rank AUROC 0.582 [0.516, 0.639]

Per-species AP (AUROC):

| method | bs | bt | fj | ka | mx | pf |
|---|---|---|---|---|---|---|
| proto_mean | 0.54 (0.55) | 0.39 (0.56) | 0.48 (0.48) | 0.82 (0.91) | 0.44 (0.37) | 0.44 (0.40) |
| proto_vote | 0.38 (0.50) | 0.35 (0.47) | 0.45 (0.46) | 0.78 (0.83) | 0.52 (0.48) | 0.54 (0.57) |
| proto_knn | 0.48 (0.60) | 0.40 (0.54) | 0.44 (0.44) | 0.83 (0.92) | 0.60 (0.57) | 0.56 (0.58) |
| mean_lin | 0.35 (0.46) | 0.38 (0.50) | 0.38 (0.37) | 0.79 (0.83) | 0.70 (0.68) | 0.47 (0.49) |
| pos/neg combos | 10/16 | 10/16 | 12/14 | 13/13 | 14/12 | 13/13 |

Paired difference against `proto_mean` (same bags, same combination resamples):

| method | Δ macro AP | P(Δ>0) | Δ mean AUROC | Δ mean FPR |
|---|---|---|---|---|
| proto_vote | -0.016 [-0.086, 0.052] | 0.30 | 0.006 [-0.072, 0.089] | -0.208 [-0.297, -0.124] |
| proto_knn | 0.032 [-0.043, 0.108] | 0.75 | 0.064 [-0.019, 0.151] | -0.113 [-0.174, -0.048] |
| mean_lin | -0.011 [-0.089, 0.060] | 0.36 | 0.010 [-0.090, 0.108] | -0.413 [-0.477, -0.346] |

Context sensitivity (per-combination mean score: spread over positive combos S_pos, positive-negative gap, combination-level AUROC):

| method | bs | bt | fj | ka | mx | pf |
|---|---|---|---|---|---|---|
| proto_mean | 0.04 / +0.00 / 0.54 | 0.06 / +0.02 / 0.54 | 0.03 / -0.00 / 0.49 | 0.08 / +0.20 / 0.92 | 0.07 / -0.03 / 0.33 | 0.03 / -0.01 / 0.38 |
| proto_vote | 0.02 / +0.00 / 0.53 | 0.11 / -0.03 / 0.46 | 0.21 / -0.04 / 0.42 | 0.17 / +0.20 / 0.88 | 0.07 / -0.00 / 0.46 | 0.04 / +0.02 / 0.59 |
| proto_knn | 0.01 / +0.00 / 0.64 | 0.01 / +0.00 / 0.59 | 0.01 / -0.00 / 0.43 | 0.05 / +0.14 / 0.92 | 0.01 / +0.00 / 0.60 | 0.01 / +0.00 / 0.60 |
| mean_lin | 0.25 / -0.04 / 0.44 | 0.25 / -0.01 / 0.48 | 0.28 / -0.16 / 0.35 | 0.32 / +0.60 / 0.85 | 0.17 / +0.21 / 0.76 | 0.21 / -0.02 / 0.50 |

Own/context attribution (logit of per-combination mean score regressed on the presence vector; own = diagonal [95 % CI over combinations], context = mean |off-diagonal|, inverted = own < 0 (resolved: CI below 0), positive resolved = CI above 0):

| method | own bs | own bt | own fj | own ka | own mx | own pf | context | inverted (resolved) | positive resolved |
|---|---|---|---|---|---|---|---|---|---|
| proto_mean | +0.04 [-0.1, +0.2] | +0.12 [-0.2, +0.5] | -0.01 [-0.1, +0.1] | +0.90 [+0.4, +1.3] | -0.07 [-0.3, +0.2] | +0.01 [-0.1, +0.1] | 0.10 | 2 (0) | 1 |
| proto_vote | +0.17 [-1.1, +1.2] | +0.04 [-0.5, +0.6] | -0.28 [-2.0, +1.4] | +1.80 [+0.6, +2.9] | -0.05 [-1.0, +0.9] | +0.26 [-0.4, +0.8] | 0.48 | 2 (0) | 1 |
| proto_knn | +0.05 [-0.1, +0.2] | +0.04 [-0.1, +0.2] | -0.03 [-0.2, +0.1] | +0.84 [+0.4, +1.2] | +0.02 [-0.1, +0.1] | +0.02 [-0.0, +0.1] | 0.05 | 1 (0) | 1 |
| mean_lin | -0.93 [-3.2, +1.3] | +0.07 [-1.4, +1.5] | -2.51 [-5.3, +0.4] | +5.72 [+1.6, +9.6] | +3.72 [+0.7, +7.0] | -0.00 [-1.9, +1.9] | 0.79 | 3 (0) | 2 |

## n_fields = 4

| method | macro AP | mean AUROC | F1 margin | macro F1 | mean FPR | #pred / #true | params |
|---|---|---|---|---|---|---|---|
| proto_mean | 0.530 [0.457, 0.630] | 0.541 [0.465, 0.622] | -0.003 [-0.045, 0.033] | 0.637 [0.576, 0.679] | 0.765 | 4.91 / 2.77 | 2304 | split-rank AUROC 0.523 [0.456, 0.594]
| proto_vote | 0.518 [0.446, 0.604] | 0.558 [0.466, 0.650] | -0.033 [-0.091, 0.028] | 0.599 [0.530, 0.646] | 0.694 | 4.40 / 2.77 | 2304 | split-rank AUROC 0.580 [0.497, 0.667]
| proto_knn | 0.562 [0.491, 0.658] | 0.622 [0.543, 0.699] | 0.020 [-0.022, 0.060] | 0.655 [0.602, 0.693] | 0.692 | 4.66 / 2.77 | 2304 | split-rank AUROC 0.635 [0.555, 0.714]
| mean_lin | 0.516 [0.466, 0.592] | 0.569 [0.479, 0.645] | -0.112 [-0.194, -0.043] | 0.522 [0.441, 0.580] | 0.391 | 2.75 / 2.77 | 2310 | split-rank AUROC 0.587 [0.507, 0.659]

Per-species AP (AUROC):

| method | bs | bt | fj | ka | mx | pf |
|---|---|---|---|---|---|---|
| proto_mean | 0.59 (0.55) | 0.38 (0.55) | 0.51 (0.49) | 0.82 (0.92) | 0.43 (0.35) | 0.45 (0.39) |
| proto_vote | 0.40 (0.50) | 0.34 (0.46) | 0.47 (0.45) | 0.79 (0.86) | 0.55 (0.47) | 0.57 (0.61) |
| proto_knn | 0.50 (0.62) | 0.41 (0.56) | 0.46 (0.44) | 0.82 (0.92) | 0.61 (0.59) | 0.57 (0.60) |
| mean_lin | 0.35 (0.46) | 0.40 (0.52) | 0.37 (0.36) | 0.75 (0.83) | 0.76 (0.74) | 0.47 (0.50) |
| pos/neg combos | 10/16 | 10/16 | 12/14 | 13/13 | 14/12 | 13/13 |

Paired difference against `proto_mean` (same bags, same combination resamples):

| method | Δ macro AP | P(Δ>0) | Δ mean AUROC | Δ mean FPR |
|---|---|---|---|---|
| proto_vote | -0.013 [-0.106, 0.075] | 0.34 | 0.017 [-0.083, 0.119] | -0.071 [-0.162, 0.016] |
| proto_knn | 0.031 [-0.063, 0.129] | 0.71 | 0.081 [-0.020, 0.188] | -0.073 [-0.170, 0.025] |
| mean_lin | -0.014 [-0.110, 0.081] | 0.39 | 0.028 [-0.093, 0.141] | -0.373 [-0.456, -0.287] |

Context sensitivity (per-combination mean score: spread over positive combos S_pos, positive-negative gap, combination-level AUROC):

| method | bs | bt | fj | ka | mx | pf |
|---|---|---|---|---|---|---|
| proto_mean | 0.04 / +0.00 / 0.54 | 0.06 / +0.02 / 0.54 | 0.03 / -0.00 / 0.50 | 0.08 / +0.20 / 0.92 | 0.07 / -0.03 / 0.33 | 0.03 / -0.01 / 0.39 |
| proto_vote | 0.02 / +0.00 / 0.51 | 0.11 / -0.03 / 0.46 | 0.21 / -0.03 / 0.42 | 0.17 / +0.20 / 0.88 | 0.07 / -0.00 / 0.46 | 0.04 / +0.02 / 0.58 |
| proto_knn | 0.01 / +0.00 / 0.64 | 0.01 / +0.00 / 0.59 | 0.01 / -0.00 / 0.43 | 0.05 / +0.14 / 0.92 | 0.01 / +0.00 / 0.58 | 0.01 / +0.00 / 0.59 |
| mean_lin | 0.36 / -0.04 / 0.41 | 0.37 / +0.04 / 0.51 | 0.37 / -0.18 / 0.35 | 0.36 / +0.61 / 0.83 | 0.26 / +0.31 / 0.77 | 0.30 / +0.00 / 0.50 |

Own/context attribution (logit of per-combination mean score regressed on the presence vector; own = diagonal [95 % CI over combinations], context = mean |off-diagonal|, inverted = own < 0 (resolved: CI below 0), positive resolved = CI above 0):

| method | own bs | own bt | own fj | own ka | own mx | own pf | context | inverted (resolved) | positive resolved |
|---|---|---|---|---|---|---|---|---|---|
| proto_mean | +0.04 [-0.1, +0.1] | +0.12 [-0.2, +0.5] | -0.01 [-0.1, +0.1] | +0.91 [+0.4, +1.3] | -0.07 [-0.3, +0.2] | +0.01 [-0.1, +0.1] | 0.10 | 2 (0) | 1 |
| proto_vote | +0.15 [-0.9, +1.1] | +0.03 [-0.5, +0.6] | -0.24 [-2.0, +1.4] | +1.79 [+0.6, +2.8] | -0.04 [-1.0, +0.9] | +0.31 [-0.3, +0.8] | 0.47 | 2 (0) | 1 |
| proto_knn | +0.05 [-0.1, +0.2] | +0.03 [-0.1, +0.2] | -0.03 [-0.2, +0.1] | +0.84 [+0.4, +1.2] | +0.02 [-0.1, +0.1] | +0.02 [-0.0, +0.1] | 0.05 | 1 (0) | 1 |
| mean_lin | -1.58 [-4.6, +2.0] | +0.39 [-2.3, +3.0] | -4.21 [-9.3, +1.1] | +8.81 [+4.1, +13.4] | +5.02 [+0.6, +10.0] | -0.19 [-3.6, +3.2] | 1.28 | 3 (0) | 2 |
