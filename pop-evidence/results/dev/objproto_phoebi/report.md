# Development, pooled dev folds [0, 1, 2, 3, 4] — objproto_phoebi, units=cls_obj

Intervals: 95 % bootstrap over combinations; point estimates and intervals averaged over training seeds. Primary endpoint: macro AP. `F1 margin` is per-sample F1 minus the all-present predictor's. Thresholds: per-species argmax-F1 on in-distribution calibration bags.

## n_fields = 1

| method | macro AP | mean AUROC | F1 margin | macro F1 | mean FPR | #pred / #true | params |
|---|---|---|---|---|---|---|---|
| proto_mean | 0.533 [0.486, 0.592] | 0.579 [0.527, 0.635] | 0.004 [-0.037, 0.040] | 0.647 [0.586, 0.685] | 0.765 | 4.96 / 2.77 | 2304 | split-rank AUROC 0.574 [0.526, 0.630]
| proto_vote | 0.522 [0.465, 0.594] | 0.573 [0.498, 0.646] | -0.059 [-0.107, -0.009] | 0.575 [0.519, 0.612] | 0.697 | 4.31 / 2.77 | 2304 | split-rank AUROC 0.577 [0.506, 0.650]
| proto_knn | 0.554 [0.509, 0.616] | 0.581 [0.522, 0.639] | 0.000 [-0.039, 0.035] | 0.639 [0.576, 0.680] | 0.771 | 4.96 / 2.77 | 2304 | split-rank AUROC 0.584 [0.529, 0.638]
| mean_lin | 0.531 [0.477, 0.600] | 0.582 [0.528, 0.637] | -0.104 [-0.167, -0.046] | 0.534 [0.478, 0.568] | 0.378 | 2.70 / 2.77 | 2310 | split-rank AUROC 0.597 [0.548, 0.640]

Per-species AP (AUROC):

| method | bs | bt | fj | ka | mx | pf |
|---|---|---|---|---|---|---|
| proto_mean | 0.52 (0.66) | 0.33 (0.42) | 0.42 (0.43) | 0.88 (0.90) | 0.56 (0.53) | 0.49 (0.53) |
| proto_vote | 0.40 (0.54) | 0.44 (0.56) | 0.42 (0.45) | 0.79 (0.79) | 0.54 (0.50) | 0.54 (0.61) |
| proto_knn | 0.51 (0.63) | 0.33 (0.38) | 0.41 (0.42) | 0.88 (0.89) | 0.62 (0.58) | 0.57 (0.59) |
| mean_lin | 0.53 (0.70) | 0.37 (0.44) | 0.45 (0.47) | 0.75 (0.82) | 0.61 (0.56) | 0.47 (0.51) |
| pos/neg combos | 10/16 | 10/16 | 12/14 | 13/13 | 14/12 | 13/13 |

Paired difference against `proto_mean` (same bags, same combination resamples):

| method | Δ macro AP | P(Δ>0) | Δ mean AUROC | Δ mean FPR |
|---|---|---|---|---|
| proto_vote | -0.011 [-0.057, 0.041] | 0.32 | -0.006 [-0.062, 0.057] | -0.069 [-0.133, -0.010] |
| proto_knn | 0.021 [-0.008, 0.055] | 0.92 | 0.002 [-0.034, 0.039] | 0.006 [-0.014, 0.028] |
| mean_lin | -0.002 [-0.063, 0.066] | 0.48 | 0.003 [-0.071, 0.073] | -0.387 [-0.466, -0.320] |

Context sensitivity (per-combination mean score: spread over positive combos S_pos, positive-negative gap, combination-level AUROC):

| method | bs | bt | fj | ka | mx | pf |
|---|---|---|---|---|---|---|
| proto_mean | 0.01 / +0.02 / 0.66 | 0.02 / -0.01 / 0.36 | 0.04 / -0.01 / 0.43 | 0.02 / +0.05 / 0.92 | 0.02 / +0.00 / 0.54 | 0.02 / +0.01 / 0.56 |
| proto_vote | 0.05 / +0.01 / 0.56 | 0.16 / +0.04 / 0.55 | 0.15 / -0.03 / 0.42 | 0.14 / +0.13 / 0.83 | 0.01 / +0.00 / 0.49 | 0.07 / +0.05 / 0.59 |
| proto_knn | 0.02 / +0.01 / 0.65 | 0.02 / -0.01 / 0.38 | 0.03 / -0.01 / 0.42 | 0.02 / +0.03 / 0.92 | 0.03 / +0.01 / 0.58 | 0.02 / +0.01 / 0.62 |
| mean_lin | 0.22 / +0.23 / 0.76 | 0.24 / -0.05 / 0.44 | 0.26 / -0.03 / 0.45 | 0.28 / +0.57 / 0.82 | 0.23 / +0.06 / 0.59 | 0.16 / -0.00 / 0.51 |

Own/context attribution (logit of per-combination mean score regressed on the presence vector; own = diagonal [95 % CI over combinations], context = mean |off-diagonal|, inverted = own < 0 (resolved: CI below 0), positive resolved = CI above 0):

| method | own bs | own bt | own fj | own ka | own mx | own pf | context | inverted (resolved) | positive resolved |
|---|---|---|---|---|---|---|---|---|---|
| proto_mean | +0.10 [-0.0, +0.2] | -0.02 [-0.1, +0.1] | -0.08 [-0.3, +0.2] | +0.25 [+0.1, +0.4] | -0.04 [-0.2, +0.2] | +0.04 [-0.1, +0.2] | 0.06 | 3 (0) | 1 |
| proto_vote | +0.21 [-0.5, +1.0] | +0.19 [-0.5, +1.0] | -0.74 [-1.8, +0.7] | +1.54 [+0.4, +2.6] | +0.12 [-0.2, +0.4] | +0.23 [-0.3, +0.9] | 0.36 | 1 (0) | 1 |
| proto_knn | +0.12 [-0.1, +0.3] | -0.08 [-0.3, +0.1] | -0.14 [-0.4, +0.2] | +0.22 [+0.1, +0.3] | -0.02 [-0.3, +0.3] | +0.06 [-0.2, +0.3] | 0.13 | 3 (0) | 1 |
| mean_lin | +0.91 [-0.5, +2.5] | -0.72 [-2.3, +1.0] | -0.49 [-3.4, +2.3] | +4.83 [+1.2, +8.4] | +1.01 [+0.0, +2.2] | +0.08 [-0.9, +1.0] | 0.60 | 2 (0) | 2 |

## n_fields = 4

| method | macro AP | mean AUROC | F1 margin | macro F1 | mean FPR | #pred / #true | params |
|---|---|---|---|---|---|---|---|
| proto_mean | 0.536 [0.489, 0.607] | 0.584 [0.520, 0.646] | 0.011 [-0.034, 0.048] | 0.656 [0.591, 0.697] | 0.743 | 4.90 / 2.77 | 2304 | split-rank AUROC 0.576 [0.517, 0.641]
| proto_vote | 0.526 [0.466, 0.611] | 0.572 [0.489, 0.654] | -0.040 [-0.101, 0.022] | 0.577 [0.510, 0.629] | 0.721 | 4.45 / 2.77 | 2304 | split-rank AUROC 0.580 [0.497, 0.666]
| proto_knn | 0.568 [0.522, 0.633] | 0.588 [0.524, 0.650] | -0.004 [-0.050, 0.039] | 0.637 [0.571, 0.678] | 0.714 | 4.68 / 2.77 | 2304 | split-rank AUROC 0.587 [0.529, 0.645]
| mean_lin | 0.543 [0.484, 0.631] | 0.607 [0.539, 0.676] | -0.097 [-0.176, -0.019] | 0.547 [0.475, 0.599] | 0.347 | 2.59 / 2.77 | 2310 | split-rank AUROC 0.617 [0.551, 0.679]

Per-species AP (AUROC):

| method | bs | bt | fj | ka | mx | pf |
|---|---|---|---|---|---|---|
| proto_mean | 0.52 (0.67) | 0.32 (0.40) | 0.42 (0.43) | 0.90 (0.92) | 0.56 (0.52) | 0.50 (0.56) |
| proto_vote | 0.41 (0.54) | 0.45 (0.55) | 0.41 (0.43) | 0.81 (0.82) | 0.55 (0.48) | 0.53 (0.61) |
| proto_knn | 0.55 (0.64) | 0.33 (0.38) | 0.43 (0.42) | 0.90 (0.91) | 0.61 (0.57) | 0.59 (0.61) |
| mean_lin | 0.58 (0.77) | 0.35 (0.44) | 0.46 (0.50) | 0.75 (0.82) | 0.62 (0.55) | 0.50 (0.55) |
| pos/neg combos | 10/16 | 10/16 | 12/14 | 13/13 | 14/12 | 13/13 |

Paired difference against `proto_mean` (same bags, same combination resamples):

| method | Δ macro AP | P(Δ>0) | Δ mean AUROC | Δ mean FPR |
|---|---|---|---|---|
| proto_vote | -0.010 [-0.064, 0.052] | 0.35 | -0.012 [-0.076, 0.058] | -0.022 [-0.093, 0.049] |
| proto_knn | 0.032 [-0.003, 0.068] | 0.96 | 0.004 [-0.036, 0.047] | -0.028 [-0.049, -0.006] |
| mean_lin | 0.007 [-0.066, 0.097] | 0.57 | 0.023 [-0.072, 0.115] | -0.396 [-0.494, -0.306] |

Context sensitivity (per-combination mean score: spread over positive combos S_pos, positive-negative gap, combination-level AUROC):

| method | bs | bt | fj | ka | mx | pf |
|---|---|---|---|---|---|---|
| proto_mean | 0.01 / +0.02 / 0.66 | 0.02 / -0.01 / 0.37 | 0.04 / -0.01 / 0.43 | 0.02 / +0.05 / 0.92 | 0.02 / +0.00 / 0.54 | 0.02 / +0.01 / 0.59 |
| proto_vote | 0.05 / +0.01 / 0.54 | 0.16 / +0.04 / 0.55 | 0.15 / -0.03 / 0.41 | 0.14 / +0.13 / 0.83 | 0.01 / +0.00 / 0.42 | 0.07 / +0.05 / 0.61 |
| proto_knn | 0.02 / +0.01 / 0.65 | 0.02 / -0.01 / 0.38 | 0.03 / -0.01 / 0.42 | 0.02 / +0.03 / 0.92 | 0.03 / +0.01 / 0.58 | 0.02 / +0.01 / 0.62 |
| mean_lin | 0.25 / +0.38 / 0.85 | 0.34 / -0.02 / 0.46 | 0.33 / +0.01 / 0.51 | 0.34 / +0.56 / 0.82 | 0.31 / +0.05 / 0.53 | 0.26 / +0.04 / 0.57 |

Own/context attribution (logit of per-combination mean score regressed on the presence vector; own = diagonal [95 % CI over combinations], context = mean |off-diagonal|, inverted = own < 0 (resolved: CI below 0), positive resolved = CI above 0):

| method | own bs | own bt | own fj | own ka | own mx | own pf | context | inverted (resolved) | positive resolved |
|---|---|---|---|---|---|---|---|---|---|
| proto_mean | +0.10 [-0.0, +0.2] | -0.01 [-0.1, +0.1] | -0.08 [-0.3, +0.2] | +0.25 [+0.1, +0.4] | -0.04 [-0.2, +0.2] | +0.04 [-0.1, +0.2] | 0.06 | 3 (0) | 1 |
| proto_vote | +0.22 [-0.5, +1.0] | +0.19 [-0.5, +1.0] | -0.76 [-1.9, +0.7] | +1.52 [+0.4, +2.6] | +0.08 [-0.2, +0.4] | +0.24 [-0.3, +0.9] | 0.36 | 1 (0) | 1 |
| proto_knn | +0.12 [-0.1, +0.3] | -0.08 [-0.3, +0.1] | -0.13 [-0.4, +0.2] | +0.22 [+0.1, +0.3] | -0.02 [-0.3, +0.3] | +0.06 [-0.2, +0.3] | 0.13 | 3 (0) | 1 |
| mean_lin | +1.75 [-0.7, +4.2] | -1.13 [-4.6, +2.3] | +0.07 [-5.9, +6.1] | +8.26 [+3.0, +13.2] | +1.38 [-0.8, +3.8] | +0.39 [-1.8, +2.1] | 1.23 | 1 (0) | 1 |
