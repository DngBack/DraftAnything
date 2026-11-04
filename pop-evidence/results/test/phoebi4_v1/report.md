# Held-out test ['phoebi4'] (read once) — phoebi4_v1

Intervals: 95 % bootstrap over combinations; point estimates and intervals averaged over training seeds. Primary endpoint: macro AP. `F1 margin` is per-sample F1 minus the all-present predictor's. Thresholds: per-species argmax-F1 on in-distribution calibration bags.

## n_fields = 1

| method | macro AP | mean AUROC | F1 margin | macro F1 | mean FPR | #pred / #true | params |
|---|---|---|---|---|---|---|---|
| mean_lin | 0.688 [0.583, 0.785] | 0.682 [0.535, 0.809] | 0.162 [0.037, 0.302] | 0.427 [0.366, 0.471] | 0.232 | 1.97 / 2.07 | 2310 |
| meanstd_lin | 0.674 [0.581, 0.754] | 0.659 [0.522, 0.777] | 0.104 [-0.009, 0.246] | 0.373 [0.334, 0.392] | 0.244 | 1.87 / 2.07 | 4614 |
| deepsets | 0.648 [0.554, 0.750] | 0.629 [0.488, 0.753] | -0.044 [-0.201, 0.146] | 0.278 [0.215, 0.327] | 0.303 | 1.99 / 2.07 | 231686 |
| abmil | 0.644 [0.552, 0.732] | 0.622 [0.517, 0.720] | 0.021 [-0.106, 0.159] | 0.405 [0.316, 0.470] | 0.425 | 2.84 / 2.07 | 166668 |
| abmil_cla1 | 0.635 [0.554, 0.720] | 0.620 [0.513, 0.724] | -0.008 [-0.147, 0.146] | 0.334 [0.266, 0.387] | 0.349 | 2.36 / 2.07 | 166668 |
| abmil_tile1 | 0.674 [0.595, 0.748] | 0.664 [0.571, 0.760] | 0.089 [-0.014, 0.207] | 0.421 [0.348, 0.470] | 0.422 | 2.91 / 2.07 | 166668 |
| tileonly_mean | 0.648 [0.566, 0.755] | 0.638 [0.498, 0.768] | 0.045 [-0.088, 0.194] | 0.384 [0.300, 0.438] | 0.358 | 2.45 / 2.07 | 165942 |
| protomatch | 0.651 [0.580, 0.747] | 0.642 [0.542, 0.740] | -0.124 [-0.210, -0.027] | 0.355 [0.245, 0.445] | 0.509 | 3.31 / 2.07 | 2304 |

Per-species AP (AUROC):

| method | bs | bt | fj | ka | mx | pf |
|---|---|---|---|---|---|---|
| mean_lin | 0.68 (0.72) | nan (nan) | 0.50 (0.45) | 0.84 (0.83) | nan (nan) | 0.73 (0.73) |
| meanstd_lin | 0.72 (0.74) | nan (nan) | 0.46 (0.36) | 0.85 (0.85) | nan (nan) | 0.66 (0.68) |
| deepsets | 0.76 (0.75) | nan (nan) | 0.50 (0.43) | 0.89 (0.89) | nan (nan) | 0.44 (0.44) |
| abmil | 0.56 (0.64) | nan (nan) | 0.46 (0.28) | 0.87 (0.89) | nan (nan) | 0.68 (0.68) |
| abmil_cla1 | 0.59 (0.66) | nan (nan) | 0.54 (0.35) | 0.88 (0.91) | nan (nan) | 0.53 (0.56) |
| abmil_tile1 | 0.64 (0.65) | nan (nan) | 0.62 (0.56) | 0.92 (0.92) | nan (nan) | 0.52 (0.52) |
| tileonly_mean | 0.69 (0.72) | nan (nan) | 0.47 (0.38) | 0.94 (0.94) | nan (nan) | 0.50 (0.52) |
| protomatch | 0.60 (0.70) | nan (nan) | 0.47 (0.36) | 0.89 (0.87) | nan (nan) | 0.65 (0.64) |
| pos/neg combos | 7/7 | 0/14 | 8/6 | 7/7 | 0/14 | 7/7 |

Paired difference against `mean_lin` (same bags, same combination resamples):

| method | Δ macro AP | P(Δ>0) | Δ mean AUROC | Δ mean FPR |
|---|---|---|---|---|
| meanstd_lin | -0.014 [-0.080, 0.047] | 0.35 | -0.023 [-0.083, 0.035] | 0.012 [-0.010, 0.041] |
| deepsets | -0.040 [-0.113, 0.057] | 0.24 | -0.053 [-0.134, 0.039] | 0.070 [-0.018, 0.155] |
| abmil | -0.044 [-0.150, 0.064] | 0.26 | -0.060 [-0.181, 0.065] | 0.193 [0.069, 0.337] |
| abmil_cla1 | -0.053 [-0.150, 0.057] | 0.20 | -0.061 [-0.183, 0.064] | 0.117 [0.023, 0.224] |
| abmil_tile1 | -0.014 [-0.110, 0.062] | 0.43 | -0.018 [-0.116, 0.084] | 0.190 [0.098, 0.283] |
| tileonly_mean | -0.040 [-0.102, 0.047] | 0.23 | -0.043 [-0.118, 0.045] | 0.126 [0.014, 0.226] |
| protomatch | -0.037 [-0.128, 0.079] | 0.33 | -0.040 [-0.137, 0.067] | 0.277 [0.205, 0.355] |

Context sensitivity (per-combination mean score: spread over positive combos S_pos, positive-negative gap, combination-level AUROC):

| method | bs | bt | fj | ka | mx | pf |
|---|---|---|---|---|---|---|
| mean_lin | 0.38 / +0.36 / 0.76 | nan / +nan / nan | 0.25 / -0.01 / 0.35 | 0.30 / +0.47 / 0.88 | nan / +nan / nan | 0.25 / +0.23 / 0.73 |
| meanstd_lin | 0.41 / +0.34 / 0.78 | nan / +nan / nan | 0.25 / -0.13 / 0.33 | 0.23 / +0.50 / 0.90 | nan / +nan / nan | 0.16 / +0.08 / 0.63 |
| deepsets | 0.43 / +0.37 / 0.80 | nan / +nan / nan | 0.12 / -0.17 / 0.44 | 0.06 / +0.44 / 0.88 | nan / +nan / nan | 0.12 / -0.17 / 0.37 |
| abmil | 0.25 / +0.22 / 0.67 | nan / +nan / nan | 0.21 / -0.18 / 0.21 | 0.04 / +0.49 / 0.94 | nan / +nan / nan | 0.18 / +0.13 / 0.69 |
| abmil_cla1 | 0.26 / +0.25 / 0.67 | nan / +nan / nan | 0.24 / -0.10 / 0.29 | 0.03 / +0.51 / 0.94 | nan / +nan / nan | 0.09 / +0.01 / 0.59 |
| abmil_tile1 | 0.20 / +0.15 / 0.69 | nan / +nan / nan | 0.14 / +0.04 / 0.62 | 0.04 / +0.50 / 0.96 | nan / +nan / nan | 0.14 / +0.01 / 0.47 |
| tileonly_mean | 0.09 / +0.08 / 0.69 | nan / +nan / nan | 0.02 / -0.05 / 0.38 | 0.40 / +0.60 / 0.96 | nan / +nan / nan | 0.05 / -0.01 / 0.55 |
| protomatch | 0.03 / +0.03 / 0.73 | nan / +nan / nan | 0.05 / -0.03 / 0.38 | 0.05 / +0.17 / 0.90 | nan / +nan / nan | 0.04 / +0.02 / 0.65 |

Own/context attribution (logit of per-combination mean score regressed on the presence vector; own = diagonal [95 % CI over combinations], context = mean |off-diagonal|, inverted = own < 0 (resolved: CI below 0), positive resolved = CI above 0):

| method | own bs | own bt | own fj | own ka | own mx | own pf | context | inverted (resolved) | positive resolved |
|---|---|---|---|---|---|---|---|---|---|
| mean_lin | +4.35 [+nan, +nan] | +0.00 [+nan, +nan] | +0.57 [+nan, +nan] | +4.93 [+nan, +nan] | +0.00 [+nan, +nan] | +2.91 [+nan, +nan] | 1.16 | 0 (0) | 0 |
| meanstd_lin | +4.57 [+nan, +nan] | +0.00 [+nan, +nan] | -0.61 [+nan, +nan] | +5.85 [+nan, +nan] | +0.00 [+nan, +nan] | +2.34 [+nan, +nan] | 1.17 | 1 (0) | 0 |
| deepsets | +4.31 [+nan, +nan] | -0.00 [+nan, +nan] | -0.31 [+nan, +nan] | +5.73 [+nan, +nan] | +0.00 [+nan, +nan] | +1.05 [+nan, +nan] | 1.45 | 2 (0) | 0 |
| abmil | +1.32 [+nan, +nan] | +0.00 [+nan, +nan] | -0.61 [+nan, +nan] | +4.43 [+nan, +nan] | +0.00 [+nan, +nan] | +1.04 [+nan, +nan] | 0.60 | 1 (0) | 0 |
| abmil_cla1 | +1.63 [+nan, +nan] | +0.00 [+nan, +nan] | -0.44 [+nan, +nan] | +4.79 [+nan, +nan] | +0.00 [+nan, +nan] | +0.69 [+nan, +nan] | 0.68 | 1 (0) | 0 |
| abmil_tile1 | +0.77 [+nan, +nan] | +0.00 [+nan, +nan] | +0.33 [+nan, +nan] | +4.63 [+nan, +nan] | +0.00 [+nan, +nan] | +0.55 [+nan, +nan] | 0.48 | 0 (0) | 0 |
| tileonly_mean | +1.85 [+nan, +nan] | -0.00 [+nan, +nan] | -0.09 [+nan, +nan] | +8.49 [+nan, +nan] | +0.00 [+nan, +nan] | +1.23 [+nan, +nan] | 1.13 | 2 (0) | 0 |
| protomatch | +0.14 [+nan, +nan] | -0.00 [+nan, +nan] | -0.09 [+nan, +nan] | +0.75 [+nan, +nan] | +0.00 [+nan, +nan] | +0.17 [+nan, +nan] | 0.11 | 2 (0) | 0 |

## n_fields = 4

| method | macro AP | mean AUROC | F1 margin | macro F1 | mean FPR | #pred / #true | params |
|---|---|---|---|---|---|---|---|
| mean_lin | 0.724 [0.627, 0.795] | 0.680 [0.560, 0.786] | 0.188 [0.083, 0.303] | 0.453 [0.377, 0.506] | 0.289 | 2.32 / 2.07 | 2310 |
| meanstd_lin | 0.699 [0.591, 0.778] | 0.644 [0.508, 0.767] | 0.109 [-0.013, 0.250] | 0.388 [0.328, 0.424] | 0.269 | 1.98 / 2.07 | 4614 |
| deepsets | 0.647 [0.531, 0.760] | 0.605 [0.451, 0.744] | -0.048 [-0.206, 0.136] | 0.299 [0.214, 0.373] | 0.269 | 1.92 / 2.07 | 231686 |
| abmil | 0.602 [0.488, 0.730] | 0.582 [0.466, 0.707] | 0.002 [-0.145, 0.162] | 0.387 [0.287, 0.461] | 0.423 | 2.79 / 2.07 | 166668 |
| abmil_cla1 | 0.621 [0.531, 0.718] | 0.590 [0.477, 0.703] | 0.009 [-0.127, 0.161] | 0.374 [0.281, 0.448] | 0.393 | 2.69 / 2.07 | 166668 |
| abmil_tile1 | 0.677 [0.599, 0.756] | 0.669 [0.566, 0.769] | 0.082 [-0.012, 0.187] | 0.429 [0.360, 0.476] | 0.401 | 2.86 / 2.07 | 166668 |
| tileonly_mean | 0.632 [0.560, 0.755] | 0.629 [0.486, 0.761] | 0.042 [-0.091, 0.193] | 0.400 [0.310, 0.453] | 0.354 | 2.49 / 2.07 | 165942 |
| protomatch | 0.655 [0.582, 0.763] | 0.654 [0.545, 0.764] | -0.134 [-0.238, -0.013] | 0.350 [0.240, 0.444] | 0.417 | 2.86 / 2.07 | 2304 |

Per-species AP (AUROC):

| method | bs | bt | fj | ka | mx | pf |
|---|---|---|---|---|---|---|
| mean_lin | 0.76 (0.72) | nan (nan) | 0.49 (0.41) | 0.86 (0.85) | nan (nan) | 0.80 (0.74) |
| meanstd_lin | 0.77 (0.74) | nan (nan) | 0.42 (0.25) | 0.87 (0.87) | nan (nan) | 0.74 (0.72) |
| deepsets | 0.73 (0.74) | nan (nan) | 0.55 (0.45) | 0.82 (0.80) | nan (nan) | 0.49 (0.44) |
| abmil | 0.56 (0.64) | nan (nan) | 0.44 (0.22) | 0.72 (0.79) | nan (nan) | 0.69 (0.68) |
| abmil_cla1 | 0.56 (0.64) | nan (nan) | 0.51 (0.29) | 0.81 (0.84) | nan (nan) | 0.60 (0.58) |
| abmil_tile1 | 0.65 (0.68) | nan (nan) | 0.60 (0.55) | 0.87 (0.91) | nan (nan) | 0.59 (0.54) |
| tileonly_mean | 0.64 (0.70) | nan (nan) | 0.46 (0.35) | 0.95 (0.95) | nan (nan) | 0.48 (0.52) |
| protomatch | 0.62 (0.72) | nan (nan) | 0.47 (0.37) | 0.89 (0.88) | nan (nan) | 0.64 (0.65) |
| pos/neg combos | 7/7 | 0/14 | 8/6 | 7/7 | 0/14 | 7/7 |

Paired difference against `mean_lin` (same bags, same combination resamples):

| method | Δ macro AP | P(Δ>0) | Δ mean AUROC | Δ mean FPR |
|---|---|---|---|---|
| meanstd_lin | -0.025 [-0.082, 0.039] | 0.21 | -0.037 [-0.101, 0.030] | -0.019 [-0.056, 0.017] |
| deepsets | -0.078 [-0.165, 0.022] | 0.07 | -0.076 [-0.162, 0.015] | -0.020 [-0.108, 0.085] |
| abmil | -0.123 [-0.212, -0.006] | 0.02 | -0.099 [-0.209, 0.015] | 0.135 [0.016, 0.263] |
| abmil_cla1 | -0.103 [-0.179, -0.019] | 0.01 | -0.091 [-0.192, 0.008] | 0.105 [-0.007, 0.229] |
| abmil_tile1 | -0.047 [-0.106, 0.021] | 0.08 | -0.012 [-0.082, 0.066] | 0.112 [0.028, 0.204] |
| tileonly_mean | -0.093 [-0.148, 0.008] | 0.04 | -0.052 [-0.139, 0.039] | 0.065 [-0.056, 0.180] |
| protomatch | -0.069 [-0.139, 0.044] | 0.13 | -0.026 [-0.099, 0.060] | 0.128 [0.017, 0.249] |

Context sensitivity (per-combination mean score: spread over positive combos S_pos, positive-negative gap, combination-level AUROC):

| method | bs | bt | fj | ka | mx | pf |
|---|---|---|---|---|---|---|
| mean_lin | 0.09 / +0.22 / 0.69 | nan / +nan / nan | 0.32 / +0.04 / 0.38 | 0.44 / +0.43 / 0.92 | nan / +nan / nan | 0.39 / +0.38 / 0.78 |
| meanstd_lin | 0.36 / +0.21 / 0.78 | nan / +nan / nan | 0.31 / -0.18 / 0.19 | 0.39 / +0.53 / 0.88 | nan / +nan / nan | 0.22 / +0.19 / 0.71 |
| deepsets | 0.40 / +0.35 / 0.71 | nan / +nan / nan | 0.26 / +0.00 / 0.44 | 0.32 / +0.31 / 0.80 | nan / +nan / nan | 0.23 / -0.09 / 0.47 |
| abmil | 0.35 / +0.27 / 0.67 | nan / +nan / nan | 0.29 / -0.28 / 0.19 | 0.07 / +0.43 / 0.78 | nan / +nan / nan | 0.28 / +0.18 / 0.69 |
| abmil_cla1 | 0.30 / +0.29 / 0.63 | nan / +nan / nan | 0.28 / -0.17 / 0.23 | 0.05 / +0.48 / 0.92 | nan / +nan / nan | 0.24 / +0.08 / 0.59 |
| abmil_tile1 | 0.28 / +0.23 / 0.71 | nan / +nan / nan | 0.17 / +0.03 / 0.58 | 0.02 / +0.51 / 0.92 | nan / +nan / nan | 0.17 / +0.03 / 0.53 |
| tileonly_mean | 0.06 / +0.04 / 0.73 | nan / +nan / nan | 0.01 / -0.06 / 0.38 | 0.38 / +0.51 / 0.96 | nan / +nan / nan | 0.03 / -0.01 / 0.47 |
| protomatch | 0.03 / +0.03 / 0.73 | nan / +nan / nan | 0.05 / -0.03 / 0.38 | 0.05 / +0.17 / 0.84 | nan / +nan / nan | 0.03 / +0.02 / 0.65 |

Own/context attribution (logit of per-combination mean score regressed on the presence vector; own = diagonal [95 % CI over combinations], context = mean |off-diagonal|, inverted = own < 0 (resolved: CI below 0), positive resolved = CI above 0):

| method | own bs | own bt | own fj | own ka | own mx | own pf | context | inverted (resolved) | positive resolved |
|---|---|---|---|---|---|---|---|---|---|
| mean_lin | +3.03 [+nan, +nan] | +0.00 [+nan, +nan] | -0.20 [+nan, +nan] | +7.89 [+nan, +nan] | +0.00 [+nan, +nan] | +2.99 [+nan, +nan] | 1.09 | 1 (0) | 0 |
| meanstd_lin | +4.96 [+nan, +nan] | +0.00 [+nan, +nan] | -3.02 [+nan, +nan] | +7.70 [+nan, +nan] | +0.00 [+nan, +nan] | +2.42 [+nan, +nan] | 1.15 | 1 (0) | 0 |
| deepsets | +4.07 [+nan, +nan] | -0.00 [+nan, +nan] | -0.31 [+nan, +nan] | +6.76 [+nan, +nan] | +0.00 [+nan, +nan] | -0.18 [+nan, +nan] | 1.45 | 3 (0) | 0 |
| abmil | +1.96 [+nan, +nan] | +0.00 [+nan, +nan] | -1.17 [+nan, +nan] | +4.71 [+nan, +nan] | +0.00 [+nan, +nan] | +1.39 [+nan, +nan] | 0.83 | 1 (0) | 0 |
| abmil_cla1 | +2.04 [+nan, +nan] | +0.00 [+nan, +nan] | -0.70 [+nan, +nan] | +5.89 [+nan, +nan] | +0.00 [+nan, +nan] | +0.92 [+nan, +nan] | 0.93 | 1 (0) | 0 |
| abmil_tile1 | +1.22 [+nan, +nan] | +0.00 [+nan, +nan] | +0.39 [+nan, +nan] | +4.94 [+nan, +nan] | +0.00 [+nan, +nan] | +0.53 [+nan, +nan] | 0.60 | 0 (0) | 0 |
| tileonly_mean | +1.73 [+nan, +nan] | -0.00 [+nan, +nan] | -0.66 [+nan, +nan] | +8.21 [+nan, +nan] | +0.00 [+nan, +nan] | +1.58 [+nan, +nan] | 1.10 | 2 (0) | 0 |
| protomatch | +0.15 [+nan, +nan] | -0.00 [+nan, +nan] | -0.09 [+nan, +nan] | +0.75 [+nan, +nan] | +0.00 [+nan, +nan] | +0.17 [+nan, +nan] | 0.11 | 2 (0) | 0 |
