# Held-out test ['phoebi4'] (read once) — phoebi4_bn_exploratory

Intervals: 95 % bootstrap over combinations; point estimates and intervals averaged over training seeds. Primary endpoint: macro AP. `F1 margin` is per-sample F1 minus the all-present predictor's. Thresholds: per-species argmax-F1 on in-distribution calibration bags.

## n_fields = 1

| method | macro AP | mean AUROC | F1 margin | macro F1 | mean FPR | #pred / #true | params |
|---|---|---|---|---|---|---|---|
| mean_lin | 0.693 [0.596, 0.807] | 0.693 [0.560, 0.816] | 0.048 [-0.058, 0.153] | 0.436 [0.378, 0.474] | 0.380 | 2.86 / 2.07 | 2310 |
| meanstd_lin | 0.694 [0.601, 0.781] | 0.681 [0.549, 0.793] | 0.049 [-0.051, 0.159] | 0.426 [0.376, 0.453] | 0.357 | 2.65 / 2.07 | 4614 |
| deepsets | 0.664 [0.591, 0.758] | 0.647 [0.543, 0.754] | -0.050 [-0.148, 0.064] | 0.366 [0.314, 0.405] | 0.345 | 2.47 / 2.07 | 231686 |
| abmil | 0.614 [0.531, 0.730] | 0.609 [0.518, 0.709] | -0.044 [-0.148, 0.080] | 0.374 [0.303, 0.429] | 0.361 | 2.49 / 2.07 | 166668 |
| abmil_cla1 | 0.638 [0.561, 0.737] | 0.623 [0.539, 0.723] | -0.039 [-0.144, 0.087] | 0.375 [0.304, 0.432] | 0.357 | 2.49 / 2.07 | 166668 |
| abmil_tile1 | 0.661 [0.582, 0.747] | 0.656 [0.564, 0.750] | 0.032 [-0.056, 0.135] | 0.429 [0.360, 0.475] | 0.445 | 3.15 / 2.07 | 166668 |
| tileonly_mean | 0.646 [0.563, 0.742] | 0.626 [0.480, 0.765] | 0.076 [0.001, 0.175] | 0.467 [0.411, 0.498] | 0.501 | 3.61 / 2.07 | 165942 |
| protomatch | 0.626 [0.558, 0.740] | 0.577 [0.437, 0.717] | 0.019 [0.005, 0.034] | 0.481 [0.411, 0.532] | 0.882 | 5.61 / 2.07 | 2304 |

Per-species AP (AUROC):

| method | bs | bt | fj | ka | mx | pf |
|---|---|---|---|---|---|---|
| mean_lin | 0.64 (0.72) | nan (nan) | 0.53 (0.49) | 0.84 (0.83) | nan (nan) | 0.76 (0.74) |
| meanstd_lin | 0.70 (0.72) | nan (nan) | 0.49 (0.42) | 0.85 (0.84) | nan (nan) | 0.74 (0.74) |
| deepsets | 0.68 (0.74) | nan (nan) | 0.46 (0.29) | 0.95 (0.96) | nan (nan) | 0.57 (0.60) |
| abmil | 0.61 (0.66) | nan (nan) | 0.44 (0.25) | 0.78 (0.88) | nan (nan) | 0.63 (0.65) |
| abmil_cla1 | 0.64 (0.69) | nan (nan) | 0.45 (0.26) | 0.83 (0.90) | nan (nan) | 0.63 (0.65) |
| abmil_tile1 | 0.64 (0.65) | nan (nan) | 0.55 (0.48) | 0.88 (0.90) | nan (nan) | 0.57 (0.60) |
| tileonly_mean | 0.74 (0.72) | nan (nan) | 0.50 (0.41) | 0.89 (0.89) | nan (nan) | 0.45 (0.49) |
| protomatch | 0.65 (0.60) | nan (nan) | 0.42 (0.24) | 0.93 (0.92) | nan (nan) | 0.50 (0.55) |
| pos/neg combos | 7/7 | 0/14 | 8/6 | 7/7 | 0/14 | 7/7 |

Paired difference against `mean_lin` (same bags, same combination resamples):

| method | Δ macro AP | P(Δ>0) | Δ mean AUROC | Δ mean FPR |
|---|---|---|---|---|
| meanstd_lin | 0.000 [-0.059, 0.039] | 0.46 | -0.011 [-0.049, 0.034] | -0.024 [-0.057, 0.009] |
| deepsets | -0.029 [-0.116, 0.033] | 0.23 | -0.046 [-0.113, 0.014] | -0.035 [-0.095, 0.016] |
| abmil | -0.079 [-0.202, 0.026] | 0.10 | -0.084 [-0.193, 0.033] | -0.020 [-0.122, 0.087] |
| abmil_cla1 | -0.055 [-0.167, 0.038] | 0.17 | -0.069 [-0.175, 0.045] | -0.023 [-0.120, 0.068] |
| abmil_tile1 | -0.032 [-0.154, 0.061] | 0.29 | -0.036 [-0.130, 0.076] | 0.064 [-0.052, 0.175] |
| tileonly_mean | -0.048 [-0.130, 0.033] | 0.14 | -0.066 [-0.143, 0.024] | 0.120 [0.001, 0.235] |
| protomatch | -0.067 [-0.172, 0.035] | 0.11 | -0.115 [-0.200, -0.024] | 0.502 [0.364, 0.627] |

Context sensitivity (per-combination mean score: spread over positive combos S_pos, positive-negative gap, combination-level AUROC):

| method | bs | bt | fj | ka | mx | pf |
|---|---|---|---|---|---|---|
| mean_lin | 0.36 / +0.34 / 0.73 | nan / +nan / nan | 0.26 / -0.02 / 0.50 | 0.34 / +0.46 / 0.90 | nan / +nan / nan | 0.29 / +0.31 / 0.82 |
| meanstd_lin | 0.39 / +0.34 / 0.78 | nan / +nan / nan | 0.25 / -0.10 / 0.38 | 0.30 / +0.46 / 0.90 | nan / +nan / nan | 0.28 / +0.29 / 0.82 |
| deepsets | 0.26 / +0.37 / 0.78 | nan / +nan / nan | 0.23 / -0.32 / 0.29 | 0.13 / +0.77 / 0.96 | nan / +nan / nan | 0.19 / +0.05 / 0.57 |
| abmil | 0.27 / +0.20 / 0.65 | nan / +nan / nan | 0.27 / -0.25 / 0.23 | 0.10 / +0.64 / 0.90 | nan / +nan / nan | 0.17 / +0.12 / 0.61 |
| abmil_cla1 | 0.27 / +0.27 / 0.76 | nan / +nan / nan | 0.28 / -0.26 / 0.21 | 0.09 / +0.63 / 0.94 | nan / +nan / nan | 0.18 / +0.12 / 0.65 |
| abmil_tile1 | 0.22 / +0.16 / 0.65 | nan / +nan / nan | 0.18 / -0.02 / 0.48 | 0.20 / +0.61 / 0.94 | nan / +nan / nan | 0.13 / +0.08 / 0.57 |
| tileonly_mean | 0.10 / +0.08 / 0.78 | nan / +nan / nan | 0.05 / -0.04 / 0.33 | 0.41 / +0.43 / 0.92 | nan / +nan / nan | 0.08 / -0.03 / 0.43 |
| protomatch | 0.17 / +0.06 / 0.57 | nan / +nan / nan | 0.10 / -0.29 / 0.21 | 0.13 / +0.36 / 0.94 | nan / +nan / nan | 0.18 / +0.05 / 0.57 |

Own/context attribution (logit of per-combination mean score regressed on the presence vector; own = diagonal [95 % CI over combinations], context = mean |off-diagonal|, inverted = own < 0 (resolved: CI below 0), positive resolved = CI above 0):

| method | own bs | own bt | own fj | own ka | own mx | own pf | context | inverted (resolved) | positive resolved |
|---|---|---|---|---|---|---|---|---|---|
| mean_lin | +3.96 [+nan, +nan] | +0.00 [+nan, +nan] | +1.21 [+nan, +nan] | +5.67 [+nan, +nan] | +0.00 [+nan, +nan] | +2.21 [+nan, +nan] | 1.20 | 0 (0) | 0 |
| meanstd_lin | +4.39 [+nan, +nan] | +0.00 [+nan, +nan] | -0.01 [+nan, +nan] | +6.12 [+nan, +nan] | +0.00 [+nan, +nan] | +3.21 [+nan, +nan] | 1.16 | 1 (0) | 0 |
| deepsets | +2.80 [+nan, +nan] | +0.00 [+nan, +nan] | -1.77 [+nan, +nan] | +10.38 [+nan, +nan] | +0.00 [+nan, +nan] | +2.91 [+nan, +nan] | 1.32 | 1 (0) | 0 |
| abmil | +1.59 [+nan, +nan] | +0.00 [+nan, +nan] | -1.08 [+nan, +nan] | +5.31 [+nan, +nan] | +0.00 [+nan, +nan] | +1.47 [+nan, +nan] | 0.83 | 1 (0) | 0 |
| abmil_cla1 | +1.94 [+nan, +nan] | +0.00 [+nan, +nan] | -1.18 [+nan, +nan] | +5.44 [+nan, +nan] | +0.00 [+nan, +nan] | +1.44 [+nan, +nan] | 0.86 | 1 (0) | 0 |
| abmil_tile1 | +0.95 [+nan, +nan] | +0.00 [+nan, +nan] | +0.26 [+nan, +nan] | +5.30 [+nan, +nan] | +0.00 [+nan, +nan] | +0.75 [+nan, +nan] | 0.56 | 0 (0) | 0 |
| tileonly_mean | +1.60 [+nan, +nan] | -0.00 [+nan, +nan] | +1.23 [+nan, +nan] | +6.94 [+nan, +nan] | +0.00 [+nan, +nan] | +2.07 [+nan, +nan] | 1.23 | 1 (0) | 0 |
| protomatch | +0.14 [+nan, +nan] | -0.00 [+nan, +nan] | -4.17 [+nan, +nan] | +5.51 [+nan, +nan] | +0.00 [+nan, +nan] | -0.21 [+nan, +nan] | 1.41 | 3 (0) | 0 |

## n_fields = 4

| method | macro AP | mean AUROC | F1 margin | macro F1 | mean FPR | #pred / #true | params |
|---|---|---|---|---|---|---|---|
| mean_lin | 0.706 [0.620, 0.788] | 0.675 [0.565, 0.783] | 0.003 [-0.109, 0.117] | 0.415 [0.351, 0.446] | 0.361 | 2.71 / 2.07 | 2310 |
| meanstd_lin | 0.703 [0.582, 0.798] | 0.649 [0.512, 0.774] | -0.002 [-0.106, 0.128] | 0.405 [0.318, 0.457] | 0.368 | 2.61 / 2.07 | 4614 |
| deepsets | 0.642 [0.564, 0.745] | 0.614 [0.484, 0.737] | -0.019 [-0.121, 0.100] | 0.380 [0.312, 0.425] | 0.397 | 2.71 / 2.07 | 231686 |
| abmil | 0.611 [0.515, 0.736] | 0.599 [0.487, 0.711] | -0.077 [-0.189, 0.062] | 0.343 [0.259, 0.413] | 0.347 | 2.34 / 2.07 | 166668 |
| abmil_cla1 | 0.630 [0.533, 0.748] | 0.609 [0.500, 0.730] | -0.052 [-0.168, 0.091] | 0.357 [0.276, 0.423] | 0.348 | 2.43 / 2.07 | 166668 |
| abmil_tile1 | 0.646 [0.565, 0.745] | 0.647 [0.553, 0.744] | 0.026 [-0.069, 0.132] | 0.430 [0.360, 0.479] | 0.462 | 3.28 / 2.07 | 166668 |
| tileonly_mean | 0.638 [0.562, 0.730] | 0.610 [0.461, 0.757] | 0.092 [0.019, 0.188] | 0.469 [0.408, 0.500] | 0.465 | 3.42 / 2.07 | 165942 |
| protomatch | 0.627 [0.558, 0.742] | 0.579 [0.434, 0.724] | 0.021 [0.006, 0.038] | 0.483 [0.409, 0.537] | 0.880 | 5.62 / 2.07 | 2304 |

Per-species AP (AUROC):

| method | bs | bt | fj | ka | mx | pf |
|---|---|---|---|---|---|---|
| mean_lin | 0.68 (0.68) | nan (nan) | 0.51 (0.46) | 0.85 (0.85) | nan (nan) | 0.78 (0.72) |
| meanstd_lin | 0.76 (0.75) | nan (nan) | 0.45 (0.30) | 0.87 (0.86) | nan (nan) | 0.74 (0.69) |
| deepsets | 0.66 (0.69) | nan (nan) | 0.47 (0.31) | 0.93 (0.94) | nan (nan) | 0.51 (0.52) |
| abmil | 0.61 (0.65) | nan (nan) | 0.45 (0.28) | 0.76 (0.87) | nan (nan) | 0.62 (0.60) |
| abmil_cla1 | 0.67 (0.70) | nan (nan) | 0.45 (0.27) | 0.79 (0.88) | nan (nan) | 0.61 (0.59) |
| abmil_tile1 | 0.63 (0.68) | nan (nan) | 0.50 (0.41) | 0.84 (0.89) | nan (nan) | 0.61 (0.62) |
| tileonly_mean | 0.74 (0.71) | nan (nan) | 0.46 (0.36) | 0.89 (0.87) | nan (nan) | 0.47 (0.50) |
| protomatch | 0.65 (0.60) | nan (nan) | 0.41 (0.23) | 0.94 (0.93) | nan (nan) | 0.51 (0.55) |
| pos/neg combos | 7/7 | 0/14 | 8/6 | 7/7 | 0/14 | 7/7 |

Paired difference against `mean_lin` (same bags, same combination resamples):

| method | Δ macro AP | P(Δ>0) | Δ mean AUROC | Δ mean FPR |
|---|---|---|---|---|
| meanstd_lin | -0.003 [-0.096, 0.081] | 0.46 | -0.026 [-0.116, 0.072] | 0.006 [-0.064, 0.073] |
| deepsets | -0.064 [-0.119, 0.002] | 0.03 | -0.061 [-0.136, 0.011] | 0.036 [-0.039, 0.113] |
| abmil | -0.095 [-0.182, -0.011] | 0.01 | -0.076 [-0.167, 0.016] | -0.014 [-0.118, 0.087] |
| abmil_cla1 | -0.076 [-0.158, 0.003] | 0.03 | -0.066 [-0.156, 0.026] | -0.013 [-0.102, 0.073] |
| abmil_tile1 | -0.060 [-0.133, 0.006] | 0.04 | -0.028 [-0.110, 0.056] | 0.100 [-0.009, 0.210] |
| tileonly_mean | -0.068 [-0.126, 0.013] | 0.05 | -0.065 [-0.167, 0.031] | 0.104 [-0.024, 0.229] |
| protomatch | -0.079 [-0.158, 0.013] | 0.05 | -0.096 [-0.200, 0.002] | 0.519 [0.368, 0.649] |

Context sensitivity (per-combination mean score: spread over positive combos S_pos, positive-negative gap, combination-level AUROC):

| method | bs | bt | fj | ka | mx | pf |
|---|---|---|---|---|---|---|
| mean_lin | 0.38 / +0.21 / 0.71 | nan / +nan / nan | 0.31 / -0.06 / 0.44 | 0.41 / +0.51 / 0.90 | nan / +nan / nan | 0.37 / +0.36 / 0.80 |
| meanstd_lin | 0.42 / +0.38 / 0.78 | nan / +nan / nan | 0.32 / -0.28 / 0.25 | 0.39 / +0.50 / 0.86 | nan / +nan / nan | 0.44 / +0.25 / 0.63 |
| deepsets | 0.28 / +0.23 / 0.65 | nan / +nan / nan | 0.30 / -0.29 / 0.29 | 0.13 / +0.72 / 0.98 | nan / +nan / nan | 0.26 / -0.03 / 0.47 |
| abmil | 0.33 / +0.16 / 0.67 | nan / +nan / nan | 0.34 / -0.25 / 0.31 | 0.14 / +0.61 / 0.90 | nan / +nan / nan | 0.24 / +0.08 / 0.61 |
| abmil_cla1 | 0.33 / +0.28 / 0.73 | nan / +nan / nan | 0.32 / -0.28 / 0.31 | 0.13 / +0.64 / 0.92 | nan / +nan / nan | 0.27 / +0.07 / 0.61 |
| abmil_tile1 | 0.23 / +0.24 / 0.73 | nan / +nan / nan | 0.19 / -0.08 / 0.42 | 0.21 / +0.61 / 0.92 | nan / +nan / nan | 0.22 / +0.12 / 0.65 |
| tileonly_mean | 0.07 / +0.06 / 0.76 | nan / +nan / nan | 0.03 / -0.05 / 0.31 | 0.41 / +0.37 / 0.88 | nan / +nan / nan | 0.06 / -0.03 / 0.45 |
| protomatch | 0.17 / +0.06 / 0.57 | nan / +nan / nan | 0.10 / -0.29 / 0.21 | 0.13 / +0.36 / 0.94 | nan / +nan / nan | 0.18 / +0.05 / 0.57 |

Own/context attribution (logit of per-combination mean score regressed on the presence vector; own = diagonal [95 % CI over combinations], context = mean |off-diagonal|, inverted = own < 0 (resolved: CI below 0), positive resolved = CI above 0):

| method | own bs | own bt | own fj | own ka | own mx | own pf | context | inverted (resolved) | positive resolved |
|---|---|---|---|---|---|---|---|---|---|
| mean_lin | +3.89 [+nan, +nan] | +0.00 [+nan, +nan] | +0.45 [+nan, +nan] | +7.55 [+nan, +nan] | +0.00 [+nan, +nan] | +2.58 [+nan, +nan] | 1.75 | 0 (0) | 0 |
| meanstd_lin | +4.41 [+nan, +nan] | +0.00 [+nan, +nan] | -2.66 [+nan, +nan] | +7.49 [+nan, +nan] | +0.00 [+nan, +nan] | +3.70 [+nan, +nan] | 1.74 | 1 (0) | 0 |
| deepsets | +5.33 [+nan, +nan] | +0.00 [+nan, +nan] | -2.56 [+nan, +nan] | +9.92 [+nan, +nan] | +0.00 [+nan, +nan] | +3.17 [+nan, +nan] | 1.78 | 1 (0) | 0 |
| abmil | +3.10 [+nan, +nan] | +0.00 [+nan, +nan] | -1.35 [+nan, +nan] | +6.57 [+nan, +nan] | +0.00 [+nan, +nan] | +1.89 [+nan, +nan] | 1.33 | 1 (0) | 0 |
| abmil_cla1 | +3.62 [+nan, +nan] | +0.00 [+nan, +nan] | -1.28 [+nan, +nan] | +7.00 [+nan, +nan] | +0.00 [+nan, +nan] | +1.61 [+nan, +nan] | 1.25 | 1 (0) | 0 |
| abmil_tile1 | +1.77 [+nan, +nan] | +0.00 [+nan, +nan] | -0.02 [+nan, +nan] | +6.16 [+nan, +nan] | +0.00 [+nan, +nan] | +1.02 [+nan, +nan] | 0.76 | 1 (0) | 0 |
| tileonly_mean | +1.54 [+nan, +nan] | -0.00 [+nan, +nan] | +0.55 [+nan, +nan] | +6.75 [+nan, +nan] | +0.00 [+nan, +nan] | +2.16 [+nan, +nan] | 1.38 | 1 (0) | 0 |
| protomatch | +0.13 [+nan, +nan] | -0.00 [+nan, +nan] | -4.17 [+nan, +nan] | +5.49 [+nan, +nan] | +0.00 [+nan, +nan] | -0.21 [+nan, +nan] | 1.41 | 3 (0) | 0 |
