# Development, pooled dev folds [0, 1, 2, 3, 4] — valid_tile_bbbc041, units=cls_bbbc041

Intervals: 95 % bootstrap over combinations; point estimates and intervals averaged over training seeds. Primary endpoint: macro AP. `F1 margin` is per-sample F1 minus the all-present predictor's. Thresholds: per-species argmax-F1 on in-distribution calibration bags.

## n_fields = 1

| method | macro AP | mean AUROC | F1 margin | macro F1 | mean FPR | #pred / #true | params |
|---|---|---|---|---|---|---|---|
| abmil | 0.459 [0.438, 0.491] | 0.800 [0.781, 0.819] | 0.244 [0.220, 0.270] | 0.479 [0.452, 0.506] | 0.166 | 1.34 / 1.04 | 166282 | split-rank AUROC 0.802 [0.783, 0.820]
| abmil_tileV | 0.426 [0.405, 0.457] | 0.771 [0.748, 0.791] | 0.194 [0.169, 0.219] | 0.431 [0.403, 0.452] | 0.164 | 1.26 / 1.04 | 166282 | split-rank AUROC 0.773 [0.753, 0.794]
| abmil_tileV_cla1 | 0.419 [0.399, 0.451] | 0.768 [0.746, 0.788] | 0.196 [0.169, 0.225] | 0.431 [0.402, 0.455] | 0.168 | 1.26 / 1.04 | 166282 | split-rank AUROC 0.770 [0.747, 0.789]
| pe_quantile_m1_tileV | 0.437 [0.419, 0.465] | 0.769 [0.750, 0.785] | 0.180 [0.162, 0.199] | 0.446 [0.425, 0.465] | 0.224 | 1.56 / 1.04 | 165677 | split-rank AUROC 0.771 [0.754, 0.787]
| tileonlyV_mean | 0.347 [0.332, 0.367] | 0.677 [0.655, 0.694] | 0.144 [0.120, 0.166] | 0.363 [0.342, 0.379] | 0.275 | 1.73 / 1.04 | 165677 | split-rank AUROC 0.681 [0.661, 0.698]

Per-species AP (AUROC):

| method | gametocyte | leukocyte | ring | schizont | trophozoite |
|---|---|---|---|---|---|
| abmil | 0.25 (0.76) | 0.21 (0.76) | 0.62 (0.79) | 0.32 (0.80) | 0.90 (0.89) |
| abmil_tileV | 0.17 (0.68) | 0.20 (0.74) | 0.57 (0.78) | 0.32 (0.77) | 0.88 (0.87) |
| abmil_tileV_cla1 | 0.21 (0.70) | 0.18 (0.73) | 0.53 (0.77) | 0.30 (0.77) | 0.87 (0.87) |
| pe_quantile_m1_tileV | 0.24 (0.74) | 0.18 (0.74) | 0.55 (0.76) | 0.33 (0.74) | 0.88 (0.87) |
| tileonlyV_mean | 0.15 (0.60) | 0.12 (0.65) | 0.45 (0.70) | 0.20 (0.63) | 0.82 (0.82) |
| pos/neg combos | 91/775 | 70/796 | 200/666 | 112/754 | 429/437 |

Paired difference against `abmil` (same bags, same combination resamples):

| method | Δ macro AP | P(Δ>0) | Δ mean AUROC | Δ mean FPR |
|---|---|---|---|---|
| abmil_tileV | -0.033 [-0.054, -0.013] | 0.00 | -0.030 [-0.047, -0.015] | -0.002 [-0.014, 0.011] |
| abmil_tileV_cla1 | -0.040 [-0.060, -0.019] | 0.00 | -0.033 [-0.050, -0.017] | 0.001 [-0.011, 0.015] |
| pe_quantile_m1_tileV | -0.022 [-0.041, -0.003] | 0.01 | -0.032 [-0.045, -0.018] | 0.058 [0.048, 0.068] |
| tileonlyV_mean | -0.113 [-0.136, -0.088] | 0.00 | -0.123 [-0.144, -0.103] | 0.109 [0.089, 0.127] |

Context sensitivity (per-combination mean score: spread over positive combos S_pos, positive-negative gap, combination-level AUROC):

| method | gametocyte | leukocyte | ring | schizont | trophozoite |
|---|---|---|---|---|---|
| abmil | 0.16 / +0.12 / 0.77 | 0.14 / +0.11 / 0.76 | 0.36 / +0.33 / 0.79 | 0.23 / +0.21 / 0.80 | 0.30 / +0.58 / 0.90 |
| abmil_tileV | 0.11 / +0.06 / 0.69 | 0.13 / +0.10 / 0.75 | 0.29 / +0.26 / 0.78 | 0.20 / +0.17 / 0.78 | 0.27 / +0.48 / 0.90 |
| abmil_tileV_cla1 | 0.23 / +0.18 / 0.70 | 0.18 / +0.16 / 0.74 | 0.29 / +0.29 / 0.77 | 0.23 / +0.25 / 0.78 | 0.29 / +0.47 / 0.87 |
| pe_quantile_m1_tileV | 0.08 / +0.07 / 0.75 | 0.10 / +0.08 / 0.76 | 0.21 / +0.21 / 0.81 | 0.13 / +0.12 / 0.79 | 0.26 / +0.47 / 0.91 |
| tileonlyV_mean | 0.00 / +0.00 / 0.61 | 0.00 / +0.00 / 0.66 | 0.00 / +0.00 / 0.73 | 0.00 / +0.00 / 0.64 | 0.05 / +0.02 / 0.86 |

Own/context attribution (logit of per-combination mean score regressed on the presence vector; own = diagonal [95 % CI over combinations], context = mean |off-diagonal|, inverted = own < 0 (resolved: CI below 0), positive resolved = CI above 0):

| method | own gametocyte | own leukocyte | own ring | own schizont | own trophozoite | context | inverted (resolved) | positive resolved |
|---|---|---|---|---|---|---|---|---|
| abmil | +1.57 [+1.3, +1.9] | +0.93 [+0.7, +1.2] | +2.26 [+1.9, +2.6] | +1.91 [+1.5, +2.3] | +5.31 [+4.9, +5.7] | 0.74 | 0 (0) | 5 |
| abmil_tileV | +0.80 [+0.6, +1.0] | +0.78 [+0.5, +1.0] | +1.66 [+1.4, +1.9] | +1.17 [+0.9, +1.5] | +3.21 [+3.0, +3.5] | 0.42 | 0 (0) | 5 |
| abmil_tileV_cla1 | +0.99 [+0.7, +1.3] | +0.56 [+0.4, +0.8] | +1.74 [+1.5, +2.0] | +1.15 [+0.9, +1.4] | +3.42 [+3.1, +3.7] | 0.38 | 0 (0) | 5 |
| pe_quantile_m1_tileV | +0.84 [+0.7, +1.0] | +0.72 [+0.5, +0.9] | +0.99 [+0.9, +1.1] | +0.81 [+0.6, +1.0] | +2.99 [+2.7, +3.2] | 0.42 | 0 (0) | 5 |
| tileonlyV_mean | +0.02 [-0.0, +0.1] | -0.03 [-0.0, -0.0] | +0.30 [+0.2, +0.4] | +0.03 [-0.0, +0.1] | +2.72 [+2.5, +3.0] | 0.14 | 1 (1) | 2 |
