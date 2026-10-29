# Development, pooled dev folds [0, 1, 2, 3, 4] — bbbc041_nobce, units=cls_bbbc041

Intervals: 95 % bootstrap over combinations; point estimates and intervals averaged over training seeds. Primary endpoint: macro AP. `F1 margin` is per-sample F1 minus the all-present predictor's. Thresholds: per-species argmax-F1 on in-distribution calibration bags.

## n_fields = 1

| method | macro AP | mean AUROC | F1 margin | macro F1 | mean FPR | #pred / #true | params |
|---|---|---|---|---|---|---|---|
| pe_max_tile1_m1 | 0.444 [0.425, 0.472] | 0.781 [0.762, 0.798] | 0.210 [0.189, 0.231] | 0.454 [0.431, 0.476] | 0.214 | 1.53 / 1.04 | 165652 |
| pe_max_tileonly_m1 | 0.468 [0.444, 0.503] | 0.801 [0.781, 0.822] | 0.232 [0.206, 0.256] | 0.479 [0.450, 0.505] | 0.194 | 1.46 / 1.04 | 165652 |
| pe_q90_tileonly_m1 | 0.481 [0.457, 0.515] | 0.813 [0.792, 0.831] | 0.257 [0.228, 0.286] | 0.484 [0.456, 0.510] | 0.135 | 1.19 / 1.04 | 165677 |

Per-species AP (AUROC):

| method | gametocyte | leukocyte | ring | schizont | trophozoite |
|---|---|---|---|---|---|
| pe_max_tile1_m1 | 0.19 (0.66) | 0.20 (0.78) | 0.57 (0.79) | 0.36 (0.80) | 0.89 (0.88) |
| pe_max_tileonly_m1 | 0.24 (0.75) | 0.20 (0.75) | 0.63 (0.81) | 0.38 (0.82) | 0.89 (0.88) |
| pe_q90_tileonly_m1 | 0.26 (0.76) | 0.19 (0.76) | 0.65 (0.82) | 0.39 (0.83) | 0.91 (0.90) |
| pos/neg combos | 91/775 | 70/796 | 200/666 | 112/754 | 429/437 |

Paired difference against `pe_max_tile1_m1` (same bags, same combination resamples):

| method | Δ macro AP | P(Δ>0) | Δ mean AUROC | Δ mean FPR |
|---|---|---|---|---|
| pe_max_tileonly_m1 | 0.024 [0.006, 0.041] | 0.99 | 0.020 [0.008, 0.032] | -0.019 [-0.030, -0.010] |
| pe_q90_tileonly_m1 | 0.038 [0.025, 0.051] | 1.00 | 0.032 [0.023, 0.042] | -0.079 [-0.088, -0.070] |

Context sensitivity (per-combination mean score: spread over positive combos S_pos, positive-negative gap, combination-level AUROC):

| method | gametocyte | leukocyte | ring | schizont | trophozoite |
|---|---|---|---|---|---|
| pe_max_tile1_m1 | 0.11 / +0.06 / 0.68 | 0.12 / +0.11 / 0.78 | 0.25 / +0.26 / 0.82 | 0.14 / +0.17 / 0.82 | 0.21 / +0.39 / 0.91 |
| pe_max_tileonly_m1 | 0.21 / +0.22 / 0.75 | 0.30 / +0.28 / 0.75 | 0.23 / +0.32 / 0.81 | 0.17 / +0.26 / 0.83 | 0.13 / +0.26 / 0.88 |
| pe_q90_tileonly_m1 | 0.18 / +0.16 / 0.76 | 0.20 / +0.17 / 0.76 | 0.30 / +0.35 / 0.82 | 0.19 / +0.26 / 0.83 | 0.20 / +0.40 / 0.90 |

Own/context attribution (logit of per-combination mean score regressed on the presence vector; own = diagonal [95 % CI over combinations], context = mean |off-diagonal|, inverted = own < 0 (resolved: CI below 0), positive resolved = CI above 0):

| method | own gametocyte | own leukocyte | own ring | own schizont | own trophozoite | context | inverted (resolved) | positive resolved |
|---|---|---|---|---|---|---|---|---|
| pe_max_tile1_m1 | +0.61 [+0.4, +0.8] | +0.91 [+0.7, +1.1] | +1.41 [+1.2, +1.6] | +0.90 [+0.7, +1.1] | +2.00 [+1.9, +2.2] | 0.38 | 0 (0) | 5 |
| pe_max_tileonly_m1 | +1.37 [+1.1, +1.6] | +1.10 [+0.8, +1.4] | +2.11 [+1.8, +2.4] | +1.20 [+1.0, +1.4] | +3.49 [+3.2, +3.8] | 0.63 | 0 (0) | 5 |
| pe_q90_tileonly_m1 | +1.29 [+1.0, +1.5] | +0.93 [+0.7, +1.2] | +1.99 [+1.7, +2.3] | +1.39 [+1.1, +1.6] | +3.17 [+2.9, +3.4] | 0.60 | 0 (0) | 5 |
