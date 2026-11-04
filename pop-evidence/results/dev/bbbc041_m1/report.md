# Development, pooled dev folds [0, 1, 2, 3, 4] — bbbc041_m1, units=cls_bbbc041

Intervals: 95 % bootstrap over combinations; point estimates and intervals averaged over training seeds. Primary endpoint: macro AP. `F1 margin` is per-sample F1 minus the all-present predictor's. Thresholds: per-species argmax-F1 on in-distribution calibration bags.

## n_fields = 1

| method | macro AP | mean AUROC | F1 margin | macro F1 | mean FPR | #pred / #true | params |
|---|---|---|---|---|---|---|---|
| pe_quantile_m1 | 0.485 [0.464, 0.520] | 0.818 [0.800, 0.836] | 0.246 [0.220, 0.273] | 0.472 [0.446, 0.498] | 0.147 | 1.23 / 1.04 | 165677 |
| pe_quantile_tile1_m1 | 0.481 [0.461, 0.515] | 0.816 [0.797, 0.835] | 0.268 [0.241, 0.296] | 0.481 [0.453, 0.507] | 0.136 | 1.20 / 1.04 | 165677 |
| pe_quantile_tile3_m1 | 0.480 [0.459, 0.514] | 0.813 [0.793, 0.831] | 0.275 [0.247, 0.304] | 0.476 [0.448, 0.503] | 0.119 | 1.12 / 1.04 | 165677 |
| pe_max_tile1_m1 | 0.444 [0.425, 0.472] | 0.781 [0.763, 0.798] | 0.210 [0.188, 0.230] | 0.454 [0.431, 0.476] | 0.214 | 1.53 / 1.04 | 165652 |

Per-species AP (AUROC):

| method | gametocyte | leukocyte | ring | schizont | trophozoite |
|---|---|---|---|---|---|
| pe_quantile_m1 | 0.27 (0.76) | 0.23 (0.78) | 0.63 (0.81) | 0.39 (0.83) | 0.91 (0.91) |
| pe_quantile_tile1_m1 | 0.26 (0.76) | 0.21 (0.77) | 0.64 (0.81) | 0.38 (0.82) | 0.92 (0.92) |
| pe_quantile_tile3_m1 | 0.27 (0.76) | 0.19 (0.76) | 0.64 (0.81) | 0.38 (0.82) | 0.92 (0.91) |
| pe_max_tile1_m1 | 0.19 (0.66) | 0.20 (0.78) | 0.57 (0.79) | 0.36 (0.80) | 0.89 (0.88) |
| pos/neg combos | 91/775 | 70/796 | 200/666 | 112/754 | 429/437 |

Paired difference against `pe_quantile_m1` (same bags, same combination resamples):

| method | Δ macro AP | P(Δ>0) | Δ mean AUROC | Δ mean FPR |
|---|---|---|---|---|
| pe_quantile_tile1_m1 | -0.003 [-0.014, 0.007] | 0.26 | -0.002 [-0.008, 0.004] | -0.010 [-0.017, -0.003] |
| pe_quantile_tile3_m1 | -0.005 [-0.016, 0.006] | 0.18 | -0.005 [-0.013, 0.002] | -0.028 [-0.034, -0.021] |
| pe_max_tile1_m1 | -0.041 [-0.056, -0.027] | 0.00 | -0.037 [-0.047, -0.027] | 0.067 [0.059, 0.076] |

Context sensitivity (per-combination mean score: spread over positive combos S_pos, positive-negative gap, combination-level AUROC):

| method | gametocyte | leukocyte | ring | schizont | trophozoite |
|---|---|---|---|---|---|
| pe_quantile_m1 | 0.15 / +0.13 / 0.77 | 0.13 / +0.12 / 0.78 | 0.33 / +0.32 / 0.81 | 0.21 / +0.24 / 0.84 | 0.27 / +0.55 / 0.91 |
| pe_quantile_tile1_m1 | 0.11 / +0.10 / 0.77 | 0.09 / +0.09 / 0.77 | 0.29 / +0.28 / 0.82 | 0.14 / +0.17 / 0.82 | 0.28 / +0.56 / 0.92 |
| pe_quantile_tile3_m1 | 0.12 / +0.10 / 0.76 | 0.08 / +0.07 / 0.76 | 0.31 / +0.30 / 0.81 | 0.13 / +0.16 / 0.82 | 0.28 / +0.55 / 0.92 |
| pe_max_tile1_m1 | 0.11 / +0.06 / 0.68 | 0.12 / +0.11 / 0.78 | 0.25 / +0.26 / 0.82 | 0.14 / +0.17 / 0.82 | 0.21 / +0.39 / 0.91 |

Own/context attribution (logit of per-combination mean score regressed on the presence vector; own = diagonal [95 % CI over combinations], context = mean |off-diagonal|, inverted = own < 0 (resolved: CI below 0), positive resolved = CI above 0):

| method | own gametocyte | own leukocyte | own ring | own schizont | own trophozoite | context | inverted (resolved) | positive resolved |
|---|---|---|---|---|---|---|---|---|
| pe_quantile_m1 | +1.49 [+1.2, +1.8] | +1.04 [+0.8, +1.3] | +2.10 [+1.8, +2.4] | +2.35 [+2.0, +2.7] | +4.40 [+4.1, +4.7] | 0.78 | 0 (0) | 5 |
| pe_quantile_tile1_m1 | +1.15 [+0.9, +1.4] | +0.88 [+0.7, +1.1] | +1.42 [+1.2, +1.6] | +1.32 [+1.1, +1.6] | +4.28 [+4.0, +4.6] | 0.62 | 0 (0) | 5 |
| pe_quantile_tile3_m1 | +1.12 [+0.9, +1.3] | +0.78 [+0.6, +1.0] | +1.69 [+1.5, +1.9] | +1.20 [+1.0, +1.4] | +4.05 [+3.7, +4.3] | 0.62 | 0 (0) | 5 |
| pe_max_tile1_m1 | +0.61 [+0.4, +0.8] | +0.91 [+0.7, +1.1] | +1.41 [+1.2, +1.6] | +0.90 [+0.7, +1.1] | +2.00 [+1.8, +2.1] | 0.38 | 0 (0) | 5 |
