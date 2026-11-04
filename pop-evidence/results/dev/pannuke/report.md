# Development, pooled dev folds [0, 1] — pannuke, units=cls_pannuke

Intervals: 95 % bootstrap over combinations; point estimates and intervals averaged over training seeds. Primary endpoint: macro AP. `F1 margin` is per-sample F1 minus the all-present predictor's. Thresholds: per-species argmax-F1 on in-distribution calibration bags.

## n_fields = 1

| method | macro AP | mean AUROC | F1 margin | macro F1 | mean FPR | #pred / #true | params |
|---|---|---|---|---|---|---|---|
| mean_lin | 0.814 [0.792, 0.839] | 0.892 [0.881, 0.902] | 0.267 [0.256, 0.277] | 0.738 [0.716, 0.761] | 0.213 | 2.09 / 1.94 | 1925 |
| meanstd_lin | 0.784 [0.763, 0.809] | 0.871 [0.859, 0.882] | 0.256 [0.246, 0.265] | 0.726 [0.704, 0.749] | 0.261 | 2.20 / 1.94 | 3845 |
| deepsets | 0.811 [0.789, 0.835] | 0.902 [0.892, 0.911] | 0.279 [0.268, 0.290] | 0.746 [0.726, 0.768] | 0.207 | 2.10 / 1.94 | 231429 |
| pe_quantile_m1 | 0.775 [0.755, 0.798] | 0.880 [0.869, 0.892] | 0.272 [0.261, 0.282] | 0.714 [0.695, 0.734] | 0.220 | 2.13 / 1.94 | 165677 |
| pe_quantile_tile1_m1 | 0.766 [0.748, 0.789] | 0.875 [0.863, 0.886] | 0.269 [0.259, 0.279] | 0.705 [0.685, 0.725] | 0.220 | 2.12 / 1.94 | 165677 |
| pe_max_tile1_m1 | 0.761 [0.745, 0.782] | 0.873 [0.861, 0.885] | 0.262 [0.252, 0.271] | 0.704 [0.687, 0.721] | 0.244 | 2.21 / 1.94 | 165652 |

Per-species AP (AUROC):

| method | neoplastic | inflammatory | connective | dead | epithelial |
|---|---|---|---|---|---|
| mean_lin | 0.92 (0.94) | 0.87 (0.84) | 0.95 (0.87) | 0.52 (0.89) | 0.82 (0.92) |
| meanstd_lin | 0.90 (0.92) | 0.84 (0.82) | 0.93 (0.84) | 0.46 (0.87) | 0.80 (0.90) |
| deepsets | 0.93 (0.94) | 0.88 (0.87) | 0.95 (0.88) | 0.47 (0.89) | 0.82 (0.93) |
| pe_quantile_m1 | 0.92 (0.94) | 0.88 (0.86) | 0.95 (0.87) | 0.35 (0.83) | 0.78 (0.90) |
| pe_quantile_tile1_m1 | 0.92 (0.94) | 0.87 (0.86) | 0.95 (0.88) | 0.29 (0.80) | 0.80 (0.91) |
| pe_max_tile1_m1 | 0.88 (0.91) | 0.89 (0.87) | 0.95 (0.87) | 0.33 (0.83) | 0.76 (0.88) |
| pos/neg combos | 1111/1479 | 1368/1222 | 1906/684 | 81/2509 | 557/2033 |

Paired difference against `mean_lin` (same bags, same combination resamples):

| method | Δ macro AP | P(Δ>0) | Δ mean AUROC | Δ mean FPR |
|---|---|---|---|---|
| meanstd_lin | -0.030 [-0.044, -0.017] | 0.00 | -0.021 [-0.028, -0.014] | 0.048 [0.039, 0.057] |
| deepsets | -0.004 [-0.015, 0.008] | 0.28 | 0.010 [0.003, 0.017] | -0.006 [-0.015, 0.003] |
| pe_quantile_m1 | -0.039 [-0.052, -0.026] | 0.00 | -0.011 [-0.019, -0.004] | 0.007 [-0.001, 0.016] |
| pe_quantile_tile1_m1 | -0.049 [-0.062, -0.032] | 0.00 | -0.017 [-0.025, -0.009] | 0.007 [-0.002, 0.016] |
| pe_max_tile1_m1 | -0.053 [-0.065, -0.039] | 0.00 | -0.018 [-0.027, -0.010] | 0.031 [0.022, 0.040] |

Context sensitivity (per-combination mean score: spread over positive combos S_pos, positive-negative gap, combination-level AUROC):

| method | neoplastic | inflammatory | connective | dead | epithelial |
|---|---|---|---|---|---|
| mean_lin | 0.28 / +0.65 / 0.94 | 0.29 / +0.42 / 0.84 | 0.22 / +0.44 / 0.87 | 0.23 / +0.26 / 0.89 | 0.32 / +0.57 / 0.92 |
| meanstd_lin | 0.30 / +0.66 / 0.92 | 0.34 / +0.44 / 0.82 | 0.27 / +0.43 / 0.84 | 0.24 / +0.26 / 0.87 | 0.35 / +0.59 / 0.90 |
| deepsets | 0.27 / +0.65 / 0.94 | 0.26 / +0.43 / 0.87 | 0.20 / +0.40 / 0.88 | 0.32 / +0.29 / 0.89 | 0.35 / +0.59 / 0.93 |
| pe_quantile_m1 | 0.26 / +0.63 / 0.94 | 0.25 / +0.41 / 0.87 | 0.18 / +0.39 / 0.88 | 0.21 / +0.19 / 0.84 | 0.34 / +0.52 / 0.91 |
| pe_quantile_tile1_m1 | 0.26 / +0.63 / 0.94 | 0.23 / +0.39 / 0.87 | 0.18 / +0.38 / 0.88 | 0.09 / +0.08 / 0.82 | 0.35 / +0.51 / 0.91 |
| pe_max_tile1_m1 | 0.20 / +0.47 / 0.93 | 0.24 / +0.42 / 0.88 | 0.18 / +0.37 / 0.87 | 0.17 / +0.18 / 0.86 | 0.30 / +0.44 / 0.90 |

Own/context attribution (logit of per-combination mean score regressed on the presence vector; own = diagonal [95 % CI over combinations], context = mean |off-diagonal|, inverted = own < 0 (resolved: CI below 0), positive resolved = CI above 0):

| method | own neoplastic | own inflammatory | own connective | own dead | own epithelial | context | inverted (resolved) | positive resolved |
|---|---|---|---|---|---|---|---|---|
| mean_lin | +6.27 [+6.0, +6.6] | +2.89 [+2.7, +3.1] | +3.61 [+3.4, +3.8] | +1.64 [+1.4, +1.9] | +4.58 [+4.3, +4.8] | 0.45 | 0 (0) | 5 |
| meanstd_lin | +8.83 [+8.5, +9.2] | +3.96 [+3.7, +4.2] | +4.56 [+4.3, +4.9] | +1.70 [+1.4, +2.0] | +5.60 [+5.2, +5.9] | 0.57 | 0 (0) | 5 |
| deepsets | +6.05 [+5.8, +6.3] | +2.46 [+2.3, +2.6] | +3.11 [+2.9, +3.3] | +3.23 [+2.7, +3.8] | +4.82 [+4.5, +5.1] | 0.66 | 0 (0) | 5 |
| pe_quantile_m1 | +4.86 [+4.7, +5.1] | +2.16 [+2.0, +2.3] | +2.66 [+2.5, +2.8] | +1.92 [+1.5, +2.4] | +3.47 [+3.3, +3.7] | 0.53 | 0 (0) | 5 |
| pe_quantile_tile1_m1 | +4.87 [+4.7, +5.1] | +1.99 [+1.9, +2.1] | +2.60 [+2.4, +2.8] | +1.13 [+0.9, +1.4] | +3.30 [+3.1, +3.5] | 0.50 | 0 (0) | 5 |
| pe_max_tile1_m1 | +2.41 [+2.3, +2.5] | +2.31 [+2.2, +2.4] | +2.41 [+2.3, +2.6] | +1.68 [+1.4, +2.0] | +2.39 [+2.2, +2.5] | 0.40 | 0 (0) | 5 |
