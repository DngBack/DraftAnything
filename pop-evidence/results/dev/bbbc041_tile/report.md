# Development, pooled dev folds [0, 1, 2, 3, 4] — bbbc041_tile, units=cls_bbbc041

Intervals: 95 % bootstrap over combinations; point estimates and intervals averaged over training seeds. Primary endpoint: macro AP. `F1 margin` is per-sample F1 minus the all-present predictor's. Thresholds: per-species argmax-F1 on in-distribution calibration bags.

## n_fields = 1

| method | macro AP | mean AUROC | F1 margin | macro F1 | mean FPR | #pred / #true | params |
|---|---|---|---|---|---|---|---|
| pe_quantile_s | 0.488 [0.466, 0.522] | 0.819 [0.801, 0.838] | 0.261 [0.233, 0.287] | 0.495 [0.468, 0.522] | 0.148 | 1.27 / 1.04 | 169637 |
| pe_quantile_tile1_s | 0.491 [0.468, 0.526] | 0.824 [0.805, 0.843] | 0.288 [0.261, 0.314] | 0.500 [0.471, 0.526] | 0.147 | 1.27 / 1.04 | 169637 |
| pe_quantile_tile3_s | 0.489 [0.466, 0.523] | 0.823 [0.804, 0.841] | 0.281 [0.254, 0.306] | 0.498 [0.470, 0.525] | 0.147 | 1.27 / 1.04 | 169637 |
| pe_max_tile1_s | 0.500 [0.477, 0.533] | 0.829 [0.811, 0.846] | 0.260 [0.235, 0.283] | 0.496 [0.471, 0.518] | 0.182 | 1.45 / 1.04 | 169537 |

Per-species AP (AUROC):

| method | gametocyte | leukocyte | ring | schizont | trophozoite |
|---|---|---|---|---|---|
| pe_quantile_s | 0.26 (0.76) | 0.24 (0.79) | 0.63 (0.81) | 0.38 (0.83) | 0.92 (0.91) |
| pe_quantile_tile1_s | 0.27 (0.77) | 0.22 (0.78) | 0.65 (0.81) | 0.39 (0.84) | 0.92 (0.92) |
| pe_quantile_tile3_s | 0.27 (0.77) | 0.21 (0.77) | 0.65 (0.82) | 0.39 (0.84) | 0.92 (0.92) |
| pe_max_tile1_s | 0.30 (0.81) | 0.21 (0.76) | 0.64 (0.81) | 0.43 (0.85) | 0.92 (0.92) |
| pos/neg combos | 91/775 | 70/796 | 200/666 | 112/754 | 429/437 |

Paired difference against `pe_quantile_s` (same bags, same combination resamples):

| method | Δ macro AP | P(Δ>0) | Δ mean AUROC | Δ mean FPR |
|---|---|---|---|---|
| pe_quantile_tile1_s | 0.003 [-0.006, 0.013] | 0.75 | 0.005 [0.001, 0.009] | -0.000 [-0.006, 0.006] |
| pe_quantile_tile3_s | 0.001 [-0.011, 0.012] | 0.56 | 0.004 [-0.001, 0.009] | -0.000 [-0.006, 0.006] |
| pe_max_tile1_s | 0.012 [-0.005, 0.027] | 0.91 | 0.010 [0.002, 0.019] | 0.034 [0.027, 0.042] |

Context sensitivity (per-combination mean score: spread over positive combos S_pos, positive-negative gap, combination-level AUROC):

| method | gametocyte | leukocyte | ring | schizont | trophozoite |
|---|---|---|---|---|---|
| pe_quantile_s | 0.17 / +0.13 / 0.76 | 0.13 / +0.13 / 0.79 | 0.34 / +0.33 / 0.81 | 0.22 / +0.25 / 0.84 | 0.28 / +0.57 / 0.92 |
| pe_quantile_tile1_s | 0.17 / +0.14 / 0.78 | 0.12 / +0.11 / 0.78 | 0.34 / +0.33 / 0.82 | 0.21 / +0.24 / 0.84 | 0.28 / +0.57 / 0.92 |
| pe_quantile_tile3_s | 0.16 / +0.13 / 0.77 | 0.11 / +0.10 / 0.77 | 0.34 / +0.33 / 0.82 | 0.20 / +0.24 / 0.84 | 0.28 / +0.56 / 0.92 |
| pe_max_tile1_s | 0.21 / +0.18 / 0.82 | 0.12 / +0.10 / 0.77 | 0.35 / +0.33 / 0.81 | 0.22 / +0.26 / 0.85 | 0.27 / +0.57 / 0.92 |

Own/context attribution (logit of per-combination mean score regressed on the presence vector; own = diagonal [95 % CI over combinations], context = mean |off-diagonal|, inverted = own < 0 (resolved: CI below 0), positive resolved = CI above 0):

| method | own gametocyte | own leukocyte | own ring | own schizont | own trophozoite | context | inverted (resolved) | positive resolved |
|---|---|---|---|---|---|---|---|---|
| pe_quantile_s | +1.69 [+1.4, +2.0] | +1.14 [+0.9, +1.4] | +2.23 [+2.0, +2.6] | +2.66 [+2.3, +3.0] | +4.87 [+4.5, +5.2] | 0.89 | 0 (0) | 5 |
| pe_quantile_tile1_s | +1.79 [+1.5, +2.1] | +0.97 [+0.7, +1.2] | +2.15 [+1.9, +2.4] | +2.45 [+2.1, +2.8] | +4.66 [+4.3, +5.0] | 0.84 | 0 (0) | 5 |
| pe_quantile_tile3_s | +1.72 [+1.4, +2.0] | +0.90 [+0.7, +1.1] | +2.13 [+1.9, +2.4] | +2.32 [+2.0, +2.7] | +4.41 [+4.1, +4.7] | 0.79 | 0 (0) | 5 |
| pe_max_tile1_s | +3.00 [+2.6, +3.4] | +0.94 [+0.7, +1.2] | +2.27 [+2.0, +2.6] | +2.72 [+2.4, +3.1] | +4.61 [+4.3, +4.9] | 0.99 | 0 (0) | 5 |
