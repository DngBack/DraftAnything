# Development, pooled dev folds [0, 1] — cla_tissue_pannuke, units=cls_pannuke

Intervals: 95 % bootstrap over combinations; point estimates and intervals averaged over training seeds. Primary endpoint: macro AP. `F1 margin` is per-sample F1 minus the all-present predictor's. Thresholds: per-species argmax-F1 on in-distribution calibration bags.

## n_fields = 1

| method | macro AP | mean AUROC | F1 margin | macro F1 | mean FPR | #pred / #true | params |
|---|---|---|---|---|---|---|---|
| abmil | 0.779 [0.759, 0.805] | 0.880 [0.867, 0.892] | 0.274 [0.264, 0.283] | 0.741 [0.720, 0.760] | 0.223 | 2.13 / 1.94 | 166282 | split-rank AUROC 0.880 [0.867, 0.892]
| abmil_cla1 | 0.748 [0.729, 0.771] | 0.860 [0.847, 0.873] | 0.263 [0.252, 0.274] | 0.708 [0.689, 0.728] | 0.233 | 2.11 / 1.94 | 166282 | split-rank AUROC 0.860 [0.846, 0.873]
| abmil_clatissue | 0.703 [0.690, 0.723] | 0.846 [0.833, 0.858] | 0.243 [0.232, 0.253] | 0.648 [0.633, 0.664] | 0.297 | 2.37 / 1.94 | 166282 | split-rank AUROC 0.846 [0.833, 0.858]
| abmil_tile1_clatissue | 0.703 [0.687, 0.725] | 0.852 [0.841, 0.864] | 0.245 [0.236, 0.256] | 0.670 [0.652, 0.690] | 0.298 | 2.35 / 1.94 | 166282 | split-rank AUROC 0.852 [0.841, 0.864]
| mean_lin | 0.814 [0.795, 0.838] | 0.892 [0.881, 0.901] | 0.267 [0.256, 0.277] | 0.738 [0.717, 0.761] | 0.213 | 2.09 / 1.94 | 1925 | split-rank AUROC 0.891 [0.881, 0.901]
| mean_lin_clatissue | 0.754 [0.736, 0.776] | 0.868 [0.858, 0.878] | 0.244 [0.233, 0.254] | 0.683 [0.660, 0.705] | 0.271 | 2.26 / 1.94 | 1925 | split-rank AUROC 0.868 [0.858, 0.878]

Per-species AP (AUROC):

| method | neoplastic | inflammatory | connective | dead | epithelial |
|---|---|---|---|---|---|
| abmil | 0.90 (0.93) | 0.89 (0.88) | 0.94 (0.87) | 0.41 (0.84) | 0.75 (0.89) |
| abmil_cla1 | 0.87 (0.90) | 0.88 (0.87) | 0.94 (0.86) | 0.32 (0.82) | 0.72 (0.86) |
| abmil_clatissue | 0.85 (0.89) | 0.88 (0.87) | 0.94 (0.86) | 0.17 (0.77) | 0.68 (0.84) |
| abmil_tile1_clatissue | 0.85 (0.89) | 0.87 (0.86) | 0.94 (0.86) | 0.23 (0.81) | 0.63 (0.83) |
| mean_lin | 0.92 (0.94) | 0.87 (0.84) | 0.95 (0.87) | 0.52 (0.89) | 0.82 (0.92) |
| mean_lin_clatissue | 0.89 (0.91) | 0.86 (0.83) | 0.94 (0.86) | 0.34 (0.86) | 0.74 (0.88) |
| pos/neg combos | 1111/1479 | 1368/1222 | 1906/684 | 81/2509 | 557/2033 |

Paired difference against `abmil` (same bags, same combination resamples):

| method | Δ macro AP | P(Δ>0) | Δ mean AUROC | Δ mean FPR |
|---|---|---|---|---|
| abmil_cla1 | -0.031 [-0.044, -0.019] | 0.00 | -0.020 [-0.026, -0.015] | 0.010 [0.004, 0.017] |
| abmil_clatissue | -0.077 [-0.093, -0.059] | 0.00 | -0.035 [-0.041, -0.028] | 0.074 [0.066, 0.083] |
| abmil_tile1_clatissue | -0.077 [-0.092, -0.061] | 0.00 | -0.028 [-0.035, -0.022] | 0.075 [0.067, 0.082] |
| mean_lin | 0.035 [0.021, 0.050] | 1.00 | 0.011 [0.003, 0.020] | -0.010 [-0.020, -0.000] |
| mean_lin_clatissue | -0.025 [-0.045, -0.005] | 0.01 | -0.013 [-0.023, -0.003] | 0.048 [0.038, 0.059] |

Context sensitivity (per-combination mean score: spread over positive combos S_pos, positive-negative gap, combination-level AUROC):

| method | neoplastic | inflammatory | connective | dead | epithelial |
|---|---|---|---|---|---|
| abmil | 0.26 / +0.61 / 0.93 | 0.25 / +0.43 / 0.88 | 0.18 / +0.38 / 0.87 | 0.17 / +0.17 / 0.84 | 0.33 / +0.48 / 0.89 |
| abmil_cla1 | 0.26 / +0.55 / 0.91 | 0.25 / +0.42 / 0.87 | 0.26 / +0.40 / 0.86 | 0.31 / +0.39 / 0.82 | 0.25 / +0.40 / 0.86 |
| abmil_clatissue | 0.25 / +0.50 / 0.89 | 0.25 / +0.41 / 0.87 | 0.25 / +0.40 / 0.86 | 0.26 / +0.26 / 0.77 | 0.23 / +0.36 / 0.84 |
| abmil_tile1_clatissue | 0.23 / +0.50 / 0.90 | 0.22 / +0.38 / 0.86 | 0.22 / +0.36 / 0.86 | 0.25 / +0.31 / 0.81 | 0.23 / +0.36 / 0.83 |
| mean_lin | 0.28 / +0.65 / 0.94 | 0.29 / +0.42 / 0.84 | 0.22 / +0.44 / 0.87 | 0.23 / +0.26 / 0.89 | 0.32 / +0.57 / 0.92 |
| mean_lin_clatissue | 0.27 / +0.57 / 0.91 | 0.29 / +0.39 / 0.83 | 0.30 / +0.45 / 0.86 | 0.36 / +0.47 / 0.86 | 0.29 / +0.53 / 0.88 |

Own/context attribution (logit of per-combination mean score regressed on the presence vector; own = diagonal [95 % CI over combinations], context = mean |off-diagonal|, inverted = own < 0 (resolved: CI below 0), positive resolved = CI above 0):

| method | own neoplastic | own inflammatory | own connective | own dead | own epithelial | context | inverted (resolved) | positive resolved |
|---|---|---|---|---|---|---|---|---|
| abmil | +4.76 [+4.6, +5.0] | +2.37 [+2.3, +2.5] | +2.60 [+2.4, +2.8] | +1.79 [+1.4, +2.2] | +3.09 [+2.9, +3.3] | 0.53 | 0 (0) | 5 |
| abmil_cla1 | +5.06 [+4.9, +5.3] | +2.32 [+2.2, +2.4] | +2.62 [+2.5, +2.8] | +1.77 [+1.3, +2.2] | +3.75 [+3.6, +3.9] | 0.68 | 0 (0) | 5 |
| abmil_clatissue | +4.88 [+4.7, +5.1] | +2.26 [+2.1, +2.4] | +2.58 [+2.4, +2.8] | +1.00 [+0.7, +1.3] | +3.65 [+3.5, +3.8] | 0.67 | 0 (0) | 5 |
| abmil_tile1_clatissue | +4.25 [+4.1, +4.4] | +2.06 [+1.9, +2.2] | +2.12 [+2.0, +2.3] | +1.24 [+1.0, +1.5] | +3.23 [+3.1, +3.4] | 0.59 | 0 (0) | 5 |
| mean_lin | +6.27 [+6.0, +6.6] | +2.89 [+2.7, +3.1] | +3.61 [+3.4, +3.8] | +1.64 [+1.4, +1.9] | +4.58 [+4.3, +4.8] | 0.45 | 0 (0) | 5 |
| mean_lin_clatissue | +6.41 [+6.1, +6.7] | +2.69 [+2.5, +2.9] | +3.87 [+3.6, +4.1] | +4.29 [+3.6, +5.1] | +5.96 [+5.7, +6.3] | 0.73 | 0 (0) | 5 |
