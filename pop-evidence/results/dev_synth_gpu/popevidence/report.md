# Development, pooled dev folds [0, 1, 2, 3, 4] — smoke_popevidence, units=cls_synth

Intervals: 95 % bootstrap over combinations; point estimates and intervals averaged over training seeds. Primary endpoint: macro AP. `F1 margin` is per-sample F1 minus the all-present predictor's. Thresholds: per-species argmax-F1 on in-distribution calibration bags.

## n_fields = 1

| method | macro AP | mean AUROC | F1 margin | macro F1 | mean FPR | #pred / #true | params |
|---|---|---|---|---|---|---|---|
| pe_mean | 0.850 [0.802, 0.906] | 0.872 [0.816, 0.924] | 0.194 [0.126, 0.260] | 0.818 [0.768, 0.860] | 0.175 | 2.82 / 2.73 | 88630 |
| pe_meanstd | 0.839 [0.798, 0.888] | 0.865 [0.798, 0.918] | 0.192 [0.122, 0.258] | 0.817 [0.766, 0.864] | 0.170 | 2.80 / 2.73 | 88654 |

Per-species AP (AUROC):

| method | bs | bt | fj | ka | mx | pf |
|---|---|---|---|---|---|---|
| pe_mean | 0.67 (0.74) | 0.47 (0.54) | 0.99 (0.98) | 0.98 (0.98) | 1.00 (0.99) | 0.99 (0.99) |
| pe_meanstd | 0.62 (0.69) | 0.43 (0.51) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) |
| pos/neg combos | 13/13 | 12/14 | 11/15 | 9/17 | 14/12 | 12/14 |

Paired difference against `pe_mean` (same bags, same combination resamples):

| method | Δ macro AP | P(Δ>0) | Δ mean AUROC | Δ mean FPR |
|---|---|---|---|---|
| pe_meanstd | -0.011 [-0.026, 0.001] | 0.06 | -0.007 [-0.034, 0.011] | -0.005 [-0.030, 0.019] |

Context sensitivity (per-combination mean score: spread over positive combos S_pos, positive-negative gap, combination-level AUROC):

| method | bs | bt | fj | ka | mx | pf |
|---|---|---|---|---|---|---|
| pe_mean | 0.21 / +0.27 / 0.74 | 0.17 / +0.02 / 0.55 | 0.11 / +0.92 / 1.00 | 0.15 / +0.88 / 1.00 | 0.08 / +0.90 / 1.00 | 0.08 / +0.89 / 1.00 |
| pe_meanstd | 0.22 / +0.23 / 0.69 | 0.25 / -0.02 / 0.51 | 0.04 / +0.97 / 1.00 | 0.02 / +0.97 / 1.00 | 0.01 / +0.98 / 1.00 | 0.06 / +0.94 / 1.00 |

## n_fields = 4

| method | macro AP | mean AUROC | F1 margin | macro F1 | mean FPR | #pred / #true | params |
|---|---|---|---|---|---|---|---|
| pe_mean | 0.856 [0.805, 0.918] | 0.879 [0.814, 0.935] | 0.214 [0.141, 0.284] | 0.836 [0.772, 0.888] | 0.131 | 2.66 / 2.73 | 88630 |
| pe_meanstd | 0.846 [0.801, 0.899] | 0.874 [0.801, 0.927] | 0.217 [0.139, 0.282] | 0.848 [0.788, 0.896] | 0.128 | 2.68 / 2.73 | 88654 |

Per-species AP (AUROC):

| method | bs | bt | fj | ka | mx | pf |
|---|---|---|---|---|---|---|
| pe_mean | 0.67 (0.73) | 0.48 (0.55) | 0.99 (0.99) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) |
| pe_meanstd | 0.63 (0.71) | 0.44 (0.54) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) |
| pos/neg combos | 13/13 | 12/14 | 11/15 | 9/17 | 14/12 | 12/14 |

Paired difference against `pe_mean` (same bags, same combination resamples):

| method | Δ macro AP | P(Δ>0) | Δ mean AUROC | Δ mean FPR |
|---|---|---|---|---|
| pe_meanstd | -0.011 [-0.032, 0.006] | 0.12 | -0.005 [-0.032, 0.015] | -0.004 [-0.040, 0.023] |

Context sensitivity (per-combination mean score: spread over positive combos S_pos, positive-negative gap, combination-level AUROC):

| method | bs | bt | fj | ka | mx | pf |
|---|---|---|---|---|---|---|
| pe_mean | 0.28 / +0.28 / 0.73 | 0.21 / +0.03 / 0.57 | 0.20 / +0.91 / 1.00 | 0.11 / +0.93 / 1.00 | 0.06 / +0.96 / 1.00 | 0.07 / +0.94 / 1.00 |
| pe_meanstd | 0.27 / +0.27 / 0.72 | 0.30 / +0.00 / 0.54 | 0.06 / +0.98 / 1.00 | 0.01 / +0.99 / 1.00 | 0.00 / +1.00 / 1.00 | 0.06 / +0.97 / 1.00 |
