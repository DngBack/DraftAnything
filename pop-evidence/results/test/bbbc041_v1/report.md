# Held-out test ['bbbc041'] (read once) — bbbc041_v1

Intervals: 95 % bootstrap over combinations; point estimates and intervals averaged over training seeds. Primary endpoint: macro AP. `F1 margin` is per-sample F1 minus the all-present predictor's. Thresholds: per-species argmax-F1 on in-distribution calibration bags.

## n_fields = 1

| method | macro AP | mean AUROC | F1 margin | macro F1 | mean FPR | #pred / #true | params |
|---|---|---|---|---|---|---|---|
| mean_lin | 0.272 [0.245, 0.331] | 0.390 [0.311, 0.471] | -0.031 [-0.080, 0.017] | 0.243 [0.207, 0.276] | 0.542 | 2.63 / 1.16 | 1925 |
| deepsets | 0.300 [0.263, 0.389] | 0.451 [0.388, 0.519] | -0.055 [-0.107, -0.002] | 0.190 [0.149, 0.228] | 0.407 | 2.02 / 1.16 | 231429 |
| abmil | 0.354 [0.318, 0.441] | 0.545 [0.460, 0.638] | -0.006 [-0.071, 0.063] | 0.203 [0.152, 0.246] | 0.229 | 1.20 / 1.16 | 166282 |
| abmil_tile1 | 0.334 [0.305, 0.400] | 0.548 [0.480, 0.618] | -0.065 [-0.130, 0.001] | 0.203 [0.149, 0.252] | 0.161 | 0.91 / 1.16 | 166282 |
| abmil_tile1_cla1 | 0.399 [0.346, 0.494] | 0.577 [0.498, 0.657] | -0.035 [-0.105, 0.040] | 0.198 [0.126, 0.276] | 0.122 | 0.73 / 1.16 | 166282 |
| pe_quantile_m1 | 0.272 [0.247, 0.330] | 0.439 [0.379, 0.499] | -0.038 [-0.094, 0.024] | 0.182 [0.139, 0.222] | 0.291 | 1.46 / 1.16 | 165677 |
| tileonly_mean | 0.279 [0.235, 0.356] | 0.423 [0.360, 0.488] | 0.046 [-0.026, 0.124] | 0.158 [0.120, 0.200] | 0.236 | 1.18 / 1.16 | 165677 |

Per-species AP (AUROC):

| method | gametocyte | leukocyte | ring | schizont | trophozoite |
|---|---|---|---|---|---|
| mean_lin | 0.09 (0.48) | nan (nan) | 0.50 (0.45) | 0.05 (0.19) | 0.45 (0.44) |
| deepsets | 0.20 (0.59) | nan (nan) | 0.50 (0.50) | 0.10 (0.39) | 0.40 (0.31) |
| abmil | 0.17 (0.65) | nan (nan) | 0.51 (0.50) | 0.10 (0.41) | 0.64 (0.62) |
| abmil_tile1 | 0.13 (0.65) | nan (nan) | 0.55 (0.51) | 0.09 (0.47) | 0.57 (0.57) |
| abmil_tile1_cla1 | 0.20 (0.67) | nan (nan) | 0.56 (0.52) | 0.18 (0.47) | 0.66 (0.65) |
| pe_quantile_m1 | 0.14 (0.64) | nan (nan) | 0.43 (0.41) | 0.06 (0.35) | 0.45 (0.35) |
| tileonly_mean | 0.25 (0.67) | nan (nan) | 0.41 (0.36) | 0.06 (0.33) | 0.40 (0.33) |
| pos/neg combos | 9/106 | 0/115 | 57/58 | 9/106 | 58/57 |

Paired difference against `mean_lin` (same bags, same combination resamples):

| method | Δ macro AP | P(Δ>0) | Δ mean AUROC | Δ mean FPR |
|---|---|---|---|---|
| deepsets | 0.027 [-0.019, 0.094] | 0.88 | 0.061 [-0.025, 0.149] | -0.135 [-0.168, -0.099] |
| abmil | 0.082 [0.031, 0.154] | 1.00 | 0.155 [0.041, 0.274] | -0.313 [-0.350, -0.275] |
| abmil_tile1 | 0.062 [0.017, 0.114] | 1.00 | 0.158 [0.059, 0.257] | -0.381 [-0.425, -0.339] |
| abmil_tile1_cla1 | 0.127 [0.059, 0.207] | 1.00 | 0.187 [0.075, 0.299] | -0.420 [-0.462, -0.379] |
| pe_quantile_m1 | -0.000 [-0.045, 0.045] | 0.53 | 0.049 [-0.049, 0.148] | -0.251 [-0.290, -0.211] |
| tileonly_mean | 0.007 [-0.049, 0.063] | 0.57 | 0.033 [-0.053, 0.122] | -0.306 [-0.341, -0.269] |

Context sensitivity (per-combination mean score: spread over positive combos S_pos, positive-negative gap, combination-level AUROC):

| method | gametocyte | leukocyte | ring | schizont | trophozoite |
|---|---|---|---|---|---|
| mean_lin | 0.01 / +0.01 / 0.49 | nan / +nan / nan | 0.40 / -0.08 / 0.46 | 0.00 / -0.02 / 0.19 | 0.35 / -0.05 / 0.44 |
| deepsets | 0.08 / +0.03 / 0.60 | nan / +nan / nan | 0.04 / -0.00 / 0.50 | 0.13 / -0.03 / 0.39 | 0.16 / -0.08 / 0.31 |
| abmil | 0.08 / +0.04 / 0.66 | nan / +nan / nan | 0.06 / -0.00 / 0.50 | 0.16 / -0.01 / 0.42 | 0.25 / +0.06 / 0.62 |
| abmil_tile1 | 0.07 / +0.03 / 0.65 | nan / +nan / nan | 0.12 / +0.02 / 0.50 | 0.06 / -0.01 / 0.47 | 0.17 / +0.04 / 0.57 |
| abmil_tile1_cla1 | 0.17 / +0.10 / 0.66 | nan / +nan / nan | 0.17 / +0.02 / 0.52 | 0.23 / -0.01 / 0.47 | 0.20 / +0.12 / 0.65 |
| pe_quantile_m1 | 0.08 / +0.04 / 0.64 | nan / +nan / nan | 0.02 / -0.02 / 0.41 | 0.07 / -0.05 / 0.33 | 0.13 / -0.05 / 0.36 |
| tileonly_mean | 0.04 / +0.02 / 0.68 | nan / +nan / nan | 0.01 / -0.01 / 0.36 | 0.02 / -0.02 / 0.33 | 0.09 / -0.05 / 0.32 |

Own/context attribution (logit of per-combination mean score regressed on the presence vector; own = diagonal [95 % CI over combinations], context = mean |off-diagonal|, inverted = own < 0 (resolved: CI below 0), positive resolved = CI above 0):

| method | own gametocyte | own leukocyte | own ring | own schizont | own trophozoite | context | inverted (resolved) | positive resolved |
|---|---|---|---|---|---|---|---|---|
| mean_lin | +0.19 [+nan, +nan] | +0.00 [+nan, +nan] | +1.72 [+nan, +nan] | -1.56 [+nan, +nan] | -1.51 [+nan, +nan] | 0.76 | 2 (0) | 0 |
| deepsets | +0.19 [+nan, +nan] | +0.00 [+nan, +nan] | -0.08 [+nan, +nan] | -0.14 [+nan, +nan] | -0.42 [+nan, +nan] | 0.09 | 3 (0) | 0 |
| abmil | +0.25 [+nan, +nan] | -0.00 [+nan, +nan] | +0.28 [+nan, +nan] | -0.12 [+nan, +nan] | +0.63 [+nan, +nan] | 0.21 | 2 (0) | 0 |
| abmil_tile1 | +0.29 [+nan, +nan] | +0.00 [+nan, +nan] | +0.51 [+nan, +nan] | +0.16 [+nan, +nan] | +0.47 [+nan, +nan] | 0.23 | 0 (0) | 0 |
| abmil_tile1_cla1 | +0.39 [+nan, +nan] | +0.00 [+nan, +nan] | +0.49 [+nan, +nan] | +0.31 [+nan, +nan] | +0.96 [+nan, +nan] | 0.23 | 0 (0) | 0 |
| pe_quantile_m1 | +0.28 [+nan, +nan] | +0.00 [+nan, +nan] | -0.14 [+nan, +nan] | -0.05 [+nan, +nan] | -0.28 [+nan, +nan] | 0.13 | 3 (0) | 0 |
| tileonly_mean | +0.25 [+nan, +nan] | +0.00 [+nan, +nan] | -0.03 [+nan, +nan] | -0.01 [+nan, +nan] | -0.26 [+nan, +nan] | 0.12 | 3 (0) | 0 |
