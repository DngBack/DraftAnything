# Pure-culture probe (cls)

Classifier: linear, 5 classes ['bs', 'fj', 'ka', 'mx', 'pf'], fitted on 13572 train images (217152 tiles) of the pure cultures.

(a) Pure-culture val: tile accuracy 0.828, image accuracy (mean of 16 tile posteriors) 1.000. Upper bound: one video per species, so species and session coincide.

Tile confusion (rows true, columns predicted, row-normalised):

| | bs | fj | ka | mx | pf |
|---|---|---|---|---|---|
| bs | 0.80 | 0.10 | 0.00 | 0.03 | 0.07 |
| fj | 0.06 | 0.78 | 0.00 | 0.03 | 0.13 |
| ka | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| mx | 0.04 | 0.05 | 0.00 | 0.85 | 0.05 |
| pf | 0.09 | 0.17 | 0.00 | 0.05 | 0.69 |

(b) Trained mixtures (26 combinations, orders 2-4, unseen by this classifier): fraction of tiles assigned to each species.

| species | mean fraction when present | when absent | combination AUROC [95 % CI over combinations] |
|---|---|---|---|
| bs | 0.124 | 0.104 | 0.62 [0.32, 0.88] |
| fj | 0.323 | 0.286 | 0.57 [0.32, 0.80] |
| ka | 0.285 | 0.035 | 0.89 [0.74, 0.99] resolved |
| mx | 0.152 | 0.111 | 0.60 [0.35, 0.82] |
| pf | 0.310 | 0.275 | 0.57 [0.33, 0.80] |

Content proxy (std of illumination-corrected grey level per tile) of the pure cultures (train):

| species | median tile std | share of tiles below global median |
|---|---|---|
| bs | 3.22 | 0.56 |
| fj | 2.73 | 0.64 |
| ka | 5.03 | 0.06 |
| mx | 4.91 | 0.32 |
| pf | 3.21 | 0.56 |

(c) Tile vote restricted to content tiles (std above a global quantile):

| quantile | bs | fj | ka | mx | pf |
|---|---|---|---|---|---|
| 0.0 | 0.62 | 0.57 | 0.89 | 0.60 | 0.57 |
| 0.5 | 0.62 | 0.57 | 0.88 | 0.61 | 0.63 |
| 0.75 | 0.56 | 0.56 | 0.84 | 0.60 | 0.60 |

| combo | bs | fj | ka | mx | pf |
|---|---|---|---|---|---|
| bs_fj | ✓0.27 | ✓0.33 |  0.00 |  0.08 |  0.32 |
| bs_ka | ✓0.00 |  0.42 | ✓0.50 |  0.00 |  0.08 |
| bs_mx | ✓0.17 |  0.25 |  0.00 | ✓0.15 |  0.43 |
| bt_ka |  0.10 |  0.10 | ✓0.61 |  0.04 |  0.15 |
| bt_mx |  0.12 |  0.33 |  0.00 | ✓0.05 |  0.49 |
| bt_pf |  0.16 |  0.34 |  0.00 |  0.13 | ✓0.37 |
| fj_pf |  0.11 | ✓0.28 |  0.01 |  0.11 | ✓0.50 |
| mx_fj |  0.15 | ✓0.29 |  0.00 | ✓0.10 |  0.47 |
| mx_ka |  0.00 |  0.11 | ✓0.66 | ✓0.03 |  0.20 |
| mx_pf |  0.10 |  0.35 |  0.03 | ✓0.13 | ✓0.39 |
| bs_bt_mx | ✓0.20 |  0.29 |  0.00 | ✓0.14 |  0.36 |
| bs_fj_pf | ✓0.19 | ✓0.37 |  0.00 |  0.07 | ✓0.36 |
| bs_ka_fj | ✓0.16 | ✓0.51 | ✓0.01 |  0.08 |  0.24 |
| bs_mx_ka | ✓0.02 |  0.21 | ✓0.42 | ✓0.03 |  0.31 |
| bs_mx_pf | ✓0.20 |  0.28 |  0.00 | ✓0.19 | ✓0.33 |
| bt_fj_pf |  0.02 | ✓0.12 |  0.39 |  0.33 | ✓0.14 |
| bt_ka_fj |  0.02 | ✓0.06 | ✓0.55 |  0.24 |  0.13 |
| bt_mx_ka |  0.17 |  0.54 | ✓0.03 | ✓0.06 |  0.20 |
| bt_mx_pf |  0.14 |  0.36 |  0.00 | ✓0.33 | ✓0.17 |
| ka_fj_pf |  0.11 | ✓0.60 | ✓0.00 |  0.10 | ✓0.19 |
| mx_fj_pf |  0.18 | ✓0.21 |  0.02 | ✓0.11 | ✓0.48 |
| mx_ka_pf |  0.10 |  0.18 | ✓0.02 | ✓0.40 | ✓0.29 |
| bs_ka_fj_pf | ✓0.00 | ✓0.61 | ✓0.27 |  0.03 | ✓0.09 |
| bs_mx_ka_pf | ✓0.02 |  0.24 | ✓0.36 | ✓0.07 | ✓0.31 |
| bt_ka_fj_pf |  0.11 | ✓0.36 | ✓0.01 |  0.12 | ✓0.40 |
| bt_mx_ka_fj |  0.08 | ✓0.14 | ✓0.25 | ✓0.33 |  0.20 |