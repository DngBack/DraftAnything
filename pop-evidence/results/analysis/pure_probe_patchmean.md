# Pure-culture probe (patchmean)

Classifier: linear, 5 classes ['bs', 'fj', 'ka', 'mx', 'pf'], fitted on 13572 train images (217152 tiles) of the pure cultures.

(a) Pure-culture val: tile accuracy 0.810, image accuracy (mean of 16 tile posteriors) 0.999. Upper bound: one video per species, so species and session coincide.

Tile confusion (rows true, columns predicted, row-normalised):

| | bs | fj | ka | mx | pf |
|---|---|---|---|---|---|
| bs | 0.78 | 0.10 | 0.00 | 0.04 | 0.08 |
| fj | 0.07 | 0.75 | 0.00 | 0.04 | 0.14 |
| ka | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| mx | 0.06 | 0.06 | 0.00 | 0.83 | 0.06 |
| pf | 0.09 | 0.17 | 0.00 | 0.06 | 0.68 |

(b) Trained mixtures (26 combinations, orders 2-4, unseen by this classifier): fraction of tiles assigned to each species.

| species | mean fraction when present | when absent | combination AUROC |
|---|---|---|---|
| bs | 0.124 | 0.105 | 0.61 |
| fj | 0.290 | 0.249 | 0.57 |
| ka | 0.273 | 0.031 | 0.92 |
| mx | 0.178 | 0.124 | 0.66 |
| pf | 0.323 | 0.307 | 0.53 |

| combo | bs | fj | ka | mx | pf |
|---|---|---|---|---|---|
| bs_fj | ✓0.25 | ✓0.30 |  0.00 |  0.09 |  0.36 |
| bs_ka | ✓0.00 |  0.24 | ✓0.69 |  0.00 |  0.07 |
| bs_mx | ✓0.19 |  0.22 |  0.00 | ✓0.14 |  0.45 |
| bt_ka |  0.12 |  0.10 | ✓0.53 |  0.04 |  0.21 |
| bt_mx |  0.13 |  0.31 |  0.00 | ✓0.05 |  0.50 |
| bt_pf |  0.17 |  0.31 |  0.00 |  0.14 | ✓0.37 |
| fj_pf |  0.11 | ✓0.25 |  0.01 |  0.12 | ✓0.51 |
| mx_fj |  0.16 | ✓0.25 |  0.00 | ✓0.12 |  0.47 |
| mx_ka |  0.01 |  0.10 | ✓0.58 | ✓0.06 |  0.25 |
| mx_pf |  0.10 |  0.33 |  0.02 | ✓0.15 | ✓0.40 |
| bs_bt_mx | ✓0.19 |  0.27 |  0.00 | ✓0.15 |  0.38 |
| bs_fj_pf | ✓0.22 | ✓0.33 |  0.00 |  0.10 | ✓0.35 |
| bs_ka_fj | ✓0.15 | ✓0.45 | ✓0.01 |  0.10 |  0.29 |
| bs_mx_ka | ✓0.04 |  0.14 | ✓0.37 | ✓0.08 |  0.36 |
| bs_mx_pf | ✓0.18 |  0.26 |  0.00 | ✓0.20 | ✓0.36 |
| bt_fj_pf |  0.02 | ✓0.11 |  0.37 |  0.32 | ✓0.19 |
| bt_ka_fj |  0.02 | ✓0.06 | ✓0.44 |  0.28 |  0.20 |
| bt_mx_ka |  0.17 |  0.47 | ✓0.02 | ✓0.08 |  0.26 |
| bt_mx_pf |  0.15 |  0.35 |  0.00 | ✓0.33 | ✓0.17 |
| ka_fj_pf |  0.13 | ✓0.52 | ✓0.00 |  0.11 | ✓0.24 |
| mx_fj_pf |  0.11 | ✓0.24 |  0.00 | ✓0.15 | ✓0.51 |
| mx_ka_pf |  0.11 |  0.20 | ✓0.04 | ✓0.40 | ✓0.26 |
| bs_ka_fj_pf | ✓0.00 | ✓0.52 | ✓0.34 |  0.03 | ✓0.10 |
| bs_mx_ka_pf | ✓0.02 |  0.19 | ✓0.35 | ✓0.11 | ✓0.33 |
| bt_ka_fj_pf |  0.14 | ✓0.29 | ✓0.02 |  0.15 | ✓0.41 |
| bt_mx_ka_fj |  0.06 | ✓0.15 | ✓0.16 | ✓0.45 |  0.18 |