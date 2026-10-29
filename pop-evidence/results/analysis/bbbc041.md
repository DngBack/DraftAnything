# BBBC041 analyses (unit = image, out-of-fold over 5 folds)

## pe_quantile_m1

Context leakage B[k, j] (score of k when j present, k presence and count held fixed), with 95 % image-bootstrap intervals; * = interval excludes 0:

| score of \ present | gametocyte | leukocyte | ring | schizont | trophozoite | own (y_k) |
|---|---|---|---|---|---|---|
| gametocyte | — | +0.43 [+0.21, +0.71]* | -0.26 [-0.45, -0.07]* | +1.29 [+1.07, +1.53]* | +0.65 [+0.45, +0.84]* | +0.08 [-1.25, +1.36] |
| leukocyte | +0.73 [+0.57, +0.92]* | — | -0.12 [-0.32, +0.07] | +0.94 [+0.74, +1.10]* | +1.78 [+1.62, +1.92]* | -0.01 [-1.06, +0.83] |
| ring | -0.37 [-0.69, -0.02]* | +0.02 [-0.31, +0.34] | — | -1.67 [-2.03, -1.33]* | +0.12 [-0.10, +0.35] | -1.20 [-1.97, -0.40] |
| schizont | +1.51 [+1.13, +1.87]* | -0.01 [-0.45, +0.37] | -1.63 [-1.90, -1.40]* | — | +0.30 [+0.04, +0.57]* | +0.61 [-0.58, +1.77] |
| trophozoite | +0.60 [+0.26, +0.97]* | +0.96 [+0.51, +1.39]* | +0.02 [-0.31, +0.35] | +1.05 [+0.69, +1.40]* | — | +0.43 [-0.06, +0.86] |

Context coefficients resolved (CI excludes 0): 15/20

Tile localisation AUROC (evidence vs box-centre-in-tile):

| class | all tiles | tiles of images containing the class |
|---|---|---|
| gametocyte | 0.844 [0.806, 0.875] | 0.762 |
| leukocyte | 0.600 [0.550, 0.650] | 0.425 |
| ring | 0.385 [0.358, 0.413] | 0.623 |
| schizont | 0.090 [0.072, 0.110] | 0.186 |
| trophozoite | 0.076 [0.068, 0.083] | 0.135 |

## pe_quantile_tile1_m1

Context leakage B[k, j] (score of k when j present, k presence and count held fixed), with 95 % image-bootstrap intervals; * = interval excludes 0:

| score of \ present | gametocyte | leukocyte | ring | schizont | trophozoite | own (y_k) |
|---|---|---|---|---|---|---|
| gametocyte | — | +0.17 [-0.02, +0.40] | -0.27 [-0.43, -0.11]* | +1.03 [+0.86, +1.22]* | +0.17 [+0.01, +0.31]* | -0.04 [-0.92, +0.78] |
| leukocyte | +0.73 [+0.59, +0.90]* | — | -0.18 [-0.35, +0.01] | +0.93 [+0.76, +1.09]* | +1.50 [+1.35, +1.62]* | -0.33 [-1.59, +0.57] |
| ring | -0.34 [-0.55, -0.15]* | -0.11 [-0.30, +0.12] | — | -0.91 [-1.09, -0.71]* | -0.06 [-0.20, +0.09] | -0.99 [-1.52, -0.43] |
| schizont | +0.74 [+0.51, +0.98]* | -0.43 [-0.74, -0.14]* | -1.13 [-1.30, -0.96]* | — | -0.50 [-0.68, -0.32]* | +0.26 [-0.59, +1.14] |
| trophozoite | +0.33 [-0.00, +0.65] | +0.89 [+0.43, +1.29]* | -0.06 [-0.37, +0.26] | +0.66 [+0.31, +0.96]* | — | +0.10 [-0.40, +0.56] |

Context coefficients resolved (CI excludes 0): 14/20

Tile localisation AUROC (evidence vs box-centre-in-tile):

| class | all tiles | tiles of images containing the class |
|---|---|---|
| gametocyte | 0.873 [0.848, 0.894] | 0.739 |
| leukocyte | 0.656 [0.599, 0.711] | 0.453 |
| ring | 0.691 [0.658, 0.718] | 0.452 |
| schizont | 0.837 [0.810, 0.863] | 0.590 |
| trophozoite | 0.878 [0.868, 0.889] | 0.779 |
