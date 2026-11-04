# pannuke analyses (unit = image, out-of-fold)

## mean_lin

Context leakage B[k, j] (score of k when j present, k presence and count held fixed), with 95 % image-bootstrap intervals; * = interval excludes 0:

| score of \ present | neoplastic | inflammatory | connective | dead | epithelial | own (y_k) |
|---|---|---|---|---|---|---|
| neoplastic | — | +0.70 [+0.46, +0.91]* | +0.45 [+0.24, +0.67]* | +0.07 [-0.52, +0.69] | +0.36 [+0.08, +0.67]* | +1.66 [+1.06, +2.21] |
| inflammatory | +1.27 [+1.08, +1.46]* | — | +0.90 [+0.73, +1.10]* | +0.76 [+0.33, +1.16]* | +1.18 [+0.95, +1.41]* | +0.36 [+0.12, +0.61] |
| connective | +0.27 [+0.05, +0.46]* | +0.34 [+0.18, +0.52]* | — | +0.18 [-0.26, +0.63] | +0.14 [-0.12, +0.38] | +0.74 [+0.45, +1.07] |
| dead | +0.24 [+0.16, +0.31]* | +0.23 [+0.17, +0.29]* | -0.05 [-0.11, +0.02] | — | -0.04 [-0.12, +0.03] | +1.27 [+0.81, +1.69] |
| epithelial | -0.39 [-0.60, -0.20]* | +0.36 [+0.20, +0.53]* | -0.04 [-0.24, +0.17] | -0.27 [-0.73, +0.24] | — | +3.92 [+3.09, +4.77] |

Context coefficients resolved (CI excludes 0): 13/20

## meanstd_lin

Context leakage B[k, j] (score of k when j present, k presence and count held fixed), with 95 % image-bootstrap intervals; * = interval excludes 0:

| score of \ present | neoplastic | inflammatory | connective | dead | epithelial | own (y_k) |
|---|---|---|---|---|---|---|
| neoplastic | — | +0.84 [+0.54, +1.16]* | +0.57 [+0.21, +0.92]* | +0.47 [-0.44, +1.40] | +0.22 [-0.26, +0.63] | +2.98 [+2.08, +3.79] |
| inflammatory | +1.66 [+1.36, +1.95]* | — | +1.54 [+1.25, +1.87]* | +0.83 [+0.13, +1.44]* | +1.78 [+1.40, +2.09]* | +0.99 [+0.58, +1.43] |
| connective | +0.68 [+0.33, +0.98]* | +0.50 [+0.27, +0.74]* | — | +0.24 [-0.41, +1.00] | +0.49 [+0.10, +0.86]* | +1.52 [+1.07, +2.07] |
| dead | +0.23 [+0.14, +0.30]* | +0.25 [+0.19, +0.32]* | -0.03 [-0.10, +0.04] | — | -0.11 [-0.20, -0.04]* | +1.27 [+0.75, +1.76] |
| epithelial | -0.43 [-0.68, -0.19]* | +0.58 [+0.35, +0.79]* | +0.07 [-0.19, +0.34] | -0.48 [-1.01, +0.06] | — | +3.89 [+2.81, +4.94] |

Context coefficients resolved (CI excludes 0): 14/20

## deepsets

Context leakage B[k, j] (score of k when j present, k presence and count held fixed), with 95 % image-bootstrap intervals; * = interval excludes 0:

| score of \ present | neoplastic | inflammatory | connective | dead | epithelial | own (y_k) |
|---|---|---|---|---|---|---|
| neoplastic | — | +0.66 [+0.46, +0.85]* | +0.13 [-0.09, +0.34] | -0.20 [-0.76, +0.32] | +0.24 [-0.01, +0.49] | +1.16 [+0.58, +1.65] |
| inflammatory | +0.67 [+0.54, +0.79]* | — | +0.82 [+0.70, +0.95]* | +0.27 [-0.04, +0.57] | +0.72 [+0.57, +0.88]* | +0.26 [+0.10, +0.44] |
| connective | -0.16 [-0.36, +0.06] | +0.56 [+0.39, +0.73]* | — | -0.12 [-0.46, +0.18] | -0.47 [-0.68, -0.25]* | +0.29 [+0.03, +0.58] |
| dead | +2.43 [+2.26, +2.62]* | +0.70 [+0.52, +0.85]* | -0.41 [-0.58, -0.24]* | — | +0.98 [+0.79, +1.19]* | +2.04 [+1.00, +2.99] |
| epithelial | -1.20 [-1.40, -0.99]* | +0.48 [+0.28, +0.67]* | +0.17 [-0.07, +0.41] | +0.06 [-0.56, +0.66] | — | +3.81 [+3.12, +4.67] |

Context coefficients resolved (CI excludes 0): 12/20

## pe_quantile_m1

Context leakage B[k, j] (score of k when j present, k presence and count held fixed), with 95 % image-bootstrap intervals; * = interval excludes 0:

| score of \ present | neoplastic | inflammatory | connective | dead | epithelial | own (y_k) |
|---|---|---|---|---|---|---|
| neoplastic | — | +0.40 [+0.21, +0.55]* | +0.04 [-0.15, +0.21] | +0.19 [-0.26, +0.65] | -0.20 [-0.42, +0.05] | +0.91 [+0.41, +1.30] |
| inflammatory | +0.65 [+0.53, +0.77]* | — | +0.81 [+0.69, +0.94]* | +0.74 [+0.45, +1.03]* | +0.81 [+0.67, +0.96]* | +0.37 [+0.22, +0.54] |
| connective | -0.02 [-0.18, +0.16] | +0.50 [+0.36, +0.63]* | — | +0.13 [-0.14, +0.39] | -0.12 [-0.31, +0.08] | +0.47 [+0.25, +0.73] |
| dead | +1.90 [+1.76, +2.08]* | +0.88 [+0.72, +1.02]* | -0.28 [-0.43, -0.12]* | — | -0.09 [-0.27, +0.09] | +1.11 [+0.38, +1.85] |
| epithelial | -0.64 [-0.78, -0.51]* | +0.44 [+0.31, +0.58]* | -0.09 [-0.23, +0.07] | -0.06 [-0.41, +0.29] | — | +2.86 [+2.27, +3.58] |

Context coefficients resolved (CI excludes 0): 11/20

Tile localisation AUROC (evidence vs box-centre-in-tile):

| class | all tiles | tiles of images containing the class |
|---|---|---|
| neoplastic | 0.850 [0.844, 0.857] | 0.732 |
| inflammatory | 0.751 [0.742, 0.761] | 0.711 |
| connective | 0.228 [0.218, 0.236] | 0.290 |
| dead | 0.115 [0.091, 0.146] | 0.326 |
| epithelial | 0.194 [0.179, 0.211] | 0.390 |

## pe_quantile_tile1_m1

Context leakage B[k, j] (score of k when j present, k presence and count held fixed), with 95 % image-bootstrap intervals; * = interval excludes 0:

| score of \ present | neoplastic | inflammatory | connective | dead | epithelial | own (y_k) |
|---|---|---|---|---|---|---|
| neoplastic | — | +0.31 [+0.12, +0.47]* | -0.02 [-0.20, +0.16] | +0.03 [-0.39, +0.47] | -0.02 [-0.27, +0.21] | +1.04 [+0.57, +1.46] |
| inflammatory | +0.69 [+0.57, +0.79]* | — | +0.86 [+0.75, +0.99]* | +0.85 [+0.59, +1.13]* | +0.80 [+0.66, +0.93]* | +0.39 [+0.24, +0.56] |
| connective | -0.25 [-0.40, -0.07]* | +0.39 [+0.24, +0.52]* | — | +0.12 [-0.15, +0.36] | -0.32 [-0.49, -0.14]* | +0.42 [+0.21, +0.67] |
| dead | +1.05 [+0.95, +1.15]* | +0.55 [+0.45, +0.64]* | -0.33 [-0.41, -0.23]* | — | -0.27 [-0.39, -0.15]* | +0.76 [+0.27, +1.25] |
| epithelial | -0.77 [-0.91, -0.65]* | +0.32 [+0.19, +0.45]* | -0.02 [-0.17, +0.12] | -0.28 [-0.73, +0.13] | — | +2.93 [+2.38, +3.56] |

Context coefficients resolved (CI excludes 0): 14/20

Tile localisation AUROC (evidence vs box-centre-in-tile):

| class | all tiles | tiles of images containing the class |
|---|---|---|
| neoplastic | 0.879 [0.873, 0.886] | 0.758 |
| inflammatory | 0.895 [0.890, 0.900] | 0.854 |
| connective | 0.773 [0.766, 0.782] | 0.712 |
| dead | 0.890 [0.860, 0.915] | 0.688 |
| epithelial | 0.799 [0.783, 0.816] | 0.583 |
