# bbbc041 analyses (unit = image, out-of-fold)

## pe_quantile_tile1_m1

Context leakage B[k, j] (score of k when j present, k presence and count held fixed), with 95 % image-bootstrap intervals; * = interval excludes 0:

| score of \ present | gametocyte | leukocyte | ring | schizont | trophozoite | own (y_k) |
|---|---|---|---|---|---|---|
| gametocyte | — | +0.17 [-0.02, +0.38] | -0.27 [-0.43, -0.14]* | +1.03 [+0.89, +1.21]* | +0.17 [+0.02, +0.28]* | -0.04 [-1.11, +0.66] |
| leukocyte | +0.73 [+0.55, +0.88]* | — | -0.18 [-0.34, -0.02]* | +0.93 [+0.81, +1.09]* | +1.50 [+1.37, +1.61]* | -0.33 [-1.23, +0.54] |
| ring | -0.34 [-0.57, -0.14]* | -0.11 [-0.31, +0.06] | — | -0.91 [-1.07, -0.74]* | -0.06 [-0.18, +0.09] | -0.99 [-1.44, -0.63] |
| schizont | +0.74 [+0.52, +1.00]* | -0.43 [-0.68, -0.22]* | -1.13 [-1.27, -0.97]* | — | -0.50 [-0.61, -0.35]* | +0.26 [-0.54, +1.00] |
| trophozoite | +0.33 [+0.05, +0.71]* | +0.89 [+0.44, +1.15]* | -0.06 [-0.24, +0.18] | +0.66 [+0.34, +1.01]* | — | +0.10 [-0.46, +0.58] |

Context coefficients resolved (CI excludes 0): 16/20

Tile localisation AUROC (evidence vs box-centre-in-tile):

| class | all tiles | tiles of images containing the class |
|---|---|---|
| gametocyte | 0.873 [0.850, 0.896] | 0.739 |
| leukocyte | 0.656 [0.616, 0.718] | 0.453 |
| ring | 0.691 [0.655, 0.729] | 0.452 |
| schizont | 0.837 [0.805, 0.859] | 0.590 |
| trophozoite | 0.878 [0.868, 0.889] | 0.779 |
