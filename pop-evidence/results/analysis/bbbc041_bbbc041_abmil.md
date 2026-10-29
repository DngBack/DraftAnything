# bbbc041 analyses (unit = image, out-of-fold)

## abmil

Context leakage B[k, j] (score of k when j present, k presence and count held fixed), with 95 % image-bootstrap intervals; * = interval excludes 0:

| score of \ present | gametocyte | leukocyte | ring | schizont | trophozoite | own (y_k) |
|---|---|---|---|---|---|---|
| gametocyte | — | +0.34 [+0.06, +0.63]* | -0.10 [-0.29, +0.11] | +1.53 [+1.29, +1.78]* | +0.84 [+0.65, +1.03]* | +0.25 [-1.43, +2.25] |
| leukocyte | +0.40 [+0.21, +0.63]* | — | -0.00 [-0.18, +0.19] | +0.55 [+0.35, +0.73]* | +1.44 [+1.29, +1.58]* | +0.24 [-1.56, +2.80] |
| ring | -0.46 [-0.87, -0.10]* | +0.05 [-0.37, +0.41] | — | -1.32 [-1.76, -0.87]* | +0.16 [-0.10, +0.40] | -1.21 [-2.09, -0.16] |
| schizont | +1.09 [+0.70, +1.52]* | -0.34 [-0.74, +0.10] | -1.23 [-1.50, -0.95]* | — | -0.33 [-0.59, -0.05]* | +0.18 [-1.25, +1.82] |
| trophozoite | +0.89 [+0.33, +1.41]* | +0.82 [+0.22, +1.45]* | -0.22 [-0.64, +0.18] | +1.53 [+0.95, +2.07]* | — | +1.04 [+0.34, +1.68] |

Context coefficients resolved (CI excludes 0): 14/20

Tile localisation AUROC (evidence vs box-centre-in-tile):

| class | all tiles | tiles of images containing the class |
|---|---|---|
| gametocyte | 0.603 [0.542, 0.670] | 0.628 |
| leukocyte | 0.501 [0.422, 0.574] | 0.528 |
| ring | 0.298 [0.268, 0.333] | 0.320 |
| schizont | 0.364 [0.320, 0.409] | 0.399 |
| trophozoite | 0.803 [0.785, 0.824] | 0.815 |
