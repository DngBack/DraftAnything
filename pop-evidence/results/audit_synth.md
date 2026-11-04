# Data audit

Manifests: `data/synth`
Species: ['bs', 'bt', 'fj', 'ka', 'mx', 'pf']

## Leakage checks

- test combos ∩ train combos: none
- test combos ∩ val combos: none
- train combos == val combos: True
- duplicate image paths across parts: 0
- held-out combos (manifest): ['bs_bt_fj_ka_mx_pf', 'bs_bt_ka', 'bs_bt_mx_pf', 'bs_mx_pf', 'bt', 'bt_fj', 'bt_fj_pf', 'fj_ka', 'fj_ka_mx_pf']

- fold 0: touches test combos: none; dev∩train: none
- fold 1: touches test combos: none; dev∩train: none
- fold 2: touches test combos: none; dev∩train: none
- fold 3: touches test combos: none; dev∩train: none
- fold 4: touches test combos: none; dev∩train: none

## Combinations

| combo | order | role | train | val | test |
|---|---|---|---|---|---|
| bs | 1 | trained | 267 | 33 | 0 |
| bt | 1 | test | 0 | 0 | 300 |
| fj | 1 | trained | 277 | 23 | 0 |
| ka | 1 | trained | 260 | 40 | 0 |
| mx | 1 | trained | 267 | 33 | 0 |
| pf | 1 | trained | 266 | 34 | 0 |
| bs_bt | 2 | trained | 261 | 39 | 0 |
| bs_fj | 2 | trained | 270 | 30 | 0 |
| bs_ka | 2 | trained | 274 | 26 | 0 |
| bs_mx | 2 | trained | 266 | 34 | 0 |
| bs_pf | 2 | trained | 273 | 27 | 0 |
| bt_fj | 2 | test | 0 | 0 | 300 |
| bt_ka | 2 | trained | 266 | 34 | 0 |
| bt_mx | 2 | trained | 271 | 29 | 0 |
| bt_pf | 2 | trained | 272 | 28 | 0 |
| fj_ka | 2 | test | 0 | 0 | 300 |
| fj_mx | 2 | trained | 265 | 35 | 0 |
| fj_pf | 2 | trained | 275 | 25 | 0 |
| ka_mx | 2 | trained | 270 | 30 | 0 |
| ka_pf | 2 | trained | 269 | 31 | 0 |
| mx_pf | 2 | trained | 267 | 33 | 0 |
| bs_bt_fj | 3 | trained | 273 | 27 | 0 |
| bs_bt_ka | 3 | test | 0 | 0 | 300 |
| bs_bt_mx | 3 | trained | 273 | 27 | 0 |
| bs_fj_mx | 3 | trained | 269 | 31 | 0 |
| bs_ka_pf | 3 | trained | 277 | 23 | 0 |
| bs_mx_pf | 3 | test | 0 | 0 | 300 |
| bt_fj_mx | 3 | trained | 266 | 34 | 0 |
| bt_fj_pf | 3 | test | 0 | 0 | 300 |
| bt_ka_mx | 3 | trained | 272 | 28 | 0 |
| fj_mx_pf | 3 | trained | 271 | 29 | 0 |
| bs_bt_fj_pf | 4 | trained | 265 | 35 | 0 |
| bs_bt_ka_pf | 4 | trained | 267 | 33 | 0 |
| bs_bt_mx_pf | 4 | test | 0 | 0 | 300 |
| bs_fj_ka_mx | 4 | trained | 267 | 33 | 0 |
| bs_fj_mx_pf | 4 | trained | 269 | 31 | 0 |
| bt_fj_mx_pf | 4 | trained | 277 | 23 | 0 |
| bt_ka_mx_pf | 4 | trained | 281 | 19 | 0 |
| fj_ka_mx_pf | 4 | test | 0 | 0 | 300 |
| bs_bt_fj_ka_mx_pf | 6 | test | 0 | 0 | 300 |

## Incidence in the trained pool

31 trained combinations. Diagonal: positive combinations per species; off-diagonal: combinations containing both.

| | bs | bt | fj | ka | mx | pf |
|---|---|---|---|---|---|---|
| bs | 14 | 5 | 6 | 4 | 5 | 5 |
| bt | 5 | 12 | 4 | 4 | 6 | 5 |
| fj | 6 | 4 | 12 | 1 | 7 | 5 |
| ka | 4 | 4 | 1 | 10 | 4 | 4 |
| mx | 5 | 6 | 7 | 4 | 15 | 5 |
| pf | 5 | 5 | 5 | 4 | 5 | 13 |

Species pairs where the first never appears without the second: none
Rank of the incidence matrix: 6 (K=6)
Pure cultures in the trained pool: ['bs', 'fj', 'ka', 'mx', 'pf']

## Development folds

| fold | dev combos | bs +/- | bt +/- | fj +/- | ka +/- | mx +/- | pf +/- | pure cultures in train pool |
|---|---|---|---|---|---|---|---|---|
| 0 | 6 | 3/3 | 4/2 | 3/3 | 3/3 | 5/1 | 1/5 | ['bs', 'fj', 'ka', 'mx', 'pf'] |
| 1 | 5 | 1/4 | 2/3 | 2/3 | 1/4 | 3/2 | 3/2 | ['bs', 'fj', 'ka', 'mx', 'pf'] |
| 2 | 5 | 2/3 | 2/3 | 2/3 | 2/3 | 3/2 | 4/1 | ['bs', 'fj', 'ka', 'mx', 'pf'] |
| 3 | 5 | 2/3 | 2/3 | 2/3 | 1/4 | 3/2 | 1/4 | ['bs', 'fj', 'ka', 'mx', 'pf'] |
| 4 | 5 | 5/0 | 2/3 | 2/3 | 2/3 | 0/5 | 3/2 | ['bs', 'fj', 'ka', 'mx', 'pf'] |

Pooled dev: 26 combinations, 26 distinct; per species positive/negative: bs 13/13, bt 12/14, fj 11/15, ka 9/17, mx 14/12, pf 12/14

Test: 9 combinations; positive/negative per species: bs 4/5, bt 6/3, fj 5/4, ka 4/5, mx 4/5, pf 5/4

## File names

Name patterns: {'N.jpg': 12000}

- bs: 300 files, index range 0..299, e.g. ['0.jpg', '1.jpg', '10.jpg', '100.jpg']
- fj: 300 files, index range 0..299, e.g. ['0.jpg', '1.jpg', '10.jpg', '100.jpg']
- ka: 300 files, index range 0..299, e.g. ['0.jpg', '1.jpg', '10.jpg', '100.jpg']

Frame/crop grouping is not encoded in the manifest; the dataset card says each video frame gives three crops. Verify on images (scripts/audit_images.py) before treating neighbouring indices as one frame.
