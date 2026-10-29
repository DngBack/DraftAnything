# Data audit

Manifests: `/home/jovyan/bachdx2/datasets/phoebi/repo/data`
Species: ['bs', 'bt', 'fj', 'ka', 'mx', 'pf']

## Leakage checks

- test combos ∩ train combos: none
- test combos ∩ val combos: none
- train combos == val combos: True
- duplicate image paths across parts: 0
- held-out combos (manifest): ['bs_bt_ka_fj', 'bs_bt_mx_ka_fj_pf', 'bs_ka_pf', 'bs_mx_fj', 'bs_mx_fj_pf', 'bs_pf', 'bt', 'ka_fj', 'mx_ka_fj']

- fold 0: touches test combos: none; dev∩train: none
- fold 1: touches test combos: none; dev∩train: none
- fold 2: touches test combos: none; dev∩train: none
- fold 3: touches test combos: none; dev∩train: none
- fold 4: touches test combos: none; dev∩train: none

## Combinations

| combo | order | role | train | val | test |
|---|---|---|---|---|---|
| bs | 1 | trained | 2735 | 265 | 0 |
| bt | 1 | test | 0 | 0 | 3000 |
| fj | 1 | trained | 2695 | 305 | 0 |
| ka | 1 | trained | 2714 | 286 | 0 |
| mx | 1 | trained | 2699 | 301 | 0 |
| pf | 1 | trained | 2729 | 271 | 0 |
| bs_fj | 2 | trained | 2701 | 299 | 0 |
| bs_ka | 2 | trained | 2690 | 310 | 0 |
| bs_mx | 2 | trained | 2685 | 315 | 0 |
| bs_pf | 2 | test | 0 | 0 | 3000 |
| bt_ka | 2 | trained | 2712 | 288 | 0 |
| bt_mx | 2 | trained | 2704 | 296 | 0 |
| bt_pf | 2 | trained | 2705 | 295 | 0 |
| fj_pf | 2 | trained | 2682 | 318 | 0 |
| ka_fj | 2 | test | 0 | 0 | 3000 |
| mx_fj | 2 | trained | 2709 | 291 | 0 |
| mx_ka | 2 | trained | 2691 | 309 | 0 |
| mx_pf | 2 | trained | 2706 | 294 | 0 |
| bs_bt_mx | 3 | trained | 2702 | 298 | 0 |
| bs_fj_pf | 3 | trained | 2671 | 329 | 0 |
| bs_ka_fj | 3 | trained | 2697 | 303 | 0 |
| bs_ka_pf | 3 | test | 0 | 0 | 3000 |
| bs_mx_fj | 3 | test | 0 | 0 | 3000 |
| bs_mx_ka | 3 | trained | 2690 | 310 | 0 |
| bs_mx_pf | 3 | trained | 2707 | 293 | 0 |
| bt_fj_pf | 3 | trained | 2667 | 333 | 0 |
| bt_ka_fj | 3 | trained | 2700 | 300 | 0 |
| bt_mx_ka | 3 | trained | 2711 | 289 | 0 |
| bt_mx_pf | 3 | trained | 2689 | 311 | 0 |
| ka_fj_pf | 3 | trained | 2712 | 288 | 0 |
| mx_fj_pf | 3 | trained | 2694 | 306 | 0 |
| mx_ka_fj | 3 | test | 0 | 0 | 3000 |
| mx_ka_pf | 3 | trained | 2713 | 287 | 0 |
| bs_bt_ka_fj | 4 | test | 0 | 0 | 3000 |
| bs_ka_fj_pf | 4 | trained | 2669 | 331 | 0 |
| bs_mx_fj_pf | 4 | test | 0 | 0 | 3000 |
| bs_mx_ka_pf | 4 | trained | 2705 | 295 | 0 |
| bt_ka_fj_pf | 4 | trained | 2698 | 302 | 0 |
| bt_mx_ka_fj | 4 | trained | 2718 | 282 | 0 |
| bs_bt_mx_ka_fj_pf | 6 | test | 0 | 0 | 3000 |

## Incidence in the trained pool

31 trained combinations. Diagonal: positive combinations per species; off-diagonal: combinations containing both.

| | bs | bt | fj | ka | mx | pf |
|---|---|---|---|---|---|---|
| bs | 11 | 1 | 4 | 5 | 5 | 4 |
| bt | 1 | 10 | 4 | 5 | 5 | 4 |
| fj | 4 | 4 | 13 | 6 | 3 | 7 |
| ka | 5 | 5 | 6 | 14 | 6 | 5 |
| mx | 5 | 5 | 3 | 6 | 15 | 6 |
| pf | 4 | 4 | 7 | 5 | 6 | 14 |

Species pairs where the first never appears without the second: none
Rank of the incidence matrix: 6 (K=6)
Pure cultures in the trained pool: ['bs', 'fj', 'ka', 'mx', 'pf']

## Development folds

| fold | dev combos | bs +/- | bt +/- | fj +/- | ka +/- | mx +/- | pf +/- | pure cultures in train pool |
|---|---|---|---|---|---|---|---|---|
| 0 | 6 | 1/5 | 3/3 | 4/2 | 3/3 | 3/3 | 3/3 | ['bs', 'fj', 'ka', 'mx', 'pf'] |
| 1 | 5 | 1/4 | 3/2 | 1/4 | 3/2 | 3/2 | 2/3 | ['bs', 'fj', 'ka', 'mx', 'pf'] |
| 2 | 5 | 3/2 | 1/4 | 4/1 | 2/3 | 1/4 | 3/2 | ['bs', 'fj', 'ka', 'mx', 'pf'] |
| 3 | 5 | 2/3 | 3/2 | 2/3 | 2/3 | 3/2 | 2/3 | ['bs', 'fj', 'ka', 'mx', 'pf'] |
| 4 | 5 | 3/2 | 0/5 | 1/4 | 3/2 | 4/1 | 3/2 | ['bs', 'fj', 'ka', 'mx', 'pf'] |

Pooled dev: 26 combinations, 26 distinct; per species positive/negative: bs 10/16, bt 10/16, fj 12/14, ka 13/13, mx 14/12, pf 13/13

Test: 9 combinations; positive/negative per species: bs 6/3, bt 3/6, fj 6/3, ka 5/4, mx 4/5, pf 4/5

## File names

Name patterns: {'N.jpg': 120000}

- bs_ka_fj_pf: 3000 files, index range 1..3000, e.g. ['00001.jpg', '00002.jpg', '00003.jpg', '00004.jpg']
- bs_fj: 3000 files, index range 1..3000, e.g. ['00001.jpg', '00002.jpg', '00003.jpg', '00004.jpg']
- bs: 3000 files, index range 1..3000, e.g. ['00001.jpg', '00002.jpg', '00003.jpg', '00004.jpg']

Frame/crop grouping is not encoded in the manifest; the dataset card says each video frame gives three crops. Verify on images (scripts/audit_images.py) before treating neighbouring indices as one frame.
