# EP-MIL feasibility (EXPLORATORY; train phoebi6, test phoebi4 mixtures, 10 videos)

steps 3000, R 8, batch 32, lambda_tile 1.0, shift sb 0.5 sm 0.3, seeds [0, 1, 2]

| model / references | mean AUROC [95% CI] | bs | fj | ka | pf |
|---|---|---|---|---|---|
| identity g (closed form), same-batch refs | 0.764 [0.643, 0.888] | 0.69 | 0.73 | 0.93 | 0.70 |
| identity g (closed form), other-batch refs | 0.591 [0.471, 0.690] | 0.68 | 0.40 | 0.76 | 0.52 |
| EP-MIL, episode shift, same-batch refs | 0.634 [0.526, 0.742] | 0.70 | 0.19 | 0.96 | 0.67 |
| EP-MIL, episode shift, other-batch refs | 0.621 [0.487, 0.733] | 0.60 | 0.48 | 0.98 | 0.42 |
| EP-MIL, no shift, same-batch refs | 0.640 [0.507, 0.755] | 0.75 | 0.34 | 0.95 | 0.52 |
| EP-MIL, no shift, other-batch refs | 0.659 [0.535, 0.748] | 0.68 | 0.63 | 0.89 | 0.43 |