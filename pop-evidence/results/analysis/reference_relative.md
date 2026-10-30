# Reference-relative representation probe (EXPLORATORY; train phoebi6, test phoebi4 mixtures)

| head on | mean AUROC [95% CI over 10 videos] | bs | fj | ka | pf |
|---|---|---|---|---|---|
| raw mean embedding (mean_lin, 4 species) | 0.619 [0.424, 0.761] | 0.75 | 0.35 | 0.69 | 0.68 |
| similarity to OTHER-batch references (phoebi6 protos at test) | 0.551 [0.442, 0.665] | 0.73 | 0.45 | 0.49 | 0.53 |
| similarity to SAME-batch references (reference-relative) | 0.529 [0.360, 0.670] | 0.46 | 0.28 | 0.80 | 0.58 |
| same-batch references + reference-standardised features | 0.552 [0.389, 0.689] | 0.43 | 0.30 | 0.84 | 0.62 |
| (closed form) nearest same-batch prototype, fraction of tiles | 0.702 | 0.64 | 0.66 | 0.99 | 0.52 |