# Reference anchoring probe (EXPLORATORY; phoebi4 mixtures, 10 videos)

| scorer | mean AUROC [95% CI over videos] | bs | fj | ka | pf |
|---|---|---|---|---|---|
| proto_cross | 0.589 [0.472, 0.691] | 0.69 | 0.40 | 0.75 | 0.52 |
| proto_inbatch | 0.764 [0.632, 0.894] | 0.69 | 0.73 | 0.93 | 0.70 |
| lin_cross | 0.643 [0.473, 0.759] | 0.69 | 0.43 | 0.95 | 0.51 |
| lin_inbatch | 0.685 [0.516, 0.803] | 0.66 | 0.48 | 0.99 | 0.61 |
| tvn_cross | 0.574 [0.431, 0.721] | 0.64 | 0.08 | 0.90 | 0.67 |

Mixture videos: [np.str_('p4_b_f'), np.str_('p4_b_f_k'), np.str_('p4_b_f_k_p'), np.str_('p4_b_k_p'), np.str_('p4_b_p'), np.str_('p4_f_b'), np.str_('p4_f_k'), np.str_('p4_f_k_p'), np.str_('p4_f_p'), np.str_('p4_k_p')]