# CoSolve oracle — income/spending trên Kleister-Charity dev-0

F1 (%), mean ± std qua 3 seed unary set4 (llm_greedy, oracle lặp lại theo seed). Tham số solver học trên val.

| hệ thống | money F1 | income | spending | precision |
|---|---|---|---|---|
| llm_greedy | 55.3 ± 0.0 | 56.4 ± 0.0 | 54.3 ± 0.0 | 55.5 ± 0.0 |
| indep | 49.7 ± 1.0 | 51.1 ± 0.9 | 48.3 ± 1.7 | 58.9 ± 1.6 |
| fixed | 54.3 ± 0.8 | 55.6 ± 1.3 | 53.1 ± 0.4 | 54.5 ± 0.8 |
| learned | 56.0 ± 1.2 | 56.9 ± 1.3 | 55.1 ± 1.1 | 65.8 ± 1.6 |
| evcond | 56.1 ± 1.2 | 56.9 ± 1.3 | 55.2 ± 1.1 | 65.8 ± 1.6 |
| oracle | 88.2 ± 0.0 | 88.0 ± 0.0 | 88.4 ± 0.0 | 100.0 ± 0.0 |

Paired bootstrap, Δ money F1 (pp), CI 95%:

| a | b | Δ | CI |
|---|---|---|---|
| learned | indep | +6.33 | [+4.48, +8.49] |
| evcond | learned | +0.02 | [+0.00, +0.06] |
| fixed | indep | +4.62 | [+2.23, +7.15] |
| indep | llm_greedy | -5.64 | [-8.13, -3.30] |
| learned | llm_greedy | +0.68 | [-1.81, +3.07] |

Lỗi của `learned` mà cặp dự đoán vẫn thỏa ràng buộc net (nhất quán nhưng sai): 90% trên 662 lỗi (3 seed).
