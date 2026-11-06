# H3c — qwen25vl7b — train CV (dev)

Mọi điều kiện chấm bằng EM chính thức.

## B = 2, n = 600, 527 tài liệu

| policy | EM |
|---|---|
| direct | 38.0 |
| fixed2 | 51.5 |
| H3-len | 63.3 |
| rule-len | 57.5 |
| v2-len | 64.0 |
| H3-em | 63.7 |
| rule-em | 57.7 |
| v2-em | 64.8 |

- v2-em − v2-len = +0.8 [-0.2, +1.8]
- H3-em − H3-len = +0.3 [-1.2, +1.9]
- v2-em − H3-em = +1.2 [-1.2, +3.7]
- v2-em − rule-em = +7.2 [+4.5, +10.1] *
- H3-em − rule-em = +6.0 [+2.7, +9.2] *
- v2-em − fixed2 = +13.3 [+9.7, +17.1] *
- H3-em − fixed2 = +12.2 [+8.8, +15.5] *

## B = 3, n = 600, 527 tài liệu

| policy | EM |
|---|---|
| direct | 38.0 |
| fixed2 | 51.5 |
| H3-len | 63.0 |
| rule-len | 57.7 |
| v2-len | 63.8 |
| H3-em | 64.5 |
| rule-em | 58.2 |
| v2-em | 64.8 |

- v2-em − v2-len = +1.0 [+0.2, +2.0] *
- H3-em − H3-len = +1.5 [-0.3, +3.3]
- v2-em − H3-em = +0.3 [-2.4, +3.1]
- v2-em − rule-em = +6.7 [+3.9, +9.7] *
- H3-em − rule-em = +6.3 [+3.1, +9.4] *
- v2-em − fixed2 = +13.3 [+9.5, +17.1] *
- H3-em − fixed2 = +13.0 [+9.4, +16.6] *
