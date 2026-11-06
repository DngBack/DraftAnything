# H3c — qwen25vl7b — dev2

Mọi điều kiện chấm bằng EM chính thức.

## B = 2, n = 630, 239 tài liệu

| policy | EM |
|---|---|
| direct | 35.1 |
| fixed2 | 53.0 |
| H3-len | 63.2 |
| rule-len | 59.5 |
| v2-len | 67.0 |
| H3-em | 62.9 |
| rule-em | 59.5 |
| v2-em | 66.3 |

- v2-em − v2-len = -0.6 [-1.5, +0.2]
- H3-em − H3-len = -0.3 [-1.7, +1.1]
- v2-em − H3-em = +3.5 [+0.6, +6.3] *
- v2-em − rule-em = +6.8 [+3.9, +10.1] *
- H3-em − rule-em = +3.3 [-0.2, +7.2]
- v2-em − fixed2 = +13.3 [+9.7, +17.3] *
- H3-em − fixed2 = +9.8 [+6.3, +13.6] *

## B = 3, n = 630, 239 tài liệu

| policy | EM |
|---|---|
| direct | 35.1 |
| fixed2 | 53.0 |
| H3-len | 64.0 |
| rule-len | 59.0 |
| v2-len | 68.3 |
| H3-em | 64.1 |
| rule-em | 59.0 |
| v2-em | 67.8 |

- v2-em − v2-len = -0.5 [-1.3, +0.3]
- H3-em − H3-len = +0.2 [-1.7, +2.0]
- v2-em − H3-em = +3.7 [+1.1, +6.5] *
- v2-em − rule-em = +8.7 [+5.6, +12.2] *
- H3-em − rule-em = +5.1 [+2.1, +8.2] *
- v2-em − fixed2 = +14.8 [+10.8, +18.8] *
- H3-em − fixed2 = +11.1 [+7.6, +14.7] *
