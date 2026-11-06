# H3c — qwen3vl8b — train CV (dev)

Mọi điều kiện chấm bằng EM chính thức.

## B = 2, n = 600, 527 tài liệu

| policy | EM |
|---|---|
| direct | 39.5 |
| fixed2 | 57.2 |
| H3-len | 62.5 |
| rule-len | 59.0 |
| v2-len | 64.0 |
| H3-em | 63.0 |
| rule-em | 60.2 |
| v2-em | 64.5 |

- v2-em − v2-len = +0.5 [-0.5, +1.7]
- H3-em − H3-len = +0.5 [-0.7, +1.8]
- v2-em − H3-em = +1.5 [-1.0, +3.8]
- v2-em − rule-em = +4.3 [+2.1, +6.7] *
- H3-em − rule-em = +2.8 [+0.5, +5.3] *
- v2-em − fixed2 = +7.3 [+4.2, +10.2] *
- H3-em − fixed2 = +5.8 [+3.6, +8.0] *

## B = 3, n = 600, 527 tài liệu

| policy | EM |
|---|---|
| direct | 39.5 |
| fixed2 | 57.2 |
| H3-len | 62.8 |
| rule-len | 59.5 |
| v2-len | 64.3 |
| H3-em | 62.8 |
| rule-em | 61.3 |
| v2-em | 64.8 |

- v2-em − v2-len = +0.5 [-0.5, +1.7]
- H3-em − H3-len = +0.0 [-1.5, +1.5]
- v2-em − H3-em = +2.0 [-0.5, +4.6]
- v2-em − rule-em = +3.5 [+0.8, +6.4] *
- H3-em − rule-em = +1.5 [-1.0, +4.1]
- v2-em − fixed2 = +7.7 [+4.3, +10.9] *
- H3-em − fixed2 = +5.7 [+3.0, +8.2] *
