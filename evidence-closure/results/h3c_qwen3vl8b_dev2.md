# H3c — qwen3vl8b — dev2

Mọi điều kiện chấm bằng EM chính thức.

## B = 2, n = 630, 239 tài liệu

| policy | EM |
|---|---|
| direct | 35.7 |
| fixed2 | 60.0 |
| H3-len | 64.0 |
| rule-len | 64.0 |
| v2-len | 65.4 |
| H3-em | 64.1 |
| rule-em | 64.1 |
| v2-em | 65.7 |

- v2-em − v2-len = +0.3 [-0.5, +1.2]
- H3-em − H3-len = +0.2 [-0.8, +1.2]
- v2-em − H3-em = +1.6 [-0.9, +4.0]
- v2-em − rule-em = +1.6 [-0.5, +3.6]
- H3-em − rule-em = +0.0 [-2.2, +2.2]
- v2-em − fixed2 = +5.7 [+2.8, +8.6] *
- H3-em − fixed2 = +4.1 [+1.8, +6.3] *

## B = 3, n = 630, 239 tài liệu

| policy | EM |
|---|---|
| direct | 35.7 |
| fixed2 | 60.0 |
| H3-len | 66.0 |
| rule-len | 63.2 |
| v2-len | 66.7 |
| H3-em | 66.0 |
| rule-em | 64.4 |
| v2-em | 67.0 |

- v2-em − v2-len = +0.3 [-0.5, +1.2]
- H3-em − H3-len = +0.0 [-1.6, +1.7]
- v2-em − H3-em = +1.0 [-1.6, +3.4]
- v2-em − rule-em = +2.5 [+0.3, +4.9] *
- H3-em − rule-em = +1.6 [-1.1, +4.1]
- v2-em − fixed2 = +7.0 [+3.8, +10.2] *
- H3-em − fixed2 = +6.0 [+2.9, +9.3] *
