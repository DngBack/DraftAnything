# Chấm lại bằng luật EM của evaluator TAT-QA chính thức (phân tích sau, không đổi policy đã khóa)

## qwen25vl7b, test2, n = 609; kiểm tra: gold tự chấm = 100.0 %

| policy | B | thước đo cũ (dễ dãi) | **EM chính thức** |
|---|---|---|---|
| direct | 1 | 41.4 | 37.4 |
| fixed2 | 2 | 57.8 | 56.5 |
| rule_table | 2 | 65.5 | 63.5 |
| H3-learned | 2 | 67.0 | 65.8 |
| v2 | 2 | 71.9 | 70.3 |
| rule_table | 3 | 65.0 | 62.9 |
| H3-learned | 3 | 68.3 | 67.2 |
| v2 | 3 | 73.4 | 70.9 |

- B = 2: v2 − fixed2 = +13.8 [+10.4, +17.5] *
- B = 2: v2 − rule_table = +6.7 [+4.5, +9.0] *
- B = 2: v2 − H3-learned = +4.4 [+2.0, +6.8] *
- B = 3: v2 − fixed2 = +14.4 [+11.1, +18.3] *
- B = 3: v2 − rule_table = +8.0 [+5.5, +10.7] *
- B = 3: v2 − H3-learned = +3.8 [+1.4, +6.0] *

- arithmetic (n=323): fixed2 64.4, v2 B2 72.8, v2 B3 73.4
- span (n=201): fixed2 49.8, v2 B2 65.7, v2 B3 67.7
- multi-span (n=72): fixed2 36.1, v2 B2 70.8, v2 B3 68.1
- count (n=13): fixed2 76.9, v2 B2 76.9, v2 B3 76.9

## qwen25vl7b, test + test2, n = 1209; kiểm tra: gold tự chấm = 100.0 %

| policy | B | thước đo cũ (dễ dãi) | **EM chính thức** |
|---|---|---|---|
| direct | 1 | 39.5 | 36.8 |
| fixed2 | 2 | 56.7 | 56.2 |
| rule_table | 2 | 62.6 | 62.0 |
| H3-learned | 2 | 65.6 | 65.2 |
| v2 | 2 | 70.6 | 69.3 |
| rule_table | 3 | 62.4 | 61.6 |
| H3-learned | 3 | 66.6 | 66.3 |
| v2 | 3 | 72.2 | 70.4 |

- B = 2: v2 − fixed2 = +13.1 [+10.0, +16.3] *
- B = 2: v2 − rule_table = +7.4 [+5.4, +9.4] *
- B = 2: v2 − H3-learned = +4.1 [+2.2, +6.2] *
- B = 3: v2 − fixed2 = +14.1 [+11.0, +17.4] *
- B = 3: v2 − rule_table = +8.8 [+6.7, +10.9] *
- B = 3: v2 − H3-learned = +4.1 [+2.3, +6.1] *

- arithmetic (n=642): fixed2 65.3, v2 B2 71.7, v2 B3 73.2
- span (n=400): fixed2 49.2, v2 B2 64.2, v2 B3 65.0
- multi-span (n=142): fixed2 31.7, v2 B2 71.8, v2 B3 71.8
- count (n=25): fixed2 76.0, v2 B2 76.0, v2 B3 76.0

## qwen3vl8b, test2, n = 609; kiểm tra: gold tự chấm = 100.0 %

| policy | B | thước đo cũ (dễ dãi) | **EM chính thức** |
|---|---|---|---|
| direct | 1 | 42.2 | 36.1 |
| fixed2 | 2 | 61.6 | 59.3 |
| rule_table | 2 | 64.9 | 62.1 |
| H3-learned | 2 | 64.9 | 61.9 |
| v2 | 2 | 70.9 | 65.0 |
| rule_table | 3 | 65.4 | 61.9 |
| H3-learned | 3 | 68.5 | 66.5 |
| v2 | 3 | 72.4 | 66.2 |

- B = 2: v2 − fixed2 = +5.7 [+3.3, +8.2] *
- B = 2: v2 − rule_table = +3.0 [+0.9, +5.1] *
- B = 2: v2 − H3-learned = +3.1 [+1.0, +5.3] *
- B = 3: v2 − fixed2 = +6.9 [+4.1, +9.8] *
- B = 3: v2 − rule_table = +4.3 [+1.1, +7.2] *
- B = 3: v2 − H3-learned = -0.3 [-2.4, +1.6]

- arithmetic (n=323): fixed2 65.0, v2 B2 69.0, v2 B3 68.7
- span (n=201): fixed2 54.7, v2 B2 59.2, v2 B3 62.7
- multi-span (n=72): fixed2 51.4, v2 B2 66.7, v2 B3 68.1
- count (n=13): fixed2 30.8, v2 B2 46.2, v2 B3 46.2

## qwen3vl8b, test + test2, n = 909; kiểm tra: gold tự chấm = 100.0 %

| policy | B | thước đo cũ (dễ dãi) | **EM chính thức** |
|---|---|---|---|
| direct | 1 | 42.5 | 37.8 |
| fixed2 | 2 | 61.4 | 60.2 |
| rule_table | 2 | 64.6 | 62.3 |
| H3-learned | 2 | 64.2 | 62.2 |
| v2 | 2 | 69.6 | 64.7 |
| rule_table | 3 | 64.2 | 61.6 |
| H3-learned | 3 | 68.4 | 66.7 |
| v2 | 3 | 71.6 | 66.7 |

- B = 2: v2 − fixed2 = +4.5 [+2.3, +7.0] *
- B = 2: v2 − rule_table = +2.4 [+0.6, +4.5] *
- B = 2: v2 − H3-learned = +2.5 [+0.6, +4.5] *
- B = 3: v2 − fixed2 = +6.5 [+4.0, +9.0] *
- B = 3: v2 − rule_table = +5.1 [+2.7, +7.3] *
- B = 3: v2 − H3-learned = +0.0 [-1.8, +1.7]

- arithmetic (n=484): fixed2 66.1, v2 B2 68.4, v2 B3 70.2
- span (n=300): fixed2 55.3, v2 B2 59.0, v2 B3 61.3
- multi-span (n=108): fixed2 51.9, v2 B2 67.6, v2 B3 69.4
- count (n=17): fixed2 29.4, v2 B2 41.2, v2 B3 41.2
