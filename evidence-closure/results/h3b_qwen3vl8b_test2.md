# H3b — qwen3vl8b — test2

Metric chính thức: giá trị đúng (strict) và scale = gold. Tất cả policy chỉ học trên train.

## B = 2

test2, n = 609 câu, 241 tài liệu; tham số H3: {'lam': 10.0, 'thr': 1.0, 'extra': 'verify', 'base_scale': 'direct'}

| policy | accuracy | chi phí TB |
|---|---|---|
| fixed2 | 61.6 | 2.00 |
| fixed2_ocr | 58.0 | 2.00 |
| confidence | 61.6 | 2.00 |
| errtype | 61.6 | 2.00 |
| H3-learned | 64.9 | 2.00 |
| rule_table | 64.9 | 2.00 |
| v2 | 70.9 | 2.00 |
| oracle | 76.5 | nan |

**v2 − H3-learned** = +6.1 [+3.3, +8.8] *

**v2 − rule_table** = +6.1 [+3.3, +8.9] *

## B = 3

test2, n = 609 câu, 241 tài liệu; tham số H3: {'lam': 0.0, 'thr': 0.15000000000000002, 'extra': 'verify', 'base_scale': 'direct'}

| policy | accuracy | chi phí TB |
|---|---|---|
| fixed2 | 61.6 | 2.00 |
| fixed2_ocr | 58.0 | 2.00 |
| fixed3 | 61.2 | 3.00 |
| confidence | 61.2 | 2.99 |
| errtype | 60.9 | 2.61 |
| H3-learned | 68.5 | 2.56 |
| rule_table | 65.4 | 3.00 |
| v2 | 72.4 | 3.00 |
| oracle | 80.5 | nan |

**v2 − H3-learned** = +3.9 [+1.2, +6.7] *

**v2 − rule_table** = +7.1 [+3.7, +10.4] *

Recovery trên các câu Fixed-2 sai (số câu đúng):

| loại | n | fixed2 | fixed2_ocr | fixed3 | confidence | errtype | H3-learned | rule_table | v2 | oracle |
|---|---|---|---|---|---|---|---|---|---|---|
| scale | 126 | 0 | 65 | 0 | 0 | 51 | 40 | 53 | 72 | 89 |
| read | 50 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 5 | 10 |
| reason | 34 | 0 | 0 | 0 | 0 | 1 | 4 | 0 | 1 | 8 |
| span | 24 | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 10 | 8 |
