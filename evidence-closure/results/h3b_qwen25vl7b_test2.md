# H3b — qwen25vl7b — test2

Metric chính thức: giá trị đúng (strict) và scale = gold. Tất cả policy chỉ học trên train.

## B = 2

test2, n = 609 câu, 241 tài liệu; tham số H3: {'lam': 10.0, 'thr': 1.0, 'extra': 'zoom', 'base_scale': 'direct'}

| policy | accuracy | chi phí TB |
|---|---|---|
| fixed2 | 57.8 | 2.00 |
| fixed2_ocr | 57.3 | 2.00 |
| confidence | 57.8 | 2.00 |
| errtype | 57.8 | 2.00 |
| H3-learned | 67.0 | 2.00 |
| rule_table | 65.5 | 2.00 |
| v2 | 71.9 | 2.00 |
| oracle | 76.8 | nan |

**v2 − H3-learned** = +4.9 [+2.4, +7.5] *

**v2 − rule_table** = +6.4 [+3.8, +9.2] *

## B = 3

test2, n = 609 câu, 241 tài liệu; tham số H3: {'lam': 0.06, 'thr': 1.0, 'extra': 'zoom', 'base_scale': 'direct'}

| policy | accuracy | chi phí TB |
|---|---|---|
| fixed2 | 57.8 | 2.00 |
| fixed2_ocr | 57.3 | 2.00 |
| fixed3 | 57.5 | 3.00 |
| confidence | 57.5 | 3.00 |
| errtype | 57.8 | 2.00 |
| H3-learned | 68.3 | 2.17 |
| rule_table | 65.0 | 3.00 |
| v2 | 73.4 | 3.00 |
| oracle | 80.3 | nan |

**v2 − H3-learned** = +5.1 [+2.4, +7.6] *

**v2 − rule_table** = +8.4 [+5.7, +11.2] *

Recovery trên các câu Fixed-2 sai (số câu đúng):

| loại | n | fixed2 | fixed2_ocr | fixed3 | confidence | errtype | H3-learned | rule_table | v2 | oracle |
|---|---|---|---|---|---|---|---|---|---|---|
| scale | 140 | 0 | 84 | 0 | 0 | 0 | 47 | 57 | 90 | 102 |
| read | 54 | 0 | 0 | 5 | 5 | 0 | 2 | 0 | 1 | 5 |
| reason | 17 | 0 | 0 | 3 | 3 | 0 | 1 | 0 | 0 | 1 |
| span | 46 | 0 | 0 | 0 | 0 | 0 | 23 | 0 | 24 | 29 |
