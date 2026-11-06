# H3b — qwen25vl7b — train CV (dev)

Metric chính thức: giá trị đúng (strict) và scale = gold. Tất cả policy chỉ học trên train.

## B = 2

5-fold CV theo tài liệu trên train, n = 600

| policy | accuracy | chi phí TB |
|---|---|---|
| fixed2 | 55.7 | nan |
| fixed2_ocr | 51.7 | nan |
| confidence | 55.7 | nan |
| errtype | 55.7 | nan |
| H3-learned | 67.2 | nan |
| rule_table | 62.7 | nan |
| v2 | 69.3 | nan |
| oracle | 76.2 | nan |

**v2 − H3-learned** = +2.2 [-0.7, +5.2]

**v2 − rule_table** = +6.7 [+3.7, +9.8] *

## B = 3

5-fold CV theo tài liệu trên train, n = 600

| policy | accuracy | chi phí TB |
|---|---|---|
| fixed2 | 55.7 | nan |
| fixed2_ocr | 51.7 | nan |
| fixed3 | 55.3 | nan |
| confidence | 55.3 | nan |
| errtype | 53.2 | nan |
| H3-learned | 67.5 | nan |
| rule_table | 63.0 | nan |
| v2 | 69.7 | nan |
| oracle | 80.7 | nan |

**v2 − H3-learned** = +2.2 [-1.0, +5.3]

**v2 − rule_table** = +6.7 [+3.4, +10.1] *
