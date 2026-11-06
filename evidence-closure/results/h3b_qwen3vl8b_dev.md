# H3b — qwen3vl8b — train CV (dev)

Metric chính thức: giá trị đúng (strict) và scale = gold. Tất cả policy chỉ học trên train.

## B = 2

5-fold CV theo tài liệu trên train, n = 600

| policy | accuracy | chi phí TB |
|---|---|---|
| fixed2 | 62.0 | nan |
| fixed2_ocr | 54.7 | nan |
| confidence | 62.0 | nan |
| errtype | 62.0 | nan |
| H3-learned | 66.3 | nan |
| rule_table | 63.0 | nan |
| v2 | 71.8 | nan |
| oracle | 76.5 | nan |

**v2 − H3-learned** = +5.5 [+2.5, +8.5] *

**v2 − rule_table** = +8.8 [+5.8, +11.9] *

## B = 3

5-fold CV theo tài liệu trên train, n = 600

| policy | accuracy | chi phí TB |
|---|---|---|
| fixed2 | 62.0 | nan |
| fixed2_ocr | 54.7 | nan |
| fixed3_verify | 62.2 | nan |
| confidence | 62.2 | nan |
| errtype | 61.0 | nan |
| H3-learned | 66.2 | nan |
| rule_table | 64.0 | nan |
| v2 | 72.3 | nan |
| oracle | 82.3 | nan |

**v2 − H3-learned** = +6.2 [+3.0, +9.3] *

**v2 − rule_table** = +8.3 [+4.5, +11.8] *
