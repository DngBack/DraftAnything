# H3 policies — qwen3vl8b — test

train n = 600, eval n = 300 câu, 187 tài liệu. Metric chính thức: giá trị đúng (strict) **và** scale = scale gold.

## Ngân sách B = 2 lượt gọi VLM / câu (trung bình)

Tham số chọn trên train: {'lam': 10.0, 'thr': 1.0, 'extra': 'verify', 'base_scale': 'direct'}

| policy | accuracy | chi phí TB |
|---|---|---|
| fixed2 | 61.0 | 2.00 |
| fixed2_ocr | 54.3 | 2.00 |
| confidence | 61.0 | 2.00 |
| errtype | 61.0 | 2.00 |
| learned | 63.0 | 2.00 |

**learned − baseline tốt nhất trên tập đánh giá (fixed2)** = +2.0 [-1.4, +5.7]

## Ngân sách B = 3 lượt gọi VLM / câu (trung bình)

Tham số chọn trên train: {'lam': 0.0, 'thr': 0.15000000000000002, 'extra': 'verify', 'base_scale': 'direct'}

| policy | accuracy | chi phí TB |
|---|---|---|
| fixed2 | 61.0 | 2.00 |
| fixed2_ocr | 54.3 | 2.00 |
| fixed3_verify | 61.3 | 3.00 |
| confidence | 61.3 | 2.99 |
| errtype | 59.0 | 2.64 |
| learned | 68.3 | 2.63 |

**learned − baseline tốt nhất trên tập đánh giá (fixed3_verify)** = +7.0 [+3.2, +11.0] *

Recovery trên các câu Fixed-2 (scale từ direct) sai, theo loại lỗi E8 (số câu được policy trả lời đúng):

| loại | n | fixed2 | fixed2_ocr | fixed3_verify | confidence | errtype | learned |
|---|---|---|---|---|---|---|---|
| scale | 61 | 0 | 30 | 0 | 0 | 22 | 20 |
| read | 28 | 0 | 0 | 0 | 0 | 0 | 1 |
| reason | 16 | 0 | 0 | 1 | 1 | 0 | 4 |
| span | 12 | 0 | 0 | 0 | 0 | 0 | 1 |
