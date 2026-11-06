# H3 policies — qwen3vl8b — train-holdout (dev only)

train n = 480, eval n = 120 câu, 106 tài liệu. Metric chính thức: giá trị đúng (strict) **và** scale = scale gold.

## Ngân sách B = 2 lượt gọi VLM / câu (trung bình)

Tham số chọn trên train: {'lam': 10.0, 'thr': 1.0, 'extra': 'verify', 'base_scale': 'direct'}

| policy | accuracy | chi phí TB |
|---|---|---|
| fixed2 | 61.7 | 2.00 |
| fixed2_ocr | 55.8 | 2.00 |
| confidence | 61.7 | 2.01 |
| errtype | 61.7 | 2.00 |
| learned | 63.3 | 2.00 |

**learned − baseline tốt nhất trên tập đánh giá (fixed2)** = +1.7 [-3.2, +6.6]

## Ngân sách B = 3 lượt gọi VLM / câu (trung bình)

Tham số chọn trên train: {'lam': 0.2, 'thr': 0.15000000000000002, 'extra': 'verify', 'base_scale': 'direct'}

| policy | accuracy | chi phí TB |
|---|---|---|
| fixed2 | 61.7 | 2.00 |
| fixed2_ocr | 55.8 | 2.00 |
| fixed3_verify | 61.7 | 3.00 |
| confidence | 61.7 | 3.00 |
| errtype | 61.7 | 2.56 |
| learned | 63.3 | 2.03 |

**learned − baseline tốt nhất trên tập đánh giá (fixed2)** = +1.7 [-3.2, +6.5]

Recovery trên các câu Fixed-2 (scale từ direct) sai, theo loại lỗi E8 (số câu được policy trả lời đúng):

| loại | n | fixed2 | fixed2_ocr | fixed3_verify | confidence | errtype | learned |
|---|---|---|---|---|---|---|---|
| scale | 23 | 0 | 10 | 0 | 0 | 11 | 4 |
| read | 8 | 0 | 0 | 0 | 0 | 0 | 0 |
| reason | 8 | 0 | 0 | 0 | 0 | 0 | 0 |
| span | 7 | 0 | 0 | 0 | 0 | 0 | 1 |
