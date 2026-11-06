# H3 policies — qwen25vl7b — train-holdout (dev only)

train n = 480, eval n = 120 câu, 106 tài liệu. Metric chính thức: giá trị đúng (strict) **và** scale = scale gold.

## Ngân sách B = 2 lượt gọi VLM / câu (trung bình)

Tham số chọn trên train: {'lam': 10.0, 'thr': 1.0, 'extra': 'zoom', 'base_scale': 'direct'}

| policy | accuracy | chi phí TB |
|---|---|---|
| fixed2 | 62.5 | 2.00 |
| fixed2_ocr | 55.0 | 2.00 |
| confidence | 62.5 | 2.00 |
| errtype | 62.5 | 2.00 |
| learned | 75.0 | 2.00 |

**learned − baseline tốt nhất trên tập đánh giá (fixed2)** = +12.5 [+5.9, +19.2] *

## Ngân sách B = 3 lượt gọi VLM / câu (trung bình)

Tham số chọn trên train: {'lam': -0.020000000000000018, 'thr': 0.1, 'extra': 'zoom', 'base_scale': 'direct'}

| policy | accuracy | chi phí TB |
|---|---|---|
| fixed2 | 62.5 | 2.00 |
| fixed2_ocr | 55.0 | 2.00 |
| fixed3_zoom | 60.0 | 3.00 |
| confidence | 60.0 | 3.00 |
| errtype | 57.5 | 2.64 |
| learned | 70.0 | 3.00 |

**learned − baseline tốt nhất trên tập đánh giá (fixed2)** = +7.5 [+0.0, +15.3]

Recovery trên các câu Fixed-2 (scale từ direct) sai, theo loại lỗi E8 (số câu được policy trả lời đúng):

| loại | n | fixed2 | fixed2_ocr | fixed3_zoom | confidence | errtype | learned |
|---|---|---|---|---|---|---|---|
| scale | 21 | 0 | 13 | 0 | 0 | 9 | 9 |
| read | 10 | 0 | 0 | 0 | 0 | 0 | 1 |
| reason | 4 | 0 | 0 | 0 | 0 | 0 | 1 |
| span | 10 | 0 | 0 | 0 | 0 | 0 | 5 |
