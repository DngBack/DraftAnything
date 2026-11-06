# H3 policies — qwen25vl7b — test

train n = 600, eval n = 600 câu, 245 tài liệu. Metric chính thức: giá trị đúng (strict) **và** scale = scale gold.

## Ngân sách B = 2 lượt gọi VLM / câu (trung bình)

Tham số chọn trên train: {'lam': 10.0, 'thr': 1.0, 'extra': 'zoom', 'base_scale': 'direct'}

| policy | accuracy | chi phí TB |
|---|---|---|
| fixed2 | 55.5 | 2.00 |
| fixed2_ocr | 52.0 | 2.00 |
| confidence | 55.5 | 2.00 |
| errtype | 55.5 | 2.00 |
| learned | 64.2 | 2.00 |

**learned − baseline tốt nhất trên tập đánh giá (fixed2)** = +8.7 [+5.0, +12.5] *

## Ngân sách B = 3 lượt gọi VLM / câu (trung bình)

Tham số chọn trên train: {'lam': 0.06, 'thr': 1.0, 'extra': 'zoom', 'base_scale': 'direct'}

| policy | accuracy | chi phí TB |
|---|---|---|
| fixed2 | 55.5 | 2.00 |
| fixed2_ocr | 52.0 | 2.00 |
| fixed3_zoom | 54.5 | 3.00 |
| confidence | 54.5 | 3.00 |
| errtype | 55.5 | 2.00 |
| learned | 64.8 | 2.17 |

**learned − baseline tốt nhất trên tập đánh giá (fixed2)** = +9.3 [+5.4, +13.2] *

Recovery trên các câu Fixed-2 (scale từ direct) sai, theo loại lỗi E8 (số câu được policy trả lời đúng):

| loại | n | fixed2 | fixed2_ocr | fixed3_zoom | confidence | errtype | learned |
|---|---|---|---|---|---|---|---|
| scale | 141 | 0 | 79 | 0 | 0 | 0 | 42 |
| read | 54 | 0 | 0 | 5 | 5 | 0 | 2 |
| reason | 17 | 0 | 0 | 5 | 5 | 0 | 3 |
| span | 55 | 0 | 0 | 1 | 1 | 0 | 31 |
