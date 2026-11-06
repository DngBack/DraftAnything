# E9 — hành động sửa lỗi có mục tiêu, qwen25vl7b

n = 600 câu dev. Base = route thực tế của E8 (biểu thức tính được thì dùng Python, không thì dùng `direct`).

## Scale (trên các câu mà giá trị đã đúng)

| nguồn scale | đúng scale gold (kể cả rỗng) | trên câu có scale |
|---|---|---|
| từ câu trả lời `direct` | 70.3 | 66.2 |
| hỏi lại VLM (1 lượt) | 53.4 | 70.2 |
| OCR + closure quan hệ (0 lượt) | 64.7 | 82.3 |

## Chấm chính thức (giá trị + scale)

| pipeline | lượt gọi VLM | accuracy |
|---|---|---|
| direct | 1 | 39.5 |
| program→Py + scale từ direct | 2 | 54.8 |
| program→Py + scale VLM | 3 | 41.7 |
| **program→Py + scale OCR-closure** | 2 | 50.5 |
| program→Py + scale OCR-closure + verify | 3 | 50.5 |
| program→Py + scale OCR-closure + zoom | 3 | 50.2 |
| oracle: tốt nhất trong {base, zoom, verify} + scale OCR | 4 | 52.3 |
| oracle: giá trị đúng + scale gold | 2 | 78.0 |

## Hành động trên giá trị: cứu / làm hỏng theo loại lỗi của base

| hành động | cứu: read | cứu: reason | cứu: span | làm hỏng câu đúng | thay đổi ròng |
|---|---|---|---|---|---|
| zoom | 7/46 | 9/29 | 5/57 | 18/468 | +0.5 |
| verify | 1/46 | 2/29 | 0/57 | 4/468 | -0.2 |