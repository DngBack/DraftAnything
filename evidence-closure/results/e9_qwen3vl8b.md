# E9 — hành động sửa lỗi có mục tiêu, qwen3vl8b

n = 300 câu dev. Base = route thực tế của E8 (biểu thức tính được thì dùng Python, không thì dùng `direct`).

## Scale (trên các câu mà giá trị đã đúng)

| nguồn scale | đúng scale gold (kể cả rỗng) | trên câu có scale |
|---|---|---|
| từ câu trả lời `direct` | 73.3 | 70.1 |
| hỏi lại VLM (1 lượt) | 65.1 | 79.3 |
| OCR + closure quan hệ (0 lượt) | 67.8 | 83.5 |

## Chấm chính thức (giá trị + scale)

| pipeline | lượt gọi VLM | accuracy |
|---|---|---|
| direct | 1 | 42.7 |
| program→Py + scale từ direct | 2 | 62.3 |
| program→Py + scale VLM | 3 | 55.3 |
| **program→Py + scale OCR-closure** | 2 | 57.7 |
| program→Py + scale OCR-closure + verify | 3 | 57.3 |
| program→Py + scale OCR-closure + zoom | 3 | 57.3 |
| oracle: tốt nhất trong {base, zoom, verify} + scale OCR | 4 | 58.7 |
| oracle: giá trị đúng + scale gold | 2 | 85.0 |

## Hành động trên giá trị: cứu / làm hỏng theo loại lỗi của base

| hành động | cứu: read | cứu: reason | cứu: span | làm hỏng câu đúng | thay đổi ròng |
|---|---|---|---|---|---|
| zoom | 1/13 | 2/20 | 3/12 | 11/255 | -1.7 |
| verify | 0/13 | 0/20 | 0/12 | 2/255 | -0.7 |