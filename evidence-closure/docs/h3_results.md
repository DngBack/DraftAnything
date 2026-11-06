# H3: kết quả test (chạy một lần, 2026-11-05)

> **Đính chính 2026-11-06:** số trong file này dùng `value_correct`, dễ dãi hơn evaluator chính thức; chữ "metric chính thức" và "strict" ở đây là sai tên. Xem [official_metric_correction.md](official_metric_correction.md).

Protocol: [protocol_h3.md](protocol_h3.md), đã khóa trước khi có output. Chi tiết: `results/h3_policy_{qwen25vl7b,qwen3vl8b}_test.md`.

Thiết lập: policy học trên 600 câu train. Test gồm 600 câu (245 tài liệu) cho Qwen2.5-VL-7B và 300 câu (187 tài liệu) cho Qwen3-VL-8B. Chấm theo metric chính thức: giá trị đúng (strict) **và** scale đúng. CI 95 % lấy bằng bootstrap theo tài liệu.

## Tiêu chí 1: learned so với baseline tốt nhất ở cùng ngân sách

| model | B | baseline tốt nhất | learned (chi phí TB) | Δ [CI 95 %] |
|---|---|---|---|---|
| 7B | 2 | Fixed-2 55.5 | 64.2 (2.00) | **+8.7 [+5.0, +12.5]** |
| 7B | 3 | Fixed-2 55.5 | 64.8 (2.17) | **+9.3 [+5.4, +13.2]** |
| Qwen3 | 2 | Fixed-2 61.0 | 63.0 (2.00) | +2.0 [−1.4, +5.7] |
| Qwen3 | 3 | Fixed-3 verify 61.3 | 68.3 (2.63) | **+7.0 [+3.2, +11.0]** |

**Đạt**: có ít nhất một B đạt ý nghĩa thống kê ở cả hai model. Confidence-triggered và error-type không hơn Fixed-2.

## Tiêu chí 2: recovery trên các câu Fixed-2 sai (B = 3)

| loại lỗi | 7B: n / confidence / learned | Qwen3: n / confidence / learned |
|---|---|---|
| scale | 141 / 0 / **42** | 61 / 0 / **20** |
| read | 54 / 5 / 2 | 28 / 0 / 1 |
| reason | 17 / 5 / 3 | 16 / 1 / 4 |
| span | 55 / 1 / **31** | 12 / 0 / 1 |
| tổng | 267 / 11 / **78** | 117 / 1 / **26** |

**Đạt** về tổng số câu được cứu. Riêng lỗi read (đọc sai ô), confidence + zoom cứu nhiều hơn ở 7B: 5 câu so với 2.

## Tiêu chí 3: đặt cạnh số liệu đã công bố (TAT-DQA test, EM có tính scale)

| hệ thống | EM |
|---|---|
| MHST (fine-tune) | 41.5 |
| Doc2SoarGraph (fine-tune) | 59.2 |
| TAT-LLM 7B / 13B / 70B (fine-tune, dùng OCR text) | 71.3 / 72.2 / 76.5 |
| Con người | 84.1 |
| **Của mình**: Qwen2.5-VL-7B zero-shot + learned H3 | 64.8 |
| **Của mình**: Qwen3-VL-8B zero-shot + learned H3 | 68.3 |

Không so sánh trực tiếp được vì ba lý do. Thứ nhất, tập của mình là tập con đã lọc: một trang, fact khớp với OCR, có fact số trong bảng. Thứ hai, chấm theo kiểu official chứ không dùng script chính thức. Thứ ba, các hệ thống kia được fine-tune trên TAT-DQA. Nguồn: https://nextplusplus.github.io/TAT-DQA

## Diễn giải trung thực

- Phần lớn mức tăng đến từ **chọn nguồn scale** (direct hay OCR-closure, OCR không tốn lượt gọi) và **chọn nguồn giá trị** (dùng direct thay vì program cho câu span). Nó **không** đến từ việc gọi thêm hành động sửa lỗi. Zoom và verify gần như không giúp gì, đúng như E9 đã thấy.
- Vì vậy tên phù hợp là "**learned answer-source selection**" (chọn nguồn câu trả lời), chưa phải "value of information" theo nghĩa đầy đủ.
- Protocol không có baseline luật theo loại câu học trên train (ví dụ span → direct, arithmetic → program + OCR-scale). Theo quy định, tôi không thêm baseline sau khi đã chạy test. Đây là hạn chế chính: một phần mức tăng có thể chỉ cần một luật đơn giản là đạt được. Nếu làm tiếp, phải khóa baseline này trong một protocol mới và chạy trên một tập test mới.
