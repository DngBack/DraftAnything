# H3b: kết quả test2 (chạy một lần, 2026-11-05)

> **Đính chính 2026-11-06:** số trong file này dùng `value_correct`, dễ dãi hơn evaluator chính thức; chữ "metric chính thức" và "strict" ở đây là sai tên. Xem [official_metric_correction.md](official_metric_correction.md).

Protocol: [protocol_h3b.md](protocol_h3b.md), khóa trước khi đánh giá. Chi tiết: `results/h3b_{qwen25vl7b,qwen3vl8b}_test2.md`. Test2 gồm 609 câu TAT-DQA test chưa dùng, trên 241 tài liệu. Chấm theo metric chính thức (giá trị strict và scale), CI 95 % lấy bằng bootstrap theo tài liệu.

## Accuracy

| policy | 7B B2 | 7B B3 | Qwen3 B2 | Qwen3 B3 |
|---|---|---|---|---|
| Fixed-2 | 57.8 | 57.8 | 61.6 | 61.6 |
| confidence / errtype / fixed3 (tốt nhất) | 57.8 | 57.8 | 61.6 | 61.2 |
| rule_table (luật tra bảng) | 65.5 | 65.0 | 64.9 | 65.4 |
| H3-learned | 67.0 | 68.3 | 64.9 | 68.5 |
| **v2** | **71.9** | **73.4** | **70.9** | **72.4** |
| oracle chọn nguồn | 76.8 | 80.3 | 76.5 | 80.5 |

## Tiêu chí

| tiêu chí | 7B B2 | 7B B3 | Qwen3 B2 | Qwen3 B3 | kết luận |
|---|---|---|---|---|---|
| 1. v2 − rule_table | +6.4 [+3.8, +9.2] | +8.4 [+5.7, +11.2] | +6.1 [+3.3, +8.9] | +7.1 [+3.7, +10.4] | **đạt** |
| 2. v2 − baseline tốt nhất (H3-learned) | +4.9 [+2.4, +7.5] | +5.1 [+2.4, +7.6] | +6.1 [+3.3, +8.8] | +3.9 [+1.2, +6.7] | **đạt** |
| 3. H3-learned − Fixed-2 (tái lập H3) | +9.2 [+6.1, +12.6] | +10.5 [+7.4, +13.8] | +3.3 [+1.2, +5.6] | +6.9 [+4.5, +9.4] | **đạt** |

Mọi CI đều không chứa 0, ở cả hai model và cả hai mức B.

## Recovery trên các câu Fixed-2 sai (B = 3)

| loại | 7B: n / rule_table / H3 / **v2** / oracle | Qwen3: n / rule_table / H3 / **v2** / oracle |
|---|---|---|
| scale | 140 / 57 / 47 / **90** / 102 | 126 / 53 / 40 / **72** / 89 |
| read | 54 / 0 / 2 / 1 / 5 | 50 / 1 / 1 / 5 / 10 |
| reason | 17 / 0 / 1 / 0 / 1 | 34 / 0 / 4 / 1 / 8 |
| span | 46 / 0 / 23 / **24** / 29 | 24 / 0 / 5 / **10** / 8 |

## Diễn giải

- **Bộ phân loại scale là đóng góp chính.** Nó sửa 64–57 % lỗi scale, so với 34–32 % của H3-learned. Trên test2, v2 còn cách oracle chọn nguồn khoảng 5–8 điểm.
- **Luật tra bảng không đủ:** phương pháp học hơn nó 6–8 điểm. Đây là câu trả lời cho hạn chế đã nêu ở H3.
- Kết quả H3 được **tái lập** trên một tập câu hỏi mới.
- Phần còn lại chủ yếu là **lỗi đọc sai ô** (50–54 câu) và **sai phép toán**. Không hành động hiện có nào sửa được hai nhóm này; oracle cũng chỉ cứu được 5–10 câu. Đây là việc của bước tiếp theo (grounding ô qua OCR).
- Hạn chế: 582/609 câu test2 nằm trên tài liệu đã có trong test cũ. Tập con đã lọc (một trang). Chấm theo kiểu official, không dùng script chính thức.
