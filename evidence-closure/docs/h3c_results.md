# H3c: kết quả dev2 (chạy một lần, 2026-11-06)

Protocol: [protocol_h3c.md](protocol_h3c.md), khóa trước khi xem output dev2. Chi tiết: `results/h3c_{qwen25vl7b,qwen3vl8b}_dev2.md`. Dev2 gồm 630 câu, 239 tài liệu. Mọi điều kiện chấm bằng EM port từ evaluator TAT-QA. CI 95 % bootstrap theo tài liệu.

| policy | 7B B2 | 7B B3 | Qwen3 B2 | Qwen3 B3 |
|---|---|---|---|---|
| `direct` | 35.1 | 35.1 | 35.7 | 35.7 |
| Fixed-2 | 53.0 | 53.0 | 60.0 | 60.0 |
| rule-em | 59.5 | 59.0 | 64.1 | 64.4 |
| H3-em | 62.9 | 64.1 | 64.1 | 66.0 |
| v2-len | 67.0 | 68.3 | 65.4 | 66.7 |
| v2-em | 66.3 | 67.8 | 65.7 | 67.0 |

## Tiêu chí

| tiêu chí | 7B B2 | 7B B3 | Qwen3 B2 | Qwen3 B3 | kết luận |
|---|---|---|---|---|---|
| 1. v2-em − v2-len | −0.6 [−1.5, +0.2] | −0.5 [−1.3, +0.3] | +0.3 [−0.5, +1.2] | +0.3 [−0.5, +1.2] | **không đạt** |
| 2. v2-em − H3-em | **+3.5 [+0.6, +6.3]** | **+3.7 [+1.1, +6.5]** | +1.6 [−0.9, +4.0] | +1.0 [−1.6, +3.4] | **không đạt** (chỉ 7B) |
| 3a. v2-em − rule-em | **+6.8 [+3.9, +10.1]** | **+8.7 [+5.6, +12.2]** | +1.6 [−0.5, +3.6] | **+2.5 [+0.3, +4.9]** | **đạt** |
| 3b. v2-em − Fixed-2 | **+13.3 [+9.7, +17.3]** | **+14.8 [+10.8, +18.8]** | **+5.7 [+2.8, +8.6]** | **+7.0 [+3.8, +10.2]** | **đạt** |

Báo cáo thêm: H3-em − H3-len từ −0.3 đến +0.2, CI đều chứa 0.

## Kết luận theo quy tắc đã khóa

1. **Giả thuyết nhãn EM không được ủng hộ.** Học lại trên nhãn EM không giúp; nhãn huấn luyện không phải đòn bẩy. Có thể giữ policy học trên nhãn nào cũng được.
2. **Không được nói v2 tốt hơn H3-learned một cách chung.** Điều này chỉ đúng ở 7B. Mức +3.1 ở Qwen3 B2 trên test2 không lặp lại trên dev2.
3. **Phần "học" có lý do để tồn tại,** rõ ở 7B. Ở Qwen3, mức hơn luật tra bảng nhỏ (+2.5, CI sát 0, chưa hiệu chỉnh đa kiểm định).

Mức hơn Fixed-2 là kết quả vững nhất: lặp lại ở cả test2 và dev2, cả hai model.

## Hạn chế

Dev2 không phải split test; 555/630 câu nằm trên tài liệu đã dùng cho thăm dò E8/E9. EM port từ evaluator TAT-QA, giả định TAT-DQA dùng cùng luật. Tập lọc chỉ gồm câu một trang có fact khớp OCR.
