# Protocol H3b: baseline mạnh hơn + bộ chọn tách đôi (khóa ngày 2026-11-05, trước khi đánh giá trên test2)

Bối cảnh: H3 đã đạt cả ba tiêu chí trên test ([h3_results.md](h3_results.md)), nhưng còn thiếu một baseline luật đơn giản học trên train. Nếu luật đó đã đạt được mức tăng thì phương pháp học không có lý do để tồn tại. H3b bổ sung baseline này, cùng một phương pháp v2 chỉ phát triển bằng CV trên train.

## Dữ liệu

- **train**: giống H3 (`results/h3_train_*.jsonl`, 600 câu).
- **test2**: **609 câu** TAT-DQA test còn lại sau khi bỏ 600 câu đã dùng ở E3/H3 (`data/h3b_test2_set.json`, bộ lọc giống E1, 241 tài liệu). Hai model chạy đủ 609 câu. Dữ liệu được thu bằng `run_h3.py` với cùng các hành động; chỉ thêm `--name` để đổi tên file output.
- **Hạn chế đã biết**: 582/609 câu nằm trên tài liệu đã xuất hiện trong test cũ. Không policy nào được học hay tinh chỉnh trên test cũ; test cũ chỉ được xem ở mức tổng hợp. Câu hỏi trong test2 là câu mới.

## Policy (`scripts/h3b_policy.py`)

- Toàn bộ policy của H3 được giữ **nguyên**, kể cả `H3-learned`.
- **rule_table** (baseline mới):
  - Mỗi câu được xếp vào một ô theo 6 đặc trưng: có biểu thức hay không, `direct` có chứa số hay không, `direct` có phải năm hay không, scale theo `direct`, scale theo OCR, và câu hỏi có từ khóa % hay không.
  - Với mỗi ô, chọn trên train cặp nguồn (giá trị ∈ {direct, program}, scale ∈ {direct, ocr} hoặc thêm vlm khi B = 3) cho accuracy cao nhất.
  - Ô có dưới 5 câu train thì dùng cặp tốt nhất toàn cục.
- **v2** (phương pháp):
  - Chọn nguồn giá trị bằng logistic regression P(giá trị đúng | đặc trưng) cho direct và program.
  - **Bộ phân loại scale** one-vs-rest gồm 5 lớp. Đặc trưng: scale do từng nguồn đề xuất, đáp án có phải số hay năm, độ lớn của giá trị và từ khóa trong câu hỏi.
  - Khi B = 3, thêm đặc trưng `scale_vlm`, tốn 1 lượt gọi.
  - Không dùng gold làm đặc trưng.
- Chi phí: v2 và rule_table tốn đúng 2 lượt gọi (B = 2) hoặc 3 lượt (B = 3).

## Kết quả phát triển (chỉ CV trên train, trước test2)

| | 7B B2 | 7B B3 | Qwen3 B2 | Qwen3 B3 |
|---|---|---|---|---|
| rule_table | 62.7 | 63.0 | 63.0 | 64.0 |
| H3-learned | 67.2 | 67.5 | 66.3 | 66.2 |
| v2 | 69.3 | 69.7 | 71.8 | 72.3 |

## Tiêu chí (đánh giá **một lần** trên test2, metric chính thức, CI 95 % bootstrap theo tài liệu)

1. **Chính**: v2 − rule_table > 0, với CI không chứa 0, ở ít nhất một B và cả hai model.
2. v2 − (baseline tốt nhất trong mọi policy H3 và rule_table ở cùng B) > 0, với CI không chứa 0, ở ít nhất một B và cả hai model.
3. **Tái lập H3**: H3-learned − Fixed-2 > 0 trên test2, với CI không chứa 0, ở ít nhất một B và cả hai model.

Nếu (1) không đạt: kết luận rằng một luật tra bảng đã đủ, và phần "học" của H3 không có lý do để tồn tại.

## Không được làm

Không sửa đặc trưng, policy, siêu tham số hay hàm chấm sau khi đã chạy `h3b_policy.py --test`.
