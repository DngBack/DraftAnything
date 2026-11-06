# Protocol H3c: học policy trên nhãn EM chính thức (khóa ngày 2026-11-06, trước khi xem output dev2)

Câu hỏi: các policy H3/H3b học trên nhãn dễ dãi (`value_correct`). Học lại trên nhãn EM chính thức có làm tăng EM hay không? Và v2 có thật sự hơn H3-learned hay không?

## Dữ liệu

- **train**: 600 câu như H3. Không thêm câu.
- **dev2**: 630 câu TAT-DQA dev đủ điều kiện, ngoài 600 câu E1 (`data/h3c_dev2_set.json`, 239 tài liệu). Thu bằng `run_h3.py`, cùng các hành động, hai model chạy đủ 630 câu. Việc thu dữ liệu bắt đầu trước khi khóa protocol; output chưa được xem.
- **Hạn chế**: 555/630 câu nằm trên tài liệu có trong E1, vốn đã dùng cho thăm dò E8/E9. Dev2 không phải split test. Câu hỏi trong dev2 chưa từng được dùng.

## Thước đo và hàm tách đáp án (cố định)

- EM port từ evaluator TAT-QA (`evc/official_metric.py`). Không tìm thấy script riêng của TAT-DQA; giả định cùng luật.
- `direct_pred` (`scripts/official_rescore.py`), chọn **chỉ trên train** trong các phương án đã thử:
  - bỏ phần chỉ có đơn vị (`$000`, `€m`...);
  - lấy vế phải của `=`;
  - chữ scale chỉ nằm ở một chỗ: bỏ khỏi chuỗi nếu policy có nhãn scale, giữ lại nếu nhãn rỗng;
  - bỏ phần trùng lặp.

## Điều kiện (`scripts/h3c_policy.py`, mọi điều kiện chấm EM)

- Baseline: `direct`, `fixed2`.
- Học trên nhãn dễ dãi: `H3-len`, `rule-len`, `v2-len`.
- Học trên nhãn EM: `H3-em` (λ chọn bằng CV theo EM), `rule-em`, `v2-em`.
- Đặc trưng và siêu tham số không đổi so với H3/H3b. B ∈ {2, 3}.

## Kết quả phát triển (chỉ 5-fold CV trên train)

| | 7B B2 | 7B B3 | Qwen3 B2 | Qwen3 B3 |
|---|---|---|---|---|
| fixed2 | 51.5 | 51.5 | 57.2 | 57.2 |
| rule-em | 57.7 | 58.2 | 60.2 | 61.3 |
| H3-len / H3-em | 63.3 / 63.7 | 63.0 / 64.5 | 62.5 / 63.0 | 62.8 / 62.8 |
| v2-len / v2-em | 64.0 / 64.8 | 63.8 / 64.8 | 64.0 / 64.5 | 64.3 / 64.8 |

Dự đoán trước: hiệu ứng của nhãn EM nhỏ (khoảng 0 đến +1.5 điểm), nên có thể không đủ để đạt ý nghĩa thống kê trên 630 câu.

## Tiêu chí (đánh giá **một lần** trên dev2, CI 95 % bootstrap theo tài liệu)

1. **Giả thuyết chính:** v2-em − v2-len > 0, CI không chứa 0, ở ít nhất một B và cả hai model.
2. **Giữ v2:** v2-em − H3-em > 0, CI không chứa 0, ở ít nhất một B và cả hai model.
3. **Thắng baseline:** v2-em − rule-em và v2-em − fixed2 > 0, CI không chứa 0, ở ít nhất một B và cả hai model.

Báo cáo thêm H3-em − H3-len. Có 8 phép so sánh chính ở tiêu chí 1–2 mà không hiệu chỉnh đa kiểm định; đọc CI sát 0 một cách thận trọng.

Cách quyết định:

- (1) không đạt: nhãn huấn luyện không phải đòn bẩy; giả thuyết không được ủng hộ.
- (2) không đạt: không được nói v2 tốt hơn H3-learned.
- (3) không đạt: phần "học" không có lý do để tồn tại trên thước đo chính thức.

## Không được làm

Không sửa đặc trưng, policy, hàm tách đáp án hay hàm chấm sau khi chạy `h3c_policy.py --dev2`.
