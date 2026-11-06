# Protocol pilot C: khả thi của giải mã kênh nhiễu (khóa ngày 2026-11-06, trước mọi output của pilot này)

Ý tưởng: model đọc chữ số từ pixel, không dùng ngữ cảnh tài liệu (H-prior chết), nhưng có độ bất định ẩn (H-meta sống). Vậy có thể giải mã lại: P(số | ảnh) lấy từ beam của model, kết hợp với ràng buộc của bảng (các dòng cộng lại bằng tổng). Pilot này chỉ đo hai điều kiện cần, chưa xây bộ giải mã.

Dữ liệu: `data/percept_set.json` (80 ô, đã dùng ở pilot B), ảnh `orig`, prompt `copy`, mức s50, s40, s30, hai model. Beam search 8 nhánh, tối đa 16 token (`scripts/run_percept_nc.py`).

## Đại lượng

- **Độ phủ beam:** trong các ô mà nhánh top-1 đọc sai, tỷ lệ số đúng nằm trong 8 nhánh.
- **Độ phủ ràng buộc** (`scripts/explore_table_sums.py`, không dùng GPU): tỷ lệ ô đích nằm trong một quan hệ cộng dọc kiểm chứng được trên OCR: một dãy liền nhau ≥ 2 ô trong cùng cột có tổng bằng ô kế tiếp, sai số ≤ max(1, 0.1 %), tổng ≥ 10. Cột là các số có cạnh phải lệch ≤ 12 px. OCR ở đây là OCR gold của độ phân giải gốc, nên đây là cận trên; khi ảnh kém, các ô khác cũng sai.

## Tiêu chí

Hướng kênh nhiễu **đáng làm tiếp** nếu cả hai đều đạt:
1. Độ phủ beam ≥ 50 % ở s40 hoặc s30, trên cả hai model.
2. Độ phủ ràng buộc ≥ 25 % số ô đích.

Tiêu chí 1 không đạt: số đúng không có trong phân phối của model, nên giải mã lại không cứu được và cần thêm pixel. Tiêu chí 2 không đạt: ràng buộc quá hiếm trên dữ liệu này.

## Giới hạn

80 câu; tập này đã xem ở pilot B, nên đây là phép đo khả thi, không phải đánh giá. Điểm beam gồm cả token định dạng. Quan hệ cộng tìm tự động có thể có dương tính giả ở các số nhỏ. Độ phủ beam không phân biệt lỗi sai ô với lỗi đọc nét.
