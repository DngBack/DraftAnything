# Protocol T: lỗi đọc ở ảnh kém do mất thông tin hay do thiếu token? (khóa ngày 2026-11-06, trước mọi output)

Vì sao có protocol này: khi xem tạm Protocol D (đã ghi nhật ký), ảnh hạ xuống rồi phóng lại kích thước gốc được đọc đúng nhiều hơn hẳn ảnh hạ thật trên hai model Qwen, trong khi InternVL gần như không đổi. So sánh đó là thăm dò. Protocol này xác nhận hoặc bác bỏ nó trên tập ô mới.

## Dữ liệu

`data/token_set.json` (`scripts/build_token_set.py`): 200 ô, mỗi ô một tài liệu TAT-DQA train, không trùng tài liệu với `h3_train_set`, `attr_set`, `percept_set`, `decide_set`. Bộ lọc ô giống Protocol D. Prompt `copy`. Decode greedy, không repetition penalty. Mỗi output lưu số token ảnh (Qwen) hoặc số tile (InternVL).

## Điều kiện

Mọi điều kiện đều giữ nguyên lượng thông tin của ảnh đã hạ; chỉ đổi kích thước đưa vào model.

**Qwen2.5-VL-7B, Qwen3-VL-8B**, ở s40 và s30, ảnh hạ `x` được phóng thêm `k` lần bằng bicubic: `k1` (nguyên), `k1.5`, `k2`, `full` (về kích thước gốc). Thêm `full_nearest` (về kích thước gốc bằng nearest) và `s100` làm mốc. 11 lần sinh mỗi ô.

**InternVL3-8B**, ở s100, s40, s30, `crop_to_patches` với `max_patches` ∈ {1, 4, 12}. Thêm `full` với `max_patches` = 12 ở s40 và s30. 11 lần sinh mỗi ô.

## Tiêu chí (từng model; CI 95 % bootstrap theo tài liệu, ghép cặp)

| mã | câu hỏi | kết luận |
|---|---|---|
| **T1** (Qwen) | phóng to có cứu không | **có** nếu T(full) − T(k1) có CI > 0 ở cả s40 và s30, và ≥ 10 điểm ở s30 |
| **T2** (Qwen) | có theo liều không | **có** nếu ở s30, T(k2) − T(k1) có CI > 0 và T(full) − T(k2) có CI > 0 |
| **T3** (InternVL) | số tile có quyết định không | **có** nếu ở s100, T(12 tile) − T(1 tile) có CI > 0 và ≥ 10 điểm |
| **T4** (InternVL) | phóng to có vô ích khi tile cố định không | **có** nếu ở s40 và s30, CI của T(full) − T(12 tile) nằm trong ±5 điểm |

Quy tắc: nếu T1 không đạt trên cả hai model Qwen, paper không được dẫn dắt bằng luận điểm "thiếu token", và so sánh của Protocol D chỉ được nêu như thăm dò. Nếu chỉ đạt trên một model, chỉ nói về model đó.

Mô tả, không có tiêu chí: T(full_nearest) − T(full), để xem cách nội suy có phải nguyên nhân không; AUROC log-prob theo điều kiện; T theo số token.

## Không được làm

Không đổi set, điều kiện, chấm điểm, ngưỡng sau khi xem output. Chạy thử chỉ dùng ô của `percept_set`, không tính vào kết quả, chỉ kiểm tra chương trình và số tile.

## Nhật ký sai lệch

- 2026-11-06, 09:20 UTC: dừng lượt chính vì người dùng cần GPU, mới có 6–13 ô mỗi model, chưa có kết quả nào được xem hay phân tích. Các tiêu chí T1–T4 chưa được kiểm định. `run_token.py` chạy tiếp được từ chỗ dừng.

## Giới hạn biết trước

- Phóng to không thêm thông tin nhưng đổi cả số token lẫn cỡ chữ theo patch; thí nghiệm không tách được hai thứ này, và cũng không cần tách để trả lời câu "mất thông tin hay do đầu vào".
- Thiết kế được chọn sau khi đã thấy kết quả tạm của Protocol D. Tập ô mới bảo vệ khỏi việc chọn theo dữ liệu, nhưng giả thuyết không còn là giả thuyết tiên nghiệm.
- Chạy thử trước lượt chính (một ô pilot): ở s40, s30 của InternVL, `max_patches` = 4 cho ra 1 tile vì bộ xử lý chọn lưới theo kích thước ảnh, nên điều kiện này trùng `tile1` trên trang đó. Không ảnh hưởng T3, T4, vốn dùng 1 và 12 tile. Trên 7B, token của s30 là khoảng 285, của `full` khoảng 2.508 (trang 1240×1605).
- Có 4 tiêu chí, không hiệu chỉnh đa kiểm định.
- Ảnh hạ độ phân giải là giả lập; một miền dữ liệu.
