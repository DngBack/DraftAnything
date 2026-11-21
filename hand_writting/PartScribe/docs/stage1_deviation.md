# Thay đổi sau NO-GO giai đoạn 1 (18/11/2026, trước mọi kết quả của nhánh mới)

## Căn cứ

Kết quả `runs/probes/verdict.txt`, với tiêu chí đã khóa trong `stage1_probe_criteria.json`:

| Chỉ số (ký tự giữ lại, font test) | Giá trị | Kết luận |
|---|---|---|
| P1 exact | slots 0,260; global 0,416; pixel 0,369 | FAIL |
| P2 success | 0,290 (mức đoán theo dấu phổ biến nhất 0,308) | FAIL |

Slot attention không tách thân chữ và dấu thành các slot riêng.

## Quyết định

1. Nhánh chính của giai đoạn 4 đổi từ `parts` (slot học không nhãn) sang **`nfd`**: token phân rã Unicode.
   - Mỗi ký tự có các token: chữ cái gốc viết thường + embedding hoa/thường; dấu mũ/trăng/móc; dấu thanh.
   - Mọi token có role embedding, và mọi embedding dùng chung giữa các ký tự.
   - plan_03 đã liệt kê phương án này ("dùng phân rã NFD làm giám sát yếu"); nay nó trở thành nhánh chính.
2. `gen_vn_parts` dừng ở khoảng 1,2k bước. Checkpoint và log được giữ lại, không dùng để kết luận.
3. **Claim của paper thay đổi.** Bỏ claim "thành phần học không nhãn". Câu hỏi còn lại: biểu diễn nội dung phân rã có giúp generator sinh đúng và đúng phong cách tổ hợp chưa có ảnh không, so với token ký tự nguyên khối (VATr-style)?
   - Với tiếng Việt, có thể dùng nhãn Unicode vì nó có sẵn và tất định.
   - Lý do tokenizer học không nhãn thất bại được báo cáo như một kết quả null.
4. **Không đổi** tập giữ lại, chỉ số chính, ngưỡng quyết định (≥ 10 điểm phần trăm, CI 95% không chứa 0, xem report_04 mục 2.5), baseline `char`, bộ chấm và việc chỉ mở test một lần.

## Ghi chú về lỗi EMA (độc lập với quyết định trên)

EMA 0,9999 không có warmup làm mẫu sinh từ EMA ra nhiễu trong vài chục nghìn bước đầu. Đã thêm warmup cho các run mới. Hai run VN bắt đầu trước bản sửa được đánh giá bằng trọng số train trực tiếp (`--raw`) cho tới khi EMA hết nhiễm (khoảng 50k bước). Mọi so sánh giữa các nhánh dùng **cùng loại trọng số ở cùng số bước**.
