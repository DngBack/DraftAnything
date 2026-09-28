# Hướng dẫn audit stage4

Chưa có nhãn người đánh giá. Mở `results/stage4/audit-viewer.html`, xem PDF đủ trang, rồi điền CSV reviewer1. Người thứ hai tự đánh giá20 mẫu trong CSV overlap trước khi trao đổi với người thứ nhất. Không xem private mapping trước khi hoàn tất. Gold có thể sai; phải kiểm tra hình ảnh khi phân loại.

- `correct`: đáp án đúng theo nghĩa field, kể cả chênh punctuation/case không đổi nghĩa; ghi mismatch official trong notes.
- `binding`: giá trị đọc đúng từ document nhưng thuộc field/entity/role khác; ghi field/role nguồn và trang làm bằng chứng.
- `miss`: giá trị có rõ trên ảnh nhưng không được trả về.
- `hallucination`: giá trị không có bằng chứng trên ảnh.
- `partial-value`: đúng entity/role nhưng thiếu thành phần quan trọng của giá trị.
- `transcription`: đúng role nhưng lỗi đọc/chép ký tự.
- `gold/schema-ambiguity`: annotation sai, thiếu, hoặc field không xác định duy nhất theo schema. Ghi nguyên nhân.
- `other`: lỗi còn lại; ghi giải thích.

CSV gồm100 quyết định random từ600 quyết định của9B-clarified và100 lỗi official chia giữa sáu điều kiện (tối đa16 mỗi điều kiện, phần còn lại lấy ngẫu nhiên từ pool lỗi chưa chọn). Hai nhóm được trộn và ẩn model/confidence. Dùng mapping sau audit để báo prevalence trên100 random; không dùng tỷ lệ của toàn200 để suy ra prevalence. Đây là mẫu decision, chưa phải200 tài liệu độc lập: CI prevalence phải cluster theo document. Báo agreement trước adjudication, giữ cả hai nhãn gốc.20 overlap chỉ là kiểm tra ban đầu, không đủ để khẳng định agreement cao chắc chắn.

Không huấn luyện head trên100 document holdout này. Nếu chọn hypothesis từ audit, cần tập xác nhận mới sau khi khóa phương pháp.
