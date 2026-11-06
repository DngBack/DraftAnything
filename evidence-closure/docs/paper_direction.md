# Định hướng paper: quy lỗi bằng can thiệp (bản nháp, 2026-11-06)

## Câu hỏi

Khi VLM trả lời sai câu hỏi trên bảng, phần sai do **nhận thức**, **tính toán**, **đơn vị** hay **lập luận**, và tỷ lệ này thay đổi thế nào theo chất lượng ảnh?

## Khoảng trống (đã kiểm tra trên 3 bài, chưa phải rà soát đầy đủ)

WildTableBench, MirageTVQA và MMTabReal không dùng can thiệp hay Shapley để quy lỗi, và theo những gì đọc được thì không bài nào cho model dùng công cụ tính. MMTabReal dùng người gán nhãn nguyên nhân chính cho 393 câu. Phần phân tích lỗi của WildTableBench và MMTabReal mới đọc qua bản tóm tắt, chưa đọc nguyên văn. Cần rà soát có hệ thống trước khi tuyên bố tính mới.

## Phương pháp đề xuất

Mỗi giai đoạn là một "người chơi". Can thiệp sửa giai đoạn đó bằng thông tin gold, rồi tính giá trị Shapley của mức tăng EM, nên kết quả không phụ thuộc thứ tự sửa.

| người chơi | can thiệp |
|---|---|
| P: nhận thức | đưa sẵn ô gold |
| O: lập luận | đưa sẵn phép toán gold, chưa có số (chưa chạy) |
| C: tính toán | Python thực thi biểu thức |
| S: đơn vị | đưa scale gold |

## Thí nghiệm cần chạy

1. **Đường cong theo chất lượng ảnh (thử nghiệm quyết định):** làm giảm độ phân giải và làm nhiễu trang TAT-DQA, đo Shapley ở mỗi mức. Giả thuyết: phần của P tăng khi ảnh kém đi, điều này giải thích vì sao các benchmark ảnh chụp thấy lỗi nhận thức nhiều hơn.
2. **Thêm O** và kiểm tra nhiều model.
3. **Bộ dữ liệu thứ hai:** cần có ô gold hoặc bảng dạng văn bản. Ứng viên: MirageTVQA (có `evidence_cells` trong prompt sinh dữ liệu, chưa xác nhận được công bố, bảng render từ HTML), FinMMR (chưa kiểm tra).

## Cách đọc pilot

Xem `results/shapley_pilot.txt`. Đây là dữ liệu dev E1, thăm dò, chưa khóa protocol. Các giới hạn đã biết:

- Tập lọc yêu cầu fact khớp OCR, có thể làm nhẹ phần P.
- Can thiệp P chỉ là gợi ý cho model, chưa chắc sửa hết lỗi nhận thức, nên phần P có thể bị đánh giá thấp. Ở câu span, nó gần như đưa sẵn đáp án, nên có thể bị đánh giá cao.
- Nhãn scale gold có nhiễu, có thể làm phần S cao giả.

## Trạng thái

Thí nghiệm 1 đã có đủ script và protocol nháp: [protocol_attr.md](protocol_attr.md). Chưa có kết quả.

## Điều kiện dừng

Nếu phần của P không đổi khi ảnh giảm chất lượng, hoặc Shapley không ổn định giữa các model và bộ dữ liệu, hướng này chỉ còn là một phân tích trên một bộ dữ liệu, không đủ cho hội nghị rank A.

## Kết quả 7B và luận đề cập nhật (2026-11-06; Qwen3 đang chờ)

Tiêu chí chính trên 7B: tỷ phần P từ 10 % (s100) lên 50 % (s30), Δ = +39.7 [+33.2, +46.3]. Nhất quán khi giữ S cố định: 25 % lên 71 % (thăm dò).

Thăm dò, ngoài tiêu chí đã khóa: không có gợi ý gold, với Python và scale gold, EM ở s60 (khoảng 36 % token) bằng s100 (79.7 và 79.7, Δ 0.0 [−3.6, +3.6]), trong khi `direct` giảm 37.2 → 31.1. Nếu lặp lại được với scale dự đoán thay vì gold, độ phân giải tối thiểu đủ dùng phụ thuộc vào pipeline phía sau.

Luận đề ứng viên: **lượng thị giác cần thiết không phải tính chất của riêng câu hỏi hay ảnh, mà phụ thuộc vào phần nào của việc suy luận được đưa ra ngoài model.** Cần: Qwen3, bộ dữ liệu thứ hai, pipeline không dùng gold, trục x theo chiều cao chữ tính bằng pixel, và một policy chọn độ phân giải theo câu hỏi, đánh giá trên dữ liệu chưa dùng.
