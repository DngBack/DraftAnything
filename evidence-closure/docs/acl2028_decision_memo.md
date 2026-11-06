# Bản ghi quyết định: hướng nghiên cứu cho ACL 2028 (2026-11-06)

Mục đích: ghi lại các giả thuyết đã thử, kết quả áp theo tiêu chí khóa trước, và đề xuất bước tiếp theo. Mọi pilot dưới đây chỉ để quyết định có đầu tư tiếp không, chưa phải kết quả công bố.

> **Cập nhật 2026-11-06, 09:20 UTC:** Protocol D dừng sớm vì cần GPU cho việc khác (218, 260, 127/500 ô), Protocol T chưa đủ để phân tích. Trên mẫu này D1 không đạt ở cả ba model, nên quy tắc dừng cho phần "tách định vị và đọc nét" bị kích hoạt tạm thời. "Phóng to cứu việc đọc" đã được công bố (2611.01517), không còn là điểm mới. Bản nháp hiện tại: [paper/main.tex](../paper/main.tex). Cần chạy tiếp cả hai protocol trước khi rút kết luận.

## 1. Kết quả các phép thử

| giả thuyết | phép đo chính | kết quả | kết luận |
|---|---|---|---|
| ATTR: tỷ phần P tăng khi ảnh kém | Δ tỷ phần P, s30 − s100 | 7B +39.7 [+33.2, +46.3]; Qwen3 +51.4 [+45.1, +58.5] | **đạt** |
| H-instab: bản đọc đổi theo prompt (`program` so với `opprog`) | bất ổn định ở s30, ngưỡng 25 % | 7B 13.3 %; Qwen3 13.0 % | **chết** |
| H-prior: ngữ cảnh kéo bản đọc về số gốc | O − A, `copy`, s40/s30 | O cao nhất 1.2 %; CI đều chứa 0 hoặc suy biến | **chết** |
| H-task: chép đọc tốt hơn viết biểu thức | T(copy) − T(addr), ≥ 5 điểm | 7B s40 +6.2 [0.0, +12.5]; Qwen3 cao nhất +2.5 | **chết** |
| H-meta: có tín hiệu ẩn dù model không nói ra | AUROC log-prob, s40 / s30 | 7B 0.84 / 0.82; Qwen3 0.85 / 0.82; từ chối của `direct` 0 % | **sống** |
| Kênh nhiễu: số đúng có trong beam | độ phủ khi top-1 sai, s40 / s30, ngưỡng 50 % | 7B 46.7 / 27.5; Qwen3 39.4 / 10.6 | **chết** |

Điều kiện hợp lệ của pilot B đạt: ở s100, T(null) − T(orig) = −1.2 và −2.5; T(cf) = 86.2 ở cả hai model.

Kết quả thăm dò, viết sau khi xem dữ liệu:

- Cỡ chữ và độ dài số dự đoán đọc đúng kém log-prob: AUROC phần lớn gần 0.5, cao nhất 0.74 (Qwen3 s30, log-prob 0.82).
- Cặp cùng ô, cùng mức, một bản đọc đúng và một bản đọc sai: bản đúng có log-prob cao hơn ở 67 % [57, 77] (7B) và 74 % [63, 85] (Qwen3).
- Được phép trả lời UNREADABLE, model chỉ dùng 0–5 % và không theo độ đúng: 7B ở s30 không dùng lần nào trong 51 câu đọc sai.
- Mức đọc đúng 50 % nằm quanh chiều cao hộp OCR 6.7–7.4 px (7B) và 8.2–8.6 px (Qwen3), khác biệt giữa prompt nhỏ; chưa có CI.
- 66 % ô đích nằm trong quan hệ cộng dọc kiểm chứng được (cận trên, có thể có dương tính giả), vậy mà O ≈ 0: model không dùng thông tin dư này.
- Khi top-1 sai, chỉ 10–30 % bản đọc cách số đúng ≤ 1 chữ số ở s40/s30. Ví dụ `(8,690)` đọc thành `(11,863)`, gần ô bên cạnh `(11,061)`. Lỗi có vẻ gồm cả sai vị trí, không chỉ sai nét. Cần can thiệp riêng để xác nhận.

## 2. Bức tranh rút ra

Trên ô số của bảng tài chính, ở hai model đã thử: đọc bám pixel, gần như không đổi theo nhiệm vụ, không bị kéo về giá trị hợp lý, và thất bại trong im lặng dù có tín hiệu bên trong. Khung "nhận thức là một giai đoạn" của ATTR vì vậy có thêm chỗ dựa thực nghiệm.

Điều này tương phản với các bài cho thấy VLM đi theo prior trên chữ và vật thể (FaithC4, PriVE-Bench). Giải thích ứng viên, chưa kiểm định: chuỗi số trong bảng ít có prior từ vựng để bám vào.

## 3. Đề xuất luận đề cho paper

> **Cập nhật cùng ngày sau rà soát tài liệu ([related_work.md](related_work.md)):** phần 4 đã đông bài, nên không còn là điểm mới chính. Lõi của paper chuyển sang phần 1 (đường cong Shapley) và phần 3 (thí nghiệm 2×2), phần 2 làm nền. Phần 4 chỉ giữ dưới dạng "khoảng cách mở ra ở đâu trên trục độ rõ". Thí nghiệm quyết định: [protocol_decide.md](protocol_decide.md).

**"Đọc thật, thất bại im lặng": giải phẫu nhân quả của việc đọc số trong VLM.** Bốn phần, theo thứ tự rủi ro tăng dần:

1. Đường cong quy lỗi Shapley theo chất lượng ảnh (đã có dữ liệu, cần bộ dữ liệu thứ hai).
2. Bằng chứng nhân quả bằng sửa pixel: T, O − A, copy so với addr, ngưỡng theo chiều cao chữ, trên mẫu lớn.
3. Tách nhận thức thành **định vị** và **đọc nét** bằng can thiệp cùng phương thức: tô hộp quanh ô (chỉ cho vị trí), phục hồi pixel của ô (cho nét). Cách này cũng tránh được hạn chế gợi ý gold gần như đưa sẵn đáp án ở câu span.
4. Khoảng cách giữa tín hiệu bên trong và lời tự báo cáo, cùng mức lợi thực tế khi dùng tín hiệu đó để từ chối hoặc xin thêm pixel, không cần gold.

## 4. Rủi ro

- **Trùng ý.** MM-AQA đã cho thấy VLM hiếm khi từ chối và quy cho miscalibration; phần 4 chỉ mới ở chỗ quét độ rõ và so sánh trong cùng ô. Chọn lọc theo log-prob là kỹ thuật cũ, nên đóng góp phải là cơ chế và khoảng cách, không phải bộ lọc. Bài đọc biểu đồ (arXiv 2609.13745) rất gần về phương pháp. Rà soát tới nay còn hẹp; phải rà soát có hệ thống trước khi tuyên bố tính mới.
- **Độ mạnh thống kê.** 80 ô chỉ loại được hiệu ứng lớn. H-task ở 7B s40 sát ngưỡng, nên kết luận "không phụ thuộc nhiệm vụ" chưa dùng được cho hiệu ứng nhỏ. H-prior chỉ xét khớp chính xác số gốc.
- **Phạm vi.** Mới hai model cùng họ Qwen, một miền dữ liệu, ảnh giảm chất lượng giả lập, kích thích dùng font thay thế.
- **Độ phù hợp với ACL.** Phần thị giác nặng có thể khiến reviewer ACL cho là lệch trọng tâm. Nên đặt trọng tâm ở tự báo cáo và tính trung thực của ngôn ngữ, đồng thời chuẩn bị phương án EMNLP hoặc TACL.

## 5. Bước tiếp theo

**Thí nghiệm quyết định (khoảng 2 tuần).** Protocol mới, tập mới, ≥ 500 ô:

- Can thiệp định vị và đọc nét.
- Lặp lại H-prior, H-task với độ mạnh đủ.
- Lặp lại H-meta với nhiều cách cho phép từ chối, gồm cả các prompt mạnh hơn.
- Thêm ít nhất một họ model ngoài Qwen.

Dừng hoặc đổi hướng nếu: định vị và đọc nét không tách được, hoặc H-meta không lặp lại ngoài họ Qwen.

**Lịch dự kiến** (ACL 2028 chưa công bố ngày; mới biết sẽ ở khu vực EMEA; các mốc dưới đây là dự kiến và phải chỉnh khi có lịch ARR):

| giai đoạn | việc |
|---|---|
| 11/2026–01/2027 | rà soát tài liệu có hệ thống; thí nghiệm quyết định |
| 02–06/2027 | bộ dữ liệu thứ hai, ảnh chụp thật, nhiều model |
| 07–12/2027 | cơ chế (patching) và ứng dụng từ chối; bản nháp đầu, xin phản biện nội bộ |
| đầu 2028 | nộp theo chu kỳ ARR tương ứng, nếu dữ liệu đủ mạnh |

Không có gì bảo đảm được nhận; lịch này chỉ nhằm có thời gian để bỏ hướng sớm nếu cần.
