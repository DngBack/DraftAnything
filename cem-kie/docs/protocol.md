# CEM-KIE — protocol kiểm tra khả thi trên Kleister-Charity

Viết và khóa ngày 26/10/2026, **trước** khi train mô hình nào. Audit dữ liệu (bước 1) đã chạy trước khi viết protocol. Vì vậy tiêu chí G-A là mô tả, không phải dự đoán.

## Câu hỏi

Trên benchmark KIE tài liệu dài chuẩn, việc giữ nhiều giả thuyết bằng chứng riêng cho từng field (competing slots + resolver) có giúp phân giải bằng chứng cạnh tranh tốt hơn (a) chấm điểm từng ứng viên độc lập, (b) một vector bằng chứng gộp, (c) Transformer dùng chung trên cùng tập ứng viên, ở cùng encoder, feature và router hay không?

## Dữ liệu

- Kleister-Charity (applicaai/kleister-charity, clone ngày 26/10/2026). `train` 1729 tài liệu, chia 85/15 thành train/val với seed 1234. Test là **dev-0** (440 tài liệu), vì nhãn test-A không công khai.
- Text: cột `text_best` (pdf2djvu, hoặc tesseract nếu dài hơn). Không có ngắt trang, nên vị trí được đo theo tỉ lệ ký tự trong tài liệu.
- 5 field có không gian ứng viên đóng: income, spending (tiền), report_date, charity_number, postcode. Bỏ charity_name, post_town, street_line, vì đó là bài toán trích span chứ không phải chọn ứng viên.
- Ứng viên được sinh bằng regex có chuẩn hóa về format của expected.tsv. Mỗi mention tiền sinh 2 ứng viên: giá trị như viết và ×1000 (cho báo cáo ghi đơn vị £'000).
- Metric: F1 kiểu Kleister trên cặp key=value, cho phép bỏ trống. Kết quả **không so trực tiếp** với F1 công bố trên 8 field.

## Mô hình

Encoder đông cứng `BAAI/bge-base-en-v1.5`, đầu vào là context 240 ký tự trái và 80 ký tự phải quanh mention, có đánh dấu mention. Thêm 9 feature vô hướng giống nhau cho mọi biến thể (vị trí, độ lớn, scale, ngoặc, £, tần suất giá trị, hạng độ lớn, độ dài tài liệu). Trunk và router dùng chung. Biến thể top-K dùng K=64 ứng viên theo router.

| variant | phân giải |
|---|---|
| point | chỉ router (pointer độc lập) |
| pool | 1 vector gộp toàn bộ ứng viên, chấm lại top-K |
| set2 / set4 | Transformer 2 / 4 lớp trên [query; top-K] |
| cem_rR | R slot cạnh tranh (slot attention + sink loại bỏ, 3 vòng), mỗi slot có pointer bằng chứng, resolver 2 lớp với quan hệ ẩn |
| cem_r4_nores | như cem_r4, bỏ resolver |

Loss: NLL đa-positive trên giá trị cuối + NLL router + 0,05 × độ trùng pointer giữa các slot (chỉ CEM). Kleister **không có nhãn quan hệ**, nên relation loss của proposal không được huấn luyện. Quan hệ là ẩn và học qua attention bias.

Huấn luyện: AdamW lr 3e-4, 30 epoch, batch 8 tài liệu, chọn epoch theo val micro F1. Mỗi biến thể chạy 3 seed (0, 1, 2). Không tune riêng cho từng biến thể.

## Tiêu chí (khóa trước)

- **G-A (hiện tượng có tồn tại):** với income/spending, mọi heuristic first/last/frequent/max đạt dưới 10 F1 trong khi oracle ứng viên ≥ 85. *Đã quan sát ở audit.*
- **G-B (H1, chính):** cem_r4 − set2 trên nhóm money ≥ +2 pp, CI 95% bootstrap > 0, **và** trên nhóm other không kém quá 1 pp. Phải đạt cả so với set4.
- **G-C (số slot):** cem_r4 − cem_r1 trên money có CI > 0.
- **G-D (resolver):** cem_r4 − cem_r4_nores trên money có CI > 0.
- **G-E (độ dài):** lợi thế của G-B không biến mất ở tertile tài liệu dài nhất.

Nếu G-B không đạt, kết luận là: trên dữ liệu chuẩn, competing memory không hơn Transformer dùng chung khi cùng ứng viên. Khi đó không đầu tư full model CEM.

Lưu ý: ngoài các so sánh khóa ở trên, mọi phân tích khác là thăm dò.
