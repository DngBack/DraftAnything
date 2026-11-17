# Thí nghiệm cổng H1: VATr++ có dùng bằng chứng cục bộ trong support không? (IAM)

Ngày 17/11/2026. Inference-only, không train. Code: `research/relation_coverage/`. Số liệu: `runs/relation_coverage_h1/main/` (local); tổng hợp đã commit tại `h1_assets/`. Panel ảnh không commit vì chứa crop IAM (giấy phép không cho phát tán lại).

## Câu hỏi

Khi bắt chước writer A từ 15 từ support, mô hình có trung thực kém hơn trên từ đích có **nét nối (bigram) chưa xuất hiện trong support** không?

Khảo sát độ phủ trên IAM val1 cho thấy support 15 từ chứa 93,6% chữ cái nhưng chỉ 36,3% nét nối của từ đích.

## Thiết kế

Các so sánh đều nằm trong cùng một cặp (writer, từ đích). Từ đích *t* là ảnh thật của writer A và không bao giờ có mặt trong support, trừ điều kiện SELF.

| Điều kiện | Support 15 từ |
|---|---|
| COV | Từ của A, chứa **mọi** bigram của *t* và mọi chữ cái của *t* |
| UNC | Từ của A, **không** chứa bigram nào của *t*, nhưng vẫn đủ mọi chữ cái |
| RAND | 15 từ ngẫu nhiên của A |
| OTHER | Từ của writer B (mức sàn: sai người viết) |
| SELF | Như COV nhưng thay một từ bằng chính ảnh thật của *t* (mức trần: có bản để sao chép) |

Cân bằng giữa COV và UNC: độ dài từ support trung bình 4,52 ở cả hai; số lần xuất hiện chữ cái của *t* là 23,2 so với 21,7.

Quy mô: tập test writer-independent chính thức của IAM, 49 writer, 762 từ đích, 2 lần lấy support mỗi từ.

Mô hình: VATr++ English (HuggingFace, legacy layout). Cấu hình này được chọn thực nghiệm vì bản sửa layout làm hỏng weights pretrained.

Thước đo: không dùng OCR. Dùng backbone HWD (VGG16 train trên Font², BMVC 2023), độc lập với cả VATr lẫn IAM. Hai cách đo:

- `d_pair`: khoảng cách giữa ảnh sinh và ảnh thật của *t*.
- `d_centroid`: khoảng cách giữa ảnh sinh và tâm feature các từ thật của A.

Khoảng tin cậy 95% lấy bằng bootstrap theo writer, 4.000 lần.

## Kết quả

| Tương phản (`d_pair`, thấp hơn = giống hơn) | Ước lượng | 95% CI |
|---|---:|---:|
| **Hiệu ứng nét nối:** UNC − COV | −0,017 | [−0,038; +0,003] |
| Hiệu ứng sai người viết: OTHER − COV | +0,700 | [+0,610; +0,804] |
| Hiệu ứng có bản sao: COV − SELF | +0,034 | [+0,025; +0,043] |
| RAND − COV | +0,005 | [−0,016; +0,025] |

Với `d_centroid`, kết luận tương tự: UNC − COV = −0,021 [−0,039; −0,002], tức UNC còn nhỉnh hơn một chút, ngược với giả thuyết.

**Kiểm tra độ nhạy của thước đo trên ảnh thật** (996 cặp):

| Cặp ảnh thật | Khoảng cách HWD |
|---|---:|
| Cùng writer, cùng từ (2 lần viết) | 2,36 |
| Cùng writer, từ khác cùng độ dài | 3,37 |
| Writer khác | 4,11 |

HWD **có nhìn thấy** hình dạng riêng theo từ: khoảng chênh "cùng từ / khác từ" là 1,01, bằng 58% khoảng chênh do khác người viết. Vì vậy kết quả âm ở trên không phải do thước đo mù với chi tiết cục bộ.

## Diễn giải

1. **H1 theo cách phát biểu ban đầu bị bác bỏ** với VATr++: thiếu nét nối trong support không làm giảm độ trung thực.
2. Nguyên nhân là **VATr++ gần như không dùng bằng chứng cục bộ của support**:
   - Ngay cả khi support chứa chính ảnh đích, mức cải thiện chỉ là 0,034.
   - Trên ảnh thật, việc "cùng từ" có giá trị tới 1,01. Mô hình khai thác khoảng 3% phần thông tin này.
   - Trong khi đó, đổi người viết làm thay đổi 0,70. Mô hình dùng support như một **vector style toàn cục**.
3. Panel `panel.png` xác nhận trực quan: các cột COV, UNC, SELF gần như trùng khớp; chỉ OTHER khác.

## Giới hạn

- Mới có một mô hình (VATr++) và một không gian feature (HWD, lấy trung bình theo cột).
- Khoảng cách tuyệt đối giữa ảnh sinh và ảnh thật còn chứa chênh lệch miền ảnh (ảnh sinh rộng cố định 16px/ký tự). Chỉ các tương phản giữa điều kiện là so sánh hợp lệ.
- Chưa xác minh được tập test IAM chính thức có trùng writer với split train của VATr hay không.
- Một số ảnh crop từ dòng bị lỗi phân đoạn. Lỗi này ảnh hưởng như nhau đến mọi điều kiện vì thiết kế ghép cặp.
- Hồi quy quan sát trên RAND (có hiệu ứng cố định theo writer) cho hệ số join coverage là +0,11: không có hiệu ứng bảo vệ, nhưng đây chỉ là số liệu quan sát.

## Hệ quả cho hướng nghiên cứu

Câu hỏi đúng không phải "quan hệ chưa thấy làm mô hình kém đi". Câu hỏi đúng là **"các mô hình HTG few-shot không khai thác bằng chứng cục bộ hay quan hệ ngay cả khi bằng chứng đó có sẵn"**. Thước đo cho thấy còn khoảng trống lớn (headroom) ở điều kiện COV/SELF.

Bước tiếp theo:

1. Lặp lại thí nghiệm trên một mô hình khác (Emuru, DiffusionPen hoặc One-DM) để biết đây là đặc điểm chung hay riêng của VATr.
2. Thêm thước đo thứ hai (writer retrieval, khoảng cách patch cục bộ).
3. Dựng baseline "oracle truy xuất cục bộ" để ước lượng phần headroom có thể đạt được thực tế.
