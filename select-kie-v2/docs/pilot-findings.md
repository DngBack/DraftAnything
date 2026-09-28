# Nhận định từ kết quả pilot v2

**Đã có kết quả chạy thật, nhưng chưa xác nhận đóng góp phương pháp binding-aware.** Lượt này huấn luyện 16 correctness head (8 cấu hình × 2 extractor), dùng prediction/candidate scores đã có của pilot CORD. Không chạy lại VLM hay fine-tune backbone. [Bảng đầy đủ](pilot-results.md), [code](../scripts/pilot.py), [protocol](pilot-protocol.md).

## Kết quả chính

Threshold chọn trên validation với target error 5%; coverage/risk dưới đây đo trên test 100 tài liệu, 796 quyết định mỗi extractor.

| Extractor | LR fusion: coverage / risk | LR + binding | Generic HGB + binding |
|---|---:|---:|---:|
| Qwen3-VL-4B | 86.06% / 4.67% | 92.46% / 4.76% | 93.59% / 4.70% |
| Qwen2-VL-2B | 70.35% / 6.25% | 77.76% / 4.85% | 81.41% / 4.94% |

LR có binding tăng coverage +6.41 và +7.41 điểm % so với LR fusion, tái hiện pilot cũ. Tuy nhiên baseline Qwen2 có risk vượt target: đây không phải phép so sánh coverage tại cùng risk được bảo đảm.

**Phát hiện mới quyết định:** generic HGB nhận cùng feature matrix tốt hơn LR có binding về AURC. HGB chỉ nhận fusion cũng đã bắt kịp phần lớn lợi ích ranking:

| Extractor | LR fusion | LR + binding | HGB fusion | HGB + binding | 95% CI ΔAURC (HGB + binding − HGB fusion) |
|---|---:|---:|---:|---:|---:|
| Qwen3-VL-4B | 0.02648 | 0.01897 | 0.01333 | 0.01229 | [−0.00367, +0.00228] |
| Qwen2-VL-2B | 0.05656 | 0.04833 | 0.04030 | 0.03844 | [−0.00774, +0.00436] |

Cả hai CI chứa 0. Với dữ liệu và các cấu hình đã thử, chưa có bằng chứng vững cho phần ranking gain riêng của binding sau khi kiểm soát head phi tuyến. Không suy ra rằng binding information vô ích trong mọi thiết lập; baseline phi tuyến cũng chưa được tuning.

HGB fusion có risk test 6.14% và 5.83%; HGB + binding đạt 4.70% và 4.94%. Đây là tín hiệu tích cực về kết quả tại threshold của lượt này. Nhưng khi resample cả validation/test và chọn lại threshold, HGB + binding vẫn vượt target trong khoảng 48–50% bootstrap không rỗng. Không đủ cơ sở claim risk control.

## Kiểm tra giả thuyết

- **H1 — có lỗi binding:** có, nhưng cần tách lỗi field hiện diện khỏi field ABSENT. Toàn bộ 1000 tài liệu: Qwen3 có 62 lỗi `binding` thật trên 7970 quyết định (0.78%), Qwen2 có 402 (5.04%). Phần lớn lỗi gọi chung là binding nằm ở `binding_absent` (482/812). Trên test Qwen3 chỉ có 6 lỗi binding thật. Đây là nhãn tự động, chưa có audit độc lập.
- **H2 — cấu trúc binding vượt head chung cùng thông tin:** **chưa được ủng hộ**. HGB cùng thông tin tốt hơn LR + binding; HGB không có binding đạt ranking gần tương đương. Chưa xây dựng architecture quan hệ mới hay cấp raw image representation cho các head.
- **H3 — unseen-template/nguồn thứ hai:** chưa kiểm tra. Không có ảnh trùng chính xác qua split, nhưng điều đó không loại trừ template leakage.
- **H4 — vai trò thị giác/không gian:** chưa kiểm tra trực tiếp. Bỏ key-side competition gần như không ảnh hưởng Qwen3 (coverage −0.38 điểm %, AURC hơi tốt hơn); ở Qwen2 ranking giảm. Bỏ rival features ở Qwen2 lại cải thiện AURC. Chưa thấy cơ chế thống nhất giữa extractor.

Phân tích độ nhạy bỏ subtotal ABSENT mà model điền gold total: chênh coverage LR trên Qwen3 còn +0.26 điểm %, cả hai risk vượt 5%. Trên Qwen2 HGB + binding còn coverage 84.42%, risk 4.92%; vẫn cần nguồn mới để kiểm chứng.

## Quyết định nghiên cứu đề xuất

**Tiếp tục audit và đánh giá trên nguồn mới; chưa đầu tư ngay một architecture phức tạp với claim superiority.** Pilot cho thấy vấn đề confidence đáng nghiên cứu và cung cấp baseline mạnh hơn để tránh lợi ích giả do head yếu.

1. Audit `results/audit.csv`: 80 quyết định ngẫu nhiên và tối đa 60 lỗi bổ sung mỗi extractor, ghi riêng nhóm lấy mẫu. Cần hai người gán nhãn một phần giao nhau; các cột human hiện còn trống. Tập trung ABSENT/AMBIGUOUS và anchor sai nhưng value vẫn đúng.
2. Chốt protocol mới trên holdout chưa dùng, ưu tiên split theo template và OCR thật. Mức 5% đã xem trong pilot này nên không gọi là lựa chọn mù.
3. Với dữ liệu mới, so sánh binding head và generic head cùng candidate, evidence representation, nhãn và ngân sách tuning. Giữ HGB fusion như baseline rẻ nhưng mạnh.
4. Chỉ huấn luyện head quan hệ một lượt sau khi lỗi mục tiêu và lợi ích độc lập còn tồn tại. Pilot hiện dùng teacher-forcing scores nhiều lượt, chưa đo throughput trên A30.

## Artifact và kiểm chứng

Lượt chạy kết thúc thành công; self-check normalization/risk–coverage của pipeline cũ đã qua. Có 800/100/100 tài liệu cho từng extractor; hash ảnh xác nhận 0 ảnh trùng chính xác qua split. Bootstrap ghép cặp 1000 lần theo tài liệu, bao gồm chọn lại threshold; không bao gồm fitting lại train. Đã lưu metrics JSON, scores CSV, biểu đồ, 280 mẫu audit và 16 head `.joblib` trong `results/heads`.

Đầu ra extractor là artifact kế thừa: không xác minh lại thời gian/GPU ghi trong báo cáo cũ. Thí nghiệm mới chạy CPU. Dataset revision, annotation hashes, input hashes và seed nằm trong [manifest](../results/manifest.json). Nguồn dataset: [NAVER CLOVA CORD-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2).
