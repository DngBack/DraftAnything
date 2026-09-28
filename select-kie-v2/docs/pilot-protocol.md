# Pilot v2 — protocol trước lượt chạy mới

Ngày 28/09/2026. Đây là phân tích **exploratory trên test đã được dùng ở pilot cũ**, không phải preregistration trên holdout mới.

Tận dụng outputs thật của Qwen3-VL-4B và Qwen2-VL-2B trên CORD-v2 trong `../../selective-kie/outputs`. Không sinh thêm prediction và không fine-tune extractor. Train correctness heads mới trên 800 tài liệu train, threshold trên 100 validation, đánh giá trên 100 test. Giữ nguyên normalization, schema và exclusions của pilot cũ để đối chiếu, nhưng công khai hạn chế zero=ABSENT và bỏ dấu âm.

Endpoint chính của lượt chạy: AURC và coverage/risk thực tế với target validation 5%; 1%, 2%, 10% là secondary. Không tuyên bố risk guarantee. So sánh LR fusion, LR fusion+binding, generic nonlinear head (histogram gradient boosting) nhận **cùng tất cả features**. Hyperparameters cố định trước chạy, không chọn model theo test. Đây là kiểm tra generic correctness baseline; chưa có architecture binding mới để chứng minh superiority.

Ablation: field-only, bỏ field identity, bỏ key-side competition, bỏ value-side rival (margin/argmax/entropy). Report Brier; bootstrap paired theo document B=1000, resample validation và test rồi chọn lại threshold; không resample train trong CI này. Báo CI của chênh coverage và AURC, kèm tỷ lệ bootstrap risk vượt target (thống kê mô tả, không phải xác suất bảo đảm).

Taxonomy tách `binding` khỏi `binding_absent`; sensitivity bỏ nhóm subtotal ABSENT nhưng prediction bằng gold total. Loại nhóm này dùng gold chỉ cho phân tích diagnostic, không phải quy tắc deployment. Xuất mẫu annotation để người nghiên cứu audit, chưa gọi nhãn tự động là nhãn được con người xác nhận. Kiểm tra trùng ảnh chính xác giữa split và completeness của outputs.

Không claim unseen-template, nguồn thứ hai, OCR thật, visual inductive bias hay self-awareness của backbone. Các score binding có sẵn cần nhiều teacher-forcing pass, chưa phải head một lượt.

## Bổ sung sau lượt đầu

HGB cùng thông tin tốt hơn LR ở lượt đầu. Bổ sung `fusion-HGB` với đúng hyperparameters HGB đã cố định để kiểm soát capacity khi đo ích lợi tín hiệu binding. Quyết định này được đưa ra sau khi xem kết quả exploratory, không gọi là preregistered. Không thay đổi hyperparameters hay chọn alpha. Lưu các head đã fit để có artifact dùng lại.
