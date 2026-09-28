# Stage7: xác nhận cuối trên Short-Form FARA

## Kết luận

**Co-primary gate thất bại. Không có bằng chứng rằng relation head tốt hơn semantic HGB đã dừng bằng development set.** Trên 89 tài liệu mới, cả official và normalized PRESENT-only AURC đều cao hơn về số học cho relation; hai khoảng tin cậy 97.5% vẫn cắt qua 0. Không viết claim superiority/9 trên 10 cho CVPR từ kết quả này.

## Protocol và tính toàn vẹn

Stage7 protocol được đăng ký trước khi inference/scoring mới. Tất cả 89 tài liệu Short-Form cuối không trùng tên hoặc exact OCR với 900 tài liệu trước; không còn tài liệu FARA Short-Form mới trong nguồn này. Dùng Qwen3.5-9B, clarified schema và các head đã khóa. Calibration map/threshold được fit trên 100 ảnh Stage4 vốn đã được xem để phát triển phương pháp; chúng không độc lập và không dùng nhãn stage7. Frozen source hashes được xác minh. Không có output cap hay key thiếu. Bootstrap ghép cặp theo tài liệu, 2,000 lần.

Primary comparator là relation với semantic HGB được early-stop trên cùng 50 nhãn development, cùng field/null calibration. Chênh lệch tính `relation − HGB-dev`; thấp hơn là tốt hơn.

| Thước đo | Relation | HGB-dev | Δ relation−HGB-dev | CI 97.5% |
|---|---:|---:|---:|---:|
| PRESENT-only official AURC | 0.04336 | 0.03802 | +0.00533 | [−0.00373, +0.01528] |
| PRESENT-only normalized AURC | 0.03623 | 0.02901 | +0.00722 | [−0.00132, +0.01691] |

Cả hai gate yêu cầu CI hoàn toàn dưới 0; cả hai đều không đạt. Ở field/null map, relation chấp nhận 282/392 giá trị hiện diện (71.94%) và sai 16 (risk 5.67%); HGB-dev chấp nhận 283/392 (72.19%) và sai 9 (risk 3.18%). Trên mọi quyết định gồm cả ABSENT, AURC lần lượt 0.03894 và 0.03628; coverage lần lượt 70.97% và 71.16%.

| Head / map | AURC ↓ | Coverage | Lỗi / accepted |
|---|---:|---:|---:|
| Generic / field-null | 0.04251 | 70.60% | 18 / 377 |
| Relation / field-null | 0.03894 | 70.97% | 16 / 379 |
| Semantic HGB / field-null | 0.03899 | 73.22% | 13 / 391 |
| Semantic HGB-dev / field-null | **0.03628** | 71.16% | **9 / 380** |

## Diễn giải

Kết quả stage6 so với MLP generic không tái lập thành ưu thế so với một đối chứng HGB được dừng bằng cùng nhãn development. Điều này làm suy yếu claim kiến trúc relation head. Dữ liệu 89 tài liệu là xác nhận cho phép so sánh cố định đã đăng ký, nhưng không độc lập với giai đoạn phát triển head, HGB, hay calibration source. Calibration chỉ có 50 tài liệu fit và 50 tài liệu đặt ngưỡng; empirical threshold không phải risk certificate. OCR/gold vẫn là nguồn benchmark, chưa có audit người.

Không thay calibration hoặc chọn map khác sau khi xem test. Chi tiết metric theo mọi head/map, bootstrap và scores nằm trong `results/stage7/metrics.json`, `all-scores.csv`, `calibrated-scores.npz`; protocol trong [stage7-protocol.md](stage7-protocol.md).

## Bước tiếp theo đã chỉnh

Một cuộc khảo sát schema DeepForm/AdBuy đã phát hiện official corpus có line-item lặp và có split unknown-template 200/100/159. Tuy nhiên, khảo sát đã duyệt annotation toàn corpus để thống kê schema và in ví dụ. Vì vậy split test hiện tại **không còn được xem là blind holdout**; không chạy hoặc báo điểm test của split đó như confirmation. Cần một corpus/test set mới chưa mở nhãn, và protocol/evaluator phải khóa trước khi inference. Artifact nguồn chính thức được giữ trong `data/vrdu/` và evaluator Google Research trong `external/vrdu/`; xem [research-progress.md](research-progress.md) để biết tình trạng.
