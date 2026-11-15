# Tổng hợp kết quả VATr++ tiếng Việt sau 250.000 steps

Ngày tổng hợp: 15/11/2026. Nguồn: status.json, expanded/benchmark_*.json và bảng benchmarks.csv của component_continuation_250k_v1.

## Kết luận hiện tại

Training đã hoàn thành 250.000 outer steps, exit code 0, lúc 19:41 UTC ngày 14/11/2026; chạy thêm 73.500 steps từ nguồn 176.500, khoảng 7 giờ 9 phút. Không train lại từ đầu. Checkpoint nguồn được giữ nguyên.

Component-aware transcript sampling có tín hiệu tích cực trong pilot 1.000 steps, nhưng continuation dài không cải thiện bền vững. Không chọn checkpoint 250k làm bản tốt nhất. Bài toán chữ hoa hiếm có dấu chưa được giải quyết.

## Thuật toán thực sự được train

Thay cách chọn transcript: kết hợp trọng số ký tự hiếm với trọng số thành phần base/case, modifier/case, tone/case và modifier×tone×case. Trong lượt rare, trộn hai trọng số theo tỷ lệ 50/50; rare fraction 0,3 và HAT fraction 0,35.

Giữ kiến trúc native VATr++, loss và OCR frozen; khôi phục G/D/writer/OCR, Adam và RNG. Data iterator khởi động lại nên không phải tiếp diễn bit-exact. Batch/support 8/15. Đây là thay đổi phân phối transcript, chưa phải cơ chế generator lắp ghép nét Ê và dấu ngã thành Ễ. Sampler không tạo transcript mới hoặc ảnh giám sát mới. Các ý tưởng kiến trúc trong report 07 không thuộc run 250k này.

## Kết quả đo

Panel cố định 600 mẫu, 30 writer. CER càng thấp càng tốt; Exact là tỷ lệ OCR đọc đúng toàn bộ transcript.

| Mốc | CER | CER bỏ dấu | Exact |
|---|---:|---:|---:|
| Nguồn component 176.500 | 12,69% | 8,65% | 65,00% |
| 188.000 — best theo rolling selector | 13,63% | 9,42% | 62,17% |
| 214.000 — CER đơn mốc tốt nhất sau resume | 12,69% | 8,54% | 64,33% |
| Kết thúc 250.000 | 16,26% | 10,58% | 56,33% |

So với nguồn, 250k tăng CER 3,57 điểm phần trăm và giảm Exact 8,67 điểm phần trăm. Mốc 214k chỉ bằng CER nguồn, Exact vẫn thấp hơn. Best theo cửa sổ ba mốc là 188k, rolling CER 13,65%; đây là tiêu chí chọn checkpoint, không phải trung bình trọng số.

Pilot trước đó từ cùng nguồn baseline 175.500: sau 1.000 steps, legacy CER 17,95%/Exact 52,50%, component CER 12,69%/Exact 65,00%. Chỉ một seed. Continuation dài chỉ chạy component, không có nhánh legacy chạy song song tới 250k; vì vậy chưa thể quy toàn bộ suy giảm cho sampler.

## Chữ hiếm, chữ hoa và dấu

| Prompt | Nguồn đúng /30 | 250k đúng /30 |
|---|---:|---:|
| Tiếng | 24 | 17 |
| Việt | 30 | 30 |
| mã | 11 | 7 |
| mạ | 12 | 13 |
| NGUYỄN | 29 | 5 |
| Ắ | 0 | 0 |
| Ễ | 0 | 0 |
| Ỡ | 0 | 0 |
| Ự | 0 | 0 |

CER nhóm uppercase tăng 47,27% → 55,76%; nhóm other tăng 4,42% → 6,81%. Nhóm uppercase gồm 180 mẫu, có nhiều ký tự đơn khó, nên không suy rộng thành tỷ lệ lỗi của mọi từ viết hoa tiếng Việt.

Kết quả cho thấy đúng từ chứa Ễ không đồng nghĩa sinh đúng Ễ đứng riêng. Audit trước đó ghi nhận thiếu transcript ký tự đơn tương ứng. Tăng sampling không tự tạo được giám sát cho ngữ cảnh chưa có. Tuy vậy, chưa thể kết luận nguyên nhân duy nhất là dữ liệu: OCR, decoder và động lực GAN đều cần kiểm tra riêng.

## Giới hạn của kết luận

OCR dùng đánh giá cũng là content supervisor; đây là proxy nội dung, chưa phải đánh giá OCR độc lập, người đọc hay phong cách. Không coi 0/30 là chứng minh tất cả ảnh đều không đọc được bởi con người. Chưa có multi-seed, ablation dài cùng ngân sách hay kiểm định trên tập độc lập. Hiện chưa đủ bằng chứng để tuyên bố phương pháp giải quyết long-tail hoặc cải tiến đạt chuẩn bài ACL.

## Checkpoint và hình ảnh

- Nguồn component để tiếp tục thử nghiệm: `runs/weights_upload_261114/component_176500.pth`.
- Generator inference của nguồn: `runs/weights_upload_261114/component_176500_generator.pth`.
- Kết thúc 250k: `runs/component_continuation_250k_v1/checkpoint_250000.pth` (cũng là last.pth).
- Best rolling: `runs/component_continuation_250k_v1/best.pth`; không mặc định tốt hơn nguồn.
- [Panel nguồn](component_250k_assets/benchmark_176500.png).
- [Panel 250k](component_250k_assets/benchmark_250000.png).
- [Biểu đồ toàn bộ quá trình](component_250k_assets/progress.png).

## Hướng nghiên cứu tiếp theo — chưa chạy trong tổng hợp này

1. Đánh giá cặp ảnh nguồn/250k bằng OCR độc lập và người đọc, phân biệt lỗi base, modifier, tone và case; kiểm tra riêng ký tự đơn và từ chứa cùng ký tự.
2. Bổ sung tập giám sát thật cho chữ hoa hiếm và ngữ cảnh ký tự đơn; giữ riêng writer/test và đo coverage trước khi train.
3. Từ checkpoint nguồn còn tốt, chạy đối chứng cùng ngân sách, nhiều seed và lịch LR/early stopping trên validation; dành test độc lập cho kết quả cuối.
4. Thử cơ chế composition ở biểu diễn/loss hoặc chuyển tri thức từ từ sang ký tự, so sánh với sampler đơn thuần. Đây là giả thuyết cần ablation, chưa phải kết quả đã có.

## Khả năng tái lập

Code đã được đẩy tại commit `3d048b251bbee8640f047c95b6d223e9fd636c97`. Có REPRODUCE.md, catalog weights/checksum, kiểm tra phục hồi checkpoint và smoke inference. Bộ kiểm tra trước đó ghi nhận 44 tests pass; tổng hợp hôm nay đọc artifact, không chạy lại training hoặc test. Weights nguồn đã gom tại runs/weights_upload_261114 để người dùng tự upload; checkpoint cuối 250k nằm riêng ở run folder như trên.
