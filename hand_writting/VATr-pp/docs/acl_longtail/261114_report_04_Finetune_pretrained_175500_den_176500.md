# Fine-tune VATr++ từ checkpoint pretrained

Nguồn: `/home/bachdx2/DraftAnything/hand_writting/VATr-pp/runs/continuation_200k_v1/best.pth` ở **175,500**; cả hai nhánh hoàn thành **176,500**, mỗi nhánh thêm 1,000 outer steps.

Runtime gốc: **PyTorch 2.9.1+cu128**. Source SHA-256: `99c222d327fa0aec97164f0b8074bd79b784ee7271675b893bc57cc0862e5948`. Checkpoint nguồn được kiểm tra giữ nguyên sau training.

## Xác nhận continuation

- G/D/writer/OCR giữ nguyên toàn bộ tên và shape tensor native; không xây lại generator.
- OCR frozen khớp hash với nguồn ở cả hai nhánh.
- Adam của G/D tăng đúng 2.000 cập nhật và writer tăng đúng 1.000; không reset optimizer.
- Hai benchmark khởi đầu đều khớp artifact nguồn: CER 13,6257%, exact 62,5%.
- Batch/support 8/15; LR encoder/G/D/writer = 5e-6/2e-5/1e-5/2e-5; OCR/writer weight 4/0,1; các loss và kiến trúc giữ nguyên.

## Chỉ thay transcript sampling

`legacy` giữ sampling cũ. `component` trộn 50% trọng số ký tự cũ và 50% trọng số nhóm base/case, modifier/case, tone/case, modifier×tone×case trong lượt rare. Rare fraction vẫn 0,3 và HAT fraction vẫn 0,35. Không thêm ảnh giám sát hoặc HAT writer giả.

| Bước | CER legacy ↓ | CER component ↓ | Exact legacy ↑ | Exact component ↑ |
|---:|---:|---:|---:|---:|
| 175500 | 13.63% | 13.63% | 62.50% | 62.50% |
| 175750 | 13.86% | 13.27% | 61.00% | 62.83% |
| 176000 | 15.91% | 14.04% | 56.67% | 62.17% |
| 176250 | 14.44% | 13.22% | 60.17% | 62.83% |
| 176500 | 17.95% | 12.69% | 52.50% | 65.00% |

| Nhánh cuối | NGUYỄN đúng /30 | Ễ đúng /30 | Ỡ đúng /30 | Ắ đúng /30 |
|---|---:|---:|---:|---:|
| legacy | 27 | 0 | 0 | 0 |
| component | 29 | 0 | 0 | 0 |

## Diễn giải

Nhánh component tốt hơn đối chứng ở cả bốn lần đánh giá sau resume trong pilot này. Tuy nhiên, chỉ một nguồn/seed và 1.000 outer steps; chưa chứng minh lợi ích bền vững, chất lượng thẩm mỹ hay lời giải cho chữ hoa hiếm. OCR benchmark vẫn là proxy có lịch sử chung với GAN, không phải chấm người đọc độc lập.

**Ễ/Ỡ/Ắ đứng riêng vẫn thất bại**. HAT có ảnh chứa chữ nhưng không có transcript một ký tự Ễ/Ỡ; cần tách thiếu glyph khỏi thiếu ngữ cảnh ký tự đơn. Sampling hiện tại không sinh transcript mới. Xem `character_context_audit.json`.

Không diễn giải sampler thành ưu tiên trực tiếp Ễ: xác suất HAT draw chứa Ễ đổi từ 0,13669% sang 0,13337% trong corpus hiện tại. Đây là phân bổ theo thành phần, không đơn thuần oversampling chữ Ễ. Xem `pretrained_sampling_exposure.json`.

Không tự động thay checkpoint nguồn bằng model pilot. Phải so sample cùng writer/support, lặp seed và đánh giá nội dung/dấu/phong cách độc lập trước khi promote. Các thí nghiệm generator nhỏ từ đầu đã được lưu riêng và dừng theo chỉnh hướng của người dùng; không gộp vào kết quả này.

## Artifact

- Trạng thái: `runs/pretrained_longtail_v1/status.json`.
- Model mới: `runs/pretrained_longtail_v1/component/last.pth`.
- Đối chứng: `runs/pretrained_longtail_v1/legacy/last.pth`.
- Panel 600 mẫu: `runs/pretrained_longtail_v1/{legacy,component}/expanded/benchmark_176500.png`.
- Code: `util/vietnamese_longtail.py`, cờ `--text-sampler` trong `train_vietnamese.py` và launcher `scripts/run_pretrained_longtail_pilot.py`.
- Bộ test trên runtime gốc: 39 tests pass; log tại `test_run.log`.
