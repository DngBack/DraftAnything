# Thử nghiệm cải thiện VATr++ sau 100k

## Đang triển khai

Ba nhánh chạy tuần tự, cùng checkpoint 100k và cùng trạng thái Adam/RNG khởi đầu, mỗi nhánh thêm 10k steps đến tổng 110k. Nguồn 100k được giữ nguyên; SHA-256 ghi trong `runs/ablation_100k_v1/plan.json`.

| Nhánh | LR encoder / G | LR D / writer | OCR weight |
| --- | --- | --- | --- |
| A_control | 5e-6 / 2e-5 | 1e-5 / 2e-5 | 4 |
| B_half_g_lr | 2.5e-6 / 1e-5 | 1e-5 / 2e-5 | 4 |
| C_half_ocr_weight | 5e-6 / 2e-5 | 1e-5 / 2e-5 | 2 |

Writer weight 0.1, clipping 5, OCR frozen, legacy layout, sampling và batch 8/support 15 giữ nguyên. Không đổi lịch cập nhật.

## Thay đổi mã

- `--override-resume-lr` áp dụng LR CLI sau khi nạp optimizer; mặc định resume vẫn giữ LR checkpoint. Thêm `--writer-lr` để nhánh B không vô tình giảm LR writer.
- Giữ `last.pth`, `best.pth` và 3 numbered checkpoints gần nhất. Numbered checkpoint dùng hard link đến file last đã ghi atomically, tránh nhân đôi cùng trọng số trên đĩa.
- `best.pth` chọn theo trung bình CER của 3 lần đánh giá gần nhất trên panel expanded; file là trọng số bước hiện tại, không phải trung bình trọng số. `best_selection.json` ghi rõ tiêu chí.
- Giữ panel cũ 48 mẫu để so lịch sử. Panel expanded gồm 20 prompt trên 30 writer development = 600 mẫu, cố định seed và support, loại prompt khỏi support.
- Ghi CER nhóm uppercase/other và từng writer, giữ toàn bộ dự đoán để phân tích dấu sau này.
- Đánh giá bảo toàn RNG Python/NumPy/PyTorch/CUDA và mode generator, kể cả khi lỗi; không tiêu thụ chuỗi ngẫu nhiên của training.
- Thêm D(real)/D(fake) và std gradient ảnh theo từng loss trước cân bằng.
- Sáu test hiện có/được bổ sung đã pass, gồm khôi phục Adam moments khi override LR và bảo toàn RNG khi evaluation lỗi.

## Theo dõi

- Queue: `runs/ablation_100k_v1/queue_process.json`, `queue.log`.
- Trạng thái nhánh: `status.json`; chỉ một nhánh training chạy tại một thời điểm.
- Log: `<branch>/train.log`, metrics: `<branch>/metrics.jsonl`.
- Benchmark 600 mẫu: `<branch>/expanded/benchmark_*.json` và `.png`.
- Mỗi nhánh kết thúc sẽ cập nhật `comparison.json`. Nếu nhánh lỗi, queue dừng để debug; không tự chuyển tiếp và che lỗi.

## Giới hạn và bước sau

Đây là giai đoạn 1, chưa phải toàn bộ cải tiến đã được chứng minh:

- OCR benchmark vẫn là OCR giám sát gốc, không phải OCR độc lập. Panel lớn hơn không loại bỏ thiên lệch này.
- Chưa có chỉ số style độc lập; cần đọc sample và đối chiếu support, tránh chọn model chỉ vì CER thấp hơn.
- Chưa thay layout, positional encoding, sampling, clip threshold hay mở khóa OCR. Các thay đổi này cần thí nghiệm riêng sau khi có kết quả A/B/C.
- Chưa thêm norm cập nhật tham số và gradient từng nhóm layer; hiện có norm G toàn cục trước clip, gradient theo loss ở ảnh và score D.
- Các nhánh chỉ chạy một seed; kết quả tốt cần lặp lại trước khi chọn cấu hình cuối.
- Không tự chọn nhánh thắng hoặc train thêm 30–50k khi chưa xem kết quả và giới hạn đánh giá.
