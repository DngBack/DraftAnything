# Báo cáo cải tiến đánh giá và tiến độ triển khai sau fine-tuning 100k

**Thời điểm chốt tiến độ: 12/11/2026, 08:38 UTC.** Đây là snapshot; xem các file trạng thái được liên kết cuối tài liệu để biết tiến độ mới nhất.

Sau run 100k, các vấn đề cần xử lý là chất lượng dao động, không lưu riêng trọng số mốc tốt nhất, panel đánh giá nhỏ và gradient bị clip kéo dài. Kết quả run gốc được trình bày riêng tại [báo cáo fine-tuning 100k](261112_report_03_Ket_qua_finetune_100k.md).

Tài liệu này phân biệt những phần đã sửa trong code, thử nghiệm đang chạy và những cải tiến chưa được triển khai hoặc chưa được xác nhận về chất lượng.

## 1. Những cải tiến đã triển khai

| Cải tiến | Hành vi mới | Trạng thái |
| --- | --- | --- |
| Resume LR rõ ràng | `--override-resume-lr` áp dụng LR CLI sau nạp optimizer, giữ Adam moments/step counters | Đã triển khai và kiểm thử |
| LR writer riêng | `--writer-lr` giúp giảm LR G mà không vô tình giảm LR writer | Đã dùng trong A/B/C |
| Lưu checkpoint | Giữ `last`, `best` và 3 numbered checkpoint gần nhất | Đang hoạt động |
| Chọn best ít nhạy với một mốc | Xếp hạng bằng CER trung bình 3 lần đánh giá gần nhất | Đang hoạt động; vẫn là proxy |
| Panel mở rộng | 20 prompt × toàn bộ 30 writer development = 600 mẫu | Đang đánh giá mỗi 500 steps |
| Phân tích nhóm | CER uppercase/other, CER và exact rate từng writer, đầy đủ transcript | Đang lưu JSON |
| Log D và gradient | D(real), D(fake), std gradient ảnh của từng loss, effective weights, norm G trước clip | Đang ghi |
| Bảo toàn RNG khi đánh giá | Khôi phục RNG Python/NumPy/PyTorch/CUDA và mode G, kể cả khi evaluation lỗi | Đã kiểm thử |
| Queue thử nghiệm | A/B/C chạy lần lượt, dừng queue nếu có nhánh lỗi | Đang hoạt động |
| Chạy tiếp có điều kiện | Chỉ mở run đến 150k khi vượt các điều kiện định lượng đã định trước | Đang chờ A/B/C |

`best.pth` là trọng số của bước hiện tại khi cửa sổ CER đạt tốt nhất, **không phải model trung bình trọng số**. Checkpoint định kỳ dùng hard link cho file last đã lưu atomically để tránh nhân đôi cùng dữ liệu trên đĩa.

**6 unit test đã pass**, gồm các bất biến OCR/data cũ, bảo toàn Adam moments khi đổi LR và bảo toàn RNG khi benchmark lỗi. Ngoài ra đã kiểm tra logic chọn nhánh: nhánh cải thiện được chọn, nhánh không cải thiện không tự kích hoạt continuation. Đây là kiểm tra tính đúng đắn của code, không chứng minh chất lượng model tăng.

## 2. Các nhánh đang chạy và kết quả tạm thời

Cả ba nhánh cùng xuất phát từ checkpoint **100k**, cùng seed và trạng thái Adam/RNG ban đầu. Mỗi nhánh thêm 10k steps đến tổng **110k**. Checkpoint nguồn được giữ nguyên, có SHA-256 trong plan.

| Nhánh | LR encoder / G | LR D / writer | OCR weight | Trạng thái tại 08:38 UTC |
| --- | --- | --- | --- | --- |
| A_control | 5e-6 / 2e-5 | 1e-5 / 2e-5 | 4 | Đang chạy; log gần nhất **109.450/110.000** |
| B_half_g_lr | 2,5e-6 / 1e-5 | 1e-5 / 2e-5 | 4 | Chờ A |
| C_half_ocr_weight | 5e-6 / 2e-5 | 1e-5 / 2e-5 | 2 | Chờ B |

Clipping 5, writer weight 0,1, OCR frozen, legacy layout, sampling, batch và support giữ nguyên. Không thay nhiều yếu tố đồng thời trong một nhánh.

### Kết quả tạm thời nhánh A trên panel 600 mẫu

| Mốc | CER | Đúng toàn từ |
| --- | --- | --- |
| Khởi đầu 100.000 | 17,84% | 52,17% |
| 109.000 | 14,39% | 59,17% |

Best theo cửa sổ 3 lần đánh giá hiện ở **107.500**: CER trung bình cửa sổ **14,37%**, CER của riêng bước đó **14,39%**. Đây chỉ là snapshot; chưa có kết quả B/C để kết luận thay đổi cấu hình nào tốt hơn đối chứng.

**Không so trực tiếp CER panel 600 mẫu với panel 48 mẫu như cùng một tập.** Panel expanded thêm nhiều writer và chữ hoa hiếm nên khó hơn.

## 3. Tại sao thấy triển khai chậm hoặc log ngắt quãng?

Hiện chỉ một nhánh train tại một thời điểm. B và C chờ theo thiết kế, không phải ba tiến trình cùng chạy. Chạy tuần tự giúp hạn chế tranh chấp tài nguyên trong các thí nghiệm.

Ở snapshot 08:38 UTC, đo từ log nhánh A:

| Đo thời gian | Kết quả |
| --- | --- |
| 50 steps thông thường, median trên 171 khoảng | **11,38 giây** |
| Khoảng 50 steps có xen benchmark/save, median trên 18 khoảng | **24,04 giây** |
| Phần tăng thêm ước tính ở khoảng có đánh giá/save | **12,67 giây** |
| Tổng thời gian đã ghi / số steps đã thêm | **0,255 giây/step** |
| Lần resume 54k→100k trước đó, gồm cả phần kết thúc | **0,336 giây/step** |

Khoảng có benchmark/save bao gồm cả tính đánh giá, render ảnh và ghi checkpoint; không quy toàn bộ phần tăng thêm cho panel 600 mẫu. Đo hiện tại và run trước diễn ra trong điều kiện tài nguyên khác nhau, không phải benchmark tốc độ có kiểm soát. **Số đo hiện tại chưa cho thấy tốc độ trung bình chậm hơn run trước**, dù log có khoảng im lâu hơn do đánh giá/lưu file.

Log ghi mỗi 50 steps, không in mỗi step. `queue.log` và `watcher.log` có thể trống vì tiến trình chỉ đang chờ; loss được ghi vào `A_control/train.log`, sau đó là B/C. Có thể theo dõi trực tiếp bằng:

```bash
cd /home/bachdx2/DraftAnything/hand_writting/VATr-pp
tail -f runs/ablation_100k_v1/A_control/train.log
```

Chưa tự đổi tần suất đánh giá/lưu file giữa A/B/C vì sẽ làm lịch đánh giá không đồng nhất. Nếu cần tối ưu thời gian ở đợt sau, có thể giữ ghi JSON chỉ số định kỳ nhưng render ít panel ảnh hơn; phải áp dụng cùng cấu hình cho các nhánh so sánh.

## 4. Điều kiện tự chạy tiếp đến tổng 150k

Sau khi A/B/C hoàn thành thành công, chương trình xét trung bình **5 lần đánh giá expanded cuối**:

1. CER giảm ít nhất 10% tương đối so với baseline expanded ban đầu.
2. Exact transcript rate tăng ít nhất 3 điểm phần trăm.
3. CER chữ hoa không kém baseline ban đầu.
4. CER bước cuối không kém baseline; checkpoint best dự kiến dùng phải đạt giảm ít nhất 10% so với baseline.
5. Với B/C: CER trung bình phải tốt hơn A ít nhất 5% tương đối; exact và CER chữ hoa không kém A; độ dao động CER không quá ngưỡng `max(0,005; 1,25 × std của A)`.

Trong các nhánh đạt điều kiện, chọn CER trung bình cuối thấp nhất và resume từ `best.pth` của nhánh đó đến **tổng 150.000 steps**. Nếu không nhánh nào đạt, không tự chạy thêm. Nếu một trial lỗi, không tự tiếp tục để che lỗi.

**Các điều kiện này vẫn chỉ dùng OCR giám sát.** Chúng chưa xác nhận bằng người đọc, OCR độc lập hay style metric. Không có quy tắc tự động đánh giá thị giác nào đã được triển khai để bảo đảm sample giữ phong cách tốt hơn.

## 5. Các phần chưa triển khai hoặc chưa được xác nhận

- Chưa có OCR độc lập và chỉ số style độc lập; chưa đo được chất lượng tổng quát chỉ từ CER nội bộ.
- Chưa train lại OCR, thay sampling tiếng Việt, thay gradient clip hoặc sửa style layout/positional encoding.
- Chưa bổ sung norm cập nhật tham số thực tế và gradient từng nhóm layer; hiện mới có norm G và gradient ảnh theo loss.
- Chưa chứng minh cấu hình B/C tốt hơn A; cần chờ kết quả.
- Mỗi nhánh chỉ một seed, cần lặp lại cấu hình tốt để kiểm tra độ tin cậy.
- Chưa bắt đầu run 150k tại thời điểm chốt báo cáo.

Các thay đổi kiến trúc, OCR và sampling được để thành thí nghiệm riêng sau A/B/C để tránh mất khả năng xác định nguyên nhân cải thiện.

## 6. Artifact và tài liệu theo dõi

- [Kết quả fine-tuning 100k đã hoàn thành](261112_report_03_Ket_qua_finetune_100k.md).
- [Trạng thái A/B/C đang cập nhật](../runs/ablation_100k_v1/status.json).
- [Log nhánh A](../runs/ablation_100k_v1/A_control/train.log).
- [Ảnh expanded 600 mẫu ở bước 109.000](../runs/ablation_100k_v1/A_control/expanded/benchmark_109000.png).
- [Best selection nhánh A](../runs/ablation_100k_v1/A_control/best_selection.json).
- [Trạng thái chờ/chạy 150k](../runs/continuation_150k_v1/status.json).
- [Kế hoạch A/B/C](261112_report_05_Thiet_ke_ablation_sau_100k.md).

`comparison.json` được tạo/cập nhật sau khi mỗi nhánh kết thúc; `continuation_150k_v1/decision.json` chỉ xuất hiện sau khi đủ kết quả để chọn nhánh. Các trạng thái trực tiếp có thể mới hơn snapshot trong tài liệu này.


## Bổ sung: thử nghiệm giám sát ảnh HAT cho chữ Ễ

Sau snapshot A/B/C ở trên, đã triển khai pilot riêng dùng 69 ảnh HAT train và 31 ảnh giữ riêng cho tác vụ tái tạo từ tham chiếu đơn. Không dùng pseudo writer ID để giám sát writer head. Hai nhánh auxiliary/control chạy từ 107.500 đến 112.500, không tự động thay thế run 150k. Xem [thiết kế, kiểm chứng và cách theo dõi pilot chữ Ễ](261112_report_06_Thu_nghiem_bo_sung_chu_E_mu_nga.md).
