# OCR đọc đúng nhưng chữ sinh chưa đẹp: phân tích và thử nghiệm cải thiện

## 1. Vấn đề quan sát được

Trong panel pilot ở bước 112.500, từ `NGUYỄN` có thể được OCR đọc đúng dù nét chữ, khoảng cách và cụm dấu nhìn thiếu tự nhiên. Nhận xét trực quan của người dùng chỉ ra khoảng cách giữa **nhận dạng đúng nội dung** và **chất lượng thị giác/phong cách**.

![Panel người dùng phản hồi](../runs/rare_hat_Ễ_pilot_v1/auxiliary/benchmark_112500.png)

OCR CTC tối ưu xác suất transcript, không trực tiếp chấm vẻ tự nhiên của nét viết, sự cân đối khoảng cách, độ hợp lý của cụm dấu hoặc mức giống writer. Vì thế OCR đúng không đồng nghĩa ảnh đẹp. Tuy nhiên, một panel chưa đủ chứng minh generator cố tình khai thác OCR hay OCR loss là nguyên nhân duy nhất của chất lượng kém.

Đánh giá trước đây đặt CER nội bộ làm tiêu chí chọn best và mở run dài là **chưa đầy đủ cho mục tiêu chữ đẹp**. Cần thêm giám sát ảnh và kiểm tra thị giác trước khi chọn cấu hình cuối.

## 2. Bằng chứng từ code và log

### 2.1. Gradient balancing đang ưu tiên nội dung

Trong pha G+OCR:

```text
loss_G_OCR = adversarial + effective_weight × CTC

effective_weight = 4 × clamp(0,7 × std(g_adv) / (std(g_ctc) + 1e-7))
```

Khi chưa chạm clamp và gradient CTC không quá nhỏ so với epsilon, độ lệch chuẩn của gradient CTC sau nhân hệ số xấp xỉ **2,8 lần** gradient adversarial tại tensor ảnh sinh.

Ở pha G+writer, hệ số ngoài là 0,1 nên tỷ lệ tương tự xấp xỉ **0,07 lần**. Hai loss ở hai pha khác nhau, trên các batch text/fake khác nhau; không được diễn giải là OCR ảnh hưởng tham số G chính xác gấp 40 lần writer. Đường truyền qua G, hướng gradient, clipping và Adam còn làm thay đổi tác động cuối.

Trong 100 mốc log cuối pilot:

| Chỉ số trung bình | Auxiliary HAT | Control |
| --- | --- | --- |
| Effective OCR weight | 10,5646 | 14,2763 |
| OCR fake CTC | 0,1949 | 0,1475 |
| Effective writer weight | 0,0949 | 0,0921 |
| Writer fake loss | 4,4166 | 4,6012 |
| Norm G+OCR trước clip | 67,54 | 66,54 |
| Norm G+writer trước clip | 18,05 | 17,56 |
| Trung bình weight × OCR loss | 1,3382 | 1,1793 |
| Trung bình weight × writer loss | 0,4181 | 0,4211 |

Giá trị scalar của các loss khác thang đo và pha cập nhật. Bảng này không đo trực tiếp tỷ lệ ảnh hưởng lên tham số hoặc chất lượng ảnh. Tỷ lệ cân bằng gradient và clipping kéo dài là cơ sở để thiết kế ablation, không phải bằng chứng nhân quả đã hoàn tất.

### 2.2. Pilot HAT chưa giải quyết vấn đề

Cùng nguồn ở bước 107.500, hai nhánh kết thúc ở 112.500. Trên panel expanded 600 mẫu:

| Chỉ số | Auxiliary HAT | Control |
| --- | --- | --- |
| CER tổng ↓ | 13,68% | **12,98%** |
| Đúng toàn từ ↑ | 61,50% | **63,50%** |
| CER của NGUYỄN ↓ | **6,67%** | 8,33% |
| NGUYỄN đúng toàn từ ↑ | **63,33%** | 53,33% |
| Ễ đứng riêng đúng | 0/30 | 0/30 |

Auxiliary giúp OCR đọc `NGUYỄN` tốt hơn ở mốc cuối, nhưng tổng CER/exact kém control và prompt `Ễ` riêng vẫn thất bại. **Chưa có cơ sở gộp auxiliary HAT vào cấu hình dài.** Những số trên là một checkpoint cuối, một seed và OCR nội bộ.

Bài auxiliary trước dùng chính ảnh HAT làm reference rồi tái tạo lại ảnh đó. Nó có thể học sao chép mà không học chuyển ký tự sang phong cách writer thật. Điều này là giới hạn của thiết kế, không được che đi bằng loss tái tạo thấp.

### 2.3. Giới hạn dữ liệu ký tự hiếm

Train writer thật có **0 ảnh chứa `Ễ`**, HAT có 100 ảnh. OCR nhận ra `Ễ` trong 66/100 ảnh HAT nhưng chỉ đúng toàn từ ở 48/100 ảnh. Cụm `NGUYỄN` chứa ký tự `Ễ`, nên giám sát ảnh writer thật hiện cũng không cung cấp trực tiếp mẫu của từ này.

Thêm paired visual loss trên dữ liệu writer thật có thể cải thiện chất lượng hình chữ chung; **không tạo ra dữ liệu writer thật còn thiếu cho `Ễ`**. Nếu chữ hiếm tiếp tục thất bại, cần bổ sung dữ liệu tương ứng hoặc thí nghiệm content/style riêng.

## 3. Định hướng cải thiện

1. **Thêm giám sát ảnh thật cùng transcript và cùng writer:** dùng support khác target và ưu tiên transcript khác; G sinh lại nhãn của target thật để có đích so sánh hình ảnh.
2. **Thử feature matching:** so đặc trưng nhiều mức của ảnh sinh và ảnh thật, thay vì chỉ scalar của D hoặc xác suất OCR. Đây là hướng có cơ sở từ [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498), nhưng cách paired đa mức trong dự án là một thiết kế thử nghiệm, không phải bản sao chính xác công thức mean-feature gốc.
3. **Dùng pixel loss nhẹ trong nhánh riêng:** cùng người viết vẫn có biến thiên nét, nên ép L1 quá mạnh có thể làm nhòe hoặc mất đa dạng. Cần so với feature-only.
4. **Giữ OCR để bảo vệ nội dung**, nhưng không chọn model chỉ theo CER. Nhánh giảm OCR weight trước chưa thắng đối chứng nên không kết luận giảm hệ số là đủ.
5. **Đánh giá chất lượng hình ảnh riêng:** đối chiếu real/generated/support, bổ sung đặc trưng của một evaluator cố định và kiểm tra người đọc. Có thể khảo sát [HWD](https://arxiv.org/abs/2310.20316) sau này; hiện chưa cài hoặc chạy HWD.

## 4. Phần đã triển khai trong code

### 4.1. Paired visual supervision

Các cờ mới, mặc định tắt:

- `--paired-feature-weight`
- `--paired-pixel-weight`
- `--paired-every`
- `--visual-benchmark`

Ở bước auxiliary:

1. Lấy target thật và support cùng writer từ batch GAN hiện có.
2. Sinh đúng transcript của target. Target image không có trong support; loader ưu tiên support có transcript khác.
3. Khóa tham số D, writer và OCR; D ở eval mode để feature extraction không đổi trạng thái mạng.
4. So 3 mức feature cuối của D giữa fake/real. Feature thật detach, mask padding ở từng mức, chuẩn hóa theo biên độ feature thật với mẫu số tối thiểu 0,1.
5. Tùy nhánh, thêm L1 pixel có mask padding; chỉ cập nhật G.

Đây là paired feature loss có căn chỉnh thô theo chiều rộng 16px/ký tự. Nó không bảo đảm căn chỉnh nét hoàn hảo; hai lần viết cùng transcript/cùng writer có thể khác hình dạng. Các hệ số ban đầu là cấu hình pilot, chưa được chứng minh tối ưu.

### 4.2. Đánh giá ảnh thật / ảnh sinh / support

Thêm panel development **60 target = 2 target × 30 writer thật**, chọn cố định theo seed. Mỗi target có support cùng writer, khác ảnh và **khác transcript target**. Target và support chỉ lấy development, không đưa vào loss train.

Ảnh panel có ba cột: **real / generated / support mẫu**. File `references.json` lưu đầy đủ target và 15 support để kiểm tra.

Các chỉ số mới:

- Feature distance tới target thật bằng **D cố định chụp từ checkpoint nguồn 150k**, cùng một nguồn cho ba nhánh. D evaluator không cập nhật theo training.
- Pixel L1 trên cặp thật/sinh.
- CER nội bộ trên cùng 60 target để theo dõi nội dung.
- Vẫn giữ panel expanded 600 mẫu, CER tổng và chỉ số riêng `NGUYỄN`/`Ễ`.

**Feature distance là proxy của D, không phải điểm đẹp do người đọc chấm.** D nguồn từng được train trong GAN nên cũng không phải evaluator style độc lập. Pixel L1 có thể ưu tiên ảnh trung bình/nhòe; không dùng riêng nó để chọn model.

### 4.3. Lưu model và giới hạn lựa chọn

Lưu thêm `best_visual.pth` khi feature distance thấp hơn các mốc đã ghi và CER panel 60 target không vượt baseline đầu nhánh quá **2 điểm phần trăm**. Tiêu chí và các số đo nằm trong `best_visual_selection.json`.

Giữ nguyên các artifact best OCR/focus, last và 3 checkpoint định kỳ. Các best này phục vụ đối chiếu, **không tự động promote model thắng theo OCR hoặc feature distance**. Cần xem sample và so với đối chứng trước khi chạy dài tiếp.

### 4.4. Sửa lỗi fallback OCR benchmark khi resume

Smoke test phát hiện đường fallback nếu output mới chưa có `benchmark_ocr.pth`: bản cũ có thể chụp OCR trước khi nạp resume, dẫn tới dùng mạng ngẫu nhiên để chấm benchmark. Đã chuyển chụp trạng thái OCR sang sau khi khôi phục checkpoint.

Các run A/B/C, pilot HAT và continuation 150k trước đó đã copy sẵn benchmark OCR gốc nên không đi qua fallback lỗi. Smoke `visual_quality_smoke_v1` bị ảnh hưởng, **không dùng CER của nó để phân tích chất lượng**. Smoke v2 và pilot mới dùng đường đã sửa; pilot còn copy OCR benchmark gốc để bảo đảm cùng thước đo.

## 5. Thí nghiệm thực tế được khởi chạy

Run continuation **150.000 đã hoàn thành**. Nguồn chung cho pilot mới là checkpoint cuối của run đó; không dùng trọng số auxiliary HAT.

| Nhánh | Feature weight | Pixel weight | Lịch auxiliary |
| --- | --- | --- | --- |
| feature_only | 1,0 | 0 | Mỗi 4 outer steps |
| feature_pixel | 1,0 | 0,1 | Mỗi 4 outer steps |
| control | 0 | 0 | Không auxiliary |

Mỗi nhánh xuất phát từ **150.000**, chạy thêm **5.000** đến tổng **155.000**. Chạy tuần tự, giữ nguồn 150k nguyên vẹn và ghi SHA-256. Giữ LR, OCR weight 4, writer weight 0,1, clip 5, OCR frozen và kiến trúc như baseline. Không gộp HAT auxiliary.

Nhánh có visual loss có thêm cập nhật G mỗi 4 steps, nên so sánh cùng outer steps **không phải cùng số cập nhật G hay chi phí tính toán**. Thử nghiệm này nhằm kiểm tra lợi ích thực tế của giám sát ảnh; nếu có tín hiệu tốt cần đối chứng thêm theo ngân sách cập nhật và nhiều seed.

Đánh giá mỗi 500 steps. Queue dừng nếu nhánh lỗi và không tự promote.

## 6. Kiểm chứng và điều kiện đánh giá thành công

- **10 unit test pass**, gồm feature loss có gradient tới fake nhưng không tới target, mask padding, support cùng writer không chứa target/target transcript, cùng các test cũ.
- Smoke v2 chạy thật trên GPU từ 150.000 đến 150.004, có bước feature + pixel loss, đánh giá panel và lưu checkpoint. Đã hoàn tất và lưu profile; feature loss = 0,26618, pixel loss = 0,27948, norm G auxiliary trước clip = 0,78944 (xem log để lấy giá trị đầy đủ). OCR benchmark sau resume khớp toàn bộ tensor với OCR nguồn đã lưu.
- Không suy ra cải thiện chất lượng từ việc code chạy được hoặc loss auxiliary hữu hạn.

Một nhánh chỉ đáng mở rộng khi nhiều mốc liên tiếp và sample cho thấy: nét/dấu có cấu trúc tốt hơn, khoảng cách tự nhiên hơn, giữ được khác biệt writer, không chỉ tăng độ nhòe để giảm L1; CER tổng không giảm chất lượng quá mức. Cần kiểm tra trực quan `NGUYỄN` và `Ễ` ngay cả khi OCR đọc đúng.

Chưa có kết quả sau 5k steps tại thời điểm viết thiết kế; không tuyên bố feature matching đã sửa được chữ xấu.

## 7. Artifact theo dõi

- Trạng thái: `runs/visual_quality_pilot_v1/status.json`.
- Log: `runs/visual_quality_pilot_v1/<branch>/train.log`.
- Panel real/generated/support: `<branch>/visual/visual_<step>.png`.
- Số đo và transcript: `<branch>/visual/visual_<step>.json`.
- Sample prompt chung: `<branch>/expanded/benchmark_<step>.png`.
- Bảng tổng hợp: `comparison.json`, cập nhật sau mỗi nhánh hoàn tất.
- Code: `util/vietnamese_visual.py`, `train_vietnamese.py`, `scripts/run_visual_quality_pilot.py`.

Báo cáo này ghi rõ thiết kế và bằng chứng hiện có. Các chỉ số định lượng là tín hiệu sàng lọc; phản hồi thị giác của người dùng vẫn cần thiết để xác nhận mục tiêu “chữ đẹp và tự nhiên”.
