# Báo cáo dữ liệu huấn luyện tiếng Việt — VATr-pp

Ngày lập: 11/11/2026. Phạm vi: dữ liệu đã được chuẩn bị tại `files/vietnamese`, cách nạp dữ liệu và benchmark step 5.000 của `runs/vatrpp_vn_finetune`.

## 1. Kết luận chính

Dữ liệu **có dấu tiếng Việt trong nhãn**; pipeline không loại bỏ dấu và sử dụng chuẩn Unicode NFC. Hiện tượng thiếu hoặc sai dấu trong ảnh sinh không chứng minh rằng toàn bộ ảnh nguồn bị thiếu dấu.

Các hạn chế đáng chú ý là ảnh nguồn chỉ cao 32 pixel, số lượng ký tự phân bố không đều và một số ký tự chỉ có trong HAT2023. HAT2023 không có định danh người viết thực dùng cho học phong cách. Vì vậy, độ phủ của bộ ký tự OCR rộng hơn độ phủ của ảnh thật dùng để huấn luyện GAN theo người viết.

## 2. Nguồn dữ liệu gốc

| Nguồn | Số ảnh | Writer ID | Độ rộng ảnh: trung vị / lớn nhất | Độ dài nhãn: trung vị / lớn nhất | Số ký tự phân biệt |
|---|---:|---:|---:|---:|---:|
| VNOnDB | 66.955 | 106 | 48 / 176 px | 3 / 11 | 144 |
| ViHTGen | 42.642 | 189 | 36 / 543 px | 3 / 8 | 157 |
| HAT2023 | 103.000 | 103.000 ID giả, mỗi ảnh một ID | 55 / 245 px | 4 / 17 | 187 |
| Test, chưa xác định nguồn thành phần | 27.257 | 68 ID nội bộ tập test | 48 / 338 px | 3 / 10 | 147 |

Tổng dữ liệu gốc: **212.597 ảnh trong file train** và **27.257 ảnh trong file test**. Toàn bộ ảnh cao **32 px**, độ rộng biến thiên. Đây chủ yếu là ảnh từ hoặc đoạn rất ngắn; chỉ 123 ảnh ViHTGen và 31 ảnh test có khoảng trắng trong nhãn, VNOnDB và HAT2023 không có.

Phân loại nguồn trong script dựa trên writer ID của file train: dưới 106 là VNOnDB, từ 106 đến 294 là ViHTGen, từ 295 trở lên là HAT2023. ID test độc lập; trùng số ID giữa train và test không đủ để kết luận cùng người viết.

## 3. Kiểm tra và chia tập

| Tập sau lọc | Số ảnh | Số writer ID | Vai trò |
|---|---:|---:|---|
| Train | 97.734 | 265 người viết | Ảnh thật và mẫu phong cách cho GAN; dữ liệu OCR |
| Validation | 11.786 | 30 người viết | Tách theo người viết từ VNOnDB và ViHTGen |
| HAT OCR | 102.533 | 102.533 ID giả | Nguồn OCR và kho transcript; không dùng làm mẫu phong cách |
| Test | 27.256 | 68 ID nội bộ | Tập kiểm tra giữ riêng |

Chia validation dùng seed **742**, chọn khoảng 10% người viết của từng nguồn VNOnDB và ViHTGen. Bộ ký tự và kho transcript được xây từ train + HAT OCR, không lấy nhãn validation/test để mở rộng bộ ký tự.

Các bước làm sạch đã áp dụng:

- Kiểm tra cấu trúc HDF5, độ dài nhãn, ảnh rỗng/phẳng, ký tự điều khiển và khả năng căn chỉnh CTC.
- Loại 10 ảnh train gốc không đủ độ dài đầu ra CTC: 2 ViHTGen và 8 HAT2023; loại 1 ảnh test vì cùng lý do.
- Loại ảnh trùng pixel chính xác: 75 ảnh trong train và 406 ảnh trong HAT OCR.
- Phát hiện 26 nhóm ảnh giống nhau nhưng khác nhãn; cách ly 53 mẫu HAT OCR.
- Ưu tiên giữ dữ liệu theo thứ tự test → validation → train → HAT OCR khi xử lý trùng ảnh.

Audit không ghi nhận trùng pixel xuyên tập sau quy trình trên. Tuy nhiên, kiểm tra này không phát hiện đầy đủ ảnh gần giống, ảnh đã resize hoặc cùng nội dung được chụp lại. Chưa có bằng chứng xác nhận tập test độc lập người viết với train ở cấp danh tính thực.

## 4. Độ phủ ký tự và dấu tiếng Việt

Bộ ký tự có **207 ký tự**, bao gồm khoảng trắng. Không có ký tự ngoài bộ ký tự trong validation và test. Không nhãn nào trong audit cần thay đổi khi chuẩn hóa NFC.

Ví dụ số lần xuất hiện ký tự trong **nhãn** sau lọc; đây không phải số ảnh hay số từ:

| Ký tự | Train dùng học phong cách | HAT OCR |
|---|---:|---:|
| a | 9.114 | 15.078 |
| á | 4.620 | 4.581 |
| à | 6.833 | 3.326 |
| ả | 2.388 | 1.691 |
| ã | 1.113 | 528 |
| ạ | 2.471 | 1.272 |
| ẵ | 14 | 51 |
| ỵ | 0 | 66 |
| Ễ | 0 | 100 |
| Ỹ | 0 | 13 |

Có **44 ký tự chỉ xuất hiện trong HAT OCR**, không có trong nhãn train dùng học phong cách:

```text
Z Ã È É Ê Ì Ò Õ Ù Ý Ũ Ạ Ẩ Ẫ Ắ Ằ Ẳ Ẵ Ặ Ẹ Ẻ Ẽ Ế Ễ Ỉ Ị Ỏ Ổ Ỗ Ộ Ớ Ờ Ỡ Ợ Ụ Ứ Ừ Ữ Ự Ỳ Ỵ ỵ Ỷ Ỹ
```

Điều này đặc biệt liên quan đến benchmark `NGUYỄN` và `Ỹ`: nhãn train phong cách không có `Ễ` và `Ỹ`. OCR/kho transcript có thể biết những ký tự đó, nhưng GAN thiếu ví dụ ảnh thật tương ứng trong tập phong cách. Đây là một hạn chế dữ liệu, chưa đủ để quy toàn bộ lỗi sinh ảnh cho một nguyên nhân duy nhất.

## 5. Cách dữ liệu đi vào mô hình

- Ảnh OCR giữ độ rộng tự nhiên, chuẩn hóa pixel về khoảng [-1, 1].
- Ảnh thật cho GAN được resize thành chiều cao 32 px, chiều rộng bằng 16 × số codepoint trong nhãn NFC.
- Mỗi mẫu GAN lấy **15 ảnh hỗ trợ cùng người viết**, ưu tiên transcript khác ảnh đích. Khi không đủ ứng viên, lấy có lặp.
- Ảnh hỗ trợ rộng hơn 192 px được nén ngang toàn ảnh; ảnh ngắn hơn được đệm nền trắng. Không cắt bỏ phần cuối ảnh.
- Sampler OCR cân bằng trọng số giữa các nguồn. Sampler GAN cân bằng nguồn và người viết trong nguồn; không lấy thuần theo tỷ lệ số ảnh gốc.
- Bộ lấy transcript kết hợp lấy ngẫu nhiên với ưu tiên ký tự hiếm; cấu hình hiện tại có `rare_fraction = 0.3`.
- Cấu hình GAN hiện tại: batch 8, OCR đóng băng, trọng số OCR 4, trọng số writer 0,1; mục tiêu 50.000 step.

HAT không được dùng làm mẫu phong cách hoặc ảnh thật GAN theo người viết. Không nên hiểu con số 212.597 ảnh train gốc là toàn bộ số ảnh thật GAN đang sử dụng.

## 6. Rủi ro về độ phân giải và vùng dấu

Audit ghi nhận số ảnh có mực ở sát biên trên/dưới:

| Nguồn | Biên trên | Biên dưới |
|---|---:|---:|
| VNOnDB | 1.722 / 66.955 (2,57%) | 1.154 / 66.955 (1,72%) |
| ViHTGen | 6.060 / 42.642 (14,21%) | 8.558 / 42.642 (20,07%) |
| HAT2023 | 61.186 / 103.000 (59,40%) | 65.628 / 103.000 (63,72%) |
| Test | 1.622 / 27.257 (5,95%) | 2.003 / 27.257 (7,35%) |

Tiêu chí là hơn 2% pixel ở hàng biên có giá trị dưới 128. **Chỉ số này không chứng minh dấu bị cắt**: nền tối hoặc nhiễu cũng kích hoạt tiêu chí. Cần kiểm tra ảnh nguồn bằng mắt để xác nhận. Ảnh đã ở độ cao 32 px; phóng lớn lại không phục hồi chi tiết dấu đã mất trước đó.

## 7. Đối chiếu benchmark step 5.000

Benchmark gồm 48 ảnh, tương ứng 12 nội dung × 4 writer: 35, 38, 111, 113.

| Chỉ số | Giá trị |
|---|---:|
| CER có tính dấu | 30,15% |
| CER sau bỏ dấu | 9,56% |
| Tỷ lệ transcript khớp hoàn toàn | 27,08% |

Một số kết quả OCR: `Tiếng → Tiêng`, `rất → rát`, `mạ → ma`, `Ỹ → Y`. Ảnh benchmark cũng cho thấy dấu thiếu hoặc chưa rõ. Chênh lệch CER cho thấy dấu và biến thể chữ tiếng Việt là điểm yếu đáng chú ý, nhưng **không phải tỷ lệ lỗi dấu thuần túy**.

OCR dùng đo benchmark cũng là OCR hướng dẫn nội dung trong quá trình train. Vì vậy, các chỉ số trên **không phải đánh giá chất lượng độc lập** và OCR có thể đọc sai cả khi ảnh có nét dấu. Step 5.000 chỉ là 10% mục tiêu của lượt train; chưa thể suy ra chất lượng cuối cùng hoặc bảo đảm chạy đủ step sẽ khắc phục lỗi.

![Benchmark step 5.000](runs/vatrpp_vn_finetune/benchmark_005000.png)

## 8. Đề xuất tiếp theo

1. Kiểm tra thủ công ảnh nguồn của các nhóm `ma/má/mà/mả/mã/mạ`, chữ có hai tầng dấu và chữ hoa hiếm để phân biệt lỗi ảnh, lỗi nhãn và lỗi OCR.
2. Bổ sung ảnh có writer ID thực cho các ký tự thiếu trong tập phong cách, ưu tiên `Ễ`, `Ỹ` và các chữ hoa có dấu phục vụ tên riêng.
3. Theo dõi cùng bộ benchmark ở các mốc sau; dùng OCR độc lập và đánh giá bằng mắt trước khi kết luận chất lượng đã đạt.
4. Nếu còn dữ liệu ảnh gốc độ phân giải cao, kiểm tra việc crop/resize có giữ đủ khoảng trống trên và dưới dấu trước khi tái tạo bộ dữ liệu.
5. Kiểm tra ảnh gần trùng và nguồn gốc người viết của tập test nếu cần báo cáo khả năng tổng quát hóa nghiêm ngặt.

## 9. Nguồn kiểm chứng trong repository

- [Thống kê audit](files/vietnamese/audit.json)
- [Tần suất ký tự theo nguồn gốc](files/vietnamese/characters.csv)
- [Ảnh mẫu dữ liệu nguồn](files/vietnamese/contact_sheet.png)
- [Manifest chia tập](files/vietnamese/manifest.json)
- [Script chuẩn bị dữ liệu](scripts/prepare_vietnamese.py)
- [Dataset và sampler](data/vietnamese.py)
- [Cấu hình lượt train](runs/vatrpp_vn_finetune/config.json)
- [Kết quả benchmark step 5.000](runs/vatrpp_vn_finetune/benchmark_005000.json)

Báo cáo dựa trên artifact audit và mã nguồn hiện có; không chạy lại audit toàn bộ ảnh hoặc thay đổi dữ liệu/training trong quá trình lập báo cáo.
