# Nhật ký huấn luyện VATr++ tiếng Việt — 11/11/2026

Tài liệu này cập nhật giai đoạn **training thực sự**, sau các run smoke chỉ
51 bước OCR và 4 outer step GAN. Không coi checkpoint smoke là mô hình sử dụng.

## 1. Nghiên cứu và lựa chọn khởi tạo

Đã đối chiếu lại [repo VATr++ của nhóm tác giả](https://github.com/EDM-Research/VATr-pp)
và [bản phát hành Hugging Face](https://huggingface.co/blowing-up-groundhogs/vatrpp).
Khả năng dùng visual archetype cho ký tự mới giúp giữ cấu trúc G khi đổi charset,
nhưng không chứng minh rằng pretrained tự viết đúng mọi dấu Việt.

Đã tải `model.safetensors` về `files/pretrained/vatrpp_english.safetensors`:

- Revision: `fb73be2be8e96ce70d9cad9675a30667e19ac376`.
- SHA-256: `3b1e4b7cae23652acd5c559117d06ef42fdd5317da2a5e0bc94ea44d8c0eb1ff`.
- Kích thước: 560.965.208 byte.
- Chỉ lấy G, không nạp D/OCR/writer classifier tiếng Anh.
- 234/234 mục state của G khớp tên/shape; missing và unexpected đều rỗng.
- Giữ layout/kiến trúc gốc, không bật `--adapt-generator-layout` trong run này.

Nguồn và checksum có sidecar JSON. Dữ liệu pretrained tiếng Anh có thể giúp
chuyển giao khả năng dựng nét; run hiện tại là fine-tune trên dữ liệu Việt,
chưa phải ablation chứng minh pretrained tốt hơn scratch ở cùng ngân sách.

## 2. OCR tiếng Việt: đã hoàn thành 20.000 bước

Lệnh:

```bash
/home/giaphk/SpiS-GAN/.venv/bin/python -u train_vietnamese.py \
  --stage ocr --output runs/vatrpp_vn_ocr_train \
  --steps 20000 --batch-size 32 --workers 2 --lr 0.0003 \
  --eval-every 1000 --save-every 1000 --log-every 100
```

Đây chỉ là dùng interpreter có thư viện đã cài. Mô hình và trainer đều thuộc
VATr-pp, không sử dụng mạng SpiS-GAN.

Tập train OCR: 200.267 mẫu đã lọc từ VNOnDB, ViHTGen và HAT, cân bằng xác suất
nguồn. Development gồm đủ 11.786 mẫu, 30 writer thật không thuộc train thật.
Không sử dụng test để chọn checkpoint.

| Bước | CER development có dấu |
|---:|---:|
| 1.000 | 53,43% |
| 2.000 | 37,09% |
| 5.000 | 25,13% |
| 9.000 | 19,57% |
| 13.000 | 17,68% |
| 16.000 | 16,74% |
| 20.000 | **16,19%** |

Mốc 20.000:

- CER: `0.16187441519358672`.
- CER bỏ dấu: `0.12879650001264445`.
- Transcript error rate: `0.3852027829628373`.
- Thời gian invocation: khoảng 780 giây trên GPU dùng chung.
- Peak allocated: 922.155.008 byte; peak reserved: 2.332.033.024 byte.

Đây là OCR nội bộ, vẫn có lỗi và chưa phải bộ giám sát hoàn hảo. Không so sánh
trực tiếp với CER 97,16% của smoke như hai phép đo cùng tập: smoke chỉ đo 64 mẫu,
run này đo toàn bộ development.

Artifact:

- [Learning curve](../runs/vatrpp_vn_ocr_train/learning_curve.png).
- [Validation 20k](../runs/vatrpp_vn_ocr_train/validation_020000.json).
- `runs/vatrpp_vn_ocr_train/best.pth`.
- `runs/vatrpp_vn_ocr_train/last.pth`.
- `runs/vatrpp_vn_ocr_train/metrics.jsonl`.

## 3. Fine-tune GAN đã khởi chạy

Mục tiêu cấu hình: 50.000 outer step, không phải cam kết hội tụ ở mốc đó.
Khởi chạy qua `scripts/start_vietnamese_finetune.py`. Script kiểm tra phải có
đánh giá **toàn development** và CER OCR không quá 0,20 trước khi khởi chạy.
0,20 là cổng kỹ thuật lựa chọn cho run này, không phải tiêu chuẩn học thuật.

Cấu hình:

| Tham số | Giá trị |
|---|---|
| G init | VATr++ pretrained, đủ 234 mục state |
| OCR init | OCR Việt tốt nhất ở 20.000 bước |
| Batch / support | 8 / 15 |
| G LR / encoder LR | 2e-5 / 5e-6 |
| D LR | 1e-5 |
| Writer coefficient | 0,1 |
| OCR trong GAN | Freeze, ablation có chủ đích để không làm trôi bộ giám sát |
| Gradient balancing | Bật |
| D crop | Bật |
| Sampling HAT / rare | 0,35 / 0,30 |
| Save / benchmark / preview | Mỗi 500 outer step |
| Giới hạn CUDA allocator | 10 GiB cho tiến trình |
| Đường output | `runs/vatrpp_vn_finetune` |

Lịch vẫn gồm G_OCR → D_OCR → G_W → D_W; do freeze OCR nên không cập nhật trọng
số R trong pha D_OCR. D và W bắt đầu mới theo miền/writer Việt. Mỗi outer step
cập nhật G hai lần, D hai lần và W một lần.

Không dừng/sửa các training job khác trên GPU. Giới hạn allocator không bảo đảm
không OOM nếu GPU dùng chung bị tiến trình khác lấp đầy; phải đọc log thực tế.

Có một lần khởi tạo thất bại do API memory fraction yêu cầu device index cụ thể,
đã sửa từ `cuda` thành CUDA index hiện hành rồi khởi chạy lại trước khi cập nhật
trọng số. Trace lần đầu còn trong đầu `train.log`; xem các dòng cuối và PID mới
trong `process.json` để xác định trạng thái hiện tại.

## 4. Benchmark cố định để theo dõi

Đã thêm `util/vietnamese_benchmark.py`:

- 4 writer development: 35, 38, 111, 113; gồm hai nguồn có metadata writer thật.
- Cùng 15 ảnh support mỗi writer giữa các checkpoint; tránh transcript prompt.
- 12 prompt: `Tiếng`, `Việt`, `rất`, `đẹp`, `ma`, `má`, `mà`, `mả`, `mã`, `mạ`,
  `NGUYỄN`, `Ỹ`.
- 48 ảnh ở mỗi mốc, kèm transcript OCR và panel quan sát.
- Lưu OCR chấm panel cố định để không thay metric khi G thay đổi hoặc resume.

**OCR này cũng là bộ giám sát nội dung của G. Điểm panel là proxy nội bộ,
không phải đánh giá độc lập, không đủ để kết luận chất lượng cuối.** Prompt cuối
cố ý chứa ký tự hiếm khó; panel nhỏ không đại diện toàn phân phối.

Baseline pretrained trước fine-tune:

- Guidance OCR CER: 50,74%.
- Guidance OCR CER bỏ dấu: 34,56%.
- Tỷ lệ transcript đúng hoàn toàn: 8,33%.
- Quan sát: đã có nét chữ và thân chữ, nhưng thiếu/sai nhiều dấu Việt; chưa thể
  gọi là mô hình viết tiếng Việt tốt.

Artifact:

- `runs/vatrpp_vn_finetune/benchmark_000000.png` và `.json`.
- `runs/vatrpp_vn_finetune/benchmark_references.json` và `.png`.
- `runs/vatrpp_vn_finetune/benchmark_ocr.pth`.
- Các mốc sau có tên `benchmark_000500.*`, `benchmark_001000.*`, … khi hoàn thành.

## 5. Cách theo dõi và resume

```bash
tail -n 20 runs/vatrpp_vn_finetune/train.log
cat runs/vatrpp_vn_finetune/process.json
tail -n 5 runs/vatrpp_vn_finetune/metrics.jsonl
```

`process.json` lưu PID đã khởi chạy, không phải heartbeat. Phải kiểm tra process
còn sống và log/step có tăng. Không suy luận training đang chạy chỉ vì PID file
hoặc checkpoint tồn tại.

Launcher dùng `start_new_session=True`, stdin đóng và stdout/stderr ghi file để
job độc lập với phiên tool. Khi cần resume một job đã dừng:

```bash
/home/giaphk/SpiS-GAN/.venv/bin/python scripts/start_vietnamese_finetune.py \
  --steps 50000 --batch-size 8
```

Launcher không khởi chạy trùng khi PID cũ còn sống; nếu có `last.pth` thì resume.
`--steps` là tổng outer step. Không resume bằng checkpoint smoke cũ.

## 6. Trạng thái và điều chưa kết luận

Đã hoàn thành training OCR 20k và bắt đầu fine-tune G pretrained; không còn dừng
ở smoke test. Tiến độ GAN là dữ liệu động trong log, không cố định theo tài liệu.
Phải xem checkpoint/panel mới nhất trước khi báo chất lượng hoặc thời gian còn lại.

Cần tiếp tục kiểm tra: dấu thanh/biến thể nguyên âm, chữ hoa hiếm, dấu nặng,
độ giống writer, background shortcuts và metric bằng OCR độc lập. Chưa có cơ sở
để nói mô hình đã hội tụ hoặc đã khắc phục hoàn toàn distribution shift.

## 7. Kiểm tra sau khi bắt đầu fine-tune

Mốc 500 outer step đã lưu checkpoint đầy đủ (khoảng 1,3 GiB). Job tiếp tục sau
mốc đó với loss hữu hạn. Panel 500 có CER nội bộ 53,68% so với baseline 50,74%,
tỷ lệ transcript đúng tăng từ 8,33% lên 16,67%; các chỉ số chưa đồng thuận về
cải thiện. Quan sát có thêm nét dấu nhưng còn sai nhiều dấu. Không tuyên bố
chất lượng đã tốt lên sau chỉ 500 bước, và không dùng một panel nhỏ để chọn
kết luận về toàn dữ liệu.

Đọc tiến độ hiện tại bằng:

```bash
/home/giaphk/SpiS-GAN/.venv/bin/python scripts/vietnamese_training_status.py
```

Công cụ kiểm tra PID có đúng command training hay không, đọc step log và
benchmark mới nhất, ước tính ETA theo tốc độ gần nhất. ETA phụ thuộc GPU dùng
chung và độ dài ảnh; không xem là cam kết thời gian hội tụ.

## 8. Điều chỉnh sau panel 1.000 bước

Ở mốc 1.000, CER panel nội bộ là 60,29%, tỷ lệ đúng transcript 10,42%: chưa cải
thiện so với pretrained baseline. Quan sát vẫn có hình chữ nhưng sai dấu và
chữ hoa hiếm. Không dùng việc training chạy được để kết luận chất lượng tốt.

Đã dừng riêng trainer của run này và resume từ `last.pth` với `--ocr-weight 4`
(thay 1) để tăng ưu tiên nội dung so với adversarial gradient. Giữ nguyên G,
OCR cố định, writer coefficient 0,1, learning rate và dữ liệu. Đây là thay đổi
hyperparameter dựa trên development panel, không phải kiến trúc mới hoặc kết
quả cải thiện đã được chứng minh. Log giữ cả các invocation; xem timestamp,
`process.json` và step sau resume thay vì giả định `seconds` luôn cộng dồn.

Các checkpoint/panel cũ được giữ. Checkpoint chỉ lưu mỗi 500 outer step nên
những bước sau checkpoint cuối trước khi dừng có thể phải chạy lại. Benchmark
sau điều chỉnh tiếp tục dùng đúng cùng reference, prompt và OCR để so sánh.

### Checkpoint thực tế khi đổi trọng số

Checkpoint được lưu cuối cùng trước restart là **1.500 outer step**; những
bước sau đó tới khoảng 1.650 chưa lưu và đã chạy lại. Panel tại 1.500 có CER
nội bộ 46,32%, CER bỏ dấu 28,68%, transcript đúng 16,67%. Đây là trọng số **trước**
điều chỉnh OCR coefficient, không được quy kết kết quả này cho coefficient 4.
Resume với coefficient 4 bắt đầu ở bước 1.501. Panel tiếp theo sau điều chỉnh
sẽ ở mốc 2.000 theo cấu hình. Trong panel 1.500 một số mục như `đẹp`, `má`, `mà`
được OCR đọc đúng ở một số writer, nhưng các nhóm dấu/chữ hoa vẫn lỗi nhiều.

## 9. Snapshot sau 2.500 outer step

Sau khi tăng ưu tiên OCR, run đã tới mốc 2.500 và tiếp tục chạy nền. Trên đúng
panel 48 ảnh cố định:

| Mốc | Guidance OCR CER | CER bỏ dấu | Transcript đúng |
|---:|---:|---:|---:|
| Pretrained, bước 0 | 50,74% | 34,56% | 8,33% |
| 1.500, trước đổi weight | 46,32% | 28,68% | 16,67% |
| 2.500 | 39,71% | 22,06% | 18,75% |

Điểm nội bộ đã giảm so với baseline nhưng ảnh vẫn sai nhiều dấu và chữ hoa.
Đây chưa phải kết quả độc lập hay kết luận hội tụ. Không xác lập quan hệ nhân
quả riêng cho coefficient 4 vì run không có nhánh đối chứng cùng ngân sách.

Xem [panel 2.500](../runs/vatrpp_vn_finetune/benchmark_002500.png) và
[JSON dự đoán](../runs/vatrpp_vn_finetune/benchmark_002500.json). Tốc độ gần nhất
khoảng 0,63 giây/outer step; ETA tới 50k lúc snapshot khoảng 8,3 giờ, phụ thuộc
GPU dùng chung và không phải thời gian bảo đảm đạt chất lượng.
