# Báo cáo các cải tiến đã triển khai và thí nghiệm đang chạy trên VATr++

**Mã report:** 261114_report_06. **Chốt số liệu:** 14/11/2026 12:49:14 UTC.

Đây là snapshot đối chiếu code, checkpoint metadata, cấu hình, benchmark và log thực tế. Training vẫn tiếp tục; số liệu mới xem ở [report 05 tự cập nhật](261114_report_05_Continuation_component_den_250k.md). [Snapshot dữ liệu của report này](report_06_assets/snapshot.json).

## 1. Kết luận chính

**Cải tiến thuật toán đang được áp dụng trong run 250k là cách lấy transcript có xét thành phần chữ, cặp dấu và hoa/thường. Generator vẫn là VATr++ pretrained gốc.**

Nhánh mới đã được fine-tune đối chứng từ 175.500 đến 176.500. Tại 176.500, CER proxy là **12,69%**, so với **17,95%** của nhánh sampling cũ và **13,63%** ở nguồn chung. Theo yêu cầu người dùng, nhánh mới đang được train tiếp từ 176.500 lên **250.000 tổng outer steps**.

Tại thời điểm chốt, trạng thái supervisor là **running**, PID trainer còn tồn tại: **có**, log mới nhất ở **181,250**. Benchmark hoàn thành gần nhất ở **181,000**. **Chưa hoàn thành 250k và chưa giải quyết được chữ Ễ đứng riêng.**

## 2. Bài toán và bằng chứng dữ liệu

Mục tiêu là sinh đúng nội dung tiếng Việt, giữ được dấu và phong cách writer, đặc biệt khi chữ hoa có dấu rất ít ảnh. Ví dụ `Ễ` gồm thân E, dấu mũ và dấu ngã. Cần phân biệt ba loại khó khăn: thiếu ảnh nguyên glyph, thiếu bằng chứng về cách ghép dấu trên chữ hoa, và thiếu cách writer mục tiêu viết tổ hợp đó.

Audit metadata xuất trực tiếp từ HDF5 cho thấy:

| Ký tự | Ảnh train writer thật có chứa | Ảnh HAT có chứa | Ảnh HAT có transcript đúng một ký tự đó |
|---|---:|---:|---:|
| Ễ | 0 | 100 | 0 |
| Ỡ | 0 | 103 | 0 |
| Ẵ | 0 | 50 | 0 |
| Ê | 0 | 961 | 4 |
| ễ | 166 | 188 | 0 |
| ỡ | 57 | 175 | 1 |
| ẵ | 14 | 51 | 0 |

Đơn vị trong bảng là **ảnh**, không phải số lần code point xuất hiện. Một ảnh có chữ lặp không được tính thành nhiều ảnh. Nguồn: [audit glyph và ngữ cảnh](character_context_audit.json).

Điểm mới của chẩn đoán là phân biệt **chữ nằm trong từ** với **chữ đứng riêng**. `NGUYỄN` đọc đúng không tự chứng minh `Ễ` đứng riêng sinh tốt. Prompt một ký tự có thể nằm ngoài độ phủ transcript huấn luyện; đồng thời OCR cũng có thể phụ thuộc ngữ cảnh. Các số đếm chưa xác định được nguyên nhân duy nhất.

HAT có ảnh chữ hiếm nhưng không có nhóm writer đáng tin cậy. Run chính tiếp tục dùng HAT cho corpus nội dung và lịch sử OCR, không biến ID từng ảnh HAT thành writer hoặc support cùng người viết.

## 3. Bản đồ cải tiến và trạng thái triển khai

| Hạng mục | Đã viết / đã kiểm chứng | Có dùng trong run 250k hiện tại? | Mục tiêu |
|---|---|---|---|
| Component-aware transcript sampler | Đã cài đặt, kiểm thử và chạy pilot đối chứng | **Có** | Phân bổ lượt học theo thân chữ, dấu, hoa/thường và cặp tương tác |
| Fine-tune từ checkpoint native | Đã xác minh trọng số, shape, OCR frozen và bước Adam | **Có** | Giữ năng lực đã học của G/D/writer/OCR |
| Audit long-tail theo ảnh, writer, thành phần | Có exporter và số đếm metadata/pixel hash | Dùng làm bằng chứng thiết kế; không thay loader native | Tránh nhầm số lần ký tự với bằng chứng hình ảnh/phong cách |
| Audit ngữ cảnh chữ đơn/trong từ | Đã xuất bảng số liệu | Phân tích, chưa thêm loss/query augmentation | Tách lỗi thiếu dữ liệu khỏi lệch phân phối ngữ cảnh |
| Quản lý job, checkpoint, panel và report | Supervisor và reporter đang hoạt động | **Có** | Theo dõi quá trình dài, giữ khả năng truy nguồn |
| Query adapter VietCompose | Có prototype và pilot trước đó; chưa thắng nguồn | **Không** | Thử biểu diễn thân/modifier/tone/case ở query |
| Renderer tách các lớp mực | Có module, smoke và pilot exploratory | **Không** | Kiểm soát thân chữ, hình dấu và vị trí dấu |
| Factorized CTC và các control recognizer | Có code và thí nghiệm exploratory riêng | **Không** | Khảo sát chia sẻ thành phần trong nhận dạng/giám sát |
| Exposure policy và bộ lọc zero-shot | Có module và kiểm thử trên nhánh sạch | **Không** trong mainline có exposure cũ | Chuẩn bị thí nghiệm khái quát hóa tổ hợp có kiểm soát |
| Memory thành phần theo writer, giám sát anchor/mask thật | Mới có thiết kế nghiên cứu | **Chưa** | Chuyển phong cách dấu từ bằng chứng của writer |

Các module prototype không được hiểu là đã tích hợp vào run đang chạy. Các mạng ResNet/Transformer/CNN, Unifont và bộ loss GAN/OCR/writer native là nền tảng kế thừa; phần mới của mainline hiện tại nằm ở **sampling transcript** và hạ tầng kiểm chứng/theo dõi.

## 4. Thuật toán sampling đang dùng

### 4.1. Đường cơ sở

`CorpusSampler` cũ lấy transcript thật từ corpus train, trộn hai cách lấy mẫu:

- 70% lấy ngẫu nhiên đều theo các phần tử corpus.
- 30% lấy theo trọng số ưu tiên ký tự hiếm.

Corpus chứa một phần tử cho mỗi nhãn ảnh hợp lệ, nên transcript trùng nhau có thể xuất hiện nhiều lần. Đây không phải lấy đều trên một từ điển các từ khác nhau.

Với N phần tử corpus, U(t) là tập ký tự khác nhau trong transcript t, và f(c) là tổng số lần ký tự xuất hiện trong corpus, trọng số cũ là:

```math
w_char(t) = min(20, mean_{c in U(t)} sqrt(N / f(c)))
```

### 4.2. Bổ sung nhóm thành phần và tương tác

Mỗi ký tự được phân tích thành `(base, modifier, tone, case)` để **tính trọng số sampling**. Ví dụ:

| Ký tự | Base | Modifier | Tone | Case |
|---|---|---|---|---|
| E | e | none | none | upper |
| Ê | e | circumflex | none | upper |
| Ễ | e | circumflex | tilde | upper |
| ễ | e | circumflex | tilde | lower |

Với `Ễ`, các nhóm dùng trong code là:

```text
base:e|upper
shape:circumflex|upper
tone:tilde|upper
pair:circumflex|tilde|upper
```

Một transcript nhận hợp các nhóm của mọi ký tự. Mỗi nhóm chỉ được đếm một lần trong một phần tử corpus, kể cả khi nhiều ký tự chia sẻ nhóm đó. Gọi G(t) là hợp nhóm, n(g) là số phần tử corpus có chứa nhóm g:

```math
w_component(t) = min(20, mean_{g in G(t)} sqrt(N / n(g)))
w_new(t) = 0.5 * w_char(t) + 0.5 * w_component(t)
```

Trong lượt rare, xác suất chọn một phần tử transcript tỷ lệ với `w_new`. Công thức tổng cho một phần tử corpus là:

```math
P(t | corpus) = 0.7 / N + 0.3 * w_new(t) / sum_u w_new(u)
```

Xác suất của một chuỗi văn bản bằng tổng xác suất các phần tử corpus mang chuỗi đó. Chọn nguồn vẫn giữ kỳ vọng **65% real-writer corpus / 35% HAT corpus**, không ép tỷ lệ chính xác trong từng batch.

### 4.3. Mục tiêu và giới hạn của thay đổi

Giả thuyết cần kiểm tra: phân bổ lượt học theo các phần chung và tương tác sẽ có ích hơn chỉ nhìn tần suất nguyên ký tự, khi nhiều chữ hoa có dấu cùng chia sẻ cấu trúc.

**Parser thành phần chưa trở thành nhánh sinh ảnh mới.** Generator vẫn nhận query Unifont nguyên ký tự như native VATr++. Không có ba token thân–mũ–ngã mới được đưa vào forward của run này; không có loss riêng cho dấu hoặc cơ chế bắt buộc học ghép dấu.

Phương pháp cũng không bảo đảm tăng xác suất của mọi chữ hiếm. Audit xác suất trong HAT cho `Ễ` cho thấy tỷ lệ transcript draw có chứa chữ này đổi từ **0,13669%** ở sampler cũ thành **0,13337%** ở sampler mới. Do đó không gọi thay đổi này là oversampling trực tiếp `Ễ`. Đây là thay đổi ưu tiên theo cấu trúc; đóng góp thực nghiệm phải được kiểm tra riêng. [Bảng xác suất](pretrained_sampling_exposure.json).

Code: [sampler mới](../../util/vietnamese_longtail.py), [nhóm thành phần](../../research/longtail/protocol.py), [sampler gốc](../../data/vietnamese.py), [trainer native](../../train_vietnamese.py).

## 5. Tích hợp vào pretrained VATr++

Chuỗi checkpoint của mainline:

```text
continuation_200k_v1/best.pth ở 175.500
    ├── sampling cũ       → 176.500, đối chứng
    └── sampling mới      → 176.500, pilot
                              └── tiếp tục sampling mới → mục tiêu 250.000
```

Báo cáo 200k khuyến nghị checkpoint best ở 175.500; tên thư mục 200k không có nghĩa mọi checkpoint bên trong đều ở bước 200.000.

Các kiểm tra đã hoàn thành ở pilot:

1. Tên và shape tensor G/D/writer/OCR giữ nguyên so nguồn.
2. Hash OCR frozen khớp nguồn sau training.
3. Bộ đếm Adam tăng đúng số cập nhật; không reset optimizer.
4. Benchmark khởi đầu cả hai nhánh khớp bản lưu của nguồn: CER 13,6257%, exact 62,5%.
5. Source checkpoint không bị thay thế; runtime gốc PyTorch 2.9.1+cu128 được dùng lại.

Đây là fine-tune có lịch sử exposure cũ, không gọi là zero-shot. RNG được khôi phục nhưng iterator dữ liệu khởi động lại; không khẳng định bit-exact so với một run chưa từng gián đoạn.

## 6. Kết quả pilot đã hoàn thành

Cùng nguồn, batch/support, LR, loss và ngân sách 1.000 outer steps. Chỉ thay sampler trong hai nhánh.

| Bước | CER sampling cũ ↓ | CER sampling mới ↓ | Exact cũ ↑ | Exact mới ↑ |
|---:|---:|---:|---:|---:|
| 175.500 | 13,63% | 13,63% | 62,50% | 62,50% |
| 175.750 | 13,86% | 13,27% | 61,00% | 62,83% |
| 176.000 | 15,91% | 14,04% | 56,67% | 62,17% |
| 176.250 | 14,44% | 13,22% | 60,17% | 62,83% |
| 176.500 | 17,95% | 12,69% | 52,50% | 65,00% |

Ở cuối pilot, nhánh mới giảm **5,26 điểm phần trăm CER** so đối chứng, tăng **12,5 điểm exact**; so nguồn chung, CER giảm khoảng **0,94 điểm**. Nhánh mới tốt hơn đối chứng tại cả bốn lần đánh giá sau resume trong pilot này.

Tuy nhiên, đây là **một seed, một checkpoint nguồn và một ngân sách ngắn**. Các mốc liên tiếp không phải các lần chạy độc lập. Chưa đủ kết luận hiệu quả bền vững hoặc ưu thế thuật toán tổng quát.

Tại 176.500: `NGUYỄN` đúng 29/30 writer ở nhánh mới và 27/30 ở đối chứng; `Ễ`, `Ỡ`, `Ắ` đứng riêng vẫn 0/30 ở cả hai. CER/Exact lấy từ OCR proxy; chưa phải điểm đẹp hay đánh giá phong cách độc lập.

Xem [report 04 và xác minh optimizer](261114_report_04_Finetune_pretrained_175500_den_176500.md).

## 7. Run dài đang chạy: 176.500 → 250.000

| Thành phần | Cấu hình thực tế |
|---|---|
| Checkpoint nguồn | `runs/pretrained_longtail_v1/component/last.pth` ở 176.500 |
| Mục tiêu | 250.000 tổng outer steps, thêm 73.500 outer steps |
| Phương pháp | `--text-sampler component` |
| Batch / support | 8 / 15 |
| LR encoder / phần G còn lại | 5e-6 / 2e-5 |
| LR discriminator / writer | 1e-5 / 2e-5 |
| OCR / writer weight | 4 / 0,1; gradient balancing bật |
| OCR | Frozen/eval |
| Kiến trúc | Native legacy layout, không đổi positional query |
| HAT fraction / rare fraction | 0,35 / 0,30 |
| HAT reconstruction / paired feature / pixel auxiliary | Không bật trong run này |
| Runtime | PyTorch 2.9.1+cu128, H200 |
| Lưu và đánh giá | Mỗi 1.000 bước theo mốc global; log mỗi 50 bước |
| Checkpoint giữ lại | last, best rolling OCR, best focus và 3 checkpoint định kỳ gần nhất |

Mỗi outer step gồm hai cập nhật G, hai cập nhật D và một cập nhật writer; OCR frozen nên không cập nhật R. Cần phân biệt outer step với optimizer update khi so ngân sách.

Tại snapshot: **181,250/250,000** theo log, tương đương đã đi thêm **4,750** outer steps từ nguồn 176.500. Các benchmark đã hoàn tất:

| Bước | CER ↓ | Exact ↑ | NGUYỄN đúng /30 | Ễ đứng riêng đúng /30 |
|---:|---:|---:|---:|---:|
| 176,500 | 12.69% | 65.00% | 29 | 0 |
| 177,000 | 15.03% | 58.33% | 20 | 0 |
| 178,000 | 14.44% | 61.67% | 30 | 0 |
| 179,000 | 13.63% | 61.50% | 29 | 0 |
| 180,000 | 14.39% | 59.50% | 26 | 0 |
| 181,000 | 16.37% | 57.83% | 20 | 0 |

Benchmark mới nhất có CER **16.37%**, so với **12,69%** ở đầu continuation. Điều này cho thấy tiến độ còn dao động; **chưa có căn cứ nói train dài đã tiếp tục cải thiện**. Chưa có số liệu tại 250k.

Best đang giữ trong thư mục run này ở bước **178,000**, CER riêng **14.44%**, rolling mean **14.05%** trên 3 mốc. Best trong run mới không mặc nhiên tốt hơn checkpoint nguồn; tiêu chí rolling và cửa sổ đánh giá phải được đọc đúng. `best.pth` chứa model tại một mốc, không phải trung bình trọng số.

[Panel nguồn 176.500](../../runs/component_continuation_250k_v1/expanded/benchmark_176500.png) · [Panel tại benchmark gần nhất của snapshot](../../runs/component_continuation_250k_v1/expanded/benchmark_181000.png) · [Report động tiếp tục cập nhật](261114_report_05_Continuation_component_den_250k.md).

## 8. Hạ tầng đã bổ sung và đang vận hành

- **Resume có provenance:** ghi source checkpoint, bước nguồn, SHA-256, command và runtime; lưu bản code dùng để khởi chạy. Checkpoint nguồn được giữ riêng.
- **Supervisor chạy nền:** khóa chống launch trùng, ghi PID/trạng thái, heartbeat tiến độ, exit code và kiểm tra kết thúc tại mục tiêu.
- **Lưu model có giới hạn:** checkpoint định kỳ và các best proxy; không ghi đè source hay tự promote model thành bản cuối để sử dụng.
- **Panel cố định:** 600 mẫu, 20 prompt × 30 writer development, giữ support để so giữa các checkpoint. Không dùng điểm của panel 48/60/97 mẫu như thể cùng metric 600 mẫu.
- **Report tự cập nhật:** watcher đọc benchmark hoàn tất, tạo Markdown, CSV và biểu đồ mà không can thiệp vòng training.
- **Tổ chức docs:** tên theo `YYMMDD_report_XX_Ten_report.md`, mục lục và catalog có đối chiếu tên cũ; script tạo report đã chuyển sang tên mới.

Code: [launcher pilot](../../scripts/run_pretrained_longtail_pilot.py), [supervisor 250k](../../scripts/resume_component_250k.py), [tạo report 250k](../../scripts/report_component_250k.py), [watcher report](../../scripts/watch_component_250k_report.py).

## 9. Các module nghiên cứu đã viết nhưng không chạy trong mainline

### 9.1. Query tách thành phần

Prototype VietCompose biểu diễn base/modifier/tone/case bằng role tokens, tương tác attention và residual query khởi tạo zero trên query glyph gốc. Mục tiêu là chia sẻ thông tin giữa các chữ cùng thành phần. Pilot warm-start trước đó chỉ cập nhật adapter đã không thắng nguồn: CER nguồn 13,63%, adapter factorized 15,38%, glyph control 16,32%. Không gộp trọng số đó vào mainline.

[Code adapter](../../research/vietcompose/model.py) · [kết quả pilot](../vietcompose/261114_report_02_Ket_qua_pilot_VietCompose.md).

### 9.2. Renderer lớp mực và recognizer thành phần

Renderer riêng tạo lớp thân, modifier và tone; có gate vắng dấu và placement khả vi. Nhánh exploratory đã chạy nhưng chất lượng hình ảnh chưa đạt. Factorized CTC cộng điểm các vai trò thành điểm ký tự hoàn chỉnh trước **một** alignment CTC; có control head nguyên ký tự và canonical-glyph prototype. Đây là các module riêng, chưa thay OCR hoặc decoder trong run 250k.

Trong một diagnostic recognition ở seed 742, 5.000 cập nhật và panel 187 ảnh/30 writer, factorized CTC có CER 34,84%, tail macro-F1 26,50%; whole-character head có CER 35,16%, tail F1 0%. Head nguyên ký tự không được học hàng output của lớp giữ lại nên đối chứng này có giới hạn cấu trúc; không lấy chênh lệch đó làm bằng chứng mới hoặc ưu thế generation. Không trộn các số này với CER của native VATr++.

[Renderer](../../research/vietcompose/renderer.py) · [recognizer](../../research/longtail/recognizer.py) · [nhánh exploratory](261114_report_03_Ket_qua_pilot_long_tail.md).

### 9.3. Audit và protocol zero-shot

Đã viết exporter pixel hash, audit theo ảnh/writer/thành phần, exposure policy dùng chung cho image/OCR/corpus/support/memory, ngân sách ảnh lồng nhau và bộ chấm transcription/paired writer-bootstrap. Các thí nghiệm sạch này dùng nhánh riêng. Mainline hiện tại khởi tạo từ checkpoint có exposure trước đó nên không thuộc protocol zero-shot.

Các tiến trình exploratory còn chạy đã dừng theo yêu cầu chuyển hướng sang pretrained; artifact giữ lại để phân tích. Không tự khởi động lại nhánh scratch.

[Protocol code](../../research/longtail/protocol.py) · [metadata exporter](../../research/longtail/prepare.py) · [steering nhánh archived](../../runs/acl_longtail_v1/steering.json).

## 10. Những phần đang xây dựng ở mức thiết kế

| Hướng | Câu hỏi phải giải quyết trước khi nhận là cải tiến đã đạt |
|---|---|
| Query chia sẻ thành phần trên pretrained G | Có thắng glyph/capacity control khi fine-tune cùng dữ liệu, seed và ngân sách không? |
| Học hình học dấu có xét case | Có đặt mũ/ngã đúng trên chữ hoa, dấu nặng đúng dưới thân; có nhãn/metric vị trí đủ tin cậy không? |
| Memory thành phần theo writer | Lấy được bằng chứng dấu từ support bằng mask/alignment đã kiểm tra hay chỉ đang dùng feature toàn cục? |
| Học chữ đơn và chuyển từ ngữ cảnh | OCR có chấm chữ đơn đáng tin không; crop ký tự có giữ đủ dấu và tránh lẫn chữ cạnh bên không? |
| Few-shot ảnh thật 0/1/5/20 | Có đủ ảnh/writer cho uppercase, split độc lập, ngân sách thực sự đạt và đối chứng acquisition ngẫu nhiên không? |
| Đánh giá chất lượng độc lập | Nội dung, dấu, khoảng cách và phong cách có tốt hơn theo evaluator độc lập/người đọc không? |

Đây là hướng nghiên cứu, chưa phải các thành phần đang chạy ở GPU trong mainline. [Thiết kế long-tail](../vietcompose/LONG_TAIL_RESEARCH_VI.md) và [study protocol](STUDY_PROTOCOL.md) lưu câu hỏi, baseline và giới hạn cần xử lý.

## 11. Kiểm chứng và giới hạn kết luận

Bộ test gần nhất trên runtime gốc có **39 tests pass**, gồm kiểm tra legacy, sampling và nhánh exploratory. Riêng sampler mới kiểm tra: mix=0 giữ đúng chuỗi lấy mẫu legacy với cùng RNG; nhóm hoa/thường và cặp dấu được phân biệt; không sinh transcript ngoài corpus; xác suất analytic ở trường hợp empirical đúng như mong đợi. [Log kiểm thử](test_run.log).

Các kiểm tra này chứng minh đường chạy/tính đúng đắn kỹ thuật trong phạm vi test, không chứng minh chữ đẹp hay cải thiện thống kê. Những giới hạn chính:

1. Sampling không bổ sung ảnh thực tế của writer đang thiếu `Ễ`/`Ỡ`/`Ẵ`.
2. Cải tiến ở mức phân phối transcript chưa buộc mạng học ghép thành phần.
3. OCR benchmark có lịch sử chung với GAN; có thể đọc đúng ảnh chưa tự nhiên hoặc dựa vào ngữ cảnh.
4. Chưa có paired real-image ground truth cho một số uppercase trong development; vẫn chấm được transcription của prompt sinh, nhưng không thay thế kiểm tra fidelity theo writer.
5. Pilot pretrained là một seed; run dài hiện không có một nhánh legacy được train song song tới 250k. Không được dùng so sánh khác ngân sách để kết luận nhân quả.
6. Chưa đủ đối chứng external, nhiều seed, chấm mù và dữ liệu quyền sử dụng rõ ràng để gọi là bài ACL hoàn chỉnh.

## 12. Artifact để theo dõi và kết luận hiện tại

- [Mục lục report](../README.md).
- [Report pilot pretrained](261114_report_04_Finetune_pretrained_175500_den_176500.md).
- [Report training 250k tự cập nhật](261114_report_05_Continuation_component_den_250k.md).
- [Trạng thái run](../../runs/component_continuation_250k_v1/status.json).
- [Log training](../../runs/component_continuation_250k_v1/train.log).
- [Snapshot số liệu của report này](report_06_assets/snapshot.json).

**Kết luận:** đã có một thay đổi sampling cụ thể, được tích hợp vào checkpoint pretrained, có pilot đối chứng và một run dài đang hoạt động. Tín hiệu pilot là tích cực về OCR proxy; tiến độ dài hạn còn dao động, và bài toán chữ hoa hiếm/ghép dấu vẫn chưa được giải quyết. Những module kiến trúc thành phần đã viết hoặc đang thiết kế được quản lý riêng, chưa được mô tả như tính năng đã chạy trong model 250k.
