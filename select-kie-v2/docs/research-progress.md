# Tiến độ nghiên cứu hướng CVPR — 28/09/2026

**Đã có kết quả mới tốt hơn về ranking và một phép xác nhận trên tài liệu mới. Chưa có cơ sở gọi công trình hiện tại đạt chất lượng 9/10.** Tham vọng đó cần một đóng góp khác biệt, đánh giá công bằng và audit đáng tin, không chỉ một bảng số đẹp. CFP yêu cầu original research; hạn nộp CVPR 2027 là 16/11/2026 AoE. [CFP chính thức](https://cvpr.thecvf.com/Conferences/2027/CallForPapers)

## Cập nhật stage4–6: learned head và xác nhận mới

Đã chạy thêm700 extraction (stage4:600; stage6:100), tạo4.800 training/evaluation decisions cho prototype và huấn luyện10 head neural (5 seed/architecture). Cùng thông tin, generic38353 parameters và relation37601. Không fine-tune extractor/encoder; không có auxiliary binding labels.

Stage4:9B với schema rõ nghĩa đạt76.00% accuracy, hơn prompt dài tương đương26.50 điểm % (CI[23.33,29.50]);2B đạt70.00%. AI PDF spot checks xác nhận ví dụ lấy cá nhân thay primary registrant và phát hiện gold/span ambiguity. Human audit200 decision vẫn trống,20 overlap sẵn cho reviewer2.

Stage6 giữ nguyên head/calibration trên100 tài liệu mới: relation AURC0.05715, coverage56.33%,9/338 lỗi (2.66%); generic MLP0.06588,34.17%,7/205 (3.41%). ΔAURC CI[−0.01715,−0.00049]. Đây là gain đã xác nhận theo document so với MLP, không phải risk certificate hay binding-specific result.

**Giới hạn quyết định:**94/133 accept tăng thêm là ABSENT; normalized/present-only AURC CI chứa0. HGB nhận cùng embedding chưa bị vượt rõ ràng (ΔCI chứa0), và threshold theo field làm coverage HGB47.50% gần relation46.50%. Prototype có ích nhưng chưa đủ claim novelty hoặc9/10CVPR. Cần audit/ontology và nguồn/model family khác.

[Stage4](stage4-results.md), [Stage5 pilot](stage5-results.md), [Stage6 confirmation và đối chứng](stage6-results.md), [audit viewer](../results/stage4/audit-viewer.html), [risk–coverage figure](../results/stage6/risk-coverage.png).

## Những gì đã thực hiện trong lượt nghiên cứu tiếp

- Huấn luyện 10 head HGB mới cho hai extractor CORD, thêm OCR key/value/layout support và 5-fold cross-validation theo document. Reuse prediction extractor cũ.
- Sinh **1200 lượt extraction mới**: Qwen2.5-1.5B và Qwen3.5-2B trên 600 tài liệu VRDU, split chính thức train200/valid100/test300, test là template Short-Form chưa có trong train. Confidence heads fit train200; valid chia development50/calibration50 theo hash.
- Sinh **400 lượt extraction mới** trên **100 Short Form mới**: Qwen3.5-2B, image/OCR × schema baseline/clarified. Không dùng 100 này để fit, tuning hay chọn threshold. Hai model GPU A30 đã kết thúc; không để job chạy nền.
- Lưu raw predictions, head, metrics, bootstrap, figures, protocol và viewer audit. 2400 quyết định của holdout mới có gold và bốn prediction để review; chưa có audit người.

## Kết quả chắc nhất của lượt này

### 1. Evidence support có thông tin hữu ích ngoài baseline HGB

| Dataset / extractor | HGB fusion AURC | HGB + evidence AURC | 95% CI ΔAURC |
|---|---:|---:|---:|
| CORD / Qwen2-VL-2B | 0.04030 | 0.02746 | [−0.01894, −0.00774] |
| VRDU OCR / Qwen2.5-1.5B | 0.72344 | 0.53484 | [−0.20728, −0.17114] |
| VRDU OCR / Qwen3.5-2B | 0.23349 | 0.19919 | [−0.04412, −0.02390] |

Các CI conditional trên head đã fit, bootstrap theo document. Trên Qwen3-VL-4B CORD, gain nhỏ và CI chứa0. Rules key aliases và boxes đã có prior; kết quả xác nhận ích lợi tín hiệu, **không xác nhận novelty kiến trúc**. [Stage 2](stage2-results.md)

Coverage/risk phải được đọc riêng: trên Qwen2 CORD, cấu hình `all` đạt coverage84.17%, risk4.78% (HGB fusion:77.64%/5.83%). Trên Qwen3.5 VRDU, evidence head AURC tốt hơn nhưng chỉ chấp nhận2.89% tại ngưỡng target5%, so với14.06% của fusion. Ranking gain chưa giải quyết automation ở tail risk thấp.

### 2. Làm rõ semantic role có ảnh hưởng rất lớn trên holdout mới

| Qwen3.5-2B | Schema baseline accuracy | Schema clarified accuracy | Δ accuracy, 95% CI (điểm %) |
|---|---:|---:|---:|
| OCR thật | 61.83% | 83.67% | +21.83 [+18.33, +25.33] |
| Ảnh mọi page, cạnh dài1280 | 29.33% | 73.50% | +44.17 [+40.33, +48.00] |

Clarification phát triển từ train và hai mẫu development; gold holdout mới chỉ được mở sau inference. Kết quả cả field hiện diện và ABSENT đều cải thiện. Đây là **extraction gain từ thông tin/schema prompt**, không phải selection gain hay phương pháp mới. Hai prompt khác độ dài và chi tiết; cần length-matched controls trước khi quy kết cơ chế. [Stage 3](stage3-results.md)

Ví dụ thật trên ảnh: `registrant_name` baseline lấy cá nhân ở đầu Short Form; target là primary registrant ở mục8. Giá trị cá nhân được đọc đúng và có trên trang, nhưng không đúng vai trò target. Số đăng ký cũng có thể bị thay bằng chữ “FARA” khi mô tả schema quá ngắn. Điều này làm yếu claim rằng mọi lỗi phải được chữa bằng confidence architecture.

### 3. Confidence/risk vẫn là vấn đề mở

Với cùng HGB-fusion và threshold đã học từ OCR stage2, image-baseline trên holdout mới sai18/23 accepted (78.26% risk). Khi làm rõ schema, image-clarified sai0/115 accepted (coverage19.17%). Không gọi 0/115 là certificate; accuracy gốc đã thay đổi và cùng một head không được fit cho modality ảnh.

HGB-evidence tại threshold cũ reject toàn bộ ảnh ở cả hai schema. Điều đó tránh lỗi bằng zero coverage, chưa đáp ứng automation. Confidence cao sau clarification còn một số partial-value/ABSENT errors; chưa đủ nhãn để xác nhận lỗi binding thật còn phổ biến.

## Điều này thay đổi ý tưởng ban đầu ra sao?

**Chưa nên viết paper với claim “binding-aware head vượt generic head và đạt risk control”.** Evidence layout giúp ranking, nhưng confidence tại low-risk chưa ổn định; schema semantics và label strictness còn là confounder lớn.

Hướng đáng kiểm chứng tiếp: **confidence có điều kiện trên schema đã được grounding đúng semantic role**, phân biệt ba nguyên nhân review: bằng chứng đọc không được, target schema không rõ, và value gán sai relation dù schema rõ. Mục tiêu là giữ coverage khi schema/template/modality thay đổi, với cùng label/information budget. Đây là hypothesis cập nhật, chưa phải một đóng góp mới được xác lập.

ExtractConf đã dùng hai cách đọc và disagreement; PLV đã kết hợp perception/layout/validation; prior về selective risk đã phân tích clustering và leakage. Vì vậy two-pass fusion, geometry rules và calibration riêng lẻ không đủ novelty. [ExtractConf](https://arxiv.org/abs/2606.24420), [PLV](https://arxiv.org/abs/2609.20110), [Valid Per-Field Selective Risk Control](https://arxiv.org/abs/2608.14639)

## Các thí nghiệm cần vượt để tiếp cận mục tiêu CVPR

1. **Kiểm tra failure mode trên model mạnh hơn và một family khác.** Dùng schema rõ nghĩa ngay từ đầu, image resolution đủ. Nếu binding thật gần biến mất, dừng method claim này; đừng dùng extractor yếu để tạo cơ hội cải thiện.
2. **Control cơ chế.** So schema clarified với prompt dài tương đương không bổ sung role, và paraphrase thực sự giữ nguyên target. Image/OCR comparison phải báo khác biệt resolution/OCR quality. Không gọi prompt refinement là causal proof riêng cho binding.
3. **Audit người.** Tách partial value, punctuation, ambiguous target và missing annotations khỏi binding. Hai reviewer kiểm tra phần giao nhau. Viewer và CSV đã sẵn; hiện không có nhãn human để kết luận agreement hay prevalence.
4. **Huấn luyện representation/head khi còn target đủ lớn.** Field description, key evidence, rival value và box relation; generic head nhận cùng raw representations/candidate/supervision. Cân bằng capacity và tuning. So với linker+correctness, ExtractConf/PLV matched baseline; không chỉ so logprob.
5. **Khóa confirmation mới.** Tách fitting, development, score calibration và threshold calibration; đánh giá natural unseen templates và nguồn thứ hai trên ảnh. Báo accuracy, AURC, coverage/risk, latency và từng subtype. Không chọn alpha hoặc normalization sau khi xem confirmation.

Đây là các gate nghiên cứu; đạt gain ở một baseline yếu không bù được thiếu novelty hoặc gold mơ hồ. Chưa chấm điểm reviewer giả hay đưa xác suất acceptance.

## Kiểm chứng và artifact

Inference VRDU stage2: đủ600 unique docs/model, không input truncation; Qwen2.5 có1 output cap, Qwen3.5 không có. Stage3: đủ200 records/modality, 100 unique docs ×2 schema; không truncation/cap. Feature reconstruction kiểm tra120 quyết định stage2 khớp exported scores của hai head tới1e−12. Bootstrap stage2 B=1000; stage3 B=2000. Cột human audit để trống.

- [Stage 2 kết quả](stage2-results.md), [protocol](stage2-protocol.md).
- [Stage 3 kết quả](stage3-results.md), [protocol](stage3-protocol.md).
- [Audit viewer](../results/stage3/audit-viewer.html), [CSV](../results/stage3/audit.csv).
- [Hướng dẫn tái lập](../README.md).

Chưa đo joint/LoRA training, không phát hành/publish artifact ra ngoài workspace, không tái hiện đầy đủ prior method. Nguồn VRDU và matcher Google Research được giữ provenance/copyright trong artifact. [Dataset chính thức](https://github.com/google-research-datasets/vrdu), [matcher](https://github.com/google-research/google-research/blob/master/vrdu/match_utils.py).
