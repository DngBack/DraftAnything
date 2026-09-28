# Đánh giá hướng nghiên cứu selective KIE cho CVPR 2027

Ngày đánh giá: **28/09/2026**. Trạng thái: **đánh giá ý tưởng và đề xuất pilot; chưa có kết quả thực nghiệm**.

Đầu vào: [literature review](../literature-reviews/learning-to-reject-kie/literature-review.vi.md) và [đề cương binding-aware selective KIE hiện có](binding-aware-selective-kie-cvpr2027.md). Bản này bổ sung phản biện và điều kiện quyết định, không coi các giả thuyết trong đề cương là kết quả.

## 1. Kết luận điều hành

**Khuyến nghị: dành một pilot ngắn cho binding-aware selective KIE; chưa cam kết đây là hướng có tính khả thi cao để đạt chất lượng “9/10”.** Hướng chỉ “kết hợp Kev + P(True) + conformal prediction cho KIE” khá khả thi về triển khai, nhưng khác biệt khoa học hiện chưa đủ thuyết phục. Câu hỏi mạnh hơn là:

> Khi giá trị được đọc đúng và thực sự xuất hiện trên trang, mô hình có học được rằng nó đang gán giá trị ấy cho sai field, và nhờ đó tự động chấp nhận được nhiều kết quả hơn ở cùng mức lỗi hay không?

Điểm đáng đầu tư là kiểm chứng **bất định của quan hệ field–bằng chứng–giá trị**. Rủi ro lớn nhất là hiệu quả thực chất đến từ thêm nhãn, thêm candidate hoặc thêm capacity; một correctness head thông thường được cấp cùng thông tin có thể làm tốt tương đương.

Đây là nhận định định tính của trợ lý, không phải điểm reviewer hay xác suất acceptance. “9/10” được hiểu là tham vọng về chất lượng nghiên cứu. Chưa có cơ sở xác nhận đó là thang điểm chính thức của CVPR 2027.

**Bằng chứng về venue:** Document analysis and understanding là một chủ đề trong main technical program của CVPR 2027. Trang CFP không mô tả nó như một track riêng có chuẩn chấm thấp hơn. Hạn đăng ký bài là 10/11/2026, nộp bài 16/11/2026 AoE: còn khoảng bảy tuần tính từ ngày đánh giá. [CVPR 2027 CFP](https://cvpr.thecvf.com/Conferences/2027/CallForPapers)

## 2. Phạm vi, giả định và cách đánh giá

**Chủ quyết định:** người nghiên cứu; trợ lý đưa ra khuyến nghị. **Đối tượng đầu tiên:** ảnh tài liệu, schema cố định hoặc được mô tả bằng ngôn ngữ, field một giá trị hoặc ABSENT. Bảng nhiều dòng và câu trả lời phải suy luận số học là mở rộng sau.

**Chưa biết:** GPU/VRAM, mô hình đang chạy, số tài liệu có nhãn, quyền dùng và phát hành dữ liệu, số người gán nhãn, thời gian làm việc. Đã hỏi người dùng; các kết luận khả thi bên dưới có điều kiện, không mặc định các nguồn lực này tồn tại.

**Giả định để lập kế hoạch:** có thể chạy một extractor truy cập được đặc trưng hoặc một pipeline sinh candidate; có dữ liệu tài liệu thật để audit; có thể tách train/development/calibration/test theo tài liệu gốc và template. Nếu chỉ có API và không truy cập representation, phải thu hẹp thành bộ kiểm chứng bên ngoài; không tuyên bố đã huấn luyện khả năng tự đánh giá bên trong VLM.

Tiêu chí được đặt trước bảng đánh giá, không cộng thành một điểm tổng:

| Tiêu chí | Mạnh | Trung bình | Yếu hoặc chưa xác định |
|---|---|---|---|
| Tính mới so với nguồn đã kiểm tra | Câu hỏi/cơ chế khác biệt, có phép thử phân biệt với prior | Khác biệt có thể bảo vệ nhưng còn cạnh tranh | Ghép thành phần đã biết; hoặc tìm kiếm chưa đủ |
| Đóng góp cho hiểu tài liệu | Quan hệ ngữ nghĩa–không gian là thiết yếu và được kiểm chứng | Có ích trên ảnh tài liệu nhưng cơ chế còn chung | Chỉ đổi dataset hoặc prompt |
| Sức mạnh kiểm chứng | Có đối chứng loại trừ giải thích thay thế | Đo được nhưng khó cô lập cơ chế | Chỉ báo cáo metric tổng hợp |
| Khả thi trong khoảng bảy tuần | Pipeline và dữ liệu đã sẵn, vòng thử ngắn | Cần thêm một khâu vừa sức | Nhiều phụ thuộc chưa có; hoặc nguồn lực chưa biết |
| Giá trị khi kết quả âm | Audit/protocol vẫn trả lời được câu hỏi hữu ích | Có một phần kết quả tái sử dụng | Chỉ còn một mô hình không tốt hơn baseline |

Không dùng trọng số; không cho phép điểm cao ở tiêu chí khác bù cho thiếu dữ liệu hoặc thiếu khác biệt. Đây là đánh giá định tính có lý do, không phải thang điểm dự báo hội nghị. Hướng dẫn reviewer CVPR 2026 được dùng như tham khảo lịch sử về tính đúng kỹ thuật, originality và ý nghĩa đóng góp; không thay cho quy định 2027. [Reviewer guidelines 2026](https://cvpr.thecvf.com/Conferences/2026/ReviewerGuidelines)

## 3. Bốn hướng đã cân nhắc

| ID và ý tưởng | Đánh giá hiện tại | Quyết định đề xuất |
|---|---|---|
| I1 — Học P(correct) rồi reject và calibrate | Triển khai tương đối thuận lợi; novelty yếu vì self-evaluation, selective prediction và confidence KIE đã có trong review | Baseline bắt buộc; không dùng làm claim chính |
| I2 — Học phát hiện sai liên kết giữa field và giá trị có thật trên trang | Phù hợp tài liệu mạnh; novelty trung bình và chưa chắc; có phép thử cơ chế rõ; khả thi phụ thuộc dữ liệu và annotation | **Ưu tiên pilot có điều kiện** |
| I3 — Học khi nào nên zoom/đọc lại để sửa lỗi, tối ưu lợi ích trên chi phí | Có tiềm năng về quyết định quan sát, nhưng adaptive perception đã có prior gần; cần dữ liệu hành động và đánh giá chi phí | Giữ cho nghiên cứu dài hơn; không dùng để cứu deadline vội |
| I4 — Benchmark thiếu bằng chứng/không trả lời được trong document QA | Audit hữu ích nhưng vùng novelty đã có VRD-UQA; chỉ tạo perturbation và chấm abstention khó đủ mạnh | Đưa vào diagnostic của I2 nếu phù hợp |

Nguồn gốc: I1 là yêu cầu người dùng và literature-inspired; I2 đã có trong đề cương trước phiên đánh giá; I3 được phản biện AI đề xuất rồi rà soát lại; I4 là phương án đối chiếu từ tài liệu. Tất cả đều bị ảnh hưởng bởi ý tưởng ban đầu, không phải một vòng phát sinh ý tưởng mù hoặc độc lập của nhiều nhà nghiên cứu.

## 4. Những prior thực sự đe dọa novelty

| Nguồn đã kiểm tra | Bằng chứng liên quan | Hệ quả cho claim |
|---|---|---|
| [KVPFormer, AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/26516) | Học quan hệ key–value bằng formulation QA | Key–value linking tự nó không mới |
| [Identify, Locate, Link](https://arxiv.org/abs/2608.20868) | VLM sinh key, value, region và link trong một lượt; arXiv ghi accepted ICDAR 2026 | Grounding + linking + end-to-end không đủ là đóng góp mới; cần existing linker + correctness head |
| [FieldSwap, ICDE 2024](https://research.google/pubs/fieldswap-data-augmentation-for-effective-form-like-document-extraction/) | Thay cụm từ chỉ field và đổi nhãn tương ứng để augment tài liệu | Không gọi key swapping là kỹ thuật mới; cần đối chứng dùng cùng augmentation |
| [Reliable VQA](https://arxiv.org/abs/2204.13631) | Selective answering với chức năng ước lượng độ đúng và đánh giá risk–coverage | Rejector đa phương thức không phải ý tưởng mới |
| [Valid Per-Field Selective Risk Control](https://arxiv.org/abs/2608.14639) | Phân tích document clustering, score-refit leakage và vấn đề threshold; phân biệt các dạng bảo đảm | Không claim đầu tiên kiểm soát risk theo field; không dùng field như mẫu độc lập một cách mặc định |
| [PLV](https://arxiv.org/abs/2609.20110) và [Beyond Logprobs](https://arxiv.org/abs/2606.24420) | Confidence kết hợp bằng chứng perception/layout/validation hoặc nhiều tín hiệu | Baseline fusion phải đủ mạnh; thêm geometry đơn thuần khó bảo vệ novelty |
| [VRD-UQA](https://arxiv.org/abs/2511.11468) | Benchmark câu hỏi không trả lời được trên visually rich documents | Missing-evidence benchmark không còn là khoảng trống hiển nhiên |
| [Q-Zoom](https://arxiv.org/abs/2604.06912), [InSight-doc](https://arxiv.org/abs/2608.10628) | Điều chỉnh quan sát/độ phân giải theo yêu cầu tác vụ | “Không chắc thì nhìn kỹ hơn” chưa đủ phân biệt I3 |

Kev, Kadavath, ASPIRE, SelectiveNet và các phương pháp conformal được phân tích trong literature review đầu vào. Vai trò phù hợp của Kev là gợi ý học quyết định; P(True) là gợi ý học tính đúng của câu trả lời cụ thể. Việc kết hợp chúng chưa xác lập một đóng góp khoa học mới.

Các preprint được dùng để đánh giá cạnh tranh về ý tưởng, không mặc nhiên coi mọi claim thực nghiệm của chúng đã được xác nhận độc lập. Chưa tái lập kết quả bất kỳ bài nào trong phiên này.

## 5. Phiên bản ý tưởng nên thử

Tên làm việc: **When the Value Is Present but the Field Is Wrong: Learning Binding-Aware Confidence for Document Extraction**.

Ví dụ: ảnh có `Subtotal = 100`, `Tax = 10`, `Total = 110`; extractor trả `Total = 100`. Giá trị 100 được đọc chính xác và tìm thấy trên trang. Câu hỏi cần học là bằng chứng nào hỗ trợ nó thuộc về Total, và có candidate khác phù hợp với quan hệ đó hơn không.

**Ý tưởng I2:** với prediction đã sinh, học correctness bằng representation của field, giá trị, vùng bằng chứng và các giá trị cạnh tranh cùng kiểu. Cấu trúc cạnh tranh có thể giúp tách “đọc được chuỗi” khỏi “gán chuỗi đúng quan hệ”. Đây là giả thuyết về inductive bias, chưa có bằng chứng rằng cấu trúc ấy hơn một head chung.

### 5.1 Mục tiêu huấn luyện tối thiểu

Đóng băng extractor trong thí nghiệm đầu tiên. Với prediction thực tế từ decoding, đặt:

\[
z_{d,f}=\mathbf 1[\operatorname{norm}(\hat y_{d,f})=\operatorname{norm}(y^*_{d,f})],
\qquad
c_\phi=\sigma(h_\phi(D,f,\hat y,E,C)).
\]

Trong đó E là evidence được đề xuất, C là tập candidate/rival. Huấn luyện:

\[
\mathcal L=\operatorname{BCE}(c_\phi,z)+\lambda\mathcal L_{\mathrm{relation}}.
\]

BCE là mục tiêu học tính đúng; relation loss là lựa chọn cần ablation, không phải novelty mặc định. Có thể dùng nhãn liên kết hoặc loss phân biệt candidate đúng/sai nếu nhãn đủ tin cậy. Không lấy softmax margin của linker làm xác suất đúng đã được chứng minh. Không chỉ học trên câu trả lời gold teacher-forced.

Nếu chuyển sang LoRA/joint training để representation của chính extractor học tự đánh giá, cần sinh lại prediction/correctness label khi extractor thay đổi và tách lợi ích tăng accuracy khỏi lợi ích selection. Frozen pilot chứng minh giá trị của tín hiệu trước, chưa chứng minh backbone đã học tự nhận biết.

### 5.2 Dữ liệu và phép biến đổi

Ưu tiên lỗi tự nhiên. Chỉ thêm hard negatives sau khi có taxonomy và baseline. FieldSwap hoặc phép đổi key/value phải cập nhật nhãn đúng của tài liệu mới. Một phép thay đổi quan hệ không có nghĩa mọi confidence đều phải giảm: nếu mô hình trả đúng đáp án mới, confidence cao vẫn hợp lý. Ngược lại, thay đổi font hoặc nhiễu nhẹ có thể giữ nhãn nhưng vẫn thay đổi khả năng đọc.

Phân biệt ABSENT, UNREADABLE và CANDIDATE-NOT-CAPTURED. Chúng có thể dẫn tới review nhưng không phải cùng nguyên nhân. Khi hai field có cùng giá trị, sai anchor không nhất thiết là sai giá trị vận hành; đánh giá provenance và value correctness riêng. Không gọi synthetic intervention là chứng minh nhân quả.

### 5.3 Giả thuyết và dấu hiệu bác bỏ

| ID | Dự đoán cần kiểm chứng | Điều làm yếu hoặc bác bỏ |
|---|---|---|
| H1 | Lỗi binding đóng góp đáng kể vào lỗi confidence cao trên tài liệu thật | Lỗi hiếm, nhãn mơ hồ, hoặc heuristic vị trí đã bắt gần hết |
| H2 | Cấu trúc quan hệ giúp selection ngoài những gì head chung học được với cùng thông tin/nhãn | Lợi ích biến mất khi baseline được cấp cùng candidate, loss và label budget |
| H3 | Lợi ích còn ở template chưa thấy và nguồn tài liệu thứ hai | Chỉ thắng trên một template hoặc ảnh sửa nhân tạo |
| H4 | Có thành phần thông tin thị giác/không gian giải thích được lợi ích | Field prior hoặc candidate count đã giải thích kết quả; bỏ hình ảnh/geometry không ảnh hưởng như dự đoán |

H4 âm không tự động khiến bài ngoài phạm vi CVPR, nhưng buộc sửa claim về cơ chế thị giác. Cả bốn giả thuyết cần được phân biệt với một kết quả calibration ECE tốt hơn.

## 6. Bốn thí nghiệm quyết định trước khi đầu tư lớn

### E0 — Audit lỗi và khoảng cải thiện khả dĩ

Lấy mẫu tài liệu trước khi lọc theo confidence; có thể bổ sung một nhóm lỗi confidence cao để chẩn đoán, nhưng không trộn hai nhóm để ước lượng prevalence. Khởi đầu khoảng 200–300 tài liệu nếu khả thi chỉ là quy mô audit thăm dò, không phải sample-size đã được tính power. Phân loại value error, OCR/perception, binding, absent hallucination, ambiguous gold; hai người kiểm tra một phần giao nhau và giải quyết bất đồng.

Đo số lỗi binding còn lọt qua baseline. Dùng nhãn oracle để xem việc phát hiện hoàn hảo riêng nhóm lỗi này có thể cải thiện selection được bao nhiêu. Đây là giới hạn chẩn đoán cho cơ chế nhắm riêng binding, không phải upper bound của mọi phương pháp confidence. Nếu cơ hội cải thiện quá nhỏ so với chi phí triển khai, dừng I2.

### E1 — So sánh công bằng khi extractor cố định

Các đối chứng cần có: token/logprob khi truy cập được; verbalized hoặc learned self-evaluation; fusion perception/layout/validation; correctness head tổng quát; linker hiện có kèm correctness head; head có cấu trúc binding đề xuất.

**Đối chứng quyết định:** generic head nhận cùng field, ảnh/evidence, candidate và supervision. Nếu binding head nhận thêm nhãn anchor hoặc hard negatives, generic head cũng phải được hưởng hoặc phải báo cáo riêng đường cong theo chi phí nhãn. Cân bằng ngân sách tuning và ghi latency/memory. So sánh với một baseline cố tình thiếu thông tin không chứng minh được đóng góp kiến trúc.

### E2 — Generalization tự nhiên và protocol risk

Tách theo tài liệu gốc; không để bản biến dạng/crop của cùng trang đi qua các split. Với template generalization, khóa nhóm template trước khi huấn luyện. Tách score fitting, phát triển hyperparameter, calibration của score nếu cần, lựa chọn threshold và test theo protocol đã ghi; tránh dùng cùng nhãn nhiều lần rồi gọi calibration độc lập.

Metric chính: coverage tại threshold chọn bằng calibration, kèm accepted error đo trên test. Báo cáo thêm toàn bộ risk–coverage/AURC, Brier hoặc NLL, calibration theo nhóm, accuracy gốc và chi phí. Resample ở cấp đơn vị độc lập phù hợp — ít nhất document, hoặc template nếu claim là tổng quát qua template — để ước lượng bất định; không bootstrap các field như độc lập. Bootstrap CI mô tả bất định, không tự tạo distribution-free guarantee.

**Sửa đề cương cũ:** không cố định 10% làm mốc duy nhất. Nếu accept-all đã có error dưới 10%, bài toán tại mốc đó có thể không còn có ý nghĩa. Chọn alpha từ nhu cầu ứng dụng và development pilot trước khi mở test; 1%, 5%, 10% có thể là các mức báo cáo khi dữ liệu đủ. Không chọn mức chỉ vì phương pháp thắng đẹp nhất. Khi chưa đủ chứng minh điều kiện, chỉ báo cáo empirical risk; đưa risk certification thành mô-đun độc lập, không lấy conformal như một nhãn bảo đảm.

### E3 — Loại trừ shortcut

Thử field identity/frequency, candidate count và geometry-only; bỏ ảnh nhưng giữ OCR+boxes; bỏ geometry nhưng giữ text; bỏ rival; bỏ relation loss; bỏ synthetic edits. Đánh giá riêng candidate recall, field absent, giá trị lặp lại và ảnh chất lượng thấp. Kiểm tra liệu cải thiện có phải do reject gần hết nhóm khó hoặc nhóm hiếm. Phân tích đúng/sai theo subtype cùng với metric toàn bộ.

## 7. Điều kiện tiếp tục và điều kiện dừng

**Đề xuất tiếp tục sau pilot** khi cùng đạt: lỗi mục tiêu có ảnh hưởng thực tế; generic head công bằng chưa giải thích hết lợi ích; xu hướng tốt xuất hiện trên natural unseen-template development; dữ liệu và throughput cho phép hoàn thành phép thử chính trước deadline.

Có thể dùng **+5 điểm phần trăm coverage** làm mốc ý nghĩa vận hành tạm thời, nếu phù hợp use case. Đây không phải tiêu chuẩn CVPR và chưa phải tính toán power. Chốt mốc, alpha và cách phân tích trước final test; không điều chỉnh hậu nghiệm để biến thất bại thành thành công. Pilot nhỏ chỉ cho quyết định đầu tư, chưa xác nhận hiệu quả.

**Dừng hoặc thu hẹp** nếu binding error quá hiếm; baseline cùng thông tin bắt kịp; gains chỉ xuất hiện trên synthetic edits; annotation không đáng tin; hoặc không có nguồn dữ liệu độc lập để kiểm chứng. Một audit âm rõ ràng vẫn có giá trị, nhưng chưa đảm bảo thành bài main-conference.

## 8. Khả thi và lịch làm việc có điều kiện

| Điều kiện thực tế | Nhận định |
|---|---|
| Extractor, data loader và nhãn đã sẵn; có compute và người audit | Pilot 7–10 ngày là hợp lý; hoàn thành bài trước 16/11 vẫn căng và phụ thuộc kết quả |
| Có GPU nhưng chưa có pipeline/dataset/annotation | Không đủ cơ sở gọi mục tiêu bài xuất sắc CVPR 2027 là khả thi cao |
| Chủ yếu API, không thể fine-tune hoặc đọc representation | Làm được audit và external verifier; mục tiêu “mô hình tự học confidence” phải thu hẹp hoặc đổi nguồn lực |

Lịch dự kiến nếu bắt đầu ngay với pipeline sẵn: ngày 1–3 audit và protocol; ngày 4–7 đối chứng frozen và head đầu tiên; ngày 8–10 kiểm tra template mới và quyết định tiếp tục. Phần thời gian còn lại ưu tiên đối chứng công bằng, nguồn thứ hai, ablation, kiểm tra thống kê và viết; không mở thêm một hệ thống agent zoom lớn khi I2 chưa vượt gate.

Trước khi ước lượng GPU-hour, cần đo throughput thực tế trên một batch đại diện với độ phân giải và số candidate dự định. Chưa có benchmark máy của người dùng nên không đưa con số VRAM/GPU-hour như đã được xác nhận.

## 9. “9/10” nên chuyển thành yêu cầu gì?

Đây là tiêu chí tham vọng do phiên đánh giá đề xuất, không phải checklist chính thức hay bảo đảm điểm số:

1. Một phát hiện dễ hiểu và có sức khái quát: vì sao nhìn thấy đúng giá trị vẫn không đủ để tin extraction.
2. Bằng chứng cho biết thành phần nào tạo hiệu quả, qua đối chứng cùng thông tin và cùng chi phí nhãn.
3. Cải thiện có ý nghĩa trên lỗi tự nhiên, không chỉ stress test tự tạo.
4. Generalization qua layout/nguồn; nếu nguồn lực đủ, thêm extractor thứ hai để tránh kết luận phụ thuộc một mô hình.
5. Evaluation risk trung thực, phân biệt ranking/calibration/certification và công bố thất bại.
6. Artifact có thể tái sử dụng: taxonomy, annotation protocol, split, mã và model head trong giới hạn quyền phát hành.

Đóng góp phương pháp không nhất thiết phải phức tạp. Nếu một mô hình đơn giản cho kết quả mạnh và giải thích được một failure mode quan trọng, câu chuyện vẫn có thể tốt. Ngược lại, nhiều loss và nhiều thành phần không bù được một insight yếu.

## 10. Phản biện đối nghịch và nhật ký quyết định

Một tác nhân AI khác được giao phản biện theo yêu cầu của skill; tác nhân này không phát sinh đề cương ban đầu. Đây là kiểm tra chéo bằng AI, không phải đánh giá độc lập của reviewer con người, và không được dùng như bằng chứng đồng thuận chuyên gia.

| Phản biện ghi nhận | Điều chỉnh trong khuyến nghị | Bất định còn lại |
|---|---|---|
| Đây có thể chỉ là correctness head trên linker đã có | E1 bắt buộc existing linker + head và generic head cùng evidence | Chỉ dữ liệu mới phân biệt được |
| Binding head có lợi thế nhãn/candidate | Báo cáo chi phí và cấp cùng supervision cho baseline | Khó cân bằng hoàn toàn capacity và tuning |
| Mốc 10% có thể quá dễ | Chọn alpha theo use case trước test, báo cáo full curve | Chưa biết accuracy và yêu cầu vận hành |
| Sai anchor nhưng giá trị vẫn đúng | Tách provenance khỏi operational correctness | Annotation nhiều chỗ lặp giá trị còn khó |
| Huấn luyện joint làm nhãn correctness thay đổi | Frozen pilot trước; sinh lại nhãn nếu đổi extractor | Khả thi end-to-end chưa được đo |
| Hướng alternative quan sát lại có thể mạnh hơn | Rà soát Q-Zoom/InSight-doc rồi giữ I3 cho thời gian dài hơn | Chưa đủ tìm kiếm để khẳng định novelty của repair-gain objective |

**Quyết định đề xuất:** I2 → pilot; I1 → baseline; I3 → hoãn; I4 → diagnostic. Không có phiếu bầu hoặc điểm số số học để tổng hợp. Nhận định về originality I2 còn bất định trung bình–cao; feasibility chưa kết luận do thiếu thông tin nguồn lực. Đây là khuyến nghị chờ người nghiên cứu quyết định đầu tư, không phải phê duyệt hay cam kết triển khai thí nghiệm.

**Trigger xem xét lại:** kết quả E0/E1; người dùng cung cấp GPU/dataset/deadline; có prior mới trùng cơ chế; quyền dùng dữ liệu hoặc throughput không đáp ứng; nguồn thứ hai không tái hiện lợi ích. Nếu E1 âm, không thêm complexity tự động: xác định lỗi trong hypothesis trước.

## 11. Nguồn gốc và giới hạn tìm kiếm

Ngày tra cứu bổ sung: 28/09/2026. Nguồn ưu tiên: CVF/CVPR, arXiv, AAAI và trang công bố Google Research. Tra cứu tiêu đề chính xác của Valid Per-Field Selective Risk Control, Identify Locate Link, FieldSwap; mở rộng bằng selective VQA, document abstention, unanswerable visually rich documents, adaptive zoom, visual evidence và value of information. Nhật ký query được lưu trong [search-log](cvpr2027-idea-assessment.search-log.json).

Đã xem metadata/abstract nguồn chính cho bảng prior; đối với các công trình gần nhất, dùng thêm full text đã truy cập trong phiên đánh giá và literature review đầu vào. Độ sâu đọc không đồng đều; chưa truy vết citation toàn diện hoặc tái lập code, vì vậy không khẳng định “đầu tiên” hay tìm kiếm có tính hệ thống đầy đủ. Các claim thuật toán chi tiết cần được kiểm tra lại khi chọn baseline triển khai.

CFP 2027 nêu mốc contemporaneous work là 15/09/2026, nhưng vẫn yêu cầu cite/discuss công trình liên quan, đặc biệt công trình ảnh hưởng đề xuất. Do đó không bỏ qua PLV chỉ vì xuất hiện gần deadline. [CVPR 2027 CFP](https://cvpr.thecvf.com/Conferences/2027/CallForPapers)

Đã sử dụng [scientific-brainstorming SKILL.md](../../skills/scientific-brainstorming/SKILL.md) phiên bản 1.2 để tách giả định/bằng chứng/quyết định và tổ chức phản biện. Theo yêu cầu ghi nguồn của skill, báo cáo dẫn: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. [arXiv:2609.00065](https://doi.org/10.48550/arXiv.2609.00065). Đã kiểm tra record hiện hành ngày 28/09/2026; đây là nguồn quy trình hỗ trợ, không phải bằng chứng xác nhận ý tưởng KIE.
