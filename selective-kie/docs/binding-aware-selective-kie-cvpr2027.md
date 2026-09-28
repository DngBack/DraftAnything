# Binding-aware selective KIE: Đề cương nghiên cứu cho CVPR 2027

**Trạng thái:** Đề xuất nghiên cứu, chưa có kết quả thực nghiệm  
**Ngày rà soát tài liệu:** 27/09/2026  
**Mục tiêu:** CVPR 2027 main conference, chủ đề Document analysis and understanding  
**Tên bài dự kiến:** *When the Value Is on the Page but Bound to the Wrong Key: Learning to Reject Visual KIE Errors*

## 1. Quyết định nghiên cứu

**Quyết định:** Tập trung vào lỗi *field binding*: mô hình nhìn thấy và chép đúng một giá trị trong tài liệu nhưng gán nó cho sai trường trong schema. Nghiên cứu xem độ bất định của quan hệ giữa trường được hỏi, nhãn hoặc cấu trúc thị giác, và vùng giá trị có giúp quyết định auto-accept/review tốt hơn confidence của chuỗi đầu ra hay không.

Đây là **ý tưởng và giả thuyết**, không phải kết quả. Không thể dự đoán điểm phản biện hay xác suất được nhận từ đề cương. Điều kiện để thành một bài CVPR mạnh là: lỗi này phải đủ phổ biến trên tài liệu thật; cách đo phải đáng tin; phương pháp phải hơn các đối chứng mạnh trên cùng extractor và tập kiểm thử template chưa thấy.

**Phạm vi chính:** trường trích xuất một span hoặc không có giá trị, trên hóa đơn và biểu mẫu có nhiều giá trị cùng kiểu (ngày, tiền, tên, địa chỉ). Bài toán bảng lặp nhiều dòng, suy luận số học và câu trả lời tổng hợp để dành cho kiểm thử mở rộng.

## 2. Tóm tắt một đoạn

Các hệ thống KIE dựa trên VLM thường cần trả lời cho từng trường của schema rồi tự động duyệt câu trả lời đủ tin cậy. Một failure mode khó phát hiện là *grounded misbinding*: câu trả lời xuất hiện thật trên trang nhưng thuộc về một nhãn, cột hoặc phần khác. Ví dụ, trang có “Total: 118” và “Balance due: 18”, song mô hình trả “total = 18”. Kiểm tra giá trị có xuất hiện trên trang vẫn cho qua. Đề xuất này xây một VLM với đầu dự đoán liên kết field–anchor–value và một quyết định accept/review được học cùng quá trình trích xuất. Mô hình đối chiếu các giá trị cạnh tranh cùng kiểu ngay trong tài liệu, kể cả trạng thái absent và candidate không được OCR bao phủ. Đánh giá chính là coverage tại mức lỗi của tập auto-accept đã định trước, trên tài liệu thật và template chưa thấy, so với confidence thông thường, grounding/layout score và các mô hình reject đơn giản. Phép hoán đổi nhãn hoặc giá trị chỉ là stress test và nguồn hard negatives; nó không chứng minh tính đúng hoặc “nhân quả” của câu trả lời.

## 3. Bối cảnh và khoảng trống thực sự

### 3.1. Những gì đã được làm

| Nhánh | Bằng chứng đã tìm được | Hệ quả cho đề xuất |
| --- | --- | --- |
| Mô hình tự ước lượng độ đúng | Anthropic nghiên cứu P(True)/P(IK) [R01]; Kev huấn luyện adapter và pointer head, sau đó hiệu chỉnh nhiệt độ, nhưng README nêu rõ confidence của choice không phải tỉ lệ chính xác đo được [R02]. | Ý tưởng “cho LM thêm confidence head” tự nó không mới. |
| Học cách từ chối | SelectiveNet tích hợp reject option [R03]. | Cần chứng minh tín hiệu *binding* có ích hơn selector chung. |
| Grounding trong document KIE | LMDX đã trích xuất và định vị [R04]; KVPFormer đã học ghép key–value [R05]; Identify, Locate, Link đã xuất key, value, region và link trong một VLM [R06]. | Không được tuyên bố localization hoặc key–value linking là đóng góp mới. |
| Phép thay key | FieldSwap thay cụm từ chỉ trường và đổi nhãn huấn luyện [R07]. | “Swap key” là đối chứng bắt buộc, không phải novelty. |
| Confidence và auto-accept | ExtractConf dùng nhiều tín hiệu và hai lượt đọc [R08]; PLV dùng perception/layout/validation và selective risk [R09]; ConfBench đánh giá confidence dưới biến dạng ảnh [R10]. | Chỉ cải thiện ECE hoặc thêm tín hiệu OCR/layout khó đủ cho CVPR. |
| Lỗi sai trường và thống kê risk | Nghiên cứu IJDAR ghi nhận nhầm các trường tiền tệ [R11]; nghiên cứu về selective risk theo field chỉ ra phụ thuộc giữa các field cùng tài liệu, rò rỉ khi fit score và vấn đề threshold [R12]. | Failure mode có cơ sở, nhưng tần suất và lợi ích riêng trên VLM hiện đại phải được đo lại; không hứa bảo đảm risk nếu thiết kế thống kê chưa đủ. |

**Khoảng trống được đề xuất, chưa xác nhận là “chưa ai làm”:** đối với VLM trích xuất theo schema, chưa thấy trong tập nguồn đã rà soát một đánh giá có kiểm soát chứng minh rằng uncertainty của *liên kết semantic–spatial giữa field và giá trị* tăng coverage tại cùng mức accepted error trên tài liệu thật, sau khi giữ cố định extractor và so với reject head tổng quát. Tìm kiếm tài liệu này có giới hạn; trước khi viết claim “đầu tiên”, cần rà soát lại công trình mới tới ngày nộp.

### 3.2. Điều không nên lấy làm claim chính

- “Mô hình nhìn đúng vùng thì đáp án đúng”: vùng chứa giá trị có thể thuộc trường khác.
- “Pass counterfactual nên có chứng chỉ nhân quả”: một mô hình có thể đáp ứng phép sửa ảnh nhưng vẫn sai trên ảnh gốc; bản sửa cũng có thể tạo artifact.
- “Calibration bảo đảm rủi ro tối đa 10%”: chỉ đúng nếu thủ tục và giả định được chứng minh phù hợp. Báo cáo empirical accepted risk, khoảng tin cậy và điều kiện lấy mẫu trước.
- “Confidence 0,9 nghĩa là 90% đúng”: chỉ có thể nói như vậy sau kiểm định calibration trên phân phối liên quan.

## 4. Định nghĩa bài toán

Cho tài liệu ảnh \(D\), schema \(F=\{f_1,\ldots,f_m\}\), và nhãn chuẩn \(y^*_{d,f}\) (một giá trị chuẩn hóa hoặc ABSENT). Mô hình xuất cho mỗi field:

\[
o_{d,f}=(\hat y_{d,f},\hat r^{\,a}_{d,f},\hat r^{\,v}_{d,f},s_{d,f},a_{d,f}),
\]

trong đó \(r^a\) là vùng *anchor* (nhãn in trên trang, tiêu đề cột hoặc ngữ cảnh cấu trúc), \(r^v\) là vùng giá trị, \(s\) là score để xếp hạng độ tin cậy và \(a\in\{\text{accept},\text{review}\}\). Anchor có thể là IMPLICIT; không ép một bounding box giả cho trường không có nhãn in rõ.

**Tính đúng vận hành:** sau chuẩn hóa theo kiểu dữ liệu, \(\hat y=y^*\), kể cả ABSENT.  
**Tính đúng của bằng chứng:** vùng giá trị và anchor hỗ trợ đúng field theo đánh giá thủ công hoặc annotation có sẵn. Hai tiêu chí phải báo cáo riêng: một giá trị trùng nhau ở hai chỗ có thể đúng về nội dung nhưng provenance sai.

**Lỗi binding chính:** \(\hat y\neq y^*\); giá trị \(\hat y\) tìm thấy trên trang; giá trị ấy được gắn với một field khác hoặc anchor khác có thể xác định được; và cặp field có kiểu tương thích. Lỗi OCR, sai chữ số, absent hallucination và trường hợp mơ hồ được mã hóa riêng. Khi một giá trị xuất hiện nhiều nơi hoặc gold có tranh cãi, đưa vào nhãn AMBIGUOUS để phân tích độ nhạy thay vì cưỡng ép vào binding error.

Tại threshold \(\tau\), acceptance set \(A_\tau=\{(d,f):s_{d,f}\ge\tau\}\). Các thước đo:

\[
\text{Coverage}(\tau)=|A_\tau|/N,\qquad
\text{Risk}(\tau)=\sum_{(d,f)\in A_\tau}\mathbf{1}[\hat y_{d,f}\neq y^*_{d,f}]/|A_\tau|.
\]

Nếu \(A_\tau\) rỗng, risk không xác định và coverage bằng 0. Báo cáo kết quả theo field và tổng hợp theo tài liệu; tách trường ABSENT, giá trị tiền, ngày, và các nhóm template. Không chọn threshold bằng test set.

## 5. Câu hỏi, giả thuyết và phép bác bỏ

| ID | Phát biểu cần kiểm chứng | Bằng chứng bác bỏ hoặc làm yếu |
| --- | --- | --- |
| H1 — failure mode | Trong số các lỗi có confidence đầu ra cao trên tài liệu thật, binding error đóng góp đáng kể cho lỗi auto-accept. | Audit cho thấy loại lỗi này hiếm; hoặc một kiểm tra vị trí đơn giản đã bắt gần hết. |
| H2 — tín hiệu độc lập | Tín hiệu cạnh tranh giữa các liên kết field–anchor–value dự báo lỗi sau khi đã kiểm soát output logprob, độ rõ ảnh, OCR và khoảng cách hình học. | Khi thêm các biến đối chứng, tín hiệu không cải thiện ranking/risk–coverage. |
| H3 — lợi ích vận hành | Phương pháp đạt thêm ít nhất 5 điểm phần trăm coverage tại target empirical accepted risk 10%, so với đối chứng mạnh nhất cùng extractor, trên tập template chưa thấy. | Delta nhỏ hơn ngưỡng, khoảng tin cậy bao gồm 0, hoặc risk vượt target. |
| H4 — khả năng tổng quát | Lợi ích còn hiện diện trên ít nhất một nguồn tài liệu hoặc kiểu layout thứ hai; không chỉ ở ảnh chỉnh sửa. | Chỉ thắng trên synthetic edit hoặc một template cụ thể. |

Ngưỡng 5 điểm phần trăm và 10% là **tiêu chí quyết định đề xuất**, chưa phải kết quả hay định mức chung của ngành. Nếu use case cần mức lỗi 1–5%, sẽ báo cáo thêm nhưng không tuyên bố đạt mức đó khi số tài liệu độc lập không đủ.

## 6. Phương pháp đề xuất

### 6.1. Thiết kế khả thi

1. **Extractor chung:** chọn một VLM open-weight phù hợp GPU sẵn có, fine-tune cùng một schema và giao thức cho mọi phương pháp. Mô hình xử lý ảnh; OCR cung cấp các proposal span và box cho phương pháp của ta **và** đối chứng cần OCR. Lưu rõ chất lượng và candidate recall của OCR.
2. **Candidate set theo field:** gồm các span cùng kiểu xuất hiện trên trang, ABSENT, UNREADABLE và OTHER/UNCOVERED. OTHER tránh ép mô hình tự tin vào tập OCR candidate thiếu giá trị đúng. Chỉ dùng thông tin có ở inference; không đưa gold box vào candidate set test.
3. **Binding head trong VLM:** lấy biểu diễn của field, anchor, candidate value, vị trí và ngữ cảnh ảnh để cho điểm bộ ba \((f,r^a,r^v)\). Score cần thể hiện *cạnh tranh*: ví dụ “total” so với “balance due” khi cả hai đều có một số tiền hợp lệ. Chỉ một scalar confidence của token không cho biết quan hệ nào đang bị tranh chấp.
4. **Huấn luyện chung:** loss trích xuất giá trị, loss định vị/link khi có nhãn, và mục tiêu xếp hạng cho hard negatives lấy từ các field cùng kiểu trong **cùng tài liệu**. Thêm mục tiêu accept/review theo lỗi dự đoán thực của model trên dữ liệu fit/validation độc lập; tránh tạo nhãn “đúng/sai” chỉ bằng teacher forcing trên câu trả lời gold. Không khẳng định có bảo đảm thống kê từ loss này.
5. **Inference:** một yêu cầu suy luận cho tài liệu và schema, không dùng verifier model riêng hoặc test-time chỉnh ảnh. Trả giá trị, bằng chứng, các đối thủ liên kết mạnh nhất và score. Chọn threshold trên calibration set tách biệt.

**Đặc tả score thử nghiệm:** với field \(f\), tạo tập proposal gồm anchor \(a_i\) và giá trị \(v_j\). Binding head tính logit \(z_{fij}=g_\theta(h_f,h_{a_i},h_{v_j},\text{geometry}_{ij},h_D)\), cùng logit cho ABSENT và UNCOVERED. Chuẩn hóa trên các cặp hợp lệ để có phân phối liên kết; giữ logit tốt nhất, margin với đối thủ mạnh nhất và entropy của phân phối. Một đầu score nhỏ **nằm trong cùng mô hình** dùng các đại lượng đó và tín hiệu đọc giá trị để xếp hạng accept/review. Softmax thô không được gọi là xác suất đúng đã hiệu chỉnh. Không ép ràng buộc một-một toàn cục: hai field hợp lệ có thể chia sẻ một giá trị; chỉ thử ràng buộc loại trừ khi schema xác định nó.

**Đặc tả huấn luyện:** loss \(L=L_{\text{extract}}+\lambda_1L_{\text{link}}+\lambda_2L_{\text{hard-negative}}+\lambda_3L_{\text{select}}\). Loss link chỉ áp dụng khi có gold anchor/value phù hợp; hard negatives là các giá trị cùng kiểu nhưng thuộc field khác. Để học selector trên lỗi thật, lấy dự đoán *out-of-fold* từ extractor, gán đúng/sai bằng gold, rồi fit selector trên phần dữ liệu riêng. Nếu cập nhật extractor và selector chung sau đó, phải đo lại calibration và tránh dùng chính các dự đoán đã thấy ở test. Các \(\lambda\), kích thước proposal và giao thức cập nhật được chốt trên validation.

Đây là đặc tả kiến trúc mục tiêu. Việc có thể triển khai mọi field trong một lượt với bộ nhớ và độ trễ chấp nhận được phải được đo; nếu không, báo cáo trung thực số lượt/field và chi phí tương ứng. Một đầu liên kết tích hợp không mặc nhiên “rẻ hơn” hệ post hoc.

### 6.2. Vai trò đúng của counterfactual

Tạo ba loại cặp có gold xác định được: (i) thay nhãn field theo FieldSwap; (ii) thay một giá trị cùng kiểu mà vẫn giữ nhãn; (iii) thêm một giá trị gây nhiễu nhưng không thay quan hệ đúng. Cặp (i) và (ii) có thể yêu cầu đáp án đổi; cặp (iii) nên giữ đáp án. Mỗi edit phải được kiểm tra tính hợp lệ ngữ nghĩa, nhất quán số liệu và dấu vết render. Dùng chúng cho hard negatives và stress test; **kết luận chính phải đến từ tài liệu thật chưa chỉnh sửa**.

## 7. Kế hoạch dữ liệu và annotation

| Nguồn | Vai trò | Giới hạn cần ghi |
| --- | --- | --- |
| DocILE [R13] | KIE trên tài liệu kinh doanh, có nhãn field và box; tập chính nếu cấp quyền truy cập kịp. | Repo yêu cầu token tải dữ liệu; nhãn giá trị không tự động là nhãn key/anchor. |
| VRDU [R14] | Kiểm tra transfer, đặc biệt unseen-template split có sẵn. | Một số field lặp hoặc phức tạp nằm ngoài phạm vi chính; cần chọn trước nhóm field hợp lệ. |
| FUNSD/XFUND [R15, R16] | Tiền huấn luyện/kiểm tra module liên kết key–value. | FUNSD chỉ có 199 ảnh; không đủ làm bằng chứng chính cho accepted risk thấp. |
| ConfBench [R10] | Kiểm tra phụ về biến dạng ảnh và calibration. | Ảnh biến dạng là phân phối kiểm thử riêng; không thay cho tài liệu tự nhiên. |

**Pilot audit:** lấy mẫu tài liệu thật ngẫu nhiên, dự kiến 200–300 tài liệu hoặc số tối đa có thể gán nhãn nhanh trong tuần đầu. Chạy extractor đã đóng băng, lưu mọi field request, prediction, score, gold và vùng liên quan. Audit toàn bộ các lỗi confidence cao trong mẫu **và** một mẫu ngẫu nhiên của tất cả dự đoán để ước lượng tần suất tổng thể, tránh selection bias. Ít nhất 20% mẫu audit được hai người gán nhãn độc lập; có quy tắc giải quyết bất đồng. Không xem synthetic edit là natural error.

**Bộ dữ liệu chính:** dùng toàn bộ nhãn field có sẵn cho risk–coverage; chỉ cần gán bổ sung anchor và taxonomy trên subset được lấy mẫu có kiểm soát. Quy mô test và số field sẽ được tính lại từ pilot theo số tài liệu/template độc lập và số lỗi quan sát được; không coi hàng nghìn field trong vài trăm tài liệu là hàng nghìn mẫu độc lập.

**Quy tắc chia tập:** group theo tài liệu gốc và mọi bản render/augmentation của nó; group theo template hoặc layout cluster cho unseen-template test; loại near-duplicates xuyên tập; tách training, score fitting, threshold calibration và test. Không fit lại score hoặc threshold trên test.

## 8. Thí nghiệm tối thiểu để bài có giá trị

### E0 — Audit và kiểm tra trần lợi ích

Đo phân bố lỗi trên tài liệu thật. Trên prediction và ranking đã đóng băng, dựng một *oracle chẩn đoán* chỉ biết nhãn “binding error” để ước lượng lợi ích tối đa của việc phát hiện hoàn hảo nhóm lỗi này trong thiết lập đã khóa. Oracle không phải phương pháp có thể triển khai và không phải upper bound cho mọi phương pháp KIE. Nếu lợi ích khả dĩ dưới 5 điểm phần trăm coverage tại mức risk mục tiêu, dừng claim trung tâm.

### E1 — Đối chứng confidence cùng extractor

So với: first-token và mean-token logprob; verbalized confidence; temperature/isotonic calibration; độ khớp text/box và khoảng cách layout; một correctness head tổng quát có ngân sách tham số và nhãn tương đương; fusion score mạnh từ OCR/layout/validation; và các cấu hình mô phỏng khả thi của ExtractConf/PLV. Nếu không tái tạo được đúng giao thức của công trình gốc, ghi rõ “reimplementation” và chỉ dùng số công bố làm bối cảnh.

### E2 — Đối chứng kiến trúc và dữ liệu

So cùng backbone, resolution, OCR, train split và compute budget: base extractor; base + FieldSwap; base + localization/link; base + reject head chung; base + binding head; bản đầy đủ. Làm thêm một so sánh **frozen extractor**: chỉ thay score quyết định accept/review để biết cải thiện có thật từ uncertainty hay chỉ do extractor trích đúng hơn.

### E3 — Tổng quát và stress test

Primary: natural unseen-template. Secondary: nguồn thứ hai, OCR noise/ảnh xấu, absent field, giá trị cùng kiểu nằm gần nhau, giá trị lặp, và các cặp counterfactual được người kiểm tra. Báo cáo candidate recall và failure khi OCR bỏ sót gold. So sánh độ trễ và số GPU-seconds/tài liệu để thấy lợi ích triển khai thật.

### E4 — Phân tích lỗi

Xem các trường auto-accepted sai còn lại: binding, OCR/transcription, absent hallucination, normalization, annotation ambiguity. Lấy ví dụ đúng và sai theo từng nhóm. Nếu các cải thiện chỉ đến từ loại lỗi không liên quan đến binding, sửa câu chuyện paper thay vì gán công lao cho giả thuyết đã chọn.

## 9. Phân tích thống kê và tiêu chí báo cáo

- **Primary endpoint:** chênh lệch coverage tại target accepted risk 10% trên test unseen-template, threshold chọn hoàn toàn từ validation/calibration. Báo cáo cả risk thực tế; nếu risk vượt 10%, phương pháp không đạt primary target dù coverage cao.
- **Đơn vị độc lập:** tài liệu; khi kết luận về unseen templates, còn phải đánh giá biến thiên giữa template/layout cluster. Bootstrap ghép cặp theo tài liệu; phân tích độ nhạy theo template và nguồn.
- **So sánh chính định trước:** phương pháp đầy đủ với đối chứng mạnh nhất được chọn bằng validation, không chọn baseline sau khi xem test. Báo cáo khoảng tin cậy của chênh lệch coverage và risk, không chỉ p-value.
- **Chỉ số phụ:** AURC, accuracy/F1 của extractor, lỗi binding được accept, lỗi absent được accept, ECE/Brier sau calibration, candidate recall, bằng chứng đúng, latency/cost. ECE đẹp không thay thế risk–coverage.
- **Nhiều thử nghiệm:** chỉ một primary contrast; các ablation, subgroup và mức risk khác là secondary/exploratory trừ khi có kế hoạch điều chỉnh multiplicity.
- **Giới hạn bảo đảm:** field cùng tài liệu có tương quan; score refit và threshold ties có thể làm các thủ tục risk control ngây thơ mất hiệu lực [R12]. Nếu không chứng minh được giả định và cỡ mẫu, chỉ kết luận empirical risk có khoảng tin cậy. Ví dụ, với **một quyết định accept trên mỗi tài liệu độc lập**, 0 lỗi trên 300 quyết định cho cận trên một phía 95% xấp xỉ 1%; 300 field tương quan trong ít tài liệu không đủ để diễn giải như vậy.

## 10. Rủi ro phản biện và cách xử lý

| Rủi ro | Mức | Phản ứng trong thiết kế |
| --- | --- | --- |
| Reviewer nói “FieldSwap + KVPFormer + SelectiveNet” | Rất cao | Không claim từng thành phần mới. Trọng tâm là phát hiện failure mode tự tin cao có định nghĩa và audit; thí nghiệm frozen extractor; lợi ích lớn trên natural unseen-template; so với tổ hợp trực tiếp của các prior đó. |
| Binding error ít gặp | Rất cao | E0 là gate sớm. Nếu oracle cho lợi ích nhỏ, không cố cứu bằng bộ ảnh nhân tạo. |
| Synthetic edits chứa artifact | Cao | Human validity audit, render chuẩn, kiểm tra khả năng phân biệt ảnh edit/original; kết quả chính trên ảnh gốc. |
| Key/anchor không được gán nhãn | Cao | Annotation subset với hướng dẫn rõ; tách explicit, structural, implicit; không lấy heuristic làm gold mà không kiểm tra. |
| Model mạnh hơn chỉ nhờ accuracy | Cao | Frozen-extractor score comparison và accuracy-matched ablation. |
| OCR proposal bỏ sót gold | Cao | Báo cáo candidate recall và trạng thái UNCOVERED; cùng OCR cho baseline tương ứng; đánh giá riêng theo OCR quality. |
| Risk claim vượt dữ liệu | Cao | Split theo tài liệu/template, calibration tách test, CI theo cluster và ngôn ngữ claim chính xác. |
| Compute/annotation không kịp CVPR | Cao | Không xây benchmark lớn trước E0; một backbone chính, một nguồn chính, replication gọn. |

## 11. Các mốc ra quyết định

CVPR 2027 ghi hạn đăng ký bài **10/11/2026 AoE**, hạn nộp **16/11/2026 AoE** [R17]. Lịch dưới đây là kế hoạch làm việc, không phải cam kết rằng thí nghiệm sẽ thành công.

| Hạn đề xuất | Đầu ra bắt buộc | Quyết định |
| --- | --- | --- |
| 04/10 | Quyền truy cập dữ liệu; extractor nền; giao thức đánh giá; 50 tài liệu audit thử và taxonomy gán nhãn | Nếu chưa có dữ liệu thật khả dụng, chuyển ngay sang nguồn mở phù hợp hoặc thu hẹp scope. |
| 11/10 | Audit 200–300 tài liệu, estimate tỉ lệ binding error, oracle chẩn đoán, ba baseline score | **Go** chỉ khi có tín hiệu thực và trần lợi ích khoảng ≥5 điểm coverage. |
| 25/10 | Full model và ablation trên validation; kiểm tra extraction accuracy, candidate recall, chi phí | **Go** nếu lợi ích còn sau frozen-extractor control; nếu không, không đẩy claim “learning to reject binding”. |
| 06/11 | Test untouched, CI, error audit, figures và draft paper | Chốt claim theo dữ liệu; không đổi primary endpoint sau khi xem test. |
| 10/11 và 16/11 | Đăng ký và nộp bài theo CFP | Kiểm tra lại quy định hội nghị, nguồn mới và giới hạn tái lập. |

## 12. Đóng góp kỳ vọng của bài, nếu giả thuyết sống sót

1. **Đo lường:** taxonomy và tập đánh giá được audit cho grounded misbinding, có tần suất tự nhiên, mức tự tin, provenance và độ tin cậy giữa người gán nhãn.
2. **Phương pháp:** một cơ chế học accept/review từ cạnh tranh của các liên kết field–anchor–value trong VLM, với một lượt suy luận tài liệu và kiểm soát chi phí minh bạch.
3. **Bằng chứng:** so sánh risk–coverage trên tài liệu thật và template chưa thấy, tách đóng góp của extraction accuracy khỏi confidence quality, cùng giới hạn thống kê được công bố rõ.

**Một câu pitch:** *A document value can be visually grounded and still belong to the wrong field; reliable selective KIE must measure uncertainty in the binding, not only confidence in the value string.*

## 13. Quyết định nếu kết quả âm tính

Nếu binding error hiếm, hoặc mô hình chỉ thắng trên dữ liệu chỉnh sửa, hoặc reject head đơn giản đạt cùng kết quả, **không đóng gói thành claim CVPR mạnh**. Kết quả âm tính vẫn hữu ích: công bố taxonomy và audit với giới hạn rõ, hoặc chuyển trọng tâm sang failure mode có sức nặng hơn sau một vòng rà soát mới. Không đổi giả thuyết sau khi xem test rồi trình bày như đã định trước.

## 14. Tài liệu nguồn đã đối chiếu

- **[R01]** Anthropic, [Language models (mostly) know what they know](https://www.anthropic.com/research/language-models-mostly-know-what-they-know), 2022.
- **[R02]** Jared Palmer, [Kev repository and README](https://github.com/jaredpalmer/kev), truy cập 27/09/2026. Repo phân biệt choice confidence với measured accuracy.
- **[R03]** Geifman và El-Yaniv, [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html), ICML 2019.
- **[R04]** Perot và cộng sự, [LMDX: Language Model-based Document Information Extraction and Localization](https://research.google/pubs/lmdx-language-model-based-document-information-extraction-and-localization/), ACL Findings 2024.
- **[R05]** Hu và cộng sự, [A Question-Answering Approach to Key Value Pair Extraction from Form-Like Document Images](https://ojs.aaai.org/index.php/AAAI/article/view/26516), AAAI 2023.
- **[R06]** Gürbüz và cộng sự, [Identify, Locate, Link](https://arxiv.org/abs/2608.20868), ICDAR 2026.
- **[R07]** Xie và cộng sự, [FieldSwap: Data Augmentation for Effective Form-Like Document Extraction](https://jameswendt.com/publications/2024_ICDE_Xie.pdf), ICDE 2024.
- **[R08]** Kumar, [Beyond Logprobs / ExtractConf](https://arxiv.org/abs/2606.24420), 2026 preprint/workshop extended version.
- **[R09]** Jin và cộng sự, [Perception, Layout, and Validation](https://arxiv.org/abs/2609.20110), 17/09/2026 preprint. Bài xuất hiện sau mốc contemporaneous của CVPR nhưng vẫn phải trích dẫn và thảo luận vì đã ảnh hưởng đề xuất này.
- **[R10]** Roy và cộng sự, [ConfBench](https://arxiv.org/abs/2608.01792), 2026 preprint.
- **[R11]** Rombach và Mehdiyev, [Beyond Accuracy: Understanding Model Confidence in KIE with Conformal Prediction](https://link.springer.com/article/10.1007/s10032-026-00572-y), IJDAR 2026.
- **[R12]** Gurram, [Valid Per-Field Selective Risk Control for Document Extraction](https://arxiv.org/abs/2608.14639), 2026 preprint.
- **[R13]** Rossum AI, [DocILE dataset and benchmark repository](https://github.com/rossumai/docile).
- **[R14]** Google Research, [VRDU dataset repository and unseen-template splits](https://github.com/google-research-datasets/vrdu).
- **[R15]** Jaume và cộng sự, [FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents](https://arxiv.org/abs/1905.13538), 2019.
- **[R16]** Xu và cộng sự, [LayoutXLM / XFUND multilingual forms](https://arxiv.org/abs/2104.08836), 2021.
- **[R17]** CVPR, [2027 Call for Papers](https://cvpr.thecvf.com/Conferences/2027/CallForPapers), truy cập 27/09/2026.
- **[R18]** Kassis, Agarwal, He, Patel và Brueckner, [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://arxiv.org/abs/2609.00065), 2026. Skill scientific-brainstorming và scientific-critical-thinking đã được dùng để lập đề cương và phản biện, không phải nguồn chứng minh giả thuyết KIE.

**Giới hạn rà soát:** đây là kiểm tra tài liệu có chủ đích quanh novelty và thực nghiệm, chưa phải systematic review. Các kết quả của preprint được trình bày như báo cáo của tác giả, chưa được xác nhận độc lập trong đề cương này.
