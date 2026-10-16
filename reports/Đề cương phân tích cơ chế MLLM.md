# Đặt cược vào chuyển giao, không vào hiện tượng

Bài CVPR 2027 về cơ chế MLLM không thể dựa trên hiện tượng "mô hình mã hóa đúng thông tin thị giác nhưng không dùng". Hiện tượng này đã có **hơn 15 bài 2025–26, mỗi bài chỉ xét một họ lỗi**. OpenReview còn hiện **ít nhất 7 bài nộp ICLR 2027** với claim "readout muộn" gần như trùng. CVPR 2026 đã trao Highlight cho một bài cùng khung ("Same Attention, Different Truths"). Phần còn trống là một **claim chuyển giao** mà chưa bài nào thử: lỗi thuộc màu, đếm, không gian, nhận dạng tinh và chữ nhỏ dùng chung một subspace "bằng chứng → câu trả lời" hạng thấp. Hướng patch hoặc steer ước lượng ở họ lỗi này phải sửa được họ lỗi khác. Một cách sửa không cần huấn luyện, dùng **một cấu hình cố định trên ≥3 họ mô hình**, phải thắng các baseline calibration rẻ tiền. Đề cương này xây bài quanh giả thuyết **"route-then-read"**. Bằng chứng được *định tuyến* qua image token ở độ sâu giữa, rồi bị một *dải readout* muộn ở vị trí câu trả lời bỏ rơi. Khung này hòa giải được ba kết quả đang mâu thuẫn: locus ở giữa tại image token, locus muộn tại token cuối, và kết quả patching null. Bài đi kèm hai artifact: chỉ số chẩn đoán **ReadGap Score (RGS)** và cách sửa **ReRead**. Có ba chương dùng chung một pipeline: crowding (Bouma's law tính theo token), metacognition (meta-d′ và cổng "đọc lại hay nhìn lại"), và RL ladder (RL đổi cách đọc chứ không đổi cái nhìn). Kết luận là **GO có điều kiện**. Mọi thứ chạy được trên một H200 với khoảng **240 GPU-giờ** (ước lượng). Pilot phải trả lời trước **20/10**. Nếu không có chuyển giao xuyên họ lỗi, bài vẫn sống nhờ kế hoạch pivot đã định sẵn. Highlight là mục tiêu kéo dài chứ không phải kỳ vọng cơ sở: mức này chỉ chiếm khoảng 3,6% số bài nộp năm 2026 (578/16.092) ([Itzik Ben-Shabat](https://itzikbs.com/blog/posts/2026-06-01-cvpr-2026-survival-guide)).

## Hiện tượng đã chật, chỉ còn cửa chuyển giao và hòa giải

### Phán quyết novelty: phần hiện tượng đã có chủ

Bằng chứng "mã hóa nhưng không dùng" đã phủ gần hết các họ lỗi, mỗi họ do một nhóm riêng làm, với mô hình, metric và locus riêng:

- **Thuộc tính so với language prior:** probe AUC > 0,99 ở độ sâu giữa trên 10 VLM. Full-sequence patching lật 60–84% output, còn last-token patching chỉ lật 0–1% ([arXiv 2604.09364](https://arxiv.org/html/2604.09364)).
- **Đếm:** probe trên token cuối gần hoàn hảo, nhưng hướng probe GT và hướng probe output lệch nhau ([arXiv 2607.09544](https://arxiv.org/pdf/2607.09544)).
- **Không gian:** hướng không gian đọc được ở L4–8 nhưng chỉ được vận chuyển nhân quả ở L12–32 ([arXiv 2608.22916](https://arxiv.org/html/2608.22916v1)).
- **Chữ trong ảnh:** các tầng sâu nhìn đúng vùng bằng chứng ngay cả khi trả lời sai ([arXiv 2510.17771](https://arxiv.org/html/2510.17771v1)).
- **Nhãn thứ bậc:** hàng unembedding của chữ số giữ lại **< 1,15%** hướng thứ bậc ([arXiv 2608.20999](https://arxiv.org/abs/2608.20999)).
- **Tác vụ vision-centric:** VLM chỉ đạt 0,170 ở low-level matching, trong khi probe trên encoder của chính nó đạt 0,819 ([arXiv 2506.08008](https://arxiv.org/html/2506.08008v1)).

Ở tầng venue, bức tranh cũng đã rõ. "Same Attention, Different Truths" là Highlight CVPR 2026. Bài này cho thấy vật thật và vật bị hallucinate nhận attention mạnh ngang nhau ở các tầng giữa và muộn, rồi dùng logit lens để phát hiện và giảm hallucination ([CVPR 2026 Highlights](https://cvpr.thecvf.com/virtual/2026/events/Highlights2026)). "Visual Symbolic Mechanisms" thành oral ICLR 2026 với điểm chỉ 6/6/6/8 và không có cách sửa ([OpenReview](https://openreview.net/forum?id=3RQ863cRbx)). OpenReview đang hiện các bài nộp ICLR 2027 có claim gần như trùng:

- "Causal Readout Dependence": khôi phục 89,1% hiệu ứng ảnh dưới forced choice, nhưng chỉ 14,8% lúc sinh câu trả lời ([OpenReview](https://openreview.net/forum?id=dORR3BQrmq)).
- "Many Paths, One Coordinate" ([OpenReview](https://openreview.net/forum?id=HBp4uTW5b0)).
- "Object Hallucinations Are Written by Late Feed-Forward Layers" ([OpenReview](https://openreview.net/forum?id=tunVuLrXNu)).
- "Vision-Language Models Do Not Keep Looking" ([OpenReview](https://openreview.net/forum?id=DnUl5Cio8T)).
- "Now You See It, Now You Don't" ([OpenReview](https://openreview.net/forum?id=q92eqsV8QI)).
- TRACE ([OpenReview](https://openreview.net/forum?id=KS9coFbr9Q)).
- "Beyond Feature Fidelity" ([OpenReview](https://openreview.net/forum?id=1ki2Qs08aF)).

Một bài chỉ nói "MLLM thấy mà không nói" sẽ bị xếp vào Findings track. Track này được lập năm 2026 cho "technically sound papers with more incremental novelty" ([CVPR 2026 Author Guidelines](https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines)).

### Hai đối thủ trực tiếp với claim "một cơ chế chung"

Có hai bài đi ngược lại claim "một cơ chế chung", và đây là chỗ đề cương phải đối diện thẳng.

Bài thứ nhất là **arXiv 2610.00024** (1/9/2026) ([arXiv](https://arxiv.org/html/2610.00024)). Bài này đăng **trước** mốc contemporaneous 15/9/2026 của CVPR 2027, nên bắt buộc phải so sánh chứ không chỉ trích dẫn ([CVPR 2027 CFP](https://cvpr.thecvf.com/Conferences/2027/CallForPapers)). Nó gọi hiện tượng "encoded-but-disconnected" là phổ quát, nhưng báo cáo **null khi patching**:

- Cross-instance residual patching chỉ lật 0/50 lỗi trên LLaVA và 0/44 trên Qwen.
- Per-head patching trên Qwen cho 0 lần lật qua 12.600 lần chạy.
- Tác giả kết luận cơ chế khác nhau theo kiến trúc.

Bài thứ hai là **arXiv 2603.18523** ([arXiv](https://arxiv.org/html/2603.18523v1)). Bài này tìm được một mạch đếm "chia sẻ" giữa nhiều tác vụ thị giác, nhưng chuyển giao chỉ cho **+1,54%** trung bình trên tác vụ OOD (số liệu từ snippet, chưa đọc toàn văn). Ngoài ra, 2607.16094 thấy mỗi phép toán thị giác có locus riêng: chọn vật nằm ở FFN, quan hệ nhiều bước nằm ở attention muộn ([arXiv](https://arxiv.org/abs/2607.16094)).

Locus cũng đang mâu thuẫn trực tiếp giữa các bài. Ở 2604.09364, điểm phân xử nằm ở **36–71% độ sâu, tại image token**, và patch token cuối không có tác dụng. Ở 2606.06890, độ dốc ghép nối nhảy từ β = 0,123 lên **0,876** trong một **dải muộn tại token cuối**, bắt đầu từ độ sâu chuẩn hóa 0,59–0,77 ([arXiv 2606.06890](https://arxiv.org/html/2606.06890)).

### Sáu khoảng trống còn mở

Sáu khoảng trống dưới đây còn mở theo những gì ghi chú tìm được. Mỗi khoảng là một đóng góp có thể bảo vệ.

1. **Subspace hạng thấp dùng chung xuyên họ lỗi.** Phải kèm thí nghiệm chuyển giao: subspace fit trên họ A phải patch hoặc steer được họ B. Chưa bài nào báo cáo con số này.
2. **Hòa giải các locus mâu thuẫn.** Image token ở giữa và token cuối ở dải muộn có thể là hai giai đoạn của cùng một pipeline. Kèm theo đó là lời giải thích cho null của 2610.00024: patch sai vị trí hoặc sai cơ sở.
3. **Một cách sửa không huấn luyện chuyển được giữa các họ.** Cách sửa phải thắng calibration. Calibration ngưỡng thuần đã khôi phục 69,8% khoảng cách probe ở bài toán meme ([arXiv 2609.18860](https://arxiv.org/abs/2609.18860)). Một hằng số bias không cần ảnh đạt 86,86% trên POPE, vượt mọi phương pháp contrastive decoding (85,89%) ([Rethinking-CD](https://github.com/Aurnawr/Rethinking-CD)).
4. **Thang base → RL theo từng tầng.** Chưa có phân tích nào như vậy cho VLM.
5. **Tách metacognition theo loại lỗi.** Câu hỏi là meta-d′ có khác nhau giữa lỗi do không thấy và lỗi do không biết hay không. Chưa có bài VLM nào đo.
6. **Hàm psychometric của crowding.** Chưa ai đo ở VLM.

Ba khoảng sau là chương phụ. Ba khoảng đầu là xương sống của bài.

### Đánh giá go/no-go và lối thoát nếu pilot thất bại

Kết luận là **GO có điều kiện**, vì ba lý do. Chi phí thấp: chỉ cần inference, hook và patching trên mô hình 7–12B. Mỗi kết cục của pilot đều dẫn tới một bài nộp được. Và claim chuyển giao, nếu đúng, là thứ một AC có thể nhắc lại trong một câu khi đề cử Highlight.

Rủi ro lớn nhất là bản thân claim chuyển giao. Ba bằng chứng nói trên (2610.00024, 2607.16094, mức chuyển giao +1,54% của 2603.18523) đều nghiêng về các cơ chế riêng theo kiến trúc hoặc theo phép toán. Theo đánh giá của tôi, khả năng G2 (chuyển giao xuyên họ lỗi) thành công **không cao hơn một nửa**. Vì vậy kế hoạch pivot phải được viết ra trước khi chạy.

| Cổng | Ngày | Tiêu chí đạt | Nếu trượt |
|---|---|---|---|
| **G1 – pilot** | 20/10 | Trên 2 họ lỗi (màu-prior, đếm) × 2 mô hình (Qwen3-VL-8B, InternVL3.5-8B): (a) khoảng probe–output ≥ 10 điểm trên tập lỗi, với selectivity dương so với control task; (b) có ít nhất một vị trí hoặc cơ sở patch (nhóm image token hoặc subspace) lật ≥ 30% lỗi trên cặp phản thực, trong khi last-token cross-instance swap tái tạo được null của 2610.00024 | Nếu probe cũng thất bại (lỗi do tri giác), pivot sang headline **RL ladder**. Chương này không cần patch non-null mà chỉ hỏi RL thay đổi tầng nào. Nếu probe tốt nhưng không patch nào có tác dụng, pivot sang headline **metacognition gate** ("VLMs know what they don't know, not what they didn't see"). Hướng này chỉ cần tính giải mã được cộng với giá trị của cổng |
| **G2 – chuyển giao** | 27/10 | Subspace fit trên một họ lỗi đạt ≥ 50% interchange accuracy so với trong-họ, trên ≥ 2 họ khác, ở ≥ 3/4 mô hình lõi, và vượt phân vị 95 của subspace ngẫu nhiên | Đổi sang tiêu đề T2: route-vs-read theo từng họ lỗi, RGS dự đoán cách sửa nào hiệu quả. ReRead giữ một *công thức* chung nhưng subspace riêng theo họ lỗi |
| **G3 – cách sửa** | 3/11 | ReRead với một cấu hình cố định thắng max{calibration, always-reprompt}, với khoảng tin cậy bootstrap ghép cặp loại trừ 0, trên ≥ 3 họ mô hình × ≥ 2 benchmark thật. Benchmark tổng quát giảm < 0,5 điểm | Hạ ReRead xuống thành ablation. Artifact chính chuyển sang cổng metacognition RGS (đọc lại/nhìn lại/trả lời), so với CounterCredit. Nếu cả G2 và G3 đều trượt, bài có nguy cơ rơi vào Findings. Khi đó cân nhắc dời sang venue sau (ngày chưa xác minh) |

## Đề xuất: route-then-read với một subspace dùng chung

### Ba tiêu đề ứng viên theo kết quả cổng

Tiêu đề phải nêu **giai đoạn** và **hướng** của claim, và tránh các cụm đã bị chiếm. Các cụm đã bị chiếm gồm "know where to look", "seeing but not", "knowing isn't saying", "hidden in plain sight", "many paths, one coordinate" và "readout dependence". Bảng dưới là phương án theo từng kết cục. Phải đối chiếu lại với danh sách ICLR 2027 trước khi đăng ký ngày 10/11.

| Mã | Dùng khi | Tiêu đề |
|---|---|---|
| T1 | G2 và G3 đạt | **Routed, Then Dropped: One Readout Subspace Explains Five Kinds of Multimodal LLM Failure** |
| T2 | G1 đạt, G2 trượt | **Routed but Not Read: Locating Where Multimodal LLMs Lose Visual Evidence, and Predicting Which Fix Works** |
| T3 | Pivot sang RL | **RL Teaches Multimodal LLMs to Read, Not to See** |

### Abstract nháp (T1; số trong ngoặc vuông là chỗ trống chờ kết quả, không phải kết quả)

> Multimodal LLMs often get plainly visible facts wrong — they miscount, misplace and misname — even when their own hidden states linearly encode the correct answer. Prior work documents this gap one failure type at a time and disagrees on where it arises: at mid-depth image tokens, in a late band at the answer token, or nowhere a patch can reach. We show these accounts describe two stages of one pipeline. Across five failure families and [N] open MLLMs spanning [M] language backbones — including an encoder-free and a cross-attention model — visual evidence is routed into the answer position at mid-depth and then dropped by a late readout band. The dropped evidence lives in a low-rank subspace shared across failure families: a subspace estimated from colour counterfactuals recovers [X]% of in-family interchange accuracy on counting, spatial and fine-grained pairs, and position-correct patching overturns a recently reported patching null. We turn this account into two tools. ReadGap Score measures, per item, how much decodable evidence the output ignores; it predicts errors and separates "failed to read" from "failed to see" (type-2 AUROC [X] vs. [Y] for output confidence). ReRead restores the evidence subspace at the readout band when the gap is large; with a single configuration fixed on synthetic stimuli, it improves [benchmarks] by [X] points on [N] model families, beating calibration and re-prompting baselines without hurting general ability. Visual crowding makes the split visible on one curve: the encoder resolves letters at spacings where the answer fails, and ReRead helps only inside that gap. Finally, RL post-training moves the readout band, not the visual code.

### Giả thuyết trung tâm và các dự đoán bác bỏ được

**Giả thuyết trung tâm (H\*).** Ở MLLM dạng prefix-token, phần lớn lỗi tri giác "dễ" là **lỗi đọc**: bằng chứng có mặt và dùng được về mặt nhân quả ở độ sâu giữa, nhưng không được chuyển thành câu trả lời bởi một dải readout muộn. Bằng chứng này nằm trong một subspace hạng thấp dùng chung giữa các họ lỗi.

Mọi giả thuyết con đều có ngưỡng giết đặt trước. Các ngưỡng này được commit vào repo kèm timestamp trước khi chạy số chính.

| ID | Giả thuyết con | Dự đoán bác bỏ được | Kết quả giết nó |
|---|---|---|---|
| H1 | Khoảng "giải mã được nhưng không dùng" phổ biến xuyên họ lỗi | Khoảng probe–output trên tập lỗi ≥ 10 điểm ở ≥ 3/5 họ lỗi trên ≥ 3/4 mô hình lõi. Probe tuyến tính có selectivity dương so với control task kiểu Hewitt–Liang ([arXiv 1909.03368](https://arxiv.org/abs/1909.03368)). Khoảng vẫn còn sau khi hoán vị đáp án MCQ | Khoảng < 5 điểm hoặc biến mất sau control. Hoặc probe trên encoder cũng thất bại ở đa số họ lỗi, tức nút thắt là tri giác |
| H2 | Route-then-read hòa giải locus | Patch nhóm image token đạt đỉnh ở độ sâu giữa rồi giảm. Patch token cuối chỉ có tác dụng từ điểm khởi đầu dải muộn ℓ_r trở đi, và ℓ_r ≥ đỉnh image-token ở mọi mô hình. Null của 2610.00024 tái tạo được khi swap residual token cuối giữa các instance, nhưng đảo ngược khi patch bằng cặp phản thực hoặc subspace | Hiệu ứng ở token cuối xuất hiện sớm hơn đỉnh image token. Hoặc không có vị trí hay cơ sở nào vượt baseline patch ngẫu nhiên |
| H3 | Subspace dùng chung | Subspace hạng k ≤ 8 tại ℓ_r, ước lượng từ hiệu cặp phản thực của một họ lỗi, đạt ≥ 50% interchange intervention accuracy (IIA) trong-họ trên ≥ 2 họ khác | IIA xuyên họ ≤ subspace ngẫu nhiên ở ≥ 2/4 mô hình lõi |
| H4 | Sửa được bằng một cấu hình | ReRead thắng calibration và always-reprompt trên ≥ 3 họ mô hình. Benchmark tổng quát giảm < 0,5 điểm | Không vượt calibration, vì khi đó "cách sửa" chỉ là bias đầu ra |
| H5 | Cách sửa nhắm đúng cơ chế (dose–response) | Lợi ích tăng đơn điệu theo decile RGS. Lợi ích bằng 0 trên item mà probe encoder thất bại. Trong crowding, lợi ích chỉ xuất hiện trong vùng khoảng cách s_enc < s < s_beh | Lợi ích đều nhau giữa các decile, hoặc xuất hiện cả trên lỗi tri giác |
| H6 | Phạm vi kiến trúc | Gemma-4-12B (không có encoder) vẫn có khoảng và dải readout. Llama-3.2-Vision (cross-attention) có khoảng tại token trả lời, và readout nằm sau tầng cross-attention cuối | Không giết bài, chỉ thu hẹp claim xuống MLLM prefix-token có encoder |
| H7 | RL đổi cách đọc (chương) | Probe trên image token không đổi (\|Δ\| < 2 điểm) cho đến ℓ_r. Ghép riêng các block muộn của mô hình RL vào mô hình base tái tạo phần lớn lợi ích | Probe tăng ở các tầng sớm, hoặc lợi ích đến từ ViT hay các block sớm |

### Đóng góp

Bài hứa bốn đóng góp chính và ba đóng góp chương.

**Bốn đóng góp chính:**

1. **Bộ công cụ ReadGap** (phát hành công khai). Gồm protocol đường cong khoảng probe–output theo tầng, chỉ số **RGS** theo từng item, phép tách lỗi thành perception/route/read, và demo tương tác lấy mẫu ngẫu nhiên. Demo này đáp lại lời chê "cherry-picked logit lens", từng là điểm yếu được nêu ở ICLR 2025 ([OpenReview](https://openreview.net/forum?id=chanJGoa7f)). Đây là kiểu artifact mà các bài phân tích lên highlight CVPR 2026 đều có, ví dụ EmbedLens ([CVPR 2026 Highlights](https://cvpr.thecvf.com/virtual/2026/events/Highlights2026)).
2. **Bằng chứng nhân quả cho route-then-read.** Gồm phép hòa giải locus, lời giải thích cho null của 2610.00024, và bằng chứng subspace dùng chung kèm ma trận chuyển giao xuyên họ lỗi.
3. **ReRead**, một cách sửa không huấn luyện với **một cấu hình cố định trên ≥ 3, mục tiêu ≥ 4 họ mô hình**. Cấu hình được chốt chỉ trên kích thích tổng hợp, so với 6 nhóm baseline, kèm bằng chứng dose–response.
4. **Bằng chứng phạm vi kiến trúc.** Có bộ ba mô hình cùng LLM nhưng khác encoder, và hai mô hình kiểm tra bác bỏ (không encoder, cross-attention).

**Ba đóng góp chương:**

1. Hàm psychometric crowding đầu tiên cho VLM, với vùng "readout gap" đọc được trên một đường cong.
2. meta-d′ tách lỗi tri giác khỏi lỗi kiến thức, cùng một cổng không huấn luyện.
3. Thang RL theo từng tầng, có ghép block.

### ReRead và RGS cụ thể là gì

Phần này là đề xuất thiết kế, sẽ chốt trên tập tổng hợp giữ lại trong tuần 28–31/10.

**Hiệu chuẩn offline, một lần cho mỗi mô hình.** Chỉ dùng khoảng 2.000 cặp phản thực tổng hợp phủ năm họ lỗi, không đụng dữ liệu benchmark. Có ba đại lượng được xác định tự động:

- Tầng bằng chứng ℓ_e: tầng có selectivity probe giữ lại cao nhất tại vị trí trả lời.
- Tầng readout ℓ_r: điểm khởi đầu của hiệu ứng patch token cuối trong bản đồ nhân quả.
- Subspace U hạng k: PCA trên hiệu kích hoạt h(x) − h(x′) của các cặp phản thực tại vị trí trả lời. Cách này không cần nhãn đáp án, và được kiểm tra chéo bằng DAS ([arXiv 2303.02536](https://arxiv.org/abs/2303.02536)).

**Lúc suy luận.** RGS đo độ bất đồng giữa phân phối "lens" tại ℓ_e và phân phối đầu ra. Phân phối lens được giới hạn trên tập đáp án ứng viên. Nếu RGS vượt ngưỡng τ (một phân vị cố định trên tập hiệu chuẩn), ReRead thay thành phần U của trạng thái tại ℓ_r bằng thành phần U của trạng thái tại ℓ_e. Tức là h_ℓr ← h_ℓr + α·UUᵀ(h_ℓe − h_ℓr), với α = 1 là mặc định "khôi phục thuần" (cần chuẩn hóa norm, xem ablation).

**Cấu hình cố định duy nhất** là: độ sâu chọn tự động, k = 8, α = 1, τ = phân vị 80. Biến thể dự phòng là trộn logit của lens tại ℓ_e, chỉ trên token ứng viên, có cổng RGS.

Thiết kế này đáp hai lời chê từng nhắm vào các bài trước:

- "Hyperparameter chỉnh theo test set". Đây là điểm yếu lớn nhất của bài attention sink ICLR 2025 ([OpenReview](https://openreview.net/forum?id=7uDI7w5RQA)).
- "Cách sửa không chạm vào cơ chế". Vea bị chê đúng điểm này ([OpenReview](https://openreview.net/forum?id=JAI7afWA9e)). ReRead sửa đúng subspace và đúng dải đã định vị.

Cổng RGS cũng tránh được lỗi của "Re-Promote" trong 2610.00024. Phương pháp đó lật 85,9% lỗi EBD nhưng lật luôn 6/7 lỗi tri giác, vì không có cổng ([arXiv](https://arxiv.org/html/2610.00024)).

## Tám mô hình, năm họ lỗi, một pipeline bốn tầng

### Panel mô hình đã xác minh trên HF

Bộ mô hình được thiết kế để tách **LLM nền** khỏi **đường thị giác**. Lý do là ba mô hình mạnh nhất cỡ 8B hiện đều dùng Qwen3-8B:

- InternVL3.5-8B ghi `Qwen3-8B` là base ([HF](https://huggingface.co/OpenGVLab/InternVL3_5-8B-Pretrained)).
- Molmo2-8B là Qwen3-8B cộng SigLIP 2 ([HF](https://huggingface.co/allenai/Molmo2-8B)).
- Qwen3-VL-8B có 36 tầng LLM ([config](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/config.json)).

Reviewer sẽ hỏi liệu "subspace chung" có chỉ là di sản của Qwen3 hay không. Bộ ba cùng LLM nhưng khác encoder biến câu hỏi đó thành một thí nghiệm.

| Vai trò | Mô hình | Đường thị giác / fusion | Lưu ý cơ chế |
|---|---|---|---|
| Lõi, cùng LLM Qwen3-8B | **Qwen3-VL-8B-Instruct** ([HF](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)) | SigLIP2, merge 2×2, **DeepStack** | `deepstack_visual_indexes` = [8, 16, 24] bơm đặc trưng ViT vào tầng LLM 0–2 ([arXiv 2511.21631](https://arxiv.org/html/2511.21631)). Phải patch cả đầu vào DeepStack hoặc chỉ patch từ ℓ ≥ 3. Hiệu ứng muộn không phải do "bơm muộn". Cố định `max_pixels` để cặp phản thực có cùng bố cục token |
| Lõi, cùng LLM | **InternVL3.5-8B** cùng thang Pretrained → Instruct → MPO → final ([HF](https://huggingface.co/OpenGVLab/InternVL3_5-8B)) | InternViT-300M, pixel-shuffle, 256 token/tile | Cố định số tile cho mỗi cặp. Thang huấn luyện dùng cho chương RL |
| Lõi, khác LLM | **Molmo2-O-7B** (OLMo-3, dữ liệu và LLM mở hoàn toàn) ([HF](https://huggingface.co/allenai/Molmo2-O-7B)), hoặc Molmo2-8B | SigLIP2 | Attention hai chiều trên vision token ([arXiv 2601.10611](https://arxiv.org/abs/2601.10611)), nên patch tập con image token lan truyền khác. Có in-distribution với PixMo-Count |
| Lõi, không encoder | **Gemma-4-12B-it** (Apache-2.0, không gated) ([HF](https://huggingface.co/google/gemma-4-12B-it)) | Patch ảnh thô 48 px chiếu thẳng vào LLM, 280 soft token | Kiểm tra bác bỏ quan trọng nhất. Nếu không có ViT tiền huấn luyện mà dải readout vẫn có, nút thắt nằm ở phía LLM |
| Bác bỏ, cross-attention | **Llama-3.2-11B-Vision-Instruct** (**gated**) ([HF](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)) | Các tầng cross-attention xen trong Llama-3.1-8B (kiến thức nền, chưa kiểm lại) | Ảnh không vào residual stream dưới dạng token, nên patch output của cross-attention tại token trả lời |
| So sánh với tài liệu | LLaVA-OneVision-qwen2-7b-ov ([HF](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf)); LLaVA-1.5-7B | SigLIP + Qwen2; CLIP + Vicuna | Dùng để tái tạo setting của 2610.00024 và 2604.09364 |
| RL | Qwen2.5-VL-7B-Instruct ([HF](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)) và các bản RL | — | Xem chương RL |
| Kiểm tra quy mô | Qwen3-VL-32B (khoảng 64 GB bf16, vừa H200; ước lượng) | — | Reviewer sẽ hỏi sau các bài dùng 32B/72B |

Không dùng Qwen3.5-9B làm mô hình chính. Mô hình này xen 3 tầng linear-attention với 1 tầng full-attention, nên không có ma trận attention theo token để knockout ([HF](https://huggingface.co/Qwen/Qwen3.5-9B)). Engine chính là nnsight cho mọi VLM trên HF ([nnsight](https://nnsight.net/)), cùng pyvene cho DAS. TransformerLens chưa có xác nhận hỗ trợ VLM.

### Năm họ lỗi và benchmark tương ứng

Kích thích tổng hợp và cặp phản thực là **nền đo cơ chế**. Benchmark thật chỉ để đo giá trị ngoài (external validity) của ReRead.

Có ba rủi ro về benchmark:

- CV-Bench khai thác được mà không cần ảnh: huấn luyện text-only trên câu hỏi test tăng +13,1 điểm ([arXiv 2511.04655](https://arxiv.org/abs/2511.04655)).
- CountBenchQA gần bão hòa, Qwen3-VL-8B đạt khoảng 88% ([arXiv 2603.18523](https://arxiv.org/pdf/2603.18523)).
- Molmo đã học trên PixMo-Count ([arXiv 2602.00108](https://arxiv.org/pdf/2602.00108)).

| Họ lỗi | Kích thích tổng hợp (một trục độ khó) | Cặp phản thực | Benchmark thật (chính) | Chỉ làm sanity check |
|---|---|---|---|---|
| Thuộc tính vs prior | Xoay hue vật thể chuẩn (ví dụ chuối) | Hoán đổi hue; Visual CounterFact ([HF](https://huggingface.co/datasets/mgolov/Visual-Counterfact)) | Visual CounterFact, ColorBench | — |
| Đếm | Chấm N = 1–30, kiểm soát tổng diện tích và convex hull | N vs N+1, cùng bố cục trừ một chấm; VLMBias ([HF](https://huggingface.co/datasets/anvo25/vlms-are-biased)) | VLMBias (đếm), PixMo-Count (bỏ Molmo), CV-Bench-count đã lọc TsT | CountBenchQA |
| Không gian | Độ lệch Δx/Δy giữa hai vật, số distractor | Lật gương; What'sUp A/B ([HF](https://huggingface.co/datasets/amitakamath2/whatsup_vlms)) | What'sUp, SpatialEval-real, CV-Bench-relation đã lọc | VSR |
| Nhận dạng tinh | Morph giữa hai hình, Vernier | MMVP ([HF](https://huggingface.co/datasets/MMVP/MMVP)), NaturalBench ([HF](https://huggingface.co/datasets/BaiqiL/NaturalBench)) | MMVP, NaturalBench | — |
| Chữ / vật nhỏ | Cỡ glyph 0,25–2× footprint của token (32 px với Qwen3-VL, 48 px với Gemma-4) | Từ khác một glyph | V\* ([HF](https://huggingface.co/datasets/craigwu/vstar_bench)), HR-Bench-4K ([HF](https://huggingface.co/datasets/DreamMr/HR-Bench)) | TextVQA |
| Không gây hại | — | — | MMBench, MME, SEED | — |

Bộ sinh kích thích là **một file PIL khoảng 300 dòng**, có seed và tất định, sinh luôn cặp phản thực cùng bố cục token. Không dùng Qwen-Image-Edit để tạo cặp ảnh thật, trừ khi reviewer đòi ảnh photoreal, vì sẽ tốn thêm 1–2 GPU-day (ước lượng). Mẫu psychometric (fit Weibull, ngưỡng tại 50%) lấy từ nghiên cứu contrast sensitivity trên MLLM ([arXiv 2508.10367](https://arxiv.org/pdf/2508.10367)). Đại lượng cốt lõi của bài là **khoảng giữa ngưỡng probe và ngưỡng hành vi, theo từng tầng**.

### Protocol bốn tầng

Bốn tầng nối với nhau thành một chuỗi: tầng sau chỉ chạy trên kết quả của tầng trước.

**Tầng 1 – Probe có control.** Probe là logistic L2 với cross-validation, ≥ 3 seed. Probe đặt tại ba chỗ: (i) output encoder trước projector, đóng vai trần tri giác giống Fu et al.; (ii) image token gộp trung bình và theo vùng vật; (iii) token trả lời, ở mọi tầng. Báo cáo selectivity so với control task, cùng baseline mô hình khởi tạo ngẫu nhiên và nhãn xáo trộn. Amnesic probing (INLP) kiểm tra hướng giải mã được có được *dùng* hay không ([arXiv 2006.00995](https://arxiv.org/abs/2006.00995)).

**Tầng 2 – Logit lens và tuned lens.** Logit lens gần như miễn phí khi đã chạy probe. Tuned lens ([arXiv 2303.08112](https://arxiv.org/abs/2303.08112)) tốn khoảng 2–6 GPU-giờ cho mỗi mô hình (ước lượng). Tầng này xác định *khi nào* đáp án đúng vào top-k và có bị ghi đè ở dải muộn hay không. Mọi kết quả lens đều báo cáo bằng tỷ lệ khớp trên mẫu ngẫu nhiên, không dùng ví dụ chọn lọc.

**Tầng 3 – Nhân quả.** Chạy bốn thí nghiệm theo thứ tự:

- (a) **Patching bằng cặp phản thực**, so sánh giữa nhóm image token và token cuối. Corruption dùng cặp phản thực hoặc symmetric token replacement, không dùng Gaussian noise. Metric là logit difference. Chạy cả hai chiều denoising và noising, theo Zhang & Nanda ([arXiv 2309.16042](https://arxiv.org/abs/2309.16042)) và Heimersheim & Nanda ([arXiv 2404.15255](https://arxiv.org/abs/2404.15255)). Patch theo nhóm token (vùng vật / nền / câu hỏi / token cuối / lưới 4×4) × tầng, khoảng 560 forward mỗi item (ước lượng).
- (b) **Attention knockout trên cả hai đường**: image → token cuối, và image → câu hỏi → token cuối. Phải knockout cả hai cùng lúc vì mô hình có cơ chế fallback giữa hai đường ([arXiv 2607.03358](https://arxiv.org/abs/2607.03358)). Thêm control knockout ngẫu nhiên cùng kích thước để loại hiệu ứng sink hoặc high-norm token ([arXiv 2309.17453](https://arxiv.org/abs/2309.17453), [arXiv 2402.17762](https://arxiv.org/abs/2402.17762)).
- (c) **AtP\*** ([arXiv 2403.00745](https://arxiv.org/abs/2403.00745)) cho bản đồ đầy đủ tầng × vị trí × thành phần, sau đó **xác minh 1–5% ô hàng đầu bằng patching thật**.
- (d) **DAS** trên khoảng 10 tầng. Kiểm tra ảo giác subspace của Makelov et al. ([arXiv 2311.17030](https://arxiv.org/abs/2311.17030)) bằng cách tổng quát hóa ra dữ liệu giữ lại và phân tích row space / null space của trọng số phía sau.

Một thí nghiệm riêng **tái tạo đúng setting của 2610.00024**: lỗi POPE-adversarial, LLaVA-1.5 / Qwen2.5-VL / InternVL3, swap residual token cuối giữa các instance. Sau đó chạy lại cùng item với patching đúng vị trí. Đây là câu trả lời trực tiếp cho null của bài đó.

**Tầng 4 – Sửa.** ReRead cùng các baseline, mô tả ở phần sau. Thêm back-patching kiểu "Same Task, Different Circuits" làm cận trên nhân quả ([arXiv 2506.09047](https://arxiv.org/abs/2506.09047)).

### Checklist đối chứng

| Nhóm | Đối chứng | Mục đích |
|---|---|---|
| Không thị giác | Không ảnh; ảnh xám **cùng số token**; ảnh xáo từ item khác; đáp án prior-only | Loại khả năng trả lời mù. Ảnh xám giữ nguyên vị trí cho baseline patching ([arXiv 2403.20330](https://arxiv.org/abs/2403.20330)) |
| Benchmark | TsT (mô hình text-only huấn luyện trên test) cho mọi benchmark MCQ ([arXiv 2511.04655](https://arxiv.org/abs/2511.04655)) | Lọc item khai thác được mà không cần ảnh |
| Định dạng | Hoán vị vòng thứ tự đáp án MCQ, báo cáo entropy thiên lệch theo chữ cái ([arXiv 2309.03882](https://arxiv.org/abs/2309.03882)); ≥ 3 paraphrase prompt ([arXiv 2310.11324](https://arxiv.org/abs/2310.11324)); vừa sinh tự do vừa chấm logit | "Lỗi readout" mà biến mất khi hoán vị thì chỉ là thiên lệch chữ cái |
| Giải mã | Greedy, seed cố định. Mô hình Thinking chạy cả bật và tắt thinking | Tránh nhiễu do sampling ([Rethinking-CD](https://github.com/Aurnawr/Rethinking-CD)) |
| Probe | Control task / selectivity, nhãn xáo, khởi tạo ngẫu nhiên, ≥ 3 seed | Phân biệt giải mã được với được dùng |
| Patching | Hướng ngẫu nhiên, tầng ngẫu nhiên, knockout token ngẫu nhiên, loại token sink | Loại hiệu ứng không ngữ nghĩa |
| Bố cục | Cố định độ phân giải và số tile, báo cáo số image token mỗi mẫu | Patch phải căn đúng vị trí |
| Tham chiếu | Ideal observer (đếm pixel, bộ giải giải tích) trên kích thích tổng hợp. Mốc Bouma ≈ 0,5× độ lệch tâm lấy từ tài liệu người ([PMC9525686](https://pmc.ncbi.nlm.nih.gov/articles/PMC9525686/)) | Có mốc chuẩn mà không cần nghiên cứu trên người. Nghiên cứu người cần phê duyệt đạo đức, là rủi ro tiến độ |
| Dữ liệu huấn luyện | Bỏ Molmo khỏi PixMo-Count; ghi rõ mô hình nào có dữ liệu mở | Kiểm soát rò rỉ |

### Baseline cho ReRead

ReRead phải thắng sáu nhóm baseline, với **cùng ngân sách dữ liệu hiệu chuẩn** (cùng tập tổng hợp):

1. **Calibration điểm số của chính mô hình**: temperature hoặc bias theo đáp án, kèm hằng số bias không cần ảnh. Đây là baseline nguy hiểm nhất ([arXiv 2609.18860](https://arxiv.org/abs/2609.18860), [Rethinking-CD](https://github.com/Aurnawr/Rethinking-CD)).
2. **Always-reprompt**. Ở bài đếm, cách này thường bằng hoặc hơn re-prompt có detector, ví dụ 89,02 so với 88,59 trên CountBench ([arXiv 2607.09544](https://arxiv.org/pdf/2607.09544)).
3. **Early readout kiểu DoLa/DeCo/VISTA**, theo mô tả trong ReWEIGH ([arXiv 2608.19075](https://arxiv.org/pdf/2608.19075)). ID và số liệu gốc của DeCo chưa kiểm lại.
4. **Vea**: +3,6 đến +11,1 EM trên VQA nhiều chữ ([arXiv 2510.17771](https://arxiv.org/html/2510.17771v1)).
5. **VTI**: latent steering, Spotlight ICLR 2025 ([arXiv 2410.15778](https://arxiv.org/pdf/2410.15778)).
6. **Steering theo probe hoặc SAE kiểu 2604.09364**: +1,4 đến +3,8% ([arXiv 2604.09364](https://arxiv.org/html/2604.09364)).

Thêm crop/zoom kiểu ViCrop cho item bị giới hạn bởi tri giác ([arXiv 2502.17422](https://arxiv.org/abs/2502.17422v1)). Ablation nội bộ gồm: ReRead không cổng, subspace ngẫu nhiên, tầng sai, và k, α khác.

### Metric và thống kê

Đơn vị báo cáo chính là **độ chính xác ròng** dưới greedy decoding khớp nhau, không phải tỷ lệ lật. 2610.00024 đã tránh không đưa ra claim ròng, nên chỗ này còn trống.

Kiểm định:

- Mỗi cặp mô hình × benchmark: paired bootstrap 10.000 lần lấy mẫu lại cho Δ (ReRead − baseline), cộng McNemar.
- Claim "có ý nghĩa trên ≥ 3 họ" dùng hiệu chỉnh Holm trên lưới mô hình × benchmark.
- Lưới tầng trong bản đồ nhân quả dùng Benjamini–Hochberg.

Mỗi con số là trung bình của ≥ 3 paraphrase và mọi hoán vị vòng MCQ. Probe chạy ≥ 3 seed và nhiều fold. Hàm psychometric fit Weibull kèm khoảng tin cậy bootstrap BCa. meta-d′ fit MLE kiểu HMeta-d với 4–6 bin, cũng bootstrap. Hiệu ứng nhân quả báo cáo bằng normalized patching effect trong [0, 1] và Δ logit-diff.

Cấu hình ReRead **được đóng băng ngày 31/10**: hash commit kèm timestamp, trước lần chạy đầu tiên trên benchmark thật.

## Ba chương dùng chung một pipeline

### Crowding: thấy được chỗ tách encoder–readout trên một đường cong

Chưa có bài nào đo hàm psychometric crowding hay critical spacing ở VLM. Tiền lệ gần nhất là một câu định tính trong BlindTest: mô hình đạt gần 100% "when much more space is added to separate shapes and letters" ([arXiv 2407.06581](https://arxiv.org/abs/2407.06581)). Còn nghiên cứu Flanker trên 47 VLM là flanker kiểu *attention*, không phải crowding ([arXiv 2505.18969](https://arxiv.org/abs/2505.18969)). Ở người, khoảng crowding tương ứng một khoảng cách vỏ não gần như hằng định và tỷ lệ với diện tích V4 ([Nature Communications 2025](https://www.nature.com/articles/s41467-025-59101-w)). Đây là phép tương tự trực tiếp cho câu hỏi "hằng định theo đơn vị token".

**Thiết kế.** Một chữ cái mục tiêu với 2 hoặc 4 flanker. Khoảng cách quét 12 mức theo đơn vị token, với 3 mức `max_pixels`, 3 loại flanker và 3 vị trí. Tổng khoảng 30.000 lượt trả lời một token cho mỗi mô hình, trên 3–4 mô hình. Mỗi cấu hình cho ba ngưỡng:

- s_beh: ngưỡng hành vi.
- s_enc: ngưỡng của probe trên patch encoder tại vị trí mục tiêu.
- s_ℓ: ngưỡng của probe tại token trả lời, theo tầng.

**Dự đoán.** Có bốn dự đoán. Dự đoán lõi là s_enc < s_beh: encoder phân giải được ở khoảng hẹp hơn mức hành vi đạt được, và **vùng s_enc < s < s_beh chính là "readout gap"**. Ba dự đoán còn lại:

- s_c hằng định theo token, không theo pixel, khi đổi `max_pixels`.
- Lỗi chủ yếu là *thay bằng chữ của flanker*, giống ở người.
- Flanker cùng loại làm tăng ngưỡng.

Chương này còn là **phép thử dose–response sạch nhất cho ReRead**. Nếu cách sửa nhắm đúng cơ chế, nó chỉ dời s_beh về phía s_enc, và không giúp gì dưới s_enc. Nếu kết quả ngược lại (s_enc ≈ s_beh), bài vẫn có một trường hợp nút thắt *là* tri giác. Trường hợp này là điều kiện biên, chứng minh pipeline phân biệt được hai loại lỗi. Chi phí dưới 1 H200-day (ước lượng).

### Metacognition: VLM biết mình không biết, nhưng không biết mình không thấy

Type-2 AUROC và meta-d′ đã là chuẩn cho LLM văn bản ([arXiv 2608.14552](https://arxiv.org/html/2608.14552)), nhưng chưa ai đo ở VLM theo cách tách lỗi tri giác khỏi lỗi kiến thức. Có hai kết quả gần nhất:

- Ở Qwen2-VL-2B, xác suất token nội tại phát hiện lỗi với AUROC 0,92–0,99. Nhưng **khi ảnh thiếu sáng nặng, độ chính xác rơi từ 0,99 xuống 0,22 trong khi cả hai tín hiệu confidence gần như không đổi** ([arXiv 2607.22034](https://arxiv.org/abs/2607.22034)).
- 2607.04683 quy lỗi VQA tri thức về trước hoặc sau bước nhận dạng thực thể, nhưng không dùng thước đo type-2 ([arXiv 2607.04683](https://arxiv.org/abs/2607.04683)).

**Thiết kế.** Dùng cặp item khớp nhau. Một vế bị giới hạn bởi tri giác: downsample hoặc che ROI, hoặc crowding dưới s_c. Vế kia bị giới hạn bởi kiến thức: thực thể hiếm hoặc câu hỏi phản thực trên ảnh sạch. Tín hiệu confidence gồm xác suất token, confidence nói ra bằng lời, P(True), và RGS / probe theo tầng. meta-d′ ([Fleming 2017](https://academic.oup.com/nc/article/2017/1/nix007/3748261)) được so tại cùng độ chính xác type-1.

**Dự đoán.** meta-d′/d′ của confidence đầu ra thấp hơn rõ ở lỗi tri giác. RGS và probe ở tầng giữa khôi phục type-2 AUROC cho nhóm lỗi này.

**Cổng.** Đây là sản phẩm của chương. Cổng ba nhánh, không huấn luyện:

- Confidence encoder thấp → **nhìn lại** (crop/zoom).
- RGS cao → **đọc lại** (ReRead).
- Còn lại → trả lời.

Cổng được so với **CounterCredit** trên đúng các metric của bài đó: độ chính xác, số lần gọi mỗi câu hỏi, và tỷ lệ gọi thừa. CounterCredit đăng 19/9/2026 nên là contemporaneous. Đó là phương pháp *có huấn luyện*: GRPO hai kênh, đạt V\* 89,5 và HR-4K 80,2 với 1,78 lần gọi mỗi câu ([arXiv 2609.22910](https://arxiv.org/abs/2609.22910)). Chi phí 1–2 H200-day nếu dùng chung dump (ước lượng).

### RL đổi cách đọc, không đổi cái nhìn

Bằng chứng hiện tại mới chỉ ở mức hành vi hoặc attention:

- Kiểm định McNemar trên **chỉ 50 bài MathVista** không thấy RL chỉ-thưởng-độ-chính-xác cải thiện tri giác ([arXiv 2506.07218](https://arxiv.org/html/2506.07218v3)).
- Mô hình suy luận giảm attention lên token thị giác ở các tầng sâu ([arXiv 2505.21523](https://arxiv.org/html/2505.21523v3)).
- RLVR thắng ở k nhỏ nhưng base thắng ở pass@k lớn, cả trên Qwen2.5-VL-7B. Bài này được giải á quân Best Paper NeurIPS 2025 ([NeurIPS](https://papers.nips.cc/paper_files/paper/2025/hash/537d5aa768c2d534016a4d06f87bc8fb-Abstract-Conference.html)).

Phản đề trực tiếp là PIVOT: RL làm encoder cho đặc trưng "stronger and precisely localized" ([arXiv 2510.16333](https://arxiv.org/abs/2510.16333)). Vì có phản đề này, đây là một câu hỏi thực nghiệm thật chứ không phải kết luận định sẵn. Đã có khuôn mẫu phương pháp cho cặp base/post-trained khớp trọng số: probe theo tầng cộng ablation phần ghi của MLP muộn ([arXiv 2608.08904](https://arxiv.org/abs/2608.08904)).

**Bước đầu tiên chỉ cần CPU.** Tính ‖ΔW‖/‖W‖ cho ViT, projector và từng block LLM của mọi cặp. Nếu một công thức RL **đóng băng ViT**, thì "encoder không đổi" đúng theo cấu tạo và không được trình bày như một phát hiện. Khi đó phép thử chuyển sang trạng thái image token bên trong LLM.

**Các cặp checkpoint:**

- Thang **InternVL3.5-8B**: Pretrained → Instruct → MPO → final (CascadeRL), cùng dữ liệu và kiến trúc ([HF](https://huggingface.co/OpenGVLab/InternVL3_5-8B-MPO)). Bản Pretrained cần chấm bằng logit ứng viên hoặc few-shot vì chưa theo được định dạng chat.
- **Qwen2.5-VL-7B-Instruct** so với bốn bản RL, mỗi bản đại diện cho một kiểu RL khác nhau:
  - VL-Rethinker-7B: chỉ RL ([HF](https://huggingface.co/TIGER-Lab/VL-Rethinker-7B)).
  - Perception-R1-7B: có thưởng tri giác ([HF](https://huggingface.co/tongxiao2002/Perception-R1-7B)).
  - ViGoRL-7b-Spatial: đúng miền không gian ([HF](https://huggingface.co/gsarch/ViGoRL-7b-Spatial)).
  - DeepEyes-7B: "think with images", nối với cổng metacognition ([HF](https://huggingface.co/ChenShawn/DeepEyes-7B)).
- Cặp SFT → RL cùng phòng lab: **MiMo-VL-7B-SFT** và **-RL** ([HF](https://huggingface.co/XiaomiMiMo/MiMo-VL-7B-RL)).

Tránh Vision-R1-7B vì cold-start SFT trộn lẫn với GRPO. Qwen3-VL Instruct và Thinking là hai nhánh anh em, không phải cặp base → RL.

**Đo lường.** Probe theo tầng, drift CKA và mutual-kNN, logit gap kiểu 2604.09364, phân phối RGS. Cuối cùng là **ghép block**: chỉ ghép block muộn hoặc chỉ ghép block sớm/ViT của mô hình RL vào mô hình base. Khoảng 30 mô hình lai × 2.000 item.

### Một pipeline phục vụ cả ba chương

Ba chương không phải ba bài con. Chúng dùng chung một **schema item**: họ lỗi, điều kiện, mã cặp, tham số độ khó. Chúng cũng dùng chung một **định dạng dump**: trạng thái tại vị trí trả lời, image token gộp, và vùng vật, theo từng tầng, lưu fp16 dạng parquet, khoảng 0,6 MB mỗi ảnh (ước lượng). Bộ probe, bộ khung patching và phép tính RGS cũng dùng chung.

Kích thích crowding đồng thời là item "giới hạn bởi tri giác" cho chương metacognition. Các cặp RL được chấm lại trên năm họ lỗi, trên crowding và trên cổng. Từ đó trả lời được câu hỏi RL có dời ngưỡng crowding hay hiệu suất metacognition hay không. Nhờ vậy, chi phí biên của mỗi chương chỉ là 12–25 GPU-giờ (ước lượng).

## Khác biệt với các công trình gần nhất

Bảng dưới là bảng sẽ đặt trong mục Related Work của bài, rút gọn lại. Mỗi dòng ghi đúng thứ công trình đó thiếu mà đề cương này cung cấp.

| Công trình | ID / venue | Phạm vi | Thiếu gì so với đề cương |
|---|---|---|---|
| Hidden in Plain Sight | [2506.08008](https://arxiv.org/html/2506.08008v1), COLM 2025 | Khoảng probe encoder vs VLM, Prismatic VLM | Không định vị nhân quả theo tầng; không có mô hình hiện đại; không có cách sửa |
| Arbitration Failure, Not Perceptual Blindness | [2604.09364](https://arxiv.org/html/2604.09364), arXiv 4/2026 | Màu/kích thước; 10 VLM; MAC ở 36–71% | Một họ lỗi; không có chuyển giao; steering chỉ +1,4–3,8% |
| The Count Is There, but Misaligned | [2607.09544](https://arxiv.org/pdf/2607.09544), arXiv 7/2026 | Đếm; 2 họ ≤ 8B | Một họ lỗi; re-prompt trung bình +5,3, always-reprompt ngang |
| Diagnosing Visual Ignorance | [2606.06890](https://arxiv.org/html/2606.06890), arXiv v2 9/2026 | Đếm; dải muộn ở token cuối | Patch chỉ trên Qwen-3B; không có sửa không huấn luyện; mâu thuẫn locus với 2604.09364 |
| Encoded but Disconnected | [2610.00024](https://arxiv.org/html/2610.00024), arXiv 1/9/2026 (**trước mốc 15/9**) | POPE; patching null; cơ chế theo kiến trúc | Không có claim độ chính xác ròng; patch token cuối giữa các instance |
| Knowing Isn't Always Saying | [2608.22916](https://arxiv.org/html/2608.22916v1), arXiv 8/2026 | Không gian/màu; 10 VLM, 5 họ | Không có claim cơ chế chung; không có cách sửa |
| Latent Ordinal Evidence | [2608.20999](https://arxiv.org/abs/2608.20999), EMNLP 2026 | Unembedding lọc hướng thứ bậc | Chỉ nhãn thứ bậc; lens phải huấn luyện |
| Counting Circuits | [2603.18523](https://arxiv.org/html/2603.18523v1), arXiv 2026 | Mạch đếm "chia sẻ" | Chuyển giao chỉ +1,54% OOD; chỉ quanh đếm |
| Seeing but Not Believing | [2510.17771](https://arxiv.org/html/2510.17771v1), ICLR 2026 poster | VQA nhiều chữ; 8 VLM; Vea | Sửa ở đầu vào, bị chê "không chạm cơ chế"; một họ lỗi |
| Same Attention, Different Truths | [CVPR 2026 Highlight](https://cvpr.thecvf.com/virtual/2026/events/Highlights2026) | Hallucination vật thể; logit lens trên attention | Một họ lỗi; không có subspace hay chuyển giao |
| Visual Symbolic Mechanisms | [ICLR 2026 Oral](https://openreview.net/forum?id=3RQ863cRbx) | Binding qua position ID | 2 họ mô hình; không có cách sửa; không xét readout |
| Decodable but Misrouted | [2609.18860](https://arxiv.org/abs/2609.18860), contemporaneous | Meme độc hại | Calibration khôi phục 69,8%; một miền |
| Vision-Operation Misalignment | [2607.16094](https://arxiv.org/abs/2607.16094), ACM MM 2026 | Locus theo từng phép toán | Đề cương kiểm định trực tiếp phản đề này |
| ICLR 2027 (Causal Readout Dependence; Many Paths, One Coordinate; Late FFN; Do Not Keep Looking) | [dORR3BQrmq](https://openreview.net/forum?id=dORR3BQrmq), [HBp4uTW5b0](https://openreview.net/forum?id=HBp4uTW5b0), [tunVuLrXNu](https://openreview.net/forum?id=tunVuLrXNu), [DnUl5Cio8T](https://openreview.net/forum?id=DnUl5Cio8T), đang review | Readout dependence, tọa độ ngữ nghĩa, FFN muộn | Chủ yếu là hallucination hoặc ảnh hưởng thị giác chung; không có ma trận chuyển giao xuyên họ lỗi; không có crowding, meta-d′ hay thang RL |
| CounterCredit | [2609.22910](https://arxiv.org/abs/2609.22910), contemporaneous | Khi nào gọi thị giác | Có huấn luyện (GRPO); quyết định sau khi gọi |
| RL makes MLLMs see better (PIVOT) | [2510.16333](https://arxiv.org/abs/2510.16333) | RL đổi encoder | Không phân tích theo tầng bên trong LLM |

Câu khác biệt cho intro nên là: *các bài trước cho thấy khoảng cách tồn tại trong từng họ lỗi; bài này cho thấy đó là cùng một subspace, giải thích vì sao patching có lúc null, và sửa nó bằng một cấu hình*.

## Ngân sách 240 GPU-giờ và lịch 34 ngày

### Phân bổ compute (mọi con số là ước lượng chưa đo)

Từ 14/10 đến lúc đóng băng thí nghiệm ngày 6/11 có khoảng 24 ngày, tức khoảng 570 GPU-giờ wall-clock. Kế hoạch dùng **240 GPU-giờ**. Phần dư là đệm cho kẹt hạ tầng, chạy lại và thời gian GPU rảnh trong lúc viết. Mọi sweep dài chạy nền, lưu checkpoint theo từng item ra parquet. Ngày 14/10 phải chạy hiệu chuẩn thông lượng trên 50 item, vì chưa có số đo VLM patching trên H200 nào được công bố.

| Hạng mục | GPU-giờ | Ghi chú |
|---|---|---|
| Hạ tầng, đo thông lượng, eval baseline 6–8 mô hình (lmms-eval / VLMEvalKit) | 20 | Gồm baseline mù và ảnh xám |
| Dump hidden state, probe, logit lens, tuned lens | 30 | 5 họ lỗi × 4–6 mô hình |
| Patching cặp phản thực theo nhóm, AtP\*, xác minh | 35 | Khoảng 5 GPU-giờ mỗi mô hình cho 500 item |
| Attention knockout hai đường | 8 | Đường eager/SDPA với mask 4D |
| DAS / subspace, ma trận chuyển giao | 25 | Khoảng 10 tầng × 3–4 mô hình |
| ReRead và 6 nhóm baseline trên benchmark thật | 35 | Gồm benchmark không gây hại |
| Chương RL (dump 8–10 checkpoint, ghép block) | 25 | Phần weight diff chạy trên CPU |
| Chương crowding | 12 | Dưới 1 H200-day |
| Chương metacognition | 15 | Dùng lại dump |
| Llama-3.2-V, Qwen3-VL-32B, tái tạo setting 2610.00024 | 10 | |
| Dự phòng (seed, paraphrase, chạy lại) | 25 | |
| **Tổng** | **240** | |

### Lịch theo ngày và theo tuần

| Ngày | Việc | Cổng |
|---|---|---|
| **13/10 (hôm nay)** | Xin quyền truy cập Llama-3.2-11B-Vision và Gemma-3-12B (gated, duyệt tay). Tạo hoặc cập nhật OpenReview profile cho mọi đồng tác giả, vì email không thuộc tổ chức có thể mất đến 2 tuần để duyệt. Tải trọng số | |
| 14/10 | Môi trường (ghim phiên bản transformers, nnsight, pyvene). Hook path cho Qwen3-VL-8B và InternVL3.5-8B. Đo thông lượng 50 item. Script weight diff (CPU) cho các cặp RL | |
| 15/10 | Bộ sinh kích thích 5 họ lỗi + crowding, sinh cặp cùng bố cục token. Tải benchmark. Baseline mù chạy nền | **G0**: pipeline chạy hết từ đầu đến cuối trên 1 mô hình |
| 16–17/10 | Pilot: dump, probe có control, logit lens cho màu-prior và đếm trên 2 mô hình. Tái tạo null của 2610.00024 trên một tập con POPE | |
| 18–19/10 | Patching cặp phản thực: image token vs token cuối. AtP\* nhanh. Phép thử chuyển giao đầu tiên màu ↔ đếm bằng PCA hiệu cặp | |
| **20/10** | Họp quyết định | **G1** (xem bảng cổng) |
| 21–23/10 | Mở rộng ra 5 họ lỗi × 4 mô hình lõi. Tuned lens. Sweep crowding chạy nền ban đêm | |
| 24–26/10 | AtP\* + patching xác minh. Knockout hai đường. DAS. Dump thang InternVL3.5 | |
| **27/10** | Ma trận chuyển giao | **G2** |
| 28–31/10 | Thiết kế RGS và ReRead chỉ trên tập tổng hợp giữ lại. Dựng tập cặp metacognition. **31/10: đóng băng cấu hình, commit hash** | |
| 1–3/11 | Chạy ReRead và mọi baseline trên benchmark thật, 4 mô hình lõi. Tính meta-d′, chạy cổng. Ghép block RL. **Bắt đầu viết intro và method từ 1/11** | |
| **3/11** | Bảng kết quả cách sửa | **G3** |
| 4–6/11 | Llama-3.2-V, Qwen3-VL-32B. Chạy lại seed, paraphrase, hoán vị. Bootstrap. **6/11: đóng băng thí nghiệm cho bài chính** | |
| 7–9/11 | Hình, bảng, bản nháp đầy đủ v1 (8/11), review nội bộ | |
| **10/11** | **Đăng ký bài**: tiêu đề chọn theo kết quả cổng, abstract, danh sách tác giả | Hạn cứng |
| 11–14/11 | Sửa bài. **Quét lại OpenReview ICLR 2027 và arXiv từ 15/9 đến nay**, thêm trích dẫn concurrent. Đối chiếu Author Guidelines và chính sách LLM của CVPR 2027 nếu đã công bố | |
| 15/11 | Đọc lại lần cuối, ẩn danh hóa, kiểm tra biên dịch | |
| **16/11** | **Nộp bài** (AOE, không gia hạn) | |
| 17–23/11 | Supplementary (hạn 23/11): plot từng mô hình, gallery mẫu ngẫu nhiên, code ẩn danh | |

Các mốc chính thức gồm: đăng ký 10/11, nộp bài 16/11, supplementary 23/11, review ra 25/1/2027, rebuttal 25/1–1/2, quyết định 25/2/2027 ([CVPR 2027 CFP](https://cvpr.thecvf.com/Conferences/2027/CallForPapers)).

## Bài tám trang, teaser và phản biện đã chuẩn bị sẵn

### Phân bổ trang và hình chính

Cấu trúc theo mạch mà các bài phân tích CVPR thành công đã dùng: phủ nhận lời giải thích cũ → chẩn đoán mới → cơ chế có tên → cách sửa. "Same Attention, Different Truths" mở đầu đúng như vậy: "prior work largely attributes this to insufficient visual attention… we find… equally strong visual attention" ([CVPR 2026 Highlights](https://cvpr.thecvf.com/virtual/2026/events/Highlights2026)).

| Mục | Trang | Nội dung |
|---|---|---|
| 1. Intro + teaser | 1,25 | Ba phát hiện đánh số; câu khác biệt |
| 2. Related (bảng khác biệt rút gọn) | 0,5 | |
| 3. Setup: mô hình, họ lỗi, kích thích, đối chứng | 0,75 | |
| 4. Bằng chứng nằm ở đâu: probe, lens, crowding | 1,0 | Đường cong khoảng probe–output theo tầng |
| 5. Route vs read: bản đồ nhân quả, hòa giải locus, giải thích null | 1,25 | |
| 6. Một subspace, nhiều họ lỗi: ma trận chuyển giao | 0,75 | |
| 7. RGS và ReRead: kết quả, baseline, dose–response, cổng metacognition | 1,5 | |
| 8. RL đổi cách đọc | 0,6 | |
| 9. Giới hạn và kết luận | 0,4 | |

Giới hạn 8 trang lấy theo quy định năm 2026 ([CVPR 2026 Author Guidelines](https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines)). Trang Author Guidelines 2027 còn trả về 404 khi kiểm tra ngày 13/10.

**Teaser: "một đường cong, hai ngưỡng".**

- **Bên trái:** một chữ cái bị crowding, mô hình trả lời sai.
- **Giữa:** đường psychometric hành vi và đường probe encoder trên cùng trục khoảng cách. Vùng tô màu giữa s_enc và s_beh là phần thông tin bị mất ở readout. Kèm một ô nhỏ theo tầng cho thấy đáp án đúng lên top-1 ở độ sâu giữa rồi bị ghi đè trong dải muộn.
- **Bên phải:** đường hành vi sau ReRead dịch về phía đường probe, chỉ trong vùng tô màu.

Teaser này độc nhất vì chưa ai có crowding. Nó cũng chứa luôn lập luận dose–response.

**Các hình và bảng khác:**

- **Hình 2:** bản đồ vị trí × độ sâu cho image token và token cuối trên 4 mô hình (route-then-read).
- **Hình 3:** ma trận chuyển giao 5×5 theo mô hình, có viền đánh dấu chỗ vượt subspace ngẫu nhiên.
- **Hình 4:** thang RL, gồm đường probe và logit gap theo tầng.
- **Bảng 1:** bảng khác biệt.
- **Bảng 2:** ReRead so với 6 nhóm baseline trên ≥ 4 họ mô hình × 6–8 benchmark.
- **Bảng 3:** cổng metacognition so với CounterCredit.

### Phản biện dự kiến và câu trả lời viết sẵn trong bài

Các lời chê dưới đây lặp lại trong review công khai của các bài cùng dòng. Có thể xem ở bài oral ICLR 2026 ([OpenReview](https://openreview.net/forum?id=3RQ863cRbx)), bài attention sink ([OpenReview](https://openreview.net/forum?id=7uDI7w5RQA)), bài logit-lens editing ([OpenReview](https://openreview.net/forum?id=94kQgWXojH)) và Seeing but Not Believing ([OpenReview](https://openreview.net/forum?id=JAI7afWA9e)).

| Phản biện | Câu trả lời đặt sẵn trong bài |
|---|---|
| "Chỉ một họ mô hình" (vẫn bị chê dù là "7 VLMs, 2 families") | 6–8 mô hình, ≥ 5 LLM nền. Bộ ba cùng LLM Qwen3-8B khác encoder tách hiệu ứng LLM khỏi hiệu ứng thị giác. Có Gemma-4-12B không encoder và Llama-3.2-V cross-attention |
| "Chỉ là tương quan" | Patching bằng cặp phản thực, IIA của DAS, amnesic probing. Reviewer của bài oral từng nói một cách sửa "would… enable direct causal claims". ReRead với dose–response theo RGS và theo vùng crowding là bằng chứng nhân quả thứ hai |
| "Chỉ dữ liệu tổng hợp" | Dữ liệu tổng hợp để đo cơ chế. Cách sửa đánh giá trên benchmark thật (V\*, HR-Bench, MMVP, NaturalBench, What'sUp, VLMBias, Visual CounterFact, CV-Bench đã lọc TsT) cộng MMBench/MME/SEED để chứng minh không gây hại |
| "Chỉnh hyperparameter trên test" | Cấu hình chốt chỉ trên tập tổng hợp. Hash commit ngày 31/10 trước lần chạy thật đầu tiên. Một cấu hình cho mọi mô hình. Có biểu đồ độ nhạy |
| "Cách sửa không tác động lên cơ chế" | ReRead sửa đúng subspace tại đúng dải đã định vị. Ablation subspace ngẫu nhiên và tầng sai cho lợi ích bằng 0. Lợi ích tập trung ở decile RGS cao và trong vùng s_enc < s < s_beh |
| "Đã biết rồi" | Bảng khác biệt. Ba dự đoán mới mà các account khác không đưa ra: chuyển giao xuyên họ lỗi, null patching biến mất khi patch đúng vị trí, và tính đặc hiệu theo vùng gap. Đối diện thẳng 2610.00024 |
| "Chỉ là calibration hoặc thiên lệch chữ cái" | Baseline calibration cùng ngân sách dữ liệu, hằng số bias không ảnh, hoán vị MCQ |
| "Dải muộn chỉ là DeepStack hoặc sink" | Chỉ patch từ ℓ ≥ 3 với Qwen3-VL; có mô hình không encoder; knockout có control sink |
| "Logit lens bị chọn lọc" | Tỷ lệ khớp trên mẫu ngẫu nhiên, demo công khai. Đây chính là bước rebuttal đã cứu bài ICLR 2025 ([OpenReview](https://openreview.net/forum?id=chanJGoa7f)) |

## Rủi ro và những điều chưa kiểm chứng

**Rủi ro scoop.** Đây là rủi ro lớn nhất. Ghi chú thấy ít nhất 7 bài nộp ICLR 2027 qua OpenReview search API ngày 13/10. Các trang forum bị bot-check chặn, nên danh sách đầy đủ chưa quét được. Phải quét lại vào khoảng 11–14/11 và trích dẫn như công trình concurrent. Chính sách CVPR 2027 không từ chối bài vì không so với công trình sau 15/9, nhưng vẫn bắt buộc trích dẫn và thảo luận ([CVPR 2027 CFP](https://cvpr.thecvf.com/Conferences/2027/CallForPapers)). Việc tìm arXiv qua web search không phủ hết các listing. Cần quét tay cs.CV từ 15/9 với các từ khóa "readout", "shared subspace", "Bouma", "meta-d′ vision-language" và "RLVR probe".

**Rủi ro về tài liệu.** Nhiều bài mới chỉ đọc abstract hoặc snippet:

- 2603.18523: con số +1,54%.
- 2604.10039: chưa rõ mô hình và mức lợi của MAS.
- 2603.17228 và 2510.21740.
- 2607.03358: chưa rõ mô hình và layer index.
- 2605.13156 và 2606.28273.
- 2609.00868.
- Chi tiết CounterCredit.
- Số liệu UG-Search, lấy từ tóm tắt bên thứ ba.

Phải đọc toàn văn trước khi trích số. ID arXiv và số liệu của DeCo và MemVR chưa được xác minh. "Xiong et al. 2026" và "Hong & Quan 2026" mà 2610.00024 trích dẫn chưa tìm thấy.

**Rủi ro về mô hình và công cụ.**

- Llama-3.2-Vision và Gemma-3 cần duyệt tay, phải xin **ngay hôm nay**.
- Chưa kiểm tra Molmo2 có remote-code processor tương thích với hook của transformers mới hay không. Phương án thay thế là LLaVA-OV-1.5-8B, nhưng LLM nền của nó chưa xác minh.
- Kiến trúc "Unified" của Gemma-4-12B có lớp patch-embedding học được ngoài phép chiếu tuyến tính hay không chưa rõ. Phải đọc toàn bộ model card.
- Base của ThinkLite-VL, MM-Eureka, Vision-R1 và VLAA-Thinker chưa xác minh từ model card.
- Chưa biết công thức RL nào đóng băng ViT. Weight diff sẽ giải quyết việc này.
- Gemma Scope 2 chưa được xác nhận là hoạt động đúng trên image token, nên không dựa vào SAE cho claim chính.
- Chưa xác minh CounterCount có tải được không ([arXiv 2605.17826](https://arxiv.org/abs/2605.17826)). Bài không phụ thuộc vào nó.

**Rủi ro khoa học.** Chuyển giao xuyên họ lỗi có thể thất bại. Cả crowding lẫn metacognition có thể lộ ra lỗi ở *encoder* (probe cũng thất bại). Cả hai khả năng đã được đăng ký trước thành điều kiện biên và nhánh pivot. Toàn bộ số compute là ước lượng chưa đo.

**Rủi ro quy trình.**

- Author Guidelines 2027 (giới hạn trang, số bài mỗi tác giả) và chính sách LLM chưa công bố tại 13/10.
- Mọi tác giả phải nhận review khi được mời. Năm 2025, hơn 200 bài bị desk-reject, gồm 19 bài vì review vô trách nhiệm ([Voxel51](https://voxel51.com/blog/opening-remarks-from-cvpr-2025)).
- Nghiên cứu trên người cho crowding cần phê duyệt đạo đức. Đề cương thay bằng ideal observer và mốc Bouma lấy từ tài liệu.

## Từ khóa và subject area

Primary area nên là **Vision, language, and reasoning**, một trong các mảng đông bài nhất năm 2026 ([Robotics & Automation News](https://roboticsandautomationnews.com/2026/05/27/cvpr-2026-fields-16000-paper-submissions-on-technical-advances-in-ai/101941/)). Secondary là **Explainable computer vision** hoặc **Multimodal learning**. Tên chính xác của các area năm 2027 chưa có trong ghi chú, phải đối chiếu khi đăng ký trên OpenReview.

**Từ khóa nên dùng:** *multimodal large language models, mechanistic interpretability, readout bottleneck, activation patching, counterfactual image pairs, linear probing, logit lens, distributed alignment search, shared subspace, training-free inference-time intervention, visual crowding, psychophysics, metacognition, meta-d′, RL post-training*.

**Không nên đặt làm từ khóa chính:** *hallucination mitigation, contrastive decoding, token pruning*. Ba cụm này dẫn bài tới những reviewer đã mệt với các dòng bão hòa, vốn bị chính tài liệu 2026 chỉ ra là đầy nhiễu đánh giá.

## Kết luận

Dòng "thấy mà không nói" đã thay đổi tiêu chuẩn. Năm 2025, chỉ cần chỉ ra một khoảng cách probe–output là có bài. Đến cuối 2026, khoảng cách đó là tiền đề mà reviewer mặc định đã biết. Câu hỏi đáng giá còn lại mang tính **cấu trúc**: các khoảng cách riêng lẻ là một hiện tượng hay nhiều hiện tượng? Đề cương biến câu hỏi đó thành ba phép thử có thể thất bại: ma trận chuyển giao, null patching biến mất khi patch đúng vị trí, và lợi ích của cách sửa chỉ nằm trong vùng gap. Mỗi kết cục đều có giá trị, kể cả kết cục "không có nút thắt chung", miễn là nó được đo chặt hơn 2610.00024.

Điểm mới nhất trong cách đóng khung là **dùng crowding làm dụng cụ đo chứ không phải benchmark**. Đây là một kích thích có núm vặn liên tục, trên đó encoder, readout và cách sửa hiện ra thành ba đường trên cùng một trục. Một AC có thể nhìn hình này và nhắc lại claim trong một câu. Nếu kết quả cổng G2 thuận lợi, đó là cơ hội thực tế nhất để chạm mức Highlight với một H200. Nếu không, cùng pipeline đó vẫn cho ra một bài nộp được qua nhánh RL ladder hoặc nhánh cổng metacognition, mà không phải bắt đầu lại từ đầu.
