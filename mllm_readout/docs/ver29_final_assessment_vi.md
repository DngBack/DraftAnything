# Báo cáo tổng kết: đánh giá và hướng nâng cấp SeeingNotSaying

**Ngày:** 16/10/2026

**Đối tượng:** bản thảo ver28, hai phản hồi review mô phỏng do tác giả cung cấp và các phép kiểm chứng ver29 chạy trong workspace.

## Kết luận điều hành

Bản ver28 được đánh giá ở mức **7/10, Weak Accept** trong phản hồi tiếng Việt; phản hồi tiếng Anh chấm **5/10, Weak Accept** trên một thang riêng. Hai review đều ghi nhận thực nghiệm rộng và nghiêm túc, đồng thời còn lo ngại về novelty, diễn giải nhân quả và khả năng tổng quát hóa.

Các phép thử mới tạo được một kết quả có tiềm năng cho paper: candidate-null real-minus-gray readout cải thiện chuyển paraphrase khoảng **4.8–4.9 điểm** trên Qwen3-VL-4B và Qwen3-VL-8B, với khoảng bootstrap ghép theo ảnh không chứa 0. Tín hiệu không xuất hiện trên Qwen2/Gemma, và gần biến mất khi giữ lại một họ illusion để kiểm tra. Đây là kết quả hẹp theo dòng mô hình, chưa phải phương pháp tổng quát.

Phương án residual phức tạp ban đầu không đạt: residual kèm consistency không hơn hidden probe tuyến tính và thua probe rõ trên cả hai cỡ Qwen3. Vì vậy, hiện có cơ sở để **bảo vệ mức 7 và đề nghị tăng nhẹ sau khi sửa paper**, có thể khoảng nửa đến một điểm nếu reviewer đánh giá cao kiểm soát và độ chính xác mới. Chưa đủ cơ sở để kỳ vọng Strong Accept hoặc tăng điểm lớn. Kết quả này chưa được đưa vào bản thảo; điểm review chính thức không thể dự đoán từ các review mô phỏng.

## Phản hồi review ban đầu nói gì

Hai feedback thống nhất ở bốn điểm:

- **Điểm mạnh:** ver28 bổ sung nhiều đối chứng, cross-fitting, kiểm tra capacity của probe, counterfactual, nested selection và báo cáo negative results rõ hơn ver25.
- **Novelty còn vừa phải:** phần lớn đóng góp vẫn là khung chẩn đoán và tổng hợp thực nghiệm; calibrator và probe tự thân là công cụ quen thuộc.
- **Cơ chế chưa được xác định duy nhất:** BIAS/LOSS/PERC là nhãn vận hành theo readout đã kiểm tra, chưa phải taxonomy nhân quả đầy đủ.
- **Transfer còn yếu:** prompt transfer có kết quả hỗn hợp; steering không đạt tiêu chí ổn định trên các model giữ lại.

Các yêu cầu review thiết thực nhất là trình bày rõ chi phí nhãn, giải thích vì sao offset-only kém, kiểm tra nhạy cảm với prompt/decoding, định lượng độ ổn định nhãn từng lỗi, và phân biệt candidate logits với toàn bộ vocabulary logits. Review 7/10 đặc biệt đánh giá cao việc ver28 đã bắt đầu kiểm tra giả thuyết theo hướng có thể bác bỏ.

## Phương pháp đã kiểm chứng

Tôi trích xuất lại state cho **132 ảnh × hai cách hỏi** trên bốn mô hình: Qwen2-VL-2B, Qwen3-VL-4B, Qwen3-VL-8B và Gemma-3-4B. Để so sánh ảnh thật với ảnh xám ở đúng cùng tiền tố, state và logits được chụp ngay trong `generate()`; ảnh xám được chạy lại bằng đúng chuỗi token. Có tổng cộng 1.056 mẫu. Hook khớp logits của bộ sinh và replay ảnh xám đều có sai lệch bằng 0.

Cache thăm dò được tạo bằng cách tính lại state sau generation từng có sai lệch logits ở Qwen3/Gemma, có thể do giữ sai vị trí đa phương thức khi nối token bằng tay. Cache đó không được dùng làm kết quả cuối ở đây. Vấn đề này thuộc phép trích xuất controlled pilot mới; cache gốc của ver28 được tạo trong `diag_gen.py` theo đường `generate()` và không bị sửa.

Phương pháp ứng viên chính là đọc nhãn từ phần

`Q [h(ảnh thật, prefix) − h(ảnh xám, cùng prefix)]`,

trong đó `Q` chiếu khỏi không gian các hàng unembedding của mọi biến thể token Yes/No. Các folds chia theo ảnh; huấn luyện một paraphrase và kiểm tra paraphrase kia trên ảnh chưa thấy. Baseline hidden probe dùng cùng ảnh và nhãn huấn luyện. Mỗi kết quả dùng ba outer split seeds; khoảng tin cậy bootstrap lấy mẫu lại 132 ảnh sau khi gộp hai cách hỏi và trung bình seeds.

## Kết quả transfer theo paraphrase

| Mô hình | Hidden probe | Candidate-null real-minus-gray | Chênh lệch | Raw candidate accuracy | Candidate-null, giữ một họ illusion |
|---|---:|---:|---:|---:|---:|
| Qwen2-VL-2B | 63.1% | 62.4% | −0.8 điểm [−5.8, 4.3] | 50.0% | 52.7% |
| Qwen3-VL-4B | 63.6% | 68.6% | +4.9 điểm [1.3, 8.7] | 57.2% | 52.7% |
| Qwen3-VL-8B | 59.5% | 64.3% | +4.8 điểm [0.8, 8.7] | 51.9% | 50.4% |
| Gemma-3-4B | 62.6% | 60.6% | −2.0 điểm [−8.3, 4.2] | 51.9% | 47.7% |

Đây là bằng chứng lặp lại trên hai cỡ Qwen3, nhưng không lặp qua kiến trúc và không chuyển sang họ illusion mới. Hiệu ứng cũng khác theo nhóm: Qwen3-4B có mức tăng candidate-null lớn ở Müller–Lyer, Ponzo và Vertical-Horizontal; ba nhóm khác gần 0. Hai prompt ở mỗi nhóm là paraphrase gần nhau, nên chưa chứng minh transfer sang format trả lời hoặc verbalizer mới.

## Kết quả residual repair và cổng an toàn

| Mô hình | Raw | Hidden probe | Residual + consistency | Gate probe | Gate residual |
|---|---:|---:|---:|---:|---:|
| Qwen2-VL-2B | 50.0% | 73.7% | 73.7% | 54.9% | 56.9% |
| Qwen3-VL-4B | 57.2% | 85.5% | 74.5% | 65.8% | 60.0% |
| Qwen3-VL-8B | 51.9% | 80.6% | 75.0% | 57.2% | 54.5% |
| Gemma-3-4B | 51.9% | 75.0% | 75.1% | 61.6% | 61.0% |

Residual sửa được nhiều lỗi raw khi không gate nhưng cũng làm hỏng khoảng 20–28% câu trả lời raw vốn đúng. So với probe thường, residual chênh 0.0 điểm trên Qwen2, −11.0 [−15.5, −6.7] trên Qwen3-4B, −5.6 [−9.2, −1.9] trên Qwen3-8B và +0.1 [−3.0, 3.2] trên Gemma.

Gate residual tăng so với raw khoảng 2.7–9.1 điểm tùy model, dưới mục tiêu tăng ít nhất 10 điểm ở mọi model. Nó làm hỏng trung bình 3.0%, 0.4%, 1.9% và 1.7% các đáp án raw đúng theo thứ tự Qwen2, Qwen3-4B, Qwen3-8B, Gemma; Qwen2 vượt mức hỏng tối đa 2% đặt trước. So với gate probe, residual chỉ hơn Qwen2 2.0 điểm [−0.4, 4.4], kém Qwen3-4B 5.8 [−8.1, −3.7], kém Qwen3-8B 2.7 [−4.9, −0.5] và gần như hòa Gemma −0.6 [−3.0, 1.5].

Khoảng bootstrap ở đây ghép theo ảnh, điều kiện trên các mô hình đã fit và chưa tính bất định do huấn luyện lại. Accuracy là dự đoán lớp tại bước quyết định token; chưa phải đánh giá câu trả lời tự do nhiều token sau khi áp dụng can thiệp.

## Ý nghĩa đối với bản thảo

**Có thể củng cố paper:** candidate-null contrast là kết quả cụ thể hơn nhận xét chung rằng hidden probe tốt. Nó gợi ý rằng ở Qwen3, tín hiệu của ảnh thật so với ảnh xám có thể giúp chuyển paraphrase khi loại phần chiếu trực tiếp vào token Yes/No. Kết quả này phù hợp với hướng review muốn kiểm tra hình học final readout.

**Chưa giải quyết được:** kết quả không chứng minh mô hình gốc sử dụng phần tín hiệu đó; một probe bên ngoài đọc được thông tin chưa đồng nghĩa model dùng nó để trả lời. Chưa có erase/restore nhân quả, benchmark độc lập, prompt format mới, model khác ngoài Qwen3 family, hoặc đánh giá end-to-end nhiều token. Chi phí suy luận cũng tăng vì cần thêm một lượt forward với ảnh xám.

**Không nên đưa thành claim:** residual + consistency + gate là phương pháp repair tổng quát; taxonomy BIAS/LOSS/PERC là nguyên nhân duy nhất; candidate-null state chứng minh thông tin “mất khỏi toàn bộ logits”; hay kết quả paraphrase là generalization rộng.

## Sửa đổi nên làm trước khi nộp

1. Giữ BIAS/LOSS/PERC như nhãn vận hành. Mô tả rõ BIAS là lỗi khôi phục được từ candidate logits bằng các calibrator đã thử; LOSS là lỗi probe khôi phục được nhưng các candidate readout đã thử chưa khôi phục; PERC là lỗi chưa được các readout đã thử khôi phục.
2. Thêm một tiểu mục controlled readout geometry. Báo cáo Qwen3-4B/8B candidate-null transfer cùng raw, hidden probe, candidate scores, random subspace/capacity controls và khoảng tin cậy theo ảnh. Đặt Qwen2/Gemma ở cùng bảng làm negative controls.
3. Giữ residual + consistency và gate như ablation âm; nêu rõ chúng không vượt hidden probe và gate còn bỏ sót nhiều lỗi.
4. Thu hẹp Discussion: kết quả hiện ủng lưng một khác biệt architecture-specific trong dòng Qwen3, không phải một nguyên lý chung về MLLM.
5. Bổ sung đúng các yêu cầu review còn thiếu: 5 labels/class ở phần chính, offset-only so với diagonal/full affine, bảng stability nhãn theo seed/layer/fold, và audit prompt/decoding. Nếu không có phép đo thì ghi rõ là giới hạn, không suy luận cơ chế.

Để theo đuổi claim phương pháp mới, bước tiếp theo cần khóa trước một dataset hoặc họ hình độc lập, prompt/verbalizer khác, rồi chạy lại candidate-null contrast cùng probe thường và random/capacity controls. Chỉ sau khi vượt baseline trên dữ liệu đó mới nên xem đây là đóng góp chính. Tăng score chỉ là khả năng nếu reviewer đánh giá cao phần bằng chứng mới và cách thu hẹp claim; các review mô phỏng không thể bảo đảm kết quả đó.

## Tệp và số liệu để tái lập

- PDF gốc: `mllm_readout/paper/versions/SeeingNotSaying_ver28_20261015-1229.pdf`
- Báo cáo kết quả kiểm chứng chi tiết: `mllm_readout/docs/ver29_validation_results_vi.md`
- Protocol: `mllm_readout/docs/ver29_verification_protocol.md`
- Trích xuất đúng luồng sinh: `mllm_readout/extract_controlled_readout_exact.py`
- Đánh giá readout: `mllm_readout/evaluate_controlled_readout.py`
- Đánh giá residual/gate: `mllm_readout/verify_residual_repair.py`
- Metrics controlled đầy đủ: `mllm_readout/results/ver29_controlled_exact_eval/metrics_complete.csv`
- Metrics residual đầy đủ: `mllm_readout/results/ver29_residual_repair_exact/metrics_complete.csv`
- Bootstrap paired theo ảnh: `mllm_readout/results/ver29_residual_repair_exact/paired_image_bootstrap_complete.csv`

Báo cáo này tổng hợp kết quả nghiên cứu và đề xuất sửa paper; file PDF ver28 chưa bị chỉnh sửa.
