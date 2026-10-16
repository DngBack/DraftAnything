# Kết quả kiểm chứng phương án cải tiến ver29

## Kết luận

Đã chạy thử residual readout có trừ ảnh xám, chiếu khỏi không gian logits ứng viên, ràng buộc nhất quán giữa hai cách hỏi và cổng an toàn. Residual có thêm không thắng hidden-state probe tuyến tính đơn giản một cách nhất quán. Tuy vậy, phép đo đúng luồng sinh hé lộ một hướng hẹp đáng kiểm tra tiếp: trên Qwen3-VL-4B, readout từ chênh lệch ảnh thật–ảnh xám đạt 68.8% khi đổi paraphrase, cao hơn hidden probe 5.2 điểm. Kết quả này không lặp lại trên Qwen2/Gemma và rơi về gần ngẫu nhiên khi giữ lại cả họ illusion, nên hiện tại là phát hiện theo mô hình, chưa phải đột phá tổng quát.

Đây là kết luận âm có ích: nó loại bớt hướng “thêm residual + consistency + gate” khỏi danh sách ứng viên chính, đồng thời chỉ ra vấn đề thực nghiệm quan trọng trong cache cũ.

## Kiểm soát tính đúng của phép đo

Cache cũ tính lại hidden state bằng cách gửi toàn bộ token prefix qua `model(...)`. Với Qwen3 và Gemma, logits tính lại lệch logits ở bước sinh thật (MAE trung bình khoảng 0.15–0.17). Vị trí RoPE đa phương thức có thể bị giữ sai khi nối token bằng tay. Vì vậy tôi viết lại bộ trích xuất để chụp trực tiếp đầu vào/đầu ra của LM head trong `generate()`, rồi chạy ảnh xám bằng cách ép đúng chuỗi token prefix qua cùng đường sinh.

Đã trích xuất lại toàn bộ 264 mẫu cho mỗi mô hình (132 ảnh × hai cách hỏi) trên Qwen2-VL-2B, Qwen3-VL-4B, Qwen3-VL-8B và Gemma-3-4B. Cả 1,056 mẫu đều có quyết định; sai lệch logits giữa hook và bộ sinh bằng 0, và replay ảnh xám cũng khớp bằng 0. Chuỗi đáp án được phân tích thành Yes/No trong 100% mẫu. Candidate argmax khớp đáp án sinh 100% trên Qwen2, 98.9% trên Qwen3-4B, 98.5% trên Qwen3-8B và 98.1% trên Gemma.

## Kết quả trên 132 ảnh

Mỗi số trong bảng là trung bình 3 outer split seed theo ảnh. “Probe” là logistic regression trên hidden state; “residual” là cùng probe sau khi dùng phần chênh trạng thái ảnh thật–ảnh xám và penalty nhất quán. “Gate” chọn khi confidence đủ cao trên validation bên trong. Các giá trị gate được đánh giá trên fold ngoài.

| Mô hình | Raw | Probe | Residual | Gate probe | Gate residual | Residual − probe, CI bootstrap theo ảnh |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2-VL-2B | 50.0% | 73.7% | 73.7% | 54.9% | 56.9% | 0.0 pp [−3.2, 3.2] |
| Qwen3-VL-4B | 57.2% | 85.5% | 74.5% | 65.8% | 60.0% | −11.0 pp [−15.5, −6.7] |
| Qwen3-VL-8B | 51.9% | 80.6% | 75.0% | 57.2% | 54.5% | −5.6 pp [−9.2, −1.9] |
| Gemma-3-4B | 51.9% | 75.0% | 75.1% | 61.6% | 61.0% | +0.1 pp [−3.0, 3.2] |

Residual sửa được nhiều lỗi raw khi không gate, nhưng cũng làm hỏng khoảng 20–25% câu trả lời raw vốn đúng. Probe đơn giản đạt ngang trên Qwen2/Gemma và cao hơn rõ trên Qwen3. Khi bật gate, residual so với raw tăng lần lượt 6.9, 2.8 và 9.1 điểm; chưa đạt mục tiêu đặt trước là +10 điểm trên cả ba mô hình. So với gate probe, residual chỉ hơn Qwen2 2.0 điểm [−0.4, 4.4], kém Qwen3 5.8 điểm [−8.1, −3.7], và gần như hòa Gemma −0.6 điểm [−3.0, 1.5].

Gate residual làm hỏng trung bình 3.0%, 0.4%, 1.9% và 1.7% các đáp án raw đúng tương ứng với Qwen2, Qwen3-4B, Qwen3-8B và Gemma. Qwen2 vượt ngưỡng tổn hại 2% đã đặt ra. Các khoảng bootstrap lấy mẫu lại 132 ảnh, sau khi trung bình ba lần chia fold; chúng điều kiện trên các mô hình đã fit và không bao gồm bất định do huấn luyện lại.

## Khả năng chuyển prompt và họ bài toán

Trên cache trích xuất đúng, probe huấn luyện với cả hai cách hỏi đạt 73.7%, 85.5%, 75.0%. Khi huấn luyện một cách hỏi và kiểm tra cách còn lại trên ảnh chưa thấy, accuracy giảm còn 63.1%, 63.6%, 62.6%. Khi giữ lại toàn bộ một họ illusion để kiểm tra, kết quả còn 53.0%, 48.5%, 49.2% trên Qwen2, Qwen3 và Gemma. Phép family holdout chỉ có sáu họ, nên đây là dấu hiệu cảnh báo chứ chưa phải ước lượng tổng quát hóa chính xác.

Các cặp prompt hiện tại là paraphrase gần nhau, ví dụ “Are the two red circles equal in size?” và “Do the two red circles have the same size?”. Do đó, cross-prompt này chưa chứng minh khả năng chuyển sang prompt, format trả lời hay verbalizer mới. Kết quả rất thấp ở family holdout cho thấy probe còn có thể học dấu hiệu riêng của từng nhóm illusion.

Ở phép cross-prompt, hướng đáng theo dõi nhất là candidate-null real-minus-gray: `Q[h(real)-h(gray)]`, với `Q` chiếu khỏi các hàng unembedding của token Yes/No và biến thể bề mặt. Qwen3-VL-4B đạt 68.6% so với hidden probe 63.6%; mức tăng ghép theo ảnh là +4.9 điểm [1.3, 8.7]. Qwen3-VL-8B đạt 64.3% so với 59.5%; tăng +4.8 [0.8, 8.7]. Như vậy có lặp lại trên hai cỡ Qwen3, dù chưa vượt ngưỡng +5 điểm đặt trước một cách rõ rệt. Chênh lệch ảnh thật–xám trước khi chiếu đạt 68.8% trên Qwen3-4B (+5.2 [1.5, 9.0]) nhưng 62.9% trên Qwen3-8B (+3.4 [−0.6, 7.3]). Qwen2 có candidate-null chênh −0.8 điểm [−5.8, 4.3], Gemma −2.0 [−8.3, 4.2], nên hiệu ứng hiện chỉ thấy trên dòng Qwen3.

Hướng này tăng cross-prompt theo cả hai hướng ở Qwen3-4B: +7.8 điểm [1.0, 14.9] khi đánh giá wording 0 và +2.5 [−2.8, 8.1] ở wording 1. Theo họ, mức tăng candidate-null của Qwen3-4B xấp xỉ −1, +11, +1, +8, +22, −3 điểm; hiệu ứng tập trung vào một phần nhóm illusion. Family holdout vẫn chỉ đạt 52.7% trên Qwen3-4B và 50.4% trên Qwen3-8B, nên chưa có bằng chứng chuyển sang họ mới.

## Đánh giá về đóng góp cho paper

1. Có bằng chứng khá mạnh rằng hidden state chứa tín hiệu liên quan đến đáp án mà logits Yes/No hiện tại chưa khai thác: probe đạt 74–86% khi chia ảnh ngoài fold, trong khi raw đạt 50–57%.
2. Residual có thêm consistency không vượt probe thường; nó còn thấp hơn probe 5.6 điểm trên Qwen3-8B và 11.0 điểm trên Qwen3-4B.
3. Riêng candidate-null real-minus-gray là ứng viên hẹp có mức tăng cross-prompt lặp lại trên hai cỡ Qwen3 (+4.9 và +4.8 điểm, CI ghép theo ảnh dương). Nó không lặp lại trên Qwen2/Gemma và không chuyển sang họ illusion mới.
4. Gate giảm số đáp án bị hỏng nhưng còn bỏ sót nhiều cơ hội sửa; ngưỡng được chọn trên validation nhỏ và mức bảo vệ chưa đạt đồng đều giữa mô hình.
5. Vì vậy, hướng cải tiến paper tốt nhất hiện tại là giới thiệu candidate-null real-minus-gray như một phát hiện transfer có điều kiện cho dòng Qwen3, giữ residual phức tạp như ablation âm, và phân biệt “đọc được từ hidden state”, “được mô hình dùng để sinh”, và “sửa được mà không gây hại”. Cần xác nhận trên model/dataset/prompt mới được khóa trước trước khi tuyên bố đây là phương pháp tổng quát.

## Giới hạn còn lại

Thử nghiệm này dùng bốn mô hình 2B–8B, sáu nhóm hình illusion, 132 ảnh độc lập và câu trả lời nhị phân. Đây là đánh giá readout ở bước quyết định token; chưa chứng minh cải thiện trên câu trả lời tự do nhiều token hoặc benchmark đa dạng. Kết quả qua ba seed là độ ổn định của outer split, không phải ba lần lặp dữ liệu độc lập. Chưa có can thiệp nhân quả chứng minh probe tương ứng với cơ chế thị giác mà mô hình dùng.

## Tái lập

Trích xuất trạng thái đúng luồng sinh: `extract_controlled_readout_exact.py`; đánh giá readout: `evaluate_controlled_readout.py`; fit residual/gate: `verify_residual_repair.py`. Cache exact nằm trong `results/ver29_controlled_exact/`; metrics nằm trong `results/ver29_controlled_exact_eval/` và `results/ver29_residual_repair_exact/`. Bản trích xuất cũ được giữ nguyên để so sánh, không ghi đè.
