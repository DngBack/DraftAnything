# Đính chính thước đo: H3/H3b chấm lại bằng EM chính thức (2026-11-06)

## Vấn đề

`docs/h3_results.md`, `docs/h3b_results.md` và `results/h3*` gọi thước đo là "metric chính thức", giá trị "strict". Tên đó sai. `value_correct` dễ dãi hơn evaluator thật ở các điểm sau:

| | `value_correct` | evaluator TAT-QA chính thức |
|---|---|---|
| dấu âm/dương | bỏ qua | có tính |
| sai số số học | làm tròn theo gold, hoặc lệch ≤ 0,5 % | làm tròn 2 chữ số rồi so bằng |
| nhiều số trong câu trả lời | chỉ cần một số khớp | phải khớp cả chuỗi |
| đáp án chữ | F1 ≥ 0,5 hoặc gold là chuỗi con | khớp chính xác sau chuẩn hóa |
| scale | so nhãn với gold | nhân vào giá trị (5 thousand = 5000 với scale rỗng) |

Protocol đã khóa ghi rõ hàm `value_correct`, nên các kết luận theo protocol vẫn đúng *với thước đo đó*. Chỉ không được gọi đó là số chính thức, và không được đặt cạnh leaderboard.

## Cách chấm lại

- `evc/official_metric.py`: port luật EM từ [NExTplusplus/TAT-QA](https://github.com/NExTplusplus/TAT-QA) (`tatqa_metric.py`, `tatqa_utils.py`). Không tìm thấy script riêng của TAT-DQA ở các đường dẫn đã thử; giả định hai bộ dùng cùng luật. Kiểm tra: gold tự chấm = 100 %.
- `scripts/official_rescore.py`: chấm lại lựa chọn của các policy. **Không fit lại**: mô hình, λ và bảng tra giữ nguyên như bản đã khóa, vốn học trên nhãn dễ dãi.
- Đây là **phân tích sau khi đã mở test**. Câu trả lời `direct` được tách bằng một hàm viết sau khi đã có output, không dùng gold, nên số EM phụ thuộc phần nào vào hàm này.
- Chi tiết: `results/official_rescore.md`.

## Lần sửa thứ hai: hàm tách đáp án (2026-11-06)

Bản đầu của `direct_pred` có lỗi làm EM thấp oan: coi đơn vị (`53; €m`) là đáp án thứ hai, và cắt chữ scale kể cả khi nhãn scale rỗng. Bản sửa được chọn chỉ trên train (xem [protocol_h3c.md](protocol_h3c.md)). Số dưới đây là bản sửa; bản đầu đã báo cho người dùng và **một kết luận đã đổi**: v2 − H3-learned ở Qwen3 B2 từ +2.0 (chưa có ý nghĩa) thành +3.1 (có ý nghĩa). Kết quả EM vì vậy có độ nhạy thật với cách tách đáp án.

## Kết quả trên test2 (609 câu)

| policy | 7B B2 | 7B B3 | Qwen3 B2 | Qwen3 B3 |
|---|---|---|---|---|
| Fixed-2 (2 lượt) | 56.5 | 56.5 | 59.3 | 59.3 |
| rule_table | 63.5 | 62.9 | 62.1 | 61.9 |
| H3-learned | 65.8 | 67.2 | 61.9 | 66.5 |
| **v2** | **70.3** | **70.9** | **65.0** | **66.2** |

| so sánh | 7B B2 | 7B B3 | Qwen3 B2 | Qwen3 B3 |
|---|---|---|---|---|
| v2 − Fixed-2 | +13.8 [+10.4, +17.5] | +14.4 [+11.1, +18.3] | +5.7 [+3.3, +8.2] | +6.9 [+4.1, +9.8] |
| v2 − rule_table | +6.7 [+4.5, +9.0] | +8.0 [+5.5, +10.7] | +3.0 [+0.9, +5.1] | +4.3 [+1.1, +7.2] |
| v2 − H3-learned | +4.4 [+2.0, +6.8] | +3.8 [+1.4, +6.0] | +3.1 [+1.0, +5.3] | −0.3 [−2.4, +1.6] |

Áp ba tiêu chí H3b lên EM với hàm tách đã sửa: tiêu chí 1 và 2 đạt, tiêu chí 3 chưa tính lại với hàm tách này (phiên bản đầu: đạt). Đây vẫn là phân tích sau khi đã mở test, và hàm tách được sửa sau khi đã xem kết quả test của bản đầu; xác nhận độc lập nằm ở H3c trên dev2 ([h3c_results.md](h3c_results.md)): v2 hơn H3-learned ở 7B lặp lại, còn ở Qwen3 B2 thì không.

## Đặt cạnh leaderboard

Toàn bộ tập lọc, B = 3, EM: 7B v2 **70.4** (1209 câu), Qwen3 v2 **66.7** (909 câu). Vẫn chưa so trực tiếp được với TAT-LLM 7B 71.3 hay Doc2SoarGraph 59.2: tập lọc chỉ có 1209/1663 câu (72.7 %), bỏ câu nhiều trang và câu không khớp OCR, tỷ lệ câu arithmetic cũng khác (53 % so với 42 %). Các số 64.8 và 68.3 từng nêu ở `h3_results.md` là thước đo dễ dãi, không dùng để so.

## Đính chính khác

Cột "oracle" chỉ chọn trong các cặp nguồn có sẵn, trong khi bộ phân loại của v2 có thể đưa ra scale mà không nguồn nào đề xuất (Qwen3, span: v2 cứu 10 câu, oracle 8). Vì vậy oracle **không phải trần** thực sự của v2.

Toàn bộ 1209 câu test đủ điều kiện đã được dùng, nên không còn test sạch trên tập lọc này. Mọi thay đổi policy tiếp theo phải xác nhận trên dữ liệu khác, ví dụ phần dev chưa dùng.
