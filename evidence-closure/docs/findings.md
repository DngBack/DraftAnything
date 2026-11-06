# Relational Evidence Closure — kết quả cuối (FINAL, sau khi sửa lỗi chấm attention)

**Kết luận một dòng:** bằng chứng trong tài liệu có tính quan hệ, và đo được (giá trị + nhãn hàng/cột/đơn vị thắng giá trị + vùng lân cận từ +10 đến +14 điểm ở cùng số token). Tuy vậy, **khi relevance theo attention được tính đúng, việc thêm quan hệ một cách tường minh không cải thiện nén visual token** ở cả 3 model. Theo tiêu chí bác bỏ của đề xuất (mục 4.4), H2 *như một phương pháp nén* là **no-go**.

Thiết lập: TAT-DQA **test** (chưa ai xem trong suốt quá trình thiết kế), câu hỏi trên bảng tài chính 1 trang. Qwen2.5-VL-7B và Qwen2.5-VL-3B chạy 600 câu / 245 tài liệu; Qwen3-VL-8B chạy 300 câu đầu (deviation #11). Cắt token sau merger, giữ M-RoPE gốc (Qwen3: cắt cả DeepStack theo cùng chỉ số), decode greedy (khớp `generate` 8/8 với từng model, deviation #9). Relevance = attention từ token câu hỏi tới visual token, trung bình layer (16, 20, 24), chọn trên dev theo fact recall (deviation #12). Mọi điều kiện và claim được khóa trước khi chạy (deviation #13). Số là value accuracy (%); CI 95 % bootstrap theo tài liệu. Chi tiết: `results/final_{qwen25vl7b,qwen25vl3b,qwen3vl8b}.md`.

## 1. Claim (a) — khoảng cách cấu trúc: **đạt ở cả 3 model**

| | 7B | 3B | Qwen3-8B |
|---|---|---|---|
| `values` (chỉ ô giá trị, ~10 token) | 41.8 | 34.5 | 42.0 |
| `local` (giá trị + lân cận, ~42 token) | 38.3 | 32.7 | 39.0 |
| `closure` (giá trị + nhãn hàng/cột/đơn vị, ~42 token) | **49.8** | **41.8** | **52.7** |
| closure − local | **+11.5** [+8.1, +15.3] | **+9.2** [+6.1, +12.4] | **+13.7** [+8.3, +19.5] |
| C_gold − L_gold @0.02 | **+10.5** [+7.3, +13.8] | **+8.0** [+4.9, +11.4] | **+10.0** [+5.3, +14.8] |
| `full` | 60.8 | 47.7 | 60.3 |

Với cùng các ô giá trị đúng, ~40 token "đúng quan hệ" đạt 82–88 % accuracy của cả trang; cùng số token đó chọn theo lân cận không gian thì chỉ đạt 63–69 %.

## 2. Claim chính — H50 thắng pruning theo attention: **thất bại ở cả 3 model**

| ρ = 0.05 | full | B_attn | H50 | **H50 − B_attn** | G_learn − B_attn | C_attn − B_attn |
|---|---|---|---|---|---|---|
| 7B | 60.8 | 58.2 | 57.3 | **−0.8** [−3.3, +1.6] | −14.5 * | −7.0 * |
| 3B | 47.7 | 42.7 | 42.8 | **+0.2** [−2.5, +3.0] | −6.7 * | −6.3 * |
| Qwen3-8B | 60.3 | 54.7 | 53.0 | **−1.7** [−5.9, +2.5] | −10.3 * | −6.7 * |

Ở ρ = 0.02 và 0.10 bức tranh giống hệt: H50 − B_attn nằm trong khoảng [−3.3, +0.2] ở mọi model; ở 0.10 thì H50 kém hơn có ý nghĩa với 7B và 3B. Mọi biến thể chỉ dùng quan hệ (C_attn, G_attn, G_learn) đều **kém hơn có ý nghĩa** so với B_attn.

## 3. Vì sao: attention đúng đã ngầm giữ quan hệ

- **Baseline mạnh.** Chỉ với 5 % token, B_attn đạt 96 % (7B), 90 % (3B) và 91 % (Qwen3) accuracy của trang đầy đủ. Ở 10 %, B_attn ≥ full ở 7B và Qwen3.
- **Trần oracle thấp hơn kỳ vọng.** Ngay cả khi biết đúng giá trị và thêm closure (C_gold), mức hơn B_attn ở 0.02 chỉ là +3.6 (7B), +7.5 (3B) và +8.3 (Qwen3). Ở 0.05 còn +2.3 / +2.1 / +5.6. Mọi bộ chọn thực tế của chúng tôi đều chưa lấy lại được phần dư địa nhỏ này.
- **Thay attention bằng nhóm thì hại.** G_learn chọn đúng nhóm gold ở 76 % câu trong top-4, nhưng vẫn kém B_attn 7–15 điểm. Model cần cả ngữ cảnh mà attention tìm ra (tiêu đề bảng, các dòng liên quan, các giá trị khác cho câu tính toán), không chỉ cần bộ ba giá trị–nhãn–đơn vị.

## 4. Những phát hiện trước đây đã bị thay thế

Các kết quả trong [findings_superseded.md](findings_superseded.md) dựa trên bước chấm attention có lỗi (deviations #12). Lỗi (a): attention được lấy từ toàn bộ văn bản sau ảnh thay vì từ câu hỏi. Lỗi (b): ở Qwen3, văn bản bị chặn không nhìn được ảnh. Hai lỗi này làm B_attn yếu đi một cách nhân tạo, nên các khẳng định sau **không còn đúng**:

- "H50 thắng ở cả 2 model (+3.5 / +23.7)";
- "3B có attention yếu nên nhóm quan hệ giúp +20–25";
- "closure khuếch đại seed sai ở 7B nhưng giúp 3B" (sau khi sửa, C_attn kém B_attn ở **mọi** model).

Phần không dùng attention vẫn giữ nguyên: `values` / `closure` / `local`, recall của selector, và kết luận rằng VLM tự chấm nhóm kém hơn khớp nhãn.

## 5. Ý nghĩa và hướng đi

1. **Viết được một bài phân tích (negative/diagnostic result).** Bằng chứng tài liệu có tính quan hệ; tuy nhiên relevance pruning bằng attention ở layer giữa, tính từ token câu hỏi, đã bảo toàn phần lớn quan hệ đó. Thêm cấu trúc tường minh không giúp ở các ngân sách 2–25 %. Bài này còn kèm một bài học phương pháp: hai lỗi kỹ thuật âm thầm (định vị token câu hỏi; mask "packed sequence" với M-RoPE) có thể làm baseline yếu đi và tạo ra phần thắng giả từ +3 đến +25 điểm.
2. **Dư địa thực sự không nằm ở nén bằng chứng.** Ở 7B, giữ 10 % token đã ngang full (~61 %). Phần lỗi còn lại nằm ở đọc và suy luận trên câu nhiều fact và câu tính toán, nên H1 (Proof-State) hoặc H3 (Value-of-Information) trong đề xuất đáng thử hơn.
3. Nếu vẫn muốn theo đuổi nén theo quan hệ, chỉ còn vùng ngân sách cực thấp (≤ 2 %), nơi oracle hơn B_attn 4–8 điểm. Lợi ích tối đa ở đó nhỏ, và các bộ chọn hiện tại chưa với tới.

## 6. Giới hạn

- Một miền dữ liệu (bảng tài chính TAT-DQA); chưa có biểu đồ hay tài liệu nhiều trang. Kết luận "attention đã đủ" có thể khác với bố cục khó hơn.
- Qwen3 chỉ chạy 300 câu. Layer chấm điểm được chọn trên dev; điều này có lợi cho baseline, đúng như chủ đích.
- Việc chấm attention cần một lượt forward riêng trên toàn chuỗi, nên chưa đo tiết kiệm thời gian thực. Các con số là chẩn đoán "giữ token nào" chứ không phải latency.
- Hàm chấm nới lỏng (bỏ dấu, sai số ≤ 0.5 %), áp dụng như nhau cho mọi điều kiện; decode greedy không dùng repetition penalty.
