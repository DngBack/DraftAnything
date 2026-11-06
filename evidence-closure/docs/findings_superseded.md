> **SUPERSEDED (2026-11-05).** Tài liệu này dựa trên bước chấm attention có lỗi (deviations #12). Xem [findings.md](findings.md) cho kết quả đã sửa. Giữ lại để truy vết.

# Relational Evidence Closure — kết quả E1/E2 (dev) và kết luận go/no-go

> **⚠ Cảnh báo (2026-11-05):** phát hiện hai lỗi trong bước chấm attention ([deviations.md #12](deviations.md)). Mọi so sánh *với B_attn* và mọi điều kiện dựa trên attention ở các mục 2–10 (C_attn, G_attn, G_learn, H25/H50, phần lấp của *_gold) đều dùng tín hiệu relevance yếu hơn protocol mô tả, nên **bị thay thế** bởi lượt FINAL (deviation #13, đang chạy). Đáng chú ý: "attention yếu của 3B" phần lớn là artefact của lỗi. Các kết quả không dùng attention vẫn giữ nguyên: `values`, `closure`, `local`, recall của selector.

Tập: 600 câu TAT-DQA dev / 233 tài liệu, báo cáo tài chính 1 trang, fact nằm trong dòng bảng. Mô hình: Qwen2.5-VL-7B (chính) và 3B (lặp lại). Cắt visual token sau merger, giữ vị trí M-RoPE gốc; N ≈ 2 560 token mỗi trang. Mọi số là value accuracy (%); hiệu cặp có CI 95 % bootstrap theo tài liệu. Chi tiết: [`results/e1_qwen25vl7b.md`](../results/e1_qwen25vl7b.md), [`results/e1_qwen25vl3b.md`](../results/e1_qwen25vl3b.md), [`results/e2_qwen25vl7b.md`](../results/e2_qwen25vl7b.md), [`results/closure_eval.md`](../results/closure_eval.md). Protocol: [protocol.md](protocol.md), sai lệch: [deviations.md](deviations.md).

## 1. Có khoảng cách cấu trúc (structural information gap)

Ở ~41 token, cùng các ô giá trị đúng (seed gold):

| | 7B | 3B |
|---|---|---|
| `values` — chỉ ô giá trị (~10 token) | 43.3 | 34.5 |
| `local` — giá trị + ô lân cận (41 token) | 39.2 | 35.5 |
| `closure` — giá trị + nhãn hàng + header cột + đơn vị (41 token) | **54.2** | **43.5** |
| closure − values | +10.8 [+8.0, +13.9] | +9.0 [+6.3, +11.8] |
| closure − local | **+15.0** [+11.6, +18.5] | **+8.0** [+4.8, +11.2] |
| scale acc: closure − values (7B) | +40.6 [+34.1, +46.9] | (3B gần như không ghi scale) |
| `full` (2 563 token) | 63.5 | 51.8 |

Giữ thêm vùng lân cận không giúp gì; giữ đúng *quan hệ* thì giúp. Lợi ích tập trung ở câu `single` và `span` (7B: closure − local = +19.8 trên câu single). Chỉ dùng 1.6 % token, `closure` đạt 85 % accuracy của `full` ở 7B.

Khi lấp thêm token bằng attention tới ngân sách K, khoảng cách chỉ còn ở ngân sách rất chặt:

| C_gold − L_gold | ρ = 0.02 | 0.05 | 0.10 | 0.25 |
|---|---|---|---|---|
| 7B | **+10.8** [+7.8, +14.2] | +3.2 [+0.8, +5.5] | +0.7 | −0.3 |
| 3B | **+8.2** [+5.1, +11.2] | +5.7 [+2.9, +8.5] | +2.2 | −1.3 |

## 2. Endpoint chính: closure theo attention so với pruning độc lập — **không đạt ở 7B**

| C_attn − B_attn | ρ = 0.02 | 0.05 | 0.10 | 0.25 | 0.50 |
|---|---|---|---|---|---|
| 7B | −8.0 * | −4.3 * | **−6.3** [−9.4, −3.2] | **−7.7** [−10.6, −4.6] | −1.7 |
| 3B | +2.0 * | +2.7 * | **+2.7** [+0.2, +5.1] | **+5.7** [+2.7, +8.8] | +3.8 * |

Dấu đảo theo model:

- **3B có attention yếu** (B_attn chỉ đạt 10–25 % ở ρ ≤ 0.25, so với full 51.8). Closure kéo được ô giá trị đúng vào cùng với nhãn của chúng, nên có ích.
- **7B có attention tốt.** C_attn lấy seed là các số có attention cao, kể cả *số gây nhiễu*, rồi bổ sung nhãn hàng/cột cho chúng. Số nhiễu trở thành bằng chứng "có nghĩa" cạnh tranh với đáp án. Lỗi điển hình là đọc lẫn chữ số giữa các giá trị (1,609,000 → 1,069,000; 8,736 → 8,376). Xem [`results/viz/tokens_001_crop.png`](../results/viz/tokens_001_crop.png).

Kết luận: **closure khuếch đại cả seed đúng lẫn seed sai.** Bảo toàn quan hệ chỉ có ích khi đã chọn đúng giá trị.

## 3. Thử sửa (E2, thăm dò vì thiết kế sau khi xem dev)

| Thay đổi | Kết quả (7B) |
|---|---|
| **Group scoring** `G_attn`: chấm cả nhóm (giá trị + nhãn) theo attention trung bình | Tốt hơn C_attn ở ρ = 0.02 (+5.3 [+2.0, +8.7]) nhưng vẫn ≤ B_attn; ở 0.10 thì G − B = −6.3 [−9.6, −3.0] |
| **Learned closure `geo`** (MLP trên đặc trưng layout, nhãn yếu từ 8 823 câu train) | F1 nhãn yếu 0.45 so với 0.30 của heuristic (tài liệu dev chưa thấy, cùng kích thước), nhưng accuracy VLM **không hơn**: closure_geo − closure = −0.8 [−3.2, +1.5]; scale +4.2 [+0.3, +8.1] |
| **Learned closure `query`** (thêm đặc trưng trùng từ với câu hỏi) | **Tệ hơn**: −4.3 [−7.1, −1.8]; nó bỏ mất header/đơn vị không xuất hiện trong câu hỏi (scale 68.6 so với 82.3) |
| C_attn_geo (learned closure thay heuristic trong C_attn) | Không đổi hoặc tệ hơn: so với C_attn là 0.0 ở 0.10, −3.3 ở 0.02 |

Proxy "relation recall" theo nhãn yếu **không dự báo** được accuracy của VLM. Nhãn yếu chỉ đo việc giữ đúng word mà câu hỏi nhắc tới, không đo việc giữ đủ để VLM đọc đúng.

## 4. Đối chiếu tiêu chí go/no-go trong protocol

| Tiêu chí | Kết quả |
|---|---|
| Gap tồn tại: C_attn − B_attn > 0 ở 0.10/0.25 **và** C_gold − L_gold > 0 | **7B: không đạt** (âm, có ý nghĩa). 3B: đạt. Riêng C_gold − L_gold > 0 thì đạt ở ρ = 0.02 với cả hai model |
| Learned closure đáng làm (còn khoảng cách C_attn so với full / C_gold) | Còn khoảng cách lớn (7B: full − C_attn@0.25 = +9.7), **nhưng** learned closure theo hướng hiện tại không thu hẹp được nó |
| Điều kiện bác bỏ ở mục 4.4 (heuristic rẻ ≈ learned) | **Bị kích hoạt**: learned không hơn heuristic |

**Quyết định đề xuất:** giữ giả thuyết *structural gap* (có bằng chứng tốt, nhất quán ở hai model), nhưng **bỏ phiên bản "relation closure không phụ thuộc câu hỏi"** như một phương pháp nén. Nút thắt nằm ở **chọn đúng nhóm bằng chứng** (seed/group selection), không nằm ở việc bảo toàn quan hệ của nhóm đã chọn. Điều này khớp với ghi chú cuối của đề xuất: "chỉ mở rộng sang Proof-State nếu bảo toàn quan hệ chưa đủ giải quyết evidence completeness". Đây chính là trường hợp đó.

## 5. Hướng đi tiếp có căn cứ từ dữ liệu

1. **Answerability-scored groups.** Chấm điểm nhóm (giá trị, nhãn hàng, header, đơn vị) bằng mức khớp giữa *nhãn* và câu hỏi, và/hoặc bằng chính VLM (log-prob của câu trả lời khi chỉ giữ nhóm đó). Đây là "query-conditioned group selector" của đề xuất, nhưng tín hiệu phải nằm ở cấp nhóm chứ không phải ở cấp token. Trên dev, bằng chứng gián tiếp là C_gold@0.02 (seed đúng + closure) đạt 56.3 % ở 51 token, so với B_attn@0.02 chỉ 40.7 %. Tức là còn ~16 điểm có thể lấy lại nếu chọn đúng nhóm.
2. **Supervision cho learned closure theo tác động lên VLM** (xóa thử từng word trong closure, xem câu trả lời có đổi không), thay cho nhãn yếu từ câu hỏi.
3. **Proof-state (H1) như bộ chọn nhóm:** mỗi nghĩa vụ chứng minh ứng với một nhóm (giá trị + nhãn) phải có mặt. Kết quả ở đây cho thấy biểu diễn theo nhóm là đơn vị đúng để làm việc đó.

## 6. E3 — xác nhận trên TAT-DQA test (chưa ai xem, điều kiện khóa trước, deviation 6)

600 câu / 245 tài liệu. Chi tiết: [`results/e3_qwen25vl7b.md`](../results/e3_qwen25vl7b.md), [`results/e3_qwen25vl3b.md`](../results/e3_qwen25vl3b.md).

| Claim | 7B | 3B | Lặp lại dev? |
|---|---|---|---|
| (a) closure − values | +7.7 [+4.7, +10.8] | +7.3 [+4.7, +10.2] | có |
| (a) closure − local | **+10.8** [+7.5, +14.6] | **+8.7** [+5.6, +11.9] | có |
| (a) C_gold − L_gold @0.02 | **+9.3** [+6.1, +12.8] | **+8.8** [+5.6, +12.3] | có |
| (b) C_attn − B_attn @0.10 | −3.8 [−7.2, −0.5] | +5.5 [+3.1, +8.2] | có (đổi dấu theo model) |
| (b) C_attn − B_attn @0.25 | −5.8 [−8.8, −2.9] | +6.5 [+3.3, +9.9] | có |
| (c) G_attn − B_attn @0.10 | −4.3 [−7.6, −1.0] | **+19.7** [+16.2, +23.1] | 7B có; 3B chưa đo ở dev |
| (c) G_attn − B_attn @0.25 | −5.5 [−8.2, −2.9] | **+17.2** [+13.2, +21.1] | như trên |
| scale acc: closure − values (7B) | +41.5 | – | có |

Kết luận đã xác nhận: (1) khoảng cách cấu trúc là có thật và tổng quát sang tài liệu mới; (2) lợi ích của closure phụ thuộc chất lượng seed. Với model có attention yếu (3B), chọn theo nhóm tăng accuracy ở ngân sách 10 % từ 13.0 lên 32.7 %. Với model có attention tốt (7B), closure theo attention vẫn thua pruning độc lập.

## 7. E4 — chấm điểm nhóm theo khả năng trả lời (dev, thăm dò)

Ứng viên: mỗi số trên trang cùng closure heuristic của nó (~75 nhóm mỗi trang). Bộ chấm `lex` đếm số từ nhãn trùng với câu hỏi. Bộ chấm `vlm` chỉ giữ riêng nhóm đó rồi hỏi VLM "có đủ thông tin không?" và lấy logit Yes − No. Chọn nhóm theo điểm tới K, phần còn lại lấp bằng attention. Chi tiết: [`results/e4_qwen25vl7b.md`](../results/e4_qwen25vl7b.md), [`results/e4_qwen25vl3b.md`](../results/e4_qwen25vl3b.md).

| | recall@4 nhóm gold (7B / 3B) | G_lex − B_attn @0.02 / 0.05 / 0.10 (7B) | (3B) |
|---|---|---|---|
| lex | 68.4 / 64.0 | −1.5 / **−4.2** / **−7.7** | **+20.2 / +24.5 / +23.2** |
| vlm | 41.1 / 51.8 | **−7.8 / −9.7 / −11.5** | **+21.5 / +24.2 / +20.7** |

- VLM tự chấm nhóm có recall kém hơn khớp nhãn đơn giản, nên hướng này bị loại.
- Với 3B, chọn theo nhóm gần chạm trần oracle (C_gold − G_lex @0.10 = +4.7).
- Với 7B, khoảng cách tới oracle còn 10–17 điểm, và chọn theo nhóm vẫn thua attention. Nút thắt là độ chính xác khi xếp hạng nhóm: các nhóm sai chiếm ngân sách mà attention của 7B lẽ ra dùng tốt hơn.

Tiếp theo:
- **E5** (đã khóa, đang chạy): bộ chọn nhóm học từ train (recall@4 trên test 76.1).
- **E6** (dev, thăm dò): lai, dành α·K token cho attention trước rồi mới thêm nhóm.

## 8. E5 — bộ chọn nhóm học được (khóa trước, deviation 7)

`evc/groupsel.py` là MLP listwise trên đặc trưng OCR/layout của nhóm (khớp nhãn hàng và header với câu hỏi, năm, đơn vị, vị trí), học từ fact gold của TAT-DQA train. Recall@4 nhóm gold: test 76.1 so với 65.2 của lex. Chi tiết: `results/e5_*.md` (dev), `results/e5test_*.md` (test).

| Test | B_attn @0.05 / @0.10 | G_learn @0.05 / @0.10 | G_learn − B_attn @0.05 / @0.10 | top4_learn (~150 token, chọn trước VLM) |
|---|---|---|---|---|
| 3B | 11.0 / 13.3 | 36.0 / 38.8 | **+25.0 / +25.5** | 35.2 (full 47.7) |
| 7B | 48.3 / 55.0 | 43.5 / 47.8 | **−4.8 / −7.2** | 40.7 (full 60.7) |

- Claim khóa trước **đạt với 3B, không đạt với 7B**; dev cho kết quả giống hệt.
- Ở 7B, recall nhóm gold cao hơn không cải thiện accuracy: G_learn @0.10 đạt 47.8 trên test, so với G_lex 55.4 trên dev. Cách *thay* attention bằng nhóm làm mất ngữ cảnh mà attention của 7B giữ được (tiêu đề bảng, các dòng liên quan khi cần tính toán). C_gold còn hơn G_learn 15 điểm ở 0.05, nên vấn đề nằm ở *cả* việc chọn nhóm lẫn ngữ cảnh ngoài nhóm.
- **E6** (dev, thăm dò, đang chạy): lai, dành α·K token cho attention trước rồi mới thêm nhóm, với α ∈ {0.25, 0.5}.

## 9. E6 — lai attention + nhóm quan hệ (dev thăm dò → test khóa trước, deviation 8)

H50: 50 % ngân sách cho top attention, phần còn lại là các nhóm theo learned group selector, cuối cùng lấp bằng attention. Chi tiết: `results/e6_*.md` (dev), `results/e6test_*.md` (test).

| value acc | B_attn @0.02 / 0.05 / 0.10 | G_learn | **H50** | **H50 − B_attn @0.05** |
|---|---|---|---|---|
| 7B dev | 41.0 / 51.5 / 57.8 | 39.3 / 46.3 / 50.0 | 44.2 / 56.7 / 58.5 | +5.2 [+2.6, +7.9] |
| **7B test** | 38.5 / 48.5 / 54.8 | 37.3 / 43.5 / 47.0 | 41.0 / **52.0** / 54.7 | **+3.5 [+0.8, +6.6]** |
| 3B dev | 10.7 / 12.7 / 17.7 | 34.5 / 38.0 / 40.7 | 26.7 / 34.8 / 41.2 | +22.2 [+18.6, +25.9] |
| **3B test** | 9.5 / 11.2 / 13.0 | 32.0 / 36.0 / 39.2 | 26.2 / **34.8** / 38.7 | **+23.7 [+19.7, +27.5]** |

- **Claim khóa trước đạt với cả hai model.** Đây là phương pháp đầu tiên trong nghiên cứu này thắng pruning thuần attention ở cả model có attention tốt lẫn model có attention yếu.
- Với 7B, lợi ích khiêm tốn và chỉ có ý nghĩa ở ρ = 0.05. Ở 0.02 chênh lệch là +2.5 [−0.7, +5.7]; ở 0.10 là −0.2, vì B_attn đã đạt ~90 % mức full.
- Với 3B, H50 ≈ G_learn (−1.2 ở 0.05): khi attention vô dụng thì phần nửa ngân sách cho attention lãng phí. H25 cho kết quả cân bằng hơn ở 3B (+24.5), nhưng α được khóa là 0.5.
- H50 − G_learn ở 7B là +8.5 [+5.6, +11.7] trên test: nhóm quan hệ và attention **bổ sung** cho nhau, không thay thế được nhau.

## 10. Kết luận tổng thể (tính tới 2026-11-05)

1. **Bằng chứng tài liệu có tính quan hệ** (H2, phần chẩn đoán). Đã xác nhận trên test với 2 model: ở cùng số token, giá trị + nhãn hàng/cột/đơn vị thắng giá trị + vùng lân cận từ +9 đến +11 điểm; scale accuracy tăng khoảng 40 điểm.
2. **Closure không phụ thuộc câu hỏi khuếch đại cả seed sai.** Nó có ích với model có attention yếu và có hại với model có attention tốt (đổi dấu theo model, đã xác nhận trên test).
3. **Thay attention bằng nhóm** (G) cho lợi ích rất lớn với 3B (+25) nhưng thua với 7B (−5 tới −8). Chọn đúng nhóm giá trị là chưa đủ; model mạnh còn cần ngữ cảnh mà attention của nó tìm ra.
4. **Kết hợp attention với nhóm quan hệ học được (H50) thắng ở cả hai model** tại ngân sách 5 %: +3.5 (7B) và +23.7 (3B) trên test. Đây là phương pháp đề xuất.
5. Khoảng cách còn lại tới oracle C_gold ở 7B @0.05 vẫn là ~7 điểm (52.0 so với 58.8). Phần lớn đến từ xếp hạng nhóm (recall@4 76 %) và từ câu nhiều fact.

## 11. Giới hạn

- Một miền (báo cáo tài chính TAT-DQA), chủ yếu là bảng; chưa có biểu đồ hay tham chiếu xuyên trang.
- Seed oracle lấy từ `block_mapping`, vốn có lỗi map nhỏ. Closure dùng OCR có sẵn; chi phí OCR chưa tính vào latency.
- Accuracy dùng hàm chấm nới lỏng (bỏ dấu, sai số ≤ 0.5 %), áp dụng như nhau cho mọi điều kiện. 3B hầu như không ghi scale nên scale acc chỉ có ý nghĩa với 7B.
- G_attn và learned closure được thiết kế sau khi xem dev, nên là kết quả thăm dò. Kết quả xác nhận nằm ở E3 (TAT-DQA test, điều kiện đã khóa trong deviation 6).
- Pruning sau encoder: vision encoder vẫn xử lý toàn trang, nên lợi ích FLOPs chỉ nằm ở phía LLM.
