# Protocol — Relational Evidence Closure, thí nghiệm E1 (structural information gap)

Khóa trước lượt chạy chính. Mọi thay đổi sau khi xem kết quả ghi vào `docs/deviations.md`.

## Câu hỏi

Ở cùng ngân sách visual token, giữ **nhóm token có quan hệ** (giá trị + nhãn hàng + tiêu đề cột + đơn vị/scale) có trả lời đúng hơn giữ **token độc lập theo độ liên quan** không? (H2 trong đề xuất, mục 8 "thí nghiệm đầu tiên": điều kiện A/B/C/D.)

Chưa train mô hình. Closure ở đây là heuristic layout rẻ tiền; nếu heuristic đã bịt được gap thì đó là kết quả bác bỏ phần "learned" của H2 (điều kiện bác bỏ ở 4.4).

## Dữ liệu

- TAT-DQA dev (`next-tat/TAT-DQA`), báo cáo tài chính thật, OCR word + bbox, `block_mapping` chỉ ra word chứa từng fact, trường `scale`.
- Lọc: tài liệu 1 trang; mọi fact map được vào word OCR; ít nhất một fact là số; block chứa fact có ≥ 2 token số (dòng bảng).
- Lấy mẫu tối đa 600 câu, seed 0, phân tầng theo `answer_type`. Nhóm phân tích: `single` (1 fact) / `multi` (≥ 2 fact); `span` / `arithmetic`; có / không có scale.
- Ảnh render lại từ PDF khớp hệ tọa độ OCR (PNG trong zip chỉ là thumbnail 224 px).

## Mô hình và cơ chế cắt token

- Qwen2.5-VL-7B-Instruct (chính), Qwen2.5-VL-3B-Instruct (lặp lại). Greedy decoding, tối đa 48 token.
- Vision encoder chạy trên ảnh đầy đủ; cắt **sau merger** (mỗi token = ô 28×28 px). Token giữ lại giữ nguyên vị trí M-RoPE gốc; token bị bỏ bị xóa khỏi chuỗi (không phải tô trắng). Đây là kiểu post-encoder pruning như FastV/DocPrune, nên ngân sách = số visual token vào LLM.
- Relevance độc lập: attention từ các token câu hỏi tới visual token, trung bình qua head và các layer {4, 8, 12, 16}, tính một lần trên input đầy đủ. Đây là điểm số mà B dùng, và C dùng cùng điểm số đó để chọn seed và lấp phần còn lại.

## Điều kiện (ngân sách K = ρ · N_full, ρ ∈ {0.05, 0.10, 0.25, 0.50})

| Mã | Nội dung | Vai trò |
|---|---|---|
| `full` | toàn bộ token | A |
| `B_attn@ρ` | top-K theo attention | B thực tế |
| `C_attn@ρ` | duyệt word theo attention giảm dần; word số kéo theo closure; lấp bằng attention tới K | C thực tế |
| `B_gold@ρ` | ô của fact (gold) + lấp attention tới K | B với seed oracle |
| `C_gold@ρ` | ô của fact + closure(fact) + lấp attention tới K | C với seed oracle |
| `L_gold@ρ` | ô của fact + vòng lân cận không gian có cùng số ô với closure + lấp attention | đối chứng locality |
| `values` | chỉ ô của fact | "giữ giá trị, mất ý nghĩa" |
| `closure` | ô fact + closure, không lấp | D tối thiểu |
| `local` | ô fact + lân cận cùng kích thước với `closure` | đối chứng của D |

B và C ở cùng ρ có **đúng** K token. Nếu closure vượt K, cắt theo ưu tiên seed > nhãn hàng > tiêu đề cột > đơn vị.

## Closure heuristic (chỉ dùng layout OCR, không dùng câu hỏi hay nhãn)

Với word số w:
1. Nhãn hàng: các word chữ bên trái w trên cùng dòng (block hoặc block khác chồng ≥ 50 % chiều cao).
2. Tiêu đề cột: các word phía trên, x chồng với w (nới 8 px), không phải số thuần (chữ, hoặc năm 19xx/20xx), trong 600 px, tối đa 3 dòng gần nhất.
3. Đơn vị: word `%`/`)%` ngay bên phải w trên cùng dòng; và block gần nhất phía trên (≤ 900 px) có thousand/million/billion/percent/`$ in`/`€`/`£`/`%`.

## Endpoint

- **Primary:** độ chính xác giá trị (value accuracy) trên toàn tập, cặp C_attn − B_attn ở ρ = 0.10 và 0.25, Qwen2.5-VL-7B. Bootstrap theo tài liệu (2000 lần), CI 95 %.
- Secondary: các ρ khác; C_gold − B_gold; C_gold − L_gold (relation hay chỉ locality); `closure` − `values`; `closure` − `local`; scale accuracy trên câu có scale; theo nhóm single/multi, span/arithmetic; lặp lại trên 3B.
- Chấm điểm: số trong dự đoán khớp số gold (bỏ dấu, `,`, `$`, ngoặc) nếu bằng nhau sau khi làm tròn theo số chữ số thập phân của gold, hoặc sai số tương đối ≤ 0.5 %; câu multi-span cần khớp mọi phần tử; câu chữ dùng token-F1 ≥ 0.5. Scale đúng nếu dự đoán chứa đúng từ scale (percent ↔ `%`). Mọi điều kiện chấm cùng một hàm nên phép so sánh cặp không phụ thuộc mức nới lỏng.

## Go / no-go

- **Gap tồn tại** nếu CI của C_attn − B_attn ở ρ = 0.10 hoặc 0.25 nằm hoàn toàn trên 0 **và** C_gold − L_gold > 0 (lợi ích không chỉ là locality).
- **Learned closure đáng làm** chỉ khi còn khoảng cách đáng kể giữa C_attn và `full` / C_gold ở ngân sách thấp (heuristic chưa bịt gap). Nếu C_attn ≈ full ở ρ = 0.25, heuristic đã đủ → điều kiện bác bỏ của 4.4.
- Không có gap → điều chỉnh hoặc bỏ hướng relational compression.

## Giới hạn biết trước

- Seed oracle dùng `block_mapping` gold; có lỗi map nhỏ (ví dụ fact "50" map nhầm dòng).
- Chỉ một miền (báo cáo tài chính), chủ yếu bảng; không có biểu đồ hay tham chiếu xuyên trang.
- Closure dùng OCR của dataset, không tính chi phí OCR vào latency.
