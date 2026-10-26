# Stage 0 — Go 1 trên Controlled Long-KIE (token space)

**Trạng thái:** bản nháp, **chưa khóa** (xem [nhật ký pilot](stage0-pilot-log.md): variant sâu chưa train ổn định). Primary cross-page = revision + link; aggregation là secondary. Train có curriculum độ dài 4→50 trong 40% số step đầu. Pilot chỉ đánh giá trên `DEV_SEED=555000`; test (`TEST_SEED=777000`) chưa được chạy trước khi khóa.

## Câu hỏi

Go 1 của [idea](idea.md): recurrent consolidation (N=4, shared weights) có cải thiện cross-page accuracy so với N=1 ở cùng memory **và** so với updater sâu hơn compute-matched (4 block không chia sẻ, cùng FLOPs, ~1,8× params) không? Stage 0 cô lập riêng module memory: encoder là bảng embedding ngẫu nhiên đông cứng (giống nhau ở mọi variant), nên không nói gì về VLM, OCR hay ảnh thật.

## Dữ liệu ([sleepkie/data.py](../sleepkie/data.py))

Mỗi trang có 64 token `(kind, field, entity, value)` + noise. Schema gồm 8 field, rút từ 64 field type, và được cho trước khi đọc. Mỗi field mang một loại suy luận:

| loại | p | cơ chế | gold | evidence |
|---|---|---|---|---|
| lookup | .20 | `assert(f,v)` ở 1 trang | v | trang đó |
| revision | .20 | `assert` rồi 1–3 `amend` ở các trang sau, giá trị khác nhau | giá trị cuối | trang amend cuối |
| link | .25 | `ref(f,e)` và `def(e,v)` ở 2 trang ngẫu nhiên, **thứ tự bất kỳ** | v | cả 2 trang |
| aggregation | .15 | 2–3 `item(f,v_i)` | Σv_i mod 64 | mọi trang item |
| missing | .20 | không có | null | none |

Distractor: 24 `def` cho entity không ai tham chiếu (khi `def` đứng trước `ref`, mô hình buộc phải nhớ các def chưa biết có liên quan hay không, tạo áp lực memory thật), 8 field ngoài schema có assert/amend, và các token filler mang entity/value ngẫu nhiên. Cross-page = revision + link + aggregation.

- Train: T ~ U{10..50}, sinh mới ở mỗi step, entity link ∈ [1,192].
- Test: T ∈ {10, 25, 50, 100, 200}, mỗi độ dài 512 tài liệu, giống nhau cho mọi variant/seed. Có hai split: `iid` và `unseen_entity` (link entity ∈ [193,256], chưa từng xuất hiện khi train). T=100 và T=200 là ngoại suy độ dài.

## Variant ([sleepkie/model.py](../sleepkie/model.py))

d=256, 4 head, 8 field slot (khởi tạo từ tên field qua cùng encoder đông cứng) + free slot. Mỗi trang: token được ghép mã trang 32 chiều, ghi vào memory bằng cross-attn, qua N block consolidation (self-attn slot + cross-attn token trang hiện tại + MLP), rồi gate. Ở stage 0 mô hình đọc cả 64 token/trang; top-K selection (RQ1) để lại sau, xem bên dưới. Decoder: 2 block bắt đầu từ field slot → value (65 lớp, gồm null) và pointer evidence lên mã trang (+ none).

| variant | N | tied | slot tổng |
|---|---|---|---|
| A0_N1 | 1 | – | 32 |
| A1_N2 | 2 | có | 32 |
| A2_N4 | 4 | có | 32 |
| A3_N4_half | 4 | có | 16 |
| A4_N1_2x | 1 | – | 64 |
| A5_deep4 | 4 | không | 32 |
| full | – | – | không giới hạn (query đọc toàn bộ T×64 token; tham chiếu, không streaming) |

Loss: CE value + 0,5 × multi-positive evidence NLL. AdamW lr 1e-3, clip 1,0, wd .01, warmup 500, cosine, batch 64, BF16, BPTT đầy đủ qua cả tài liệu. Số step cố định (xem bên dưới) và đánh giá trọng số cuối, không chọn checkpoint. Mỗi variant chạy 3 seed (0, 1, 2).

## Endpoint và tiêu chí Go 1 (khóa trước test)

Primary: cross-page accuracy, split iid, T=50 và T=100. Paired bootstrap theo tài liệu (5000 lần), lấy trung bình qua seed.

- (a) A2_N4 − A0_N1 ≥ +5 pp, 95% CI > 0 và cả 3 seed cùng dấu, ở cả T=50 lẫn T=100.
- (b) A2_N4 − A5_deep4 ≥ +2 pp, 95% CI > 0 và cả 3 seed cùng dấu, ở cả T=50 lẫn T=100.
- **Go 1 đạt khi cả (a) và (b) đạt.** Nếu chỉ (a) đạt, tác dụng đến từ compute chứ không phải recurrence: không đầu tư adaptive sleep, chuyển sang evidence-aware streaming memory (theo §9 của idea).

Secondary (mô tả, không quyết định): unseen_entity, T=200, A1/A3/A4, đường cong khoảng cách, evidence hit/recall, null P/R, doc EM, latency và peak memory của module memory (GPU dùng chung với job khác nên latency chỉ mang tính tương đối).

## Nhật ký debug trước khi khóa (chỉ dev seed / dữ liệu train)

Pilot đầu (lr 3e-4, 2000 step) cho thấy **mọi** model, kể cả full-context, đều đoán toàn null; ngay cả lookup ở T=4 cũng đứng ở loss ln 64. Bisect (script nằm ngoài repo, chỉ dùng batch train) tìm ra bốn nguyên nhân, đều là vấn đề tối ưu chứ không phải lợi thế riêng của variant nào. Các sửa đổi áp dụng như nhau cho mọi variant:

1. Schema query là embedding học được, độc lập với encoder, nên phải học thêm một bảng tra 64 field. Sửa: tên field đi qua cùng encoder đông cứng (giống text query thật).
2. LayerNorm cuối trước value head, và LayerNorm memory sau mỗi trang, làm việc học chậm hẳn hoặc kẹt. Sửa: bỏ cả hai, giữ pre-LN trong block. Gate là tổ hợp lồi nên vẫn giới hạn cập nhật.
3. Mã trang được *cộng* vào token lấn át nội dung (kể cả khi khởi tạo bằng 0, Adam đẩy lên rất nhanh). Sửa: *ghép* mã trang 32 chiều.
4. Top-K cứng 16/64 loại token đúng khoảng 75% số lần lúc đầu, nên streaming không thoát được plateau. Tạm bỏ (đọc cả 64 token). Đây đúng là rủi ro §3.2 của idea: selection cần soft/ST hoặc supervision, và là câu hỏi riêng của RQ1.

Thêm: lr 1e-3 + clip gradient; gate bias 0. Sau các sửa đổi, A0 học lookup T=4 trong khoảng 600 step.

## Giới hạn đã biết

- Synthetic, encoder đông cứng ngẫu nhiên: không có ảnh, OCR, layout hay bbox. Evidence chỉ ở mức trang.
- A5 compute-matched về FLOPs inference/train, nhưng có nhiều tham số hơn. A2 và A0 có cùng số tham số.
- Training FLOPs của A2/A5 gấp ~2–4 lần A0 ở cùng số step. Cùng số training docs, không cùng training FLOPs.
- Chưa có adaptive stopping, consistency loss, prefix loss, baseline retrieval, top-K selection.
