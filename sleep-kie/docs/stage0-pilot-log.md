# Stage 0 — nhật ký pilot (chỉ dev seed, 25/10/2026)

Mọi số dưới đây là `DEV_SEED=555000`, 1 seed train, 256 tài liệu/độ dài. Test seed **chưa** được chạy. Kết quả thô nằm ở `results/pilot*/`.

## Pilot 1 — model không học được gì (lỗi thiết kế)

lr 3e-4, 2000 step: A0, A2 và full đều đoán toàn null, kể cả lookup T=4 (loss ln 64). Bisect ra 4 nguyên nhân, đã sửa cho mọi variant: schema query học riêng, LayerNorm cuối/memory, mã trang cộng vào token, top-K cứng. Chi tiết ở [protocol](stage0-protocol.md#nhật-ký-debug-trước-khi-khóa-chỉ-dev-seed--dữ-liệu-train).

## Pilot 2 — thêm curriculum độ dài (T: 4 → 50 trong 40% số step)

Không có curriculum, streaming vẫn kẹt ở T 10–50 dù đã học được ở T=4.

## Pilot 3 — giới hạn RMS slot, primary = revision + link

Accuracy (%), iid:

| variant | step | params | T | lookup | revision | link | aggregation | evidence hit |
|---|---|---|---|---|---|---|---|---|
| A0_N1 | 8000 | 3,86M | 50 | 100 | 100 | 1 | 1 | 100 |
| | | | 200 | 93 | 87 | 1 | 2 | 50 |
| full | 4000 | 6,49M | 50 | 100 | 100 | 5 | 1 | 100 |
| | | | 200 | 55 | 13 | 4 | 1 | 42 |
| A2_N4 (tied) | 4000 | 3,86M | mọi T | 0 | 0 | 0 | 0 | ~20 (= tỉ lệ none) |
| A5_deep4 | 4000 | 7,02M | – | NaN từ khoảng step 1000 | | | | |

Nhận định (thăm dò, 1 seed, dev):

1. **Streaming N=1 ngoại suy độ dài tốt hơn full-context**: revision 87% so với 13% ở T=200, khi train chỉ đến T=50. Đây là điểm cộng thật cho memory có giới hạn, nhưng mới là 1 seed.
2. **Link (ref/def thứ tự bất kỳ, 24 def nhiễu) là bài khó nhất**. Streaming chưa bao giờ học được. Full-context học được 95–98% ở pilot 2 nhưng chỉ 5% ở pilot 3 với cùng seed (khác biệt duy nhất là non-determinism GPU). Nghĩa là việc có thoát plateau hay không đang **quyết định kết quả nhiều hơn kiến trúc**.
3. **Variant sâu (N=4 tied và 4 block untied) chưa train được**: A2 sụp về hằng số (missing 0%), A5 NaN dù đã có giới hạn RMS. Nếu chạy ma trận khóa lúc này, Go 1 sẽ "không đạt" vì lý do tối ưu, không phải vì thiếu năng lực biểu diễn. Kết luận như vậy là không hợp lệ.
4. Aggregation mod 64 thì không model nào học được, nên chuyển thành secondary.

## Pilot 4 — fp32 + giảm bước cập nhật 1/N (phương án 1), 2 seed, 4000 step

Accuracy (%), iid, T=50 (T=200 trong ngoặc):

| run | lookup | revision | link | evidence hit | ghi chú |
|---|---|---|---|---|---|
| A0_N1 s0 | 1 (0) | 2 (0) | 1 | 87 | mới học evidence, chưa học value |
| A0_N1 s1 | 59 (6) | 50 (3) | 2 | 99 | bắt đầu thoát plateau ở khoảng step 3200 |
| A2_N4 s0 | 0 | 0 | 0 | 19 | kẹt |
| A2_N4 s1 | 2 | 1 | 2 | 93 | evidence bắt đầu học ở khoảng step 3200 |
| A5_deep4 s0 | 0 | 0 | 0 | 24 | kẹt |
| A5_deep4 s1 | – | – | – | – | OOM ở step khoảng 1400 (GPU 0 bị job khác chiếm 105 GiB) |

- **Hết NaN.** fp32 + bước 1/N giải quyết được phân kỳ của A2/A5.
- **Nhưng 4000 step không đủ cho bất kỳ variant nào**, kể cả A0, để ổn định ở fp32. Ở pilot 3 (bf16), A0 thoát plateau vào khoảng step 2400–3200; ở đây chỉ 1/2 seed của A0 bắt đầu thoát. Thời điểm thoát plateau ngẫu nhiên và có vẻ muộn hơn khi N lớn. Thứ tự luôn là evidence pointer học trước, value học sau.
- Vì vậy kết quả đến lúc này **không nói gì về Go 1**: chênh lệch hiện tại chỉ phản ánh tốc độ thoát plateau.

## Pilot 5 — fp32 + bước 1/N, 12k step, 3 seed (đánh giá 26/10, 5/9 run xong)

Accuracy (%), dev, iid (unseen-entity cho kết quả tương tự, chênh lệch dưới 3 pp):

| run | lookup T=50 | revision T=50 | revision T=100 | revision T=200 | link | evidence hit T=200 |
|---|---|---|---|---|---|---|
| A0_N1 s0 | 100 | 100 | 97,2 | 87,9 | 1,5 | 52,7 |
| A0_N1 s1 | 100 | 100 | 98,8 | 94,9 | 0,7–2,6 | 43,3 |
| A2_N4 s0 | 0 | 0 | 0 | 0 | 0 | 20 (kẹt suốt 12k step) |
| A2_N4 s1 | 100 | 100 | 92,5 | **69,0** | 0,8–2,3 | 47,7 |
| A5_deep4 s1 | 100 | 100 | 95,1 | 90,3 | 0,8–1,9 | 52,9 |

Tỉ lệ seed học được (thoát plateau): A0 2/2, A2 1/2, A5 1/1 (s0 bị OOM, đang chạy lại và đã thoát ở step 3800). Thời gian: A0 khoảng 45 phút, A2/A5 khoảng 85–90 phút trên GPU chia sẻ.

Đánh giá:

- **Benchmark hiện tại không phân biệt được các variant ở vùng train.** Ở T ≤ 50, mọi run đã học đều đạt trần (lookup, revision, missing 100%), còn link ở mức đoán mò (1/64 ≈ 1,6%). Primary cross-page (≈ 43%) thực chất chỉ là "revision 100% + link 0%" chia đôi. Trên benchmark này, Go 1 không thể "đạt"; tiêu chí khóa trước sẽ cho kết quả không đạt vì thiết kế, không phải vì phương pháp.
- **Khác biệt duy nhất xuất hiện ở ngoại suy độ dài** (train ≤ 50 trang): revision T=200 là A0 88–95, A5 90, A2 69 (1 seed). Recurrence tied **không** giúp, thậm chí kém hơn. Đây là tín hiệu ngược với giả thuyết, nhưng chỉ có 1 seed học được.
- **Link, bài kiểm tra memory consolidation thật, chưa variant streaming nào học được.** Full-context học được ở 1/2 lần. Đây là nơi recurrence có cơ hội tạo khác biệt, nhưng độ khó hiện tại nằm ngoài tầm mọi model.
- Evidence hit giảm ở T > 50 chủ yếu vì pointer theo mã trang chưa từng thấy khi train (ngoại suy vị trí), không hẳn vì memory.
- Chi phí: N=4 (tied hay untied) chậm hơn N=1 khoảng 2,4 lần/tài liệu (616 so với 253 ms ở bs=1). Peak memory của module như nhau (khoảng 540 MiB ở bs=64).

## Việc cần làm trước khi khóa (cần chọn)

- **Ổn định variant sâu** (áp dụng như nhau cho mọi variant):
  (a) fp32 thay bf16 cho memory, cho A5/A2;
  (b) residual scale 1/N trong vòng lặp;
  (c) warm-start: N=4 khởi tạo từ checkpoint A0 (block tied dùng chung trọng số nên làm được trực tiếp). Cách này thay đổi so sánh: A2 khi đó = A0 + fine-tune thêm compute, nên A0 cũng phải được fine-tune cùng số step.
- **Giảm phương sai plateau**: ≥ 3 seed + báo cáo tỉ lệ run thoát plateau như một metric riêng, hoặc giảm độ khó link (ít def nhiễu hơn) để mọi variant đều vào được vùng học.
- Thời gian: A0 khoảng 0,23 s/step, A2/A5 khoảng 0,45 s/step trên GPU đang chia sẻ. Ma trận 7 × 3 × 6000 step mất khoảng 6–8 giờ trên 2 GPU.
