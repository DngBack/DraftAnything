# PartScribe

Sinh chữ viết tay có style, với nội dung biểu diễn bằng **thành phần nét học được**. Mục tiêu là sinh đúng và đúng phong cách những tổ hợp ký tự không có ảnh thật, và vẫn giữ chất lượng trên IAM.

- Thiết kế: [docs/261118_plan_03](docs/261118_plan_03_Kien_truc_moi_bieu_dien_thanh_phan_va_tron_style.md).
- Tổng quan tài liệu và protocol giữ lại: [docs/261118_report_04](docs/261118_report_04_Tong_quan_tai_lieu_va_protocol_giu_lai_to_hop.md).
- Protocol v1: [docs/holdout_protocol_v1.json](docs/holdout_protocol_v1.json). File được khóa bằng SHA-256 trong `partscribe/holdout.py`.
- **Report tổng hợp đợt thực nghiệm tiếng Việt (18–22/11):** [docs/261122_report_01](docs/261122_report_01_Tong_hop_thuc_nghiem_PartScribe_tieng_Viet.md). **Bản chốt hiện tại: nfd v3 350k** ([docs/final_v3_350k.json](docs/final_v3_350k.json)); mục "Bản chốt: VN v2, 300k" bên dưới là bản trước.

## Kiến trúc và vị trí trong code

```
text ──► [A] PartTokenizer (frozen) ──► ContentTable ──► [B] ContentEncoder ──► part tokens ─┐
                                                              └► WidthHead ──► layout ───────┤
support ─► [C] StyleEncoder ──► g (writer) + M (patch memory) ───────────────────────────────┼► [D] DiT ─► [E] VAE ─► image
noise ───────────────────────────────────────────────────────────────────────────────────────┘   (rectified flow)
```

| Khối | File | Ghi chú |
|---|---|---|
| [A] Part tokenizer | `partscribe/tokenizer.py` | Slot attention + FSQ (240 mã) + where. Train trên font, không dùng nhãn thành phần |
| Content table | `partscribe/content.py::build_content_table` | Token của mỗi ký tự trong 32 font train tham chiếu. Train chọn ngẫu nhiên font tham chiếu; sample dùng font 0 |
| [B] Context encoder | `partscribe/content.py::ContentEncoder` | `parts`: một token cho mỗi slot hợp lệ (mã dùng chung giữa các ký tự). `char`: baseline, một embedding cho mỗi ký tự |
| Width head | `partscribe/content.py::WidthHead` | Độ rộng từng ký tự theo `g`. Khi train, tỉ lệ được co để tổng bằng độ rộng thật. Loss chỉ đặt trên tổng độ rộng |
| [C] Style | `partscribe/style.py` | `M` = patch của 8 ảnh support (stride 8). `g` = 4 query gộp từ `M`. Có SupCon theo writer và gradient reversal chống mang nội dung |
| [D] DiT | `partscribe/dit.py` | AdaLN-zero (t + g), cross-attention tới nội dung có bias Gaussian quanh tâm ký tự, cross-attention tới `M`. Mọi cổng khởi tạo bằng 0, null token cho CFG |
| [E] VAE | `partscribe/vae.py` | f=4, 4 kênh. Loss L1 + Sobel + PatchGAN. `latent_scale` đo sau khi train |
| Flow | `partscribe/flow.py` | Rectified flow, t logit-normal, Euler. CFG tách riêng cho nội dung và style |
| Dữ liệu | `partscribe/datasets.py` | Áp protocol khi nạp. Batch gồm 16 writer × 4 ảnh. Support không chứa ảnh đích và không chứa ảnh có cùng transcript |

### Ba quyết định cần biết khi đọc kết quả

1. **Không có nhãn độ rộng từng ký tự.** Bias căn chỉnh `-½((x − tâm)/σ)²` khả vi theo độ rộng, nên loss khử nhiễu dạy được tỉ lệ rộng giữa các ký tự. Test `test_layout_gradient_reaches_widths` kiểm tra điều này.
2. **Độ cao ảnh:** IAM 64 px, cắt lại từ ảnh dòng gốc. Tiếng Việt **32 px**, vì HDF5 của VATr-pp chỉ lưu 32 px; upsample không thêm thông tin. Muốn có 64 px cho tiếng Việt thì phải render lại từ InkML của VNOnDB, việc này chưa làm.
3. **HAT2023 không được dùng.** HAT không có writer, nên không làm được style hay SupCon. VATr++ dùng HAT làm prompt cho OCR giám sát; PartScribe không có OCR giám sát.

## Chạy

Chạy mọi lệnh từ `hand_writting/PartScribe`. Đường dẫn tới dữ liệu tiếng Việt phải trỏ tới thư mục có `train.h5`/`test.h5`; hash được kiểm tra với fingerprint của VATr-pp.

```bash
# 0. dữ liệu
python scripts/fetch_fonts.py --out data/fonts --manifest data/fonts_manifest.json
python scripts/prepare_iam.py --iam /home/bachdx2/DraftAnything/handwriting/data/iam \
    --groundtruth ../VATr-pp/Groundtruth --out data/iam64.npz
# 1. tokenizer + probe (xem mục Giai đoạn 1)
python scripts/train_tokenizer.py --model slots  --variant strict --out runs/tok_slots_strict
python scripts/train_tokenizer.py --model global --variant strict --out runs/tok_global_strict
python scripts/run_probes.py --ckpt runs/tok_slots_strict/best.pt --out runs/tok_slots_strict/probes.json
python scripts/build_content_table.py --ckpt runs/tok_slots_strict/best.pt --out data/content_strict.pt
# 2. VAE
python scripts/train_vae.py --dataset iam --out runs/vae_iam64
python scripts/train_vae.py --dataset vn --vn-data <h5_dir> --out runs/vae_vn32
# 3. baseline token ký tự, 4. token thành phần (giữ nguyên mọi thứ khác)
python scripts/train_generator.py --dataset iam --content char  --vae runs/vae_iam64/final.pt --out runs/gen_iam_char
python scripts/train_generator.py --dataset iam --content parts --table data/content_strict.pt \
    --vae runs/vae_iam64/final.pt --out runs/gen_iam_parts
# mẫu
python scripts/sample.py --dataset iam --split test --ckpt runs/gen_iam_parts/last.pt \
    --table data/content_strict.pt --vae runs/vae_iam64/final.pt --texts jumping Yesterday --out samples.png
```

`--no-holdout` chỉ dùng cho run IAM parity trên dữ liệu đầy đủ. Mặc định, mọi run đều áp protocol v1 và ghi SHA-256 của protocol vào `config.json`.

Test: `PARTSCRIBE_VN_DATA=<h5_dir> python -m unittest discover -s tests -v`.

## Giai đoạn 1: probe tokenizer

Định nghĩa probe nằm trong `partscribe/probes.py`:

- **P1, phân rã:** probe huấn luyện trên ký tự **đã thấy** ở font train, đọc (thân chữ, tập dấu) của ký tự **giữ lại** ở font test.
  - Slot dùng DeepSets.
  - GlobalAE và pixel thô dùng MLP.
  - 3 seed, 40 epoch.
- **P2, gắn dấu (chỉ cho slot):**
  - Slot sở hữu pixel của một dấu phải có độ chính xác ≥ 0,5 trên dấu đó.
  - Mã FSQ của slot, tra trong bảng lập từ ký tự đã thấy, phải gọi đúng tên dấu.

**Ngưỡng đi tiếp, đề xuất. Phải chốt trước khi chạy probe trên checkpoint train đầy đủ; chưa có kết quả probe nào.**

1. P1 `test_heldout.exact` của slot cao hơn GlobalAE ít nhất 10 điểm phần trăm, và cao hơn pixel thô.
2. P2 `test_heldout.success` ≥ 0,6, và cao hơn `majority_mark_rate`.

Nếu (1) không đạt, biểu diễn slot không cho phân rã tốt hơn một vector toàn cục. Khi đó dùng bản `char` cho giai đoạn 3 và ghi kết quả null cho tokenizer.

## Bản chốt: VN v2, 300k (21/11/2026)

Checkpoint, SHA-256, cấu hình sinh và toàn bộ số liệu nằm trong [docs/final_v2_300k.json](docs/final_v2_300k.json).

- **Checkpoint báo cáo:** `runs/gen_vn_{nfd,char}_v2/snap_300000.pt`, dùng trọng số EMA.
- **Quá trình train:** đi từ `_lt` 100k, qua v2 200k, rồi nối tiếp tới 300k bằng `pipeline.sh gen_{mode}_v2_300k` (`--seed 1`, LR không đổi).
- **Cấu hình sinh để báo cáo:** 50 bước, `w_content` 3, `w_style` 1.5, không dùng `w_mark`. Đây là cấu hình mặc định, không chỉnh theo tập đánh giá.
- **Hình minh họa** dùng `scripts/showcase.py` với `w_content` 4.5 và chọn 1 mẫu tốt nhất trong 16; không dùng cho số liệu.

Ảnh sinh được giải mã bằng `partscribe.vae.decode_generated`. Hàm này bỏ vạch xám ở mép phải do decoder tạo ra tại ranh giới độ rộng. Tất cả số dưới đây dùng hàm đó; số đo trước khi sửa vẫn nằm trong JSON (`before_edgefix`).

**Ký tự bị giữ lại** (`pair_ok`, 30 writer val, đo bằng LocalOCR):

| Nhóm | char | nfd |
|---|---|---|
| C1+C2 | 0,012 | **0,829** (CI chênh +0,75..+0,87) |
| C1 | | 0,867 |
| C2 | | 0,773 |
| C3 | | 0,895 |
| Đối chứng | 0,962 | 0,967 |

**Metric chuẩn trên ký tự đã thấy** (`scripts/eval_standard.py`; quy ước của repo HWD; 2.847 từ; mỗi mô hình dùng cùng 15 ảnh mẫu):

| Mô hình | FID | BFID | KID×10³ | HWD | CER |
|---|---|---|---|---|---|
| VATr++ VN 175,5k | 38,13 | 25,27 | 20,07 | 1,713 | 1,74% |
| VATr++ VN 250k | 38,46 | 20,77 | 21,36 | 1,613 | 1,18% |
| PartScribe char 300k | 16,25 | 11,62 | 5,35 | 0,871 | 2,80% |
| **PartScribe nfd 300k** (3 seed) | **16,13 ± 0,10** | **11,31 ± 0,22** | **5,31 ± 0,01** | **0,876 ± 0,006** | 2,74% ± 0,05 |
| Ảnh thật | 11,9 (nửa so với nửa) | | | | 8,80% |

So với VATr++ 250k (CI theo writer):
- HWD tốt hơn, chênh −0,97..−0,49.
- CER kém hơn 1,1..2,2 điểm %. CER của VATr++ còn thấp hơn ảnh thật.

nfd và char ngang nhau trên ký tự đã thấy.

Sửa mép, so cùng checkpoint:
- FID 19,60 → 16,13.
- HWD 0,906 → 0,876 (CI chênh −0,046..−0,015).
- CER 3,13 → 2,74%.
- `pair_ok` 0,793 → 0,829.

Từ 200k lên 300k (trước khi sửa mép): CER giảm 3,32 → 3,13%; FID, HWD và `pair_ok` không đổi.

**Giới hạn đã biết:**
- Chấm dưới của `ẹ` phần lớn bị mất; `w_mark` không sửa được.
- Tên riêng viết thường ("Ngô Quyền") ra cùng một kiểu chữ đậm cho mọi người viết, nhiều khả năng do đích HAT không kèm style.
- Mọi số liệu đều đo trên tập val; tập test chưa mở.
- IAM đang tạm dừng.

## Trạng thái (18/11/2026, 20:30 UTC)

- **Giai đoạn 1: NO-GO.** Probe P1 trên ký tự giữ lại: slots 0,260, global 0,416, pixel 0,369. P2: 0,290. Nhánh chính đổi sang `--content nfd` ([docs/stage1_deviation.md](docs/stage1_deviation.md)).
- **Long tail theo VATr++:** `Batcher(rare_fraction=0.3)` dùng đúng trọng số `ComponentCorpusSampler` của VATr-pp, áp lên ảnh đích. Test `test_vatr_parity_if_available` kiểm tra trọng số khớp tuyệt đối. Run chính: `gen_vn_char_lt`, `gen_vn_nfd_lt` (200k bước), sau đó IAM. Hai run cũ `gen_vn_char` / `gen_vn_nfd` (tần suất tự nhiên) dừng ở 25k, giữ làm ablation sampler.
- **Kết quả đầu (run tần suất tự nhiên, 25k bước, raw, 30 writer dev, `runs/eval/vn_25000/`):**

  | Nhóm | char | nfd | nfd − char (CI 95%) | Ảnh thật |
  |---|---|---|---|---|
  | C1+C2 | 0,018 | 0,530 | +0,512 (+0,373..+0,598) | 0,799 |
  | C3 | 0,000 | 0,684 | +0,684 (+0,429..+0,867), n=19 | 1,000 |
  | Đối chứng | 0,856 | 0,882 | +0,027 (−0,007..+0,054) | 0,923 |

  **Mốc 50k, run `_lt` (`runs/eval/vn_lt_50000/`):** C1+C2 char 0,024 / nfd 0,756, chênh +0,732 (CI +0,635..+0,804); C3 +0,421 (CI +0,154..+0,600); đối chứng 0,931 / 0,942 (CI chứa 0); CER toàn bộ char 0,249 / nfd 0,112.

  **Mốc 75k, run `_lt`:** C1+C2 char 0,018 / nfd 0,848, chênh +0,829 (CI +0,710..+0,907); C3 +0,737 (CI +0,357..+0,955); đối chứng 0,950 / 0,951; CER char 0,226 / nfd 0,074.

  Vượt ngưỡng khai báo trước (+10 điểm, CI không chứa 0) nhưng mới là mốc 25k; kết luận chính chờ run `_lt` 200k và tập test.
- **v2 (19/11):** chẩn đoán ảnh mẫu ở mốc 75k cho thấy:
  - dữ liệu gần như không có dấu cách (109/96k nhãn), nên cụm như "Hà Nội", "bỡ ngỡ" bị chèn nét rác;
  - OCR chấm (BiLSTM) tự thêm dấu từ ngữ cảnh: xóa vùng dấu trên ảnh thật vẫn đọc ra `ẹ` ở 19/32 ảnh, ký tự có dấu ở trên 59/151;
  - Width head khớp tốt (độ rộng dự đoán trên thật: trung vị 1,00).

  Thay đổi:
  - `Batcher(multiword=0.2)` ghép 2–3 từ cùng writer, chỉ dùng ảnh nền sạch;
  - `mark_weight=3` tăng trọng số loss trên thành phần mực nhỏ;
  - guidance dấu khi sinh: `sample(w_mark=...)`;
  - bộ chấm `LocalOCR` vùng nhìn 29 px, có cổng `scripts/check_evaluator.py`;
  - chỉ số cặp đối chứng: `eval_holdout.py --pairs`, `pair_acc`.

  `scripts/switch_v2.sh` dừng `_lt` ở 100k (giữ làm ablation) rồi chạy `gen_vn_{char,nfd}_v2` từ snapshot 100k tới 200k.

  **Mốc 100k `_lt`, chấm lại bằng LocalOCR (`runs/eval/vn_lt_local_100000/`):**
  - C1+C2: char 0,000 / nfd 0,787 (CI chênh +0,667..+0,862).
  - Cặp đối chứng (`pair_ok`): char 0,000 / nfd 0,762 (CI +0,652..+0,842).
  - C3: nfd 0,632 (CI +0,429..+0,810).
  - Đối chứng: 0,925 / 0,947.

  Cổng LocalOCR chỉ đạt một phần: sau khi xóa dấu vẫn đọc ra chấm dưới 12/32, dấu trên 35/151 (CRNN là 19/32 và 59/151). Vì vậy số chính là `pair_ok`.
- **v2, sửa lại (19/11, 12:30 UTC):** v2 đầu tiên dừng ở khoảng 110k, giữ lại làm ablation `gen_vn_*_v2a`. Nguyên nhân là mỗi từ gốc đã được chuẩn hóa chiều cao riêng, nên khi ghép thì đường kẻ dòng và cỡ chữ lệch nhau; mẫu sinh ra từ đầu bị nhỏ hoặc lệch dòng. Thay đổi:
  - Ghép từ (`Batcher._ink`): đưa các từ không có chữ hoa về cùng chiều cao thân chữ, đặt mọi từ trên cùng một đường kẻ dòng trong khung đủ cao, rồi thu cả dòng về h (một tỉ lệ chung, bù độ đậm nét).
  - Nội dung HAT2023 (`--content-only-per-batch 16`, chỉ VN): giống nhiệm vụ HAT của VATr-pp. Mỗi batch thêm 16 ảnh HAT (98.421 ảnh, 21k từ viết hoa toàn bộ, 1.139 ảnh có Y hoa, đủ 67 chữ hoa có dấu; train VN có 0 ảnh Ễ/Ê/Ộ). HAT không có writer, nên với các ảnh này: luôn drop style, width head dùng null style, không tính SupCon/anti-content. Nền xám được chuẩn hóa (`ink_normalised`). Ảnh HAT chứa ký tự bị giữ lại ở cả dạng thường lẫn dạng hoa đều bị loại, chặt hơn protocol v1.
  - Batch được tạo ở luồng nền (`prefetch`).
- **Còn thiếu:** bộ chấm IAM; chỉ số chữ in hoa (C4, probe `upper_eval.py` cần chuyển vào `scripts/`); writer-ID và HWD; VATr++ GlyphCNN train lại dưới cùng protocol.
