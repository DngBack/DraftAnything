# PartScribe: mô tả kiến trúc và thuật toán (theo code hiện tại)

- Ngày: 20/11/2026. Nhánh `research/partscribe`.
- Tài liệu này mô tả **những gì code đang chạy**, không phải đề xuất. Chỗ nào code khác với [plan_03](261118_plan_03_Kien_truc_moi_bieu_dien_thanh_phan_va_tron_style.md) đều được ghi rõ.
- Số liệu kích thước lấy từ giá trị mặc định trong `partscribe/generator.py::Config` và `scripts/train_generator.py`.

---

## 0. Tóm tắt một đoạn

PartScribe là mô hình **sinh ảnh chữ viết tay theo phong cách một người viết** (HTG offline, few-shot). Đầu vào gồm một chuỗi văn bản và 15 ảnh mẫu (support) của người viết; đầu ra là ảnh từ/cụm từ viết bằng nét chữ đó. Mô hình là **latent diffusion dạng rectified flow**: một VAE nén ảnh, một **DiT** khử nhiễu trong latent, điều kiện bằng ba nguồn đi theo ba kênh khác nhau:

| Nguồn | Ý nghĩa | Kênh đi vào DiT |
|---|---|---|
| Token nội dung + bố cục | vẽ **cái gì**, **ở đâu** | cross-attention #1, có bias Gaussian quanh tâm từng ký tự |
| `g` (style toàn cục) | danh tính writer: độ nghiêng, độ đậm, cỡ | AdaLN-zero (cộng với timestep) |
| `M` (bộ nhớ patch style) | writer vẽ **từng nét như thế nào** | cross-attention #2 |

Câu hỏi nghiên cứu: nếu nội dung được biểu diễn bằng **thành phần dùng chung giữa các ký tự** (thân chữ, dấu mũ, dấu thanh...) thay vì một embedding nguyên khối cho mỗi ký tự, thì mô hình có viết đúng những ký tự **chưa từng có ảnh thật** khi train không (ví dụ `ễ`, `ẹ`, chữ hoa có dấu)?

---

## 1. Sơ đồ tổng thể

```
                         ┌──────────────────────── train một lần, rồi đóng băng ───────────────────────┐
 text ──► ContentEncoder ─┬─► tokens (B, Lc·S, 384) ───────────────────────────┐                        │
          (char | nfd |   └─► char_feats ─► WidthHead(+g) ─► độ rộng từng ký tự  │                        │
           parts)                              └─► layout (tâm, rộng) ─────────┤                        │
                                                                               ▼                        │
 support (15 ảnh) ─► StyleEncoder ─► M (patch memory) ─────────────────► ┌──────────┐                   │
                                 └─► g (4 query → trung bình) ─────────► │   DiT    │ ─► v (vận tốc) ─► Euler ─► z0 ─► VAE.decode ─► ảnh
 noise z1 ~ N(0, I) ───────────────────────────────────────────────────► │ 8 block  │
 t ∈ [0,1] ────────────────────────────────────────────────────────────► └──────────┘
```

Thứ tự huấn luyện (`scripts/pipeline.sh`):

1. **Giai đoạn 1** – Part tokenizer trên glyph font (chỉ cần cho mode `parts`). Kết quả: **NO-GO** (mục 3.3).
2. **Giai đoạn 2** – VAE, riêng cho từng bộ dữ liệu: `vae_vn32` (32 px), `vae_iam64` (64 px).
3. **Giai đoạn 3–4** – Generator (ContentEncoder + StyleEncoder + WidthHead + DiT) train chung trên latent của VAE đã đóng băng.
4. Evaluator OCR độc lập, train riêng, chỉ dùng để chấm.

---

## 2. [E] VAE – `partscribe/vae.py`

**Mục đích:** nén ảnh 1 kênh thành latent nhỏ để DiT làm việc nhanh, nhưng giữ được nét mảnh và dấu.

- Ảnh được biểu diễn dưới dạng **ink map** trong [0, 1]: mực = 1, nền và padding = 0. Nhờ vậy padding bằng 0 tự nhiên là "giấy trắng", và ảnh có độ rộng thay đổi.
- Encoder: Conv → 2 lần stride 2 → `2·z_ch` kênh (μ, logσ²). **f = 4, z_ch = 4.**
  - VN 32 px → latent 4 × 8 × W/4.
  - IAM 64 px → latent 4 × 16 × W/4.
- Decoder: đối xứng, upsample nearest + ResBlock, sigmoid ở đầu ra.
- Loss (`recon_losses`, `scripts/train_vae.py`):
  - `L1` có mask theo cột hợp lệ;
  - `+ 1.0 × L1(Sobel(rec), Sobel(x))` – ép biên nét sắc;
  - `+ 1e-6 × KL` – gần như AE, KL chỉ để latent không trôi;
  - `+ 0.05 × hinge PatchGAN`, bật sau 20k bước.
- Sau khi train, đo độ lệch chuẩn latent và lưu vào `latent_scale`; `to_latent` nhân với hệ số này để latent có phương sai ≈ 1 như nhiễu Gauss.

> Khác plan_03: plan có perceptual loss và OCR loss trên ảnh tái tạo; code chỉ có L1 + Sobel + PatchGAN.

---

## 3. [A] Biểu diễn nội dung – `partscribe/content.py`, `partscribe/tokenizer.py`

Đây là phần "khác biệt" của PartScribe. `ContentEncoder` có **ba mode**, mọi thứ khác trong generator giữ nguyên để so sánh công bằng.

### 3.1. Mode `char` – baseline nguyên khối (kiểu VATr)

- Mỗi ký tự có **một** embedding riêng (`nn.Embedding(n_chars, 384)`), S = 1 token/ký tự.
- Ký tự bị giữ lại (không có ảnh train) **giữ nguyên embedding khởi tạo ngẫu nhiên**. Đây là hành vi "trung thực" của mô hình nguyên tử: nó không có cách nào biết `ễ` trông ra sao.

### 3.2. Mode `nfd` – phân rã Unicode (**nhánh chính hiện tại**)

Mỗi ký tự được tách bằng `unicodedata.normalize('NFD')` (`charset.parts`) thành tối đa **3 token**:

| Slot | Nội dung | Embedding |
|---|---|---|
| 0 | chữ cái gốc **viết thường** + hoa/thường | `letter_emb[e] + case_emb[upper] + role_emb[letter]` |
| 1 | dấu biến đổi (mũ `^`, trăng `˘`, móc `ʼ`) hoặc dấu thanh | `mark_emb[...] + role_emb[modifier/tone]` |
| 2 | dấu thanh (huyền, sắc, hỏi, ngã, nặng) | như trên |

Ví dụ: `Ễ` = [`e` + upper] + [circumflex] + [tilde]. Cả ba embedding đều đã được học từ những ký tự khác có trong train (`ê`, `ẽ`, `ỗ`, `E`...). Slot không dùng bị mask.

- `đ/Đ` không có dấu tổ hợp trong Unicode nên là một chữ cái riêng.
- Dấu cách là một token học được riêng (`self.space`).

### 3.3. Mode `parts` – thành phần học không nhãn (**đã dừng sau NO-GO**)

Ý tưởng ban đầu của plan_03: học thành phần từ ảnh glyph font mà không dùng nhãn Unicode, để áp dụng được cho mọi hệ chữ.

**Part tokenizer** (`tokenizer.py::PartTokenizer`), train trên glyph render từ ~font Google (`data/fonts/`), 96×96:

1. **Encoder CoordConv**: ảnh + 2 kênh tọa độ → feature 24×24×128.
2. **Slot Attention** (Locatello 2020): 6 slot, 3 vòng lặp, slot khởi tạo học được (tất định). Softmax chạy **trên chiều slot**, nên các slot cạnh tranh nhau để "sở hữu" pixel.
3. Mỗi slot cho ra:
   - `what`: chiếu xuống 3 chiều rồi **FSQ** levels (8, 6, 5) → **240 mã** rời rạc dùng chung giữa mọi ký tự;
   - `where`: tính giải tích từ attention: tâm (cx, cy), độ trải (sx, sy), log khối lượng.
4. **Decoder spatial broadcast**: mỗi slot vẽ riêng một (ink logit, alpha logit); alpha softmax giữa các slot; ảnh = hỗn hợp. Style đưa vào decoder lấy từ **một ký tự khác của cùng font**, nên danh tính ký tự chỉ có thể đi qua slot.
5. Loss: chỉ BCE tái tạo có trọng số mực (`recon_loss`, ink_weight = 4).

> Khác plan_03: plan có thêm loss tương phản giữa font và hard negative (È/Ê); code chỉ có loss tái tạo.

`build_content_table` chạy tokenizer đóng băng trên mọi ký tự × 32 font tham chiếu, lưu (code, where, valid). Slot "hợp lệ" khi sở hữu ≥ 5% mực. Khi train generator, mỗi từ chọn ngẫu nhiên một font tham chiếu (augment); khi sample dùng font 0. Token = `code_emb[code] + MLP(where)`.

**Kết quả probe (đã khóa tiêu chí trước, `docs/stage1_probe_criteria.json`):**

| Probe | slots | GlobalAE | pixel thô | Ngưỡng |
|---|---|---|---|---|
| P1 – đọc (thân, tập dấu) của ký tự giữ lại | 0,260 | 0,416 | 0,369 | slots ≥ global + 0,10 |
| P2 – slot sở hữu dấu gọi đúng tên dấu | 0,290 | – | đoán dấu phổ biến nhất: 0,308 | ≥ 0,60 |

Slot attention **không** tách thân chữ và dấu thành slot riêng. Theo [stage1_deviation.md](stage1_deviation.md), nhánh chính đổi sang `nfd`, claim "học không nhãn" bị bỏ và được báo cáo như kết quả null. Run `gen_vn_parts` dừng ở ~1,2k bước.

### 3.4. Context encoder (chung cho cả ba mode)

- Token của tất cả ký tự trong từ được xếp thành chuỗi dài `Lc × S`, cộng **positional encoding sinusoid theo chỉ số ký tự** (các token của cùng một ký tự có cùng vị trí).
- **Transformer 2 chiều, 4 lớp, 384 chiều, 6 head**, có key mask. Nhờ vậy hình dạng một ký tự có thể phụ thuộc ký tự lân cận (nét nối, chữ hoa đầu từ).
- Đầu ra:
  - `tokens` – đi vào cross-attention nội dung của DiT;
  - `token_char` – mỗi token thuộc ký tự nào (để tính bias bố cục);
  - `char_feats` – trung bình các token hợp lệ của một ký tự → đi vào WidthHead.

---

## 4. Width head và bố cục ngang – `content.py::WidthHead`, `char_layout`

DiT không phải tự học alignment chữ ↔ cột ảnh; bố cục được đặt **tường minh**, giống duration predictor trong TTS.

- `w_i = exp(MLP([char_feats_i, g]))` – độ rộng ký tự i tính theo **cột latent**, phụ thuộc writer. Khởi tạo ≈ 4 cột latent (≈ 16 px).
- **Không có nhãn độ rộng từng ký tự.** Chỉ biết tổng độ rộng thật của ảnh (`lat_width`). Do đó:
  - `width_loss` = (log Σw − log tổng_thật)², trọng số 0,1;
  - khi train, độ rộng được **co tỉ lệ cho khớp tổng thật** (`fit_widths`, teacher forcing) trước khi tính bố cục;
  - **tỉ lệ** giữa các ký tự (i hẹp, m rộng) được học gián tiếp: bias Gaussian ở mục 6 khả vi theo độ rộng, nên gradient của loss khử nhiễu chảy ngược vào WidthHead. Test `test_layout_gradient_reaches_widths` kiểm tra điều này.
- `char_layout`: tâm_i = Σ_{j≤i} w_j − w_i/2, rộng_i = w_i.
- Khi sample: tổng độ rộng dự đoán quyết định độ rộng ảnh sinh ra (làm tròn lên bội số của patch).

---

## 5. [C] Style hai tầng – `partscribe/style.py`

Đầu vào: 15 ảnh support của **cùng writer** (VATr-pp: 15 ảnh rộng 192 px ở 32 px cao; ở chiều cao khác giữ tỉ lệ 6 × h). Ảnh rộng hơn bị nén ngang, không cắt.

- **CNN** (stride tổng = 8, dim 384) chạy độc lập trên từng ảnh, cộng PE 2D.
- **`M` – patch memory:** toàn bộ feature của 15 ảnh trải phẳng, có mask theo cột hợp lệ. Với VN 32 px: 15 × 4 × 24 = 1.440 token. `M` cố ý **giữ cả nội dung cục bộ** – nó là nơi DiT tra "writer này viết dấu mũ thế nào".
- **`g` – style toàn cục:** 4 query học được cross-attend vào `M`, rồi lấy trung bình → một vector 384 chiều.
- Hai tín hiệu huấn luyện riêng cho `g`:
  1. **SupCon theo writer** (Khosla 2020, τ = 0,1) trên một head chiếu 128 chiều; batch được dựng 16 writer × 4 ảnh để luôn có cặp dương. Trọng số 0,05.
  2. **Chống mang nội dung (anti-copy):** một head dự đoán bag-of-characters của các từ support từ `g`, nối qua **gradient reversal** (λ = 1). Head cố đoán đúng, encoder bị đẩy theo hướng ngược lại để `g` không chứa thông tin chữ. Trọng số 0,05.
- Chống sao chép ở mức dữ liệu: support **không bao giờ** chứa ảnh đích hoặc ảnh có cùng transcript (`Batcher.support_for`).

---

## 6. [D] DiT khử nhiễu – `partscribe/dit.py`

### 6.1. Token hóa latent

- Latent (4, h, w) được cắt patch 2×2 → token 384 chiều, cộng PE 2D sinusoid (nửa đầu mã hàng, nửa sau mã cột).
  - VN: 4 hàng token; IAM: 8 hàng token. Mỗi cột token = 2 cột latent = 8 px.
- Token ở cột vượt quá độ rộng thật bị mask trong self-attention.
- `cond = MLP(sinusoid(1000·t)) + MLP(g)`.

### 6.2. Một block (8 block, dim 384, 6 head)

```
x = x + g1 · SelfAttn( AdaLN(x; s1, c1) )                       # AdaLN-zero theo (t + g)
x = x + gate_c ⊙ CrossAttn( LN(x), content_tokens, bias=B )     # "vẽ cái gì, ở đâu"
x = x + gate_s ⊙ CrossAttn( LN(x), M )                          # "vẽ thế nào"
x = x + g2 · MLP( AdaLN(x; s2, c2) )
```

- **Mọi cổng khởi tạo bằng 0**: tham số AdaLN (`ada`), `gate_c`, `gate_s`, lớp ra cuối. Lúc đầu DiT là ánh xạ đồng nhất và các nhánh điều kiện "mở dần", giúp train ổn định.
- **Bias căn chỉnh B** (điểm mấu chốt để nội dung đặt đúng chỗ): với query ở cột x và key thuộc ký tự k,

  ```
  B(x, k) = −½ · ((x − tâm_k) / σ_k)²,   σ_k = (rộng_k / 2) · softplus(τ_block) + 0,5
  ```

  Mỗi cột ảnh chủ yếu nhìn các token của ký tự nằm dưới nó, nhưng mềm (không cứng) nên vẫn thấy được hàng xóm. `τ_block` học riêng cho từng block, khởi tạo sao cho hệ số = 1.
- **Null token:** luôn có một null token ở đầu ngữ cảnh nội dung và ngữ cảnh `M` (không có bias). Mỗi query luôn có ít nhất một key hợp lệ, và khi drop điều kiện thì chỉ còn null token → cùng một bộ trọng số cho ra đầu ra không điều kiện / chỉ nội dung / đầy đủ.

### 6.3. Classifier-free dropout khi train

- Drop nội dung với p = 0,1 (mask toàn bộ token nội dung).
- Drop style với p = 0,1 (`g` → `null_g`, mask toàn bộ `M`).

---

## 7. Rectified flow – `partscribe/flow.py`

- **Nội suy thẳng:** `x_t = (1 − t)·x₀ + t·ε`, vận tốc mục tiêu `v = ε − x₀`. t = 0 là dữ liệu, t = 1 là nhiễu.
- **Lấy mẫu t:** logit-normal, `t = sigmoid(N(0, 1))` (Esser và cộng sự 2024, SD3) – dồn mẫu vào vùng t giữa, nơi khó nhất.
- **Loss:** MSE giữa `v_pred` và `v`, chỉ trên các cột latent hợp lệ (`masked_mse`). Có thể nhân bản đồ trọng số (mục 9, `mark_weight`).
- **Sinh ảnh:** tích phân Euler từ t = 1 → 0, mặc định 50 bước.
- **CFG tổ hợp, tách nội dung và style** (mỗi bước chạy DiT 3 lần trong một batch):

  ```
  v = v_∅ + w_c · (v_content − v_∅) + w_s · (v_full − v_content)
  ```

  Mặc định `w_c = 3,0`, `w_s = 1,5`. Hai núm chỉnh độc lập: độ đúng chữ và độ giống writer.
- **Guidance dấu (`w_mark`, thêm ở v2):** chạy thêm lần thứ 4 với văn bản đã **bỏ dấu** (`strip_marks`: ẹ → e, Ễ → E) nhưng giữ nguyên bố cục, rồi cộng `w_mark · (v_full(text) − v_full(text không dấu))`. Hiệu này xấp xỉ "vận tốc của riêng các dấu", dùng để khuếch đại dấu.

---

## 8. Tổng loss của generator – `generator.py::PartScribe.loss`

```
L = L_flow + 0,1 · L_width + 0,05 · L_SupCon(g) + 0,05 · L_anti(g)
```

- AdamW, lr 1e-4, warmup 5k, weight decay 0,01, clip grad 1,0, bf16 autocast.
- **EMA 0,9999 có warmup** (`decay_n = min(0,9999, (1+n)/(10+n))`). Bản không warmup làm mẫu từ EMA ra nhiễu trong vài chục nghìn bước đầu; hai run VN cũ vì vậy được chấm bằng trọng số thô (`--raw`).
- VAE đóng băng; latent được nhân `latent_scale` và zero-hóa ngoài độ rộng thật.
- **Ảnh không có writer** (HAT2023, `writer = −1`): luôn drop style, WidthHead dùng `null_g`, không tính SupCon / anti-content. Các ảnh này chỉ dạy nội dung.

---

## 9. Dữ liệu và chiến lược lấy batch – `partscribe/datasets.py::Batcher`

| | Tiếng Việt | IAM |
|---|---|---|
| Nguồn | VNOnDB + ViHTGen gộp (theo VATr-pp), HDF5 | IAM words, cắt lại từ ảnh dòng |
| Chiều cao | **32 px** (HDF5 của VATr-pp chỉ có 32 px) | **64 px** |
| Writer | 265 train / 30 dev | split VATr |
| Batch | 16 writer × 4 ảnh | 8 × 4, accum 2 |
| Số bước | 200k | 100k |

Các cơ chế lấy mẫu:

1. **Cân bằng nguồn:** chọn nguồn (VNOnDB / ViHTGen) đều nhau, rồi writer đều nhau trong nguồn.
2. **Long-tail sampler** (`rare_fraction = 0,3`, chuyển từ ComponentCorpusSampler của VATr-pp): 30% slot writer là "rare draw", ảnh đích được rút theo trọng số `√(N / tần suất)` của ký tự và của nhóm thành phần (thân / dấu biến đổi / dấu thanh / cặp dấu, theo hoa/thường), trộn 50/50.
3. **Multiword (v2, p = 0,2):** ghép 2–3 từ **cùng writer** thành một cụm có dấu cách. Lý do: dữ liệu gần như không có dấu cách (109/96k nhãn), nên token dấu cách không được học. Khi ghép: đưa các từ không có chữ hoa về cùng chiều cao thân chữ, đặt trên **một đường kẻ dòng** chung, rồi thu cả dòng về h bằng một tỉ lệ chung (có bù độ đậm nét). Chỉ ghép ảnh nền sạch.
4. **Mark weight (v2, = 3):** thành phần liên thông nhỏ (diện tích ≤ 2,5%·h², cao ≤ 30%·h – tức dấu, chấm) được nhân trọng số 1 + 3 = 4 trong flow loss, để loss không bị nét lớn áp đảo.
5. **HAT2023 làm mục tiêu chỉ-nội-dung (v2, VN):** mỗi batch thêm 16 ảnh HAT (có nhiều chữ hoa có dấu mà train VN không có). Nền xám được chuẩn hóa (`ink_normalised`).
6. **Prefetch** batch ở luồng nền.

---

## 10. Protocol giữ lại tổ hợp – `partscribe/holdout.py`, `docs/holdout_protocol_v1.json`

Đây là "đề thi" của toàn bộ nghiên cứu. File JSON được **khóa bằng SHA-256** trong code; sửa file là loader báo lỗi. Mọi run ghi SHA vào `config.json`.

| Nhóm | Ký tự | Ý nghĩa |
|---|---|---|
| C1 | è é ẻ ẹ ỷ | dấu thanh đã thấy ở nguyên âm khác, thân chữ mới |
| C2 | ễ ỡ ử | cặp dấu chồng đã thấy trên thân khác |
| C3 | ẳ | cặp dấu chồng chưa từng thấy (thăm dò, n nhỏ) |
| C4 | 42 chữ hoa có dấu | 0 ảnh thật tự nhiên, chỉ đo độ đọc đúng |
| IAM | j q J Y | phép thử tính tổng quát |

- Mọi ảnh/transcript chứa ký tự giữ lại bị loại khỏi train, prompt, HAT (HAT loại cả dạng hoa) – kiểm tra bằng `assert_clean`.
- Glyph font của các ký tự này **được phép** thấy (biết hình dạng chuẩn là điều kiện bài toán).
- Tiêu chí quyết định định trước: `nfd` phải hơn `char` ≥ 10 điểm phần trăm trên C1 ∪ C2, CI 95% (bootstrap theo writer) không chứa 0.

---

## 11. Đánh giá – `partscribe/evaluator.py`, `scripts/eval_holdout.py`

- **Evaluator OCR độc lập**, train trên dữ liệu train + HAT **đầy đủ** (có ký tự giữ lại), không bao giờ thấy val/test, không truyền gradient cho generator.
  - `CRNN` (CNN + BiLSTM + CTC): bị phát hiện tự "điền" dấu từ ngữ cảnh – xóa dấu nặng trên ảnh thật `mẹ` vẫn đọc ra `ẹ` 19/32 lần.
  - `LocalOCR`: 3 block conv cuối chỉ theo chiều cao, không recurrence, mỗi cột đầu ra chỉ nhìn ~29 px. Giảm nhưng chưa hết thiên lệch (12/32), nên có cổng kiểm tra `scripts/check_evaluator.py`.
- **Chỉ số chính P1:** tỉ lệ đọc đúng tại vị trí ký tự giữ lại, sau căn chỉnh Levenshtein (`align`).
- **`pair_ok`** (chỉ số đáng tin nhất hiện tại): sinh mỗi từ hai lần, có dấu và không dấu (`--pairs`). Đúng khi bản có dấu đọc ra đúng ký tự **và** bản không dấu không đọc ra ký tự đó. Cách này loại phần OCR tự đoán dấu.
- Chỉ số phụ: tỉ lệ nhầm "anh em cùng thân" (`sibling`, ví dụ È ↔ Ê), nhóm đối chứng (từ chỉ có ký tự đã thấy, cùng writer), CER toàn bộ, ảnh thật làm chặn trên.

### Kết quả hiện có (val, 30 writer dev; test chưa mở)

| Mốc | C1+C2 char | C1+C2 nfd | Chênh (CI 95%) | Đối chứng char / nfd |
|---|---|---|---|---|
| 25k (tần suất tự nhiên) | 0,018 | 0,530 | +0,512 (+0,373..+0,598) | 0,856 / 0,882 |
| 75k `_lt` | 0,018 | 0,848 | +0,829 (+0,710..+0,907) | 0,950 / 0,951 |
| 100k `_lt`, LocalOCR, `pair_ok` | 0,000 | 0,762 | CI +0,652..+0,842 | 0,925 / 0,947 |

Đọc nhanh: biểu diễn `nfd` viết được ký tự chưa có ảnh, `char` gần như không; trên ký tự đã thấy hai bên ngang nhau. Kết luận chính vẫn chờ run v2 200k, IAM và mở test một lần.

---

## 12. Bản đồ file

| File | Vai trò |
|---|---|
| `partscribe/generator.py` | `Config`, lớp `PartScribe`: ghép mọi khối, `loss`, `sample` (CFG, guidance dấu) |
| `partscribe/content.py` | `ContentEncoder` (char/nfd/parts), `WidthHead`, bố cục, content table |
| `partscribe/style.py` | `StyleEncoder` (M, g), SupCon, anti-content |
| `partscribe/dit.py` | DiT, block 3 nhánh, bias căn chỉnh |
| `partscribe/flow.py` | rectified flow, CFG tổ hợp, Euler |
| `partscribe/vae.py` | VAE f=4, PatchGAN, loss tái tạo |
| `partscribe/tokenizer.py` | Part tokenizer (slot attention + FSQ), GlobalAE baseline |
| `partscribe/layers.py` | Attention có bias + mask (căn stride cho SDPA), PE, gradient reversal |
| `partscribe/datasets.py` | Nạp VN/IAM có protocol, `Batcher` |
| `partscribe/holdout.py`, `charset.py` | Protocol khóa SHA, bảng chữ, phân rã NFD |
| `partscribe/evaluator.py` | CRNN, LocalOCR, căn chỉnh, `sibling` |
| `partscribe/probes.py` | Probe P1/P2 của giai đoạn 1 |
| `scripts/pipeline.sh` | Thứ tự các track huấn luyện thực tế |
| `scripts/train_*.py`, `eval_*.py`, `sample.py` | Điểm vào train / chấm / sinh mẫu |

---

## 13. Những chỗ dễ hiểu nhầm

1. **Tên "PartScribe" nhưng nhánh chính không dùng thành phần học được.** Sau NO-GO giai đoạn 1, "thành phần" = phân rã Unicode NFD. Mode `parts` chỉ còn là kết quả null.
2. **Model không bao giờ thấy pixel glyph** ở mode `char`/`nfd`; chỉ thấy embedding. Ở mode `parts` nó chỉ thấy mã FSQ + where, không thấy pixel.
3. **Tiếng Việt chỉ ở 32 px**, không phải 64 px như plan; muốn 64 px phải render lại từ InkML của VNOnDB.
4. **Độ rộng từng ký tự không có nhãn**; nó được học qua gradient của bias căn chỉnh.
5. **`M` cố ý chứa nội dung**, chỉ `g` bị ép bỏ nội dung. Đây là thiết kế, không phải lỗi rò rỉ.
6. **So sánh char ↔ nfd luôn dùng cùng loại trọng số (raw hoặc EMA) ở cùng số bước.**
