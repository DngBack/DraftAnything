# Tổng hợp đợt thực nghiệm PartScribe trên tiếng Việt (18–22/11/2026)

- Nhánh: `research/partscribe`. Bản chốt hiện tại: **nfd v3 350k** ([final_v3_350k.json](final_v3_350k.json)).
- Mô tả kiến trúc và thuật toán chi tiết: [261120_explain](261120_explain_Kien_truc_va_thuat_toan_PartScribe.md). Report này không lặp lại; chỉ tóm tắt ở mục 2.
- **Mọi số liệu đều đo trên tập val** (30 writer dev) hoặc trên 5 writer PNL giữ riêng. **Tập test VN chưa mở.**

---

## 1. Tóm tắt

**Câu hỏi:** nội dung biểu diễn bằng **thành phần dùng chung** (chữ cái gốc + dấu, phân rã Unicode NFD) có cho phép sinh đúng ký tự **chưa từng có ảnh thật khi train**, theo đúng phong cách writer, mà không làm kém ký tự đã thấy hay không?

| Phép đo (val) | nfd (PartScribe) | char (baseline cùng kiến trúc) | VATr++ VN 250k |
|---|---|---|---|
| **Ký tự giữ lại, `pair_ok` C1+C2** | **0,823** | ≈ 0,01 | – (đã thấy các ký tự này khi train) |
| Ký tự đã thấy: FID / HWD / CER | **16,5 / 0,866** / 2,77% | 16,3 / 0,871 / 2,80% | 38,5 / 1,613 / **1,18%** |
| **Chữ hoa, 5 writer thật chưa thấy:** HWD / CER | **0,685 / 6,8%** | 0,697 / 6,6% (325k) | 0,985 / 12,7% |

**Kết luận chính:**

1. Biểu diễn **nfd viết được ký tự chưa có ảnh** (ẹ, ễ, ỡ, ử, ẳ…); char thì không. Chênh lệch +0,69..+0,87, vượt xa ngưỡng +10 điểm đã khai báo trước.
2. Trên ký tự đã thấy, **nfd ngang char**: khả năng tổ hợp không phải trả giá.
3. So với VATr++ tiếng Việt:
   - PartScribe **giống ảnh thật và giống writer hơn nhiều**: FID giảm khoảng 58%, HWD giảm khoảng 46%.
   - VATr++ có CER thấp hơn, nhưng **thấp hơn cả ảnh thật** (8,8%): chữ của nó "sạch quá mức".
4. Bổ sung **dữ liệu chữ hoa có writer thật (PNL)** làm chữ hoa giống writer hơn rõ rệt (HWD 0,828 → 0,685) mà không làm giảm hai kết quả trên.

---

## 2. Mô hình (tóm tắt)

| Khối | Nội dung |
|---|---|
| VAE | f = 4, một kênh, ảnh 32 px, ink map (mực = 1); L1 + Sobel + PatchGAN |
| Nội dung | `nfd`: mỗi ký tự ≤ 3 token (chữ gốc + hoa/thường, dấu biến đổi, dấu thanh); `char`: 1 embedding/ký tự. Context encoder Transformer 4 lớp |
| Bố cục | WidthHead dự đoán độ rộng từng ký tự (không có nhãn; học qua bias căn chỉnh khả vi) |
| Style | 15 ảnh support → `M` (patch memory) + `g` (style toàn cục, SupCon + chống mang nội dung) |
| Khử nhiễu | DiT 8 block, 384 chiều; `g` qua AdaLN-zero, nội dung và `M` qua hai cross-attention; bias Gaussian căn ký tự ↔ cột ảnh |
| Huấn luyện / sinh | Rectified flow, logit-normal t; Euler 50 bước; CFG tách nội dung (w = 3) và style (w = 1,5) |

**Giai đoạn 1 (part tokenizer học không nhãn) là NO-GO.** Probe P1 trên ký tự giữ lại:

| slots | global | pixel |
|---|---|---|
| 0,260 | 0,416 | 0,369 |

Vì vậy nhánh chính đổi sang `nfd`, và claim "thành phần học không nhãn" được báo cáo như kết quả null ([stage1_deviation.md](stage1_deviation.md)).

---

## 3. Dữ liệu và protocol

### 3.1. Nguồn

| Nguồn | Vai trò | Số lượng |
|---|---|---|
| VNOnDB + ViHTGen (gộp theo VATr-pp) | Ảnh có writer, train / val | 265 writer train, 30 writer val; 96k ảnh train sau lọc |
| HAT2023 | Đích **chỉ có nội dung** (không style), 16 ảnh/batch | 98.421 ảnh dùng được, gồm 21k từ viết hoa toàn bộ |
| **PNL handwriting_word_v1.0** (từ v3) | Chữ hoa có writer thật | 7.600 ảnh, 19 writer; sau lọc 6.276 |

### 3.2. Protocol giữ lại tổ hợp v1 (khóa SHA-256)

- C1 `è é ẻ ẹ ỷ`: dấu thanh đã thấy ở nguyên âm khác, đặt trên thân chữ mới.
- C2 `ễ ỡ ử`: cặp dấu chồng đã thấy ở thân khác.
- C3 `ẳ`: cặp dấu chồng chưa từng thấy.

Ảnh hoặc transcript chứa các ký tự này bị loại khỏi mọi dữ liệu train. Với HAT và PNL, loại **cả dạng hoa** (È, Ễ…), chặt hơn protocol gốc.

### 3.3. Chuẩn bị dữ liệu PNL (`scripts/prepare_pnl.py` → `data/pnl32.npz`)

- **Chuẩn hóa ảnh:** chuyển xám, resize về cao 32 px giữ tỉ lệ (BOX khi thu nhỏ). Kết quả khớp VNOnDB:

  | | Lề trên / dưới | Độ đậm nét p90 | Tỉ lệ rộng/cao |
  |---|---|---|---|
  | PNL sau resize | 0,87 / 0,92 px | 0,962 | 1,5 |
  | VNOnDB | 0,87 / 0,90 px | 0,986 | 1,5 |

- **Lọc:**
  - 1.155 ảnh bị loại vì chứa ký tự giữ lại (ở bất kỳ dạng hoa/thường nào);
  - 169 ảnh bị loại vì có cờ `overwritten` / `text_uncertain`.
- **Chia theo writer** (seed 742):
  - **14 writer train (4.623 ảnh)**, phủ **đủ 80/80 chữ hoa** mà protocol cho phép;
  - **5 writer test (NV08, NV09, NV14, NV16, NV19; 1.653 ảnh)**, không bao giờ train. Đây là tập đo chữ hoa C4.
- **Trong batch:** PNL chiếm 15% số slot writer tự nhiên; đo thực tế được 14,7% đích thật.

---

## 4. Diễn biến thực nghiệm

| Ngày | Run | Thay đổi chính | Lý do / kết quả |
|---|---|---|---|
| 18/11 | `gen_vn_{char,nfd}` | Tần suất tự nhiên | Ablation sampler; dừng ở 25k (C1+C2: 0,018 so với 0,530) |
| 18/11 | `_lt` 0 → 100k | Long-tail sampler của VATr-pp (`rare_fraction` 0,3) | Yêu cầu: xử lý đuôi dài ít nhất như VATr++. 75k: 0,018 so với 0,848 |
| 19/11 | Chấm lại | Phát hiện OCR CRNN tự điền dấu từ ngữ cảnh (xóa dấu nặng vẫn đọc ra `ẹ` 19/32 lần) | Thêm LocalOCR (vùng nhìn 29 px) và chỉ số cặp đối chứng `pair_ok` |
| 19/11 | `v2a` 100k → 110k | Multiword 0,2; mark weight 3 | Ghép từ lệch dòng (mỗi từ chuẩn hóa chiều cao riêng) → dừng, giữ làm ablation |
| 19/11 | **`v2` 100k → 200k** | Ghép từ trên cùng đường kẻ dòng; HAT2023 làm đích chỉ-nội-dung; prefetch batch | Sửa chữ hoa (train VN có 0 ảnh Ễ/Ê/Ộ); tốc độ 3.380 → khoảng 4.150 bước/giờ |
| 20/11 | IAM 0 → 3k | Track IAM tự khởi động | **Dừng theo yêu cầu**: tập trung tiếng Việt trước |
| 20/11 | **`v2` 200k → 300k** | Nối tiếp, `--seed 1`, LR không đổi | Loss còn giảm khoảng 0,011/20k bước |
| 21/11 | Sửa giải mã | Xóa vạch mép phải (mục 6.1) | Không train lại; đo lại toàn bộ |
| 21/11 | **`v3` 300k → 400k** | Thêm PNL (15% slot) từ v2 300k, `--seed 2` | Chữ hoa có writer thật |
| 21/11 | Chốt | **nfd v3 350k** | Mục 5.4 |

**Loss flow của nfd:**

| Run | Các mốc |
|---|---|
| v2 | 0,718 (100k) → 0,669 (150k) → 0,643 (200k) → 0,601 (300k) |
| v3 | tăng lên 0,645 khi thêm dữ liệu mới, rồi giảm về 0,589 (375k) |

---

## 5. Kết quả

### 5.1. Ký tự giữ lại (`pair_ok`, LocalOCR, 30 writer val)

`pair_ok` = đọc đúng ký tự trên bản sinh có dấu **và** không đọc ra ký tự đó trên bản cùng từ sinh không dấu. Chỉ số này loại phần OCR tự đoán dấu.

| Mốc | C1+C2 nfd [CI 95%] | C1 | C2 | C3 | char C1+C2 | Đối chứng char / nfd |
|---|---|---|---|---|---|---|
| `_lt` 100k | 0,762 [chênh +0,65..+0,84] | | | 0,632 | 0,000 | 0,925 / 0,947 |
| v2 125k | 0,793 | 0,816 | 0,758 | 0,737 | 0,000 | 0,956 / 0,942 |
| v2 150k | 0,726 | 0,765 | 0,667 | 0,632 | 0,000 | 0,945 / 0,933 |
| v2 175k | 0,774 | 0,765 | 0,788 | 0,895 | 0,006 | 0,950 / 0,942 |
| v2 200k | 0,805 | 0,765 | 0,864 | 0,684 | 0,000 | 0,959 / 0,959 |
| v2 225k | 0,805 | 0,806 | 0,803 | 0,895 | 0,018 | 0,944 / 0,956 |
| v2 250k | 0,780 | 0,765 | 0,803 | 0,632 | 0,012 | 0,953 / 0,953 |
| v2 275k | 0,829 | 0,816 | 0,848 | 0,895 | 0,000 | 0,948 / 0,966 |
| v2 300k | 0,793 | 0,816 | 0,758 | 0,895 | 0,018 | 0,956 / 0,956 |
| **v2 300k, đã sửa mép** | 0,829 [chênh +0,75..+0,87] | 0,867 | 0,773 | 0,895 | 0,012 | 0,962 / 0,967 |
| v3 325k | 0,829 | 0,827 | 0,833 | 0,895 | 0,006 | 0,981 / 0,962 |
| **v3 350k (chốt)** | **0,823 [0,70..0,88]** | 0,827 | 0,818 | 0,842 | 0,006 | 0,972 / 0,973 |
| v3 375k | 0,817 | 0,816 | 0,818 | 0,789 | 0,006 | 0,964 / 0,958 |
| v3 400k | 0,774 | 0,827 | 0,697 | 0,789 | 0,012 | 0,975 / 0,967 |

- Từ v2 200k trở đi, C1+C2 dao động trong khoảng 0,77–0,83, nằm trong khoảng tin cậy. C3 chỉ có 11 writer nên dao động mạnh.
- v3 sau 350k giảm dần (C2 0,83 → 0,70 ở 400k). Đây là lý do không chốt 400k.

### 5.2. Metric chuẩn trên ký tự đã thấy (`scripts/eval_standard.py`)

**Thiết lập:**
- 2.847 từ, 30 writer val. Mọi model dùng **cùng 15 ảnh support** cho mỗi từ.
- Quy ước theo repo HWD (BMVC 2023): FID/KID/BFID trên ô vuông đầu từ 32×32; HWD dùng VGG16 font_square ở cao 32.
- CER đo bằng evaluator OCR độc lập.
- Số có "±" là trung bình ± độ lệch chuẩn của 3 seed.

| Model | FID ↓ | BFID ↓ | KID×10³ ↓ | HWD ↓ | CER ↓ | Đúng cả từ ↑ |
|---|---|---|---|---|---|---|
| VATr++ VN 175,5k (khuyến nghị) | 38,13 | 25,27 | 20,07 | 1,713 | 1,74% | 94,6% |
| VATr++ VN 250k | 38,46 | 20,77 | 21,36 | 1,613 | **1,18%** | **96,4%** |
| char v2 300k | 16,25 | 11,62 | 5,35 | 0,871 | 2,80% | 92,8% |
| nfd v2 300k | **16,13 ± 0,10** | 11,31 ± 0,22 | **5,31 ± 0,01** | 0,876 ± 0,006 | 2,74% ± 0,05 | 92,7% |
| **nfd v3 350k (chốt)** | 16,47 | 11,47 | 5,45 | 0,866 | 2,77% | 92,3% |
| nfd v3 400k | 16,68 ± 0,09 | 11,37 ± 0,20 | 5,67 ± 0,07 | **0,861 ± 0,009** | 3,16% ± 0,18 | 91,5% |
| char v3 400k | 17,10 | 11,69 | 6,01 | 0,853 | 3,38% | 91,1% |
| *Ảnh thật* | *11,9 (nửa so với nửa)* | | | | *8,80%* | *79,0%* |

**So sánh cặp** (CI 95% bootstrap theo writer, B − A):

| A → B | HWD | CER |
|---|---|---|
| VATr++ 250k → nfd v2 300k | **−0,97..−0,49** | +1,1..+2,2 điểm % |
| char v2 300k → nfd v2 300k | chứa 0 | chứa 0 |
| nfd v2 300k → nfd v3 350k | −0,032..+0,005 | −0,35..+0,30 điểm % (ngang) |
| nfd v3 350k → nfd v3 400k | chứa 0 | **+0,13..+0,77 điểm %** (400k kém hơn) |

Theo số bước của v2 (trước khi sửa mép):

| Mốc | FID | CER | HWD |
|---|---|---|---|
| 125k | 20,35 | 4,07% | 0,92 |
| 160k | 19,90 | 3,45% | 0,92 |
| 200k | 19,58 | 3,32% | 0,91 |
| 300k | 19,60 | 3,13% | 0,91 |

Sau 200k, mô hình gần hội tụ: chỉ CER còn giảm.

### 5.3. Chữ hoa trên writer thật chưa thấy (C4, 5 writer PNL, 1.653 từ)

Support lấy từ chính chữ hoa của writer đó.

| Model | FID ↓ | KID×10³ ↓ | HWD ↓ | CER ↓ | Đúng cả từ ↑ |
|---|---|---|---|---|---|
| VATr++ VN 250k | 35,82 | 15,52 | 0,985 | 12,68% | 64,1% |
| char v2 300k | 25,98 | 12,55 | 0,828 | 8,14% | 80,0% |
| nfd v2 300k | 27,60 | 13,37 | 0,828 | 6,73% | 84,3% |
| char v3 325k | 24,69 | 12,86 | 0,697 | 6,57% | 85,3% |
| nfd v3 325k | 26,05 | 14,42 | 0,713 | **6,26%** | **86,0%** |
| **nfd v3 350k (chốt)** | 23,97 | 12,28 | 0,685 | 6,83% | 84,9% |
| nfd v3 400k | 21,79 | 9,76 | 0,619 | 6,46% | 85,2% |
| char v3 400k | **20,92** | **8,69** | **0,590** | 6,92% | 84,4% |
| *Ảnh thật* | *15,7* | | | *18,1%* | *60,3%* |

**So sánh cặp:**

| A → B | HWD | CER |
|---|---|---|
| nfd v2 300k → **nfd v3 350k** | **−0,21..−0,07** | chứa 0 |
| char v2 300k → char v3 325k | −0,19..−0,06 | −2,4..−0,7 điểm % |
| VATr++ 250k → nfd v3 325k | −0,42..−0,15 | **−7,1..−5,9 điểm %** |

**Lưu ý khi đọc:**
- Chữ hoa tiếp tục tốt lên khi train lâu hơn (400k tốt nhất), ngược với ký tự giữ lại.
- CER của ảnh thật cao hơn mọi model vì evaluator OCR chủ yếu học chữ thường. Chỉ nên so CER giữa các model với nhau.
- Chỉ có 5 writer, nên CI theo writer còn thô.

### 5.4. Chọn checkpoint chốt

| Tiêu chí | v2 300k | v3 350k | v3 400k |
|---|---|---|---|
| Ký tự giữ lại | 0,829 | 0,823 (ngang) | 0,774 (giảm) |
| Tiếng Việt chung (CER) | 2,74% | 2,77% (ngang) | 3,16% (kém hơn, có ý nghĩa) |
| Chữ hoa (HWD) | 0,828 | 0,685 (tốt hơn, có ý nghĩa) | 0,619 (tốt nhất) |

**v3 350k** là điểm duy nhất không kém v2 ở tiêu chí nào và tốt hơn hẳn ở chữ hoa. Checkpoint được khóa chỉ đọc:

| Checkpoint | SHA-256 |
|---|---|
| `runs/gen_vn_nfd_v3/snap_350000.pt` | `f3830dc2…` |
| `runs/gen_vn_char_v3/snap_350000.pt` | `819fd211…` |

### 5.5. Cấu hình sinh (quét 8 cấu hình trên 450 từ val)

| Cấu hình | HWD | CER |
|---|---|---|
| Mặc định (content 3, style 1,5, 50 bước) | 1,173 | 3,37% |
| content 4,5 | 1,220 | **2,38%** |
| content 2 | **1,161** | 4,82% |
| style 2,5 | 1,164 | 4,36% |
| style 1,0 | 1,204 | 3,77% |
| mark 1 / mark 2 | 1,175 / 1,166 | 4,03% / 4,36% |
| 100 bước | 1,173 | 3,30% |

- Các cấu hình chỉ đổi qua lại giữa độ đọc đúng và độ giống writer.
- Guidance dấu (`w_mark`) **không sửa được** lỗi chấm dưới.
- 100 bước không hơn 50 bước.
- **Số báo cáo dùng cấu hình mặc định** (không chỉnh theo tập đánh giá).
- Hình minh họa dùng content 4,5 và chọn mẫu tốt nhất trong N lần sinh (`scripts/showcase.py`): OCR edit distance trước, rồi HWD tới support.

---

## 6. Vấn đề phát hiện và cách xử lý

### 6.1. Vạch xám ở mép phải ảnh sinh (đã sửa)

- **Triệu chứng:** dải xám đều, cao hết ảnh, rộng 2–3 px ở mép phải. Độ đậm trung vị theo cột (0–255) ở các cột cuối là 22,6 / 11,2 / 4,8, trong khi nền khoảng 1,6.
- **Nguyên nhân:** generator đặt latent ngoài độ rộng chữ bằng 0, nhưng 0 không phải latent của giấy trắng (latent nền có |mean| ≈ 0,4). Decoder vẽ bước nhảy đó thành một cạnh. Ảnh thật và latent từ encoder không có hiện tượng này; chỉ 0,2% ảnh thật có vạch ở mép.
- **Không cắt bỏ cột mép,** vì nét chữ chạm mép ở khoảng một nửa số từ (69/128 có nét trong 3 cột cuối).
- **Cách sửa** (`partscribe.vae.decode_generated`): ở 4 cột cuối, trừ trung vị của từng cột. Vạch là dải đều nên bị xóa; nét chữ thưa và đậm hơn nên được giữ.
  - Trung vị ở mép sau khi sửa: 0,1–0,8.
  - 97,2% điểm mực của nét chữ ở mép được giữ lại.
- **Tác động lên cùng checkpoint nfd 300k:**

  | Metric | Trước | Sau |
  |---|---|---|
  | FID | 19,60 | 16,13 |
  | HWD | 0,906 | 0,876 (CI chênh −0,046..−0,015) |
  | CER | 3,13% | 2,74% |
  | `pair_ok` | 0,793 | 0,829 |
  | BFID | không đổi | không đổi (nhị phân hóa vốn đã loại vạch mờ) |

### 6.2. OCR tự điền dấu

CRNN BiLSTM đọc ra dấu từ ngữ cảnh:

| | Chấm dưới (n = 32) | Dấu ở trên (n = 151) |
|---|---|---|
| CRNN, sau khi xóa vùng dấu | 19 | 59 |
| LocalOCR, sau khi xóa vùng dấu | 12 | 35 |

LocalOCR chỉ đạt một phần cổng kiểm tra, nên chỉ số chính là `pair_ok`, không phải độ đúng OCR thô.

### 6.3. Sự cố vận hành

| Sự cố | Hậu quả | Xử lý |
|---|---|---|
| Vòng đánh giá thoát trước khi snapshot mốc cuối (`last.pt` chưa ổn định 60 s) | Thiếu đánh giá 200k của nfd | Chạy tay; `eval_loop.py` giờ chờ rồi snapshot lần cuối |
| Phiên tmux chết theo phiên Claude Code (21/11, 08:56) | v3 mất khoảng 1.400 bước | Resume từ `last.pt` 340k; tmux khởi động bằng `setsid`, job dài chạy bằng `setsid nohup` |
| Dọn checkpoint trung gian v2 | **Mất checkpoint v2 175k và 200k** (không khôi phục được) | Kết quả đánh giá và ảnh sinh của hai mốc vẫn còn; quy tắc mới: giữ các mốc quan trọng và duyệt danh sách trước khi xóa |
| Tốc độ train giảm còn 3.380 bước/giờ (ghép support lớn, batch đồng bộ) | Train chậm | Collate tại chỗ + luồng prefetch, lên khoảng 4.150 bước/giờ (khoảng 12k/giờ khi GPU rảnh) |

---

## 7. Giới hạn còn lại

1. **Chấm dưới của `ẹ`** (ký tự giữ lại) vẫn phần lớn bị mất. Với ký tự đã thấy, char vẽ chấm dưới chắc hơn nfd; v3 cải thiện chấm dưới trên chữ hoa (Ạ, Ộ). Không sửa được bằng tham số sinh.
2. **Tên riêng viết thường có chữ hoa đầu từ** ("Ngô Quyền", "Trần Hưng Đạo") ra gần như cùng một kiểu chữ đậm cho mọi writer. Nhiều khả năng do đích HAT train không kèm style. Style của chữ hoa toàn bộ cũng còn ít đa dạng hơn chữ hoa thật.
3. **IAM chưa có kết quả** (track đã tạm dừng). Chưa có bảng so sánh chuẩn với các paper khác trên IAM.
4. **Chưa có VATr++ train lại dưới cùng protocol.** VATr++ hiện có đã thấy các ký tự giữ lại, nên chưa so được trên `pair_ok`.
5. Tập C4 chỉ có 5 writer; C3 chỉ có 11 writer.
6. **Tập test VN chưa mở.** Checkpoint được chọn trên val.

---

## 8. Việc tiếp theo (cho bài nộp)

| Ưu tiên | Việc | Ghi chú |
|---|---|---|
| 1 | Mở tập test VN một lần với nfd/char v3 350k | Số chính thức |
| 2 | Train lại VATr++ VN theo protocol giữ lại, đo `pair_ok` | Khoảng 1 ngày GPU; reviewer sẽ hỏi |
| 3 | Sửa chấm dưới / style chữ hoa trong một run mới | Ví dụ: trọng số loss riêng cho token chấm dưới; HAT mượn style writer thật thay vì không style |
| 4 | IAM: train và bảng A (FID/KID/HWD/CER) | Theo yêu cầu "phải chạy được trên IAM" |

Hội nghị có thông báo kết quả trước 31/03/2027 và rank ≥ B:
- **IEEE ICME 2027**: nộp 30/12/2026, kết quả khoảng 24/03/2027.
- **IEEE IJCNN 2027**: nộp 31/01/2027, kết quả 15/03/2027.
- ICDAR 2027 (rank A, đúng chủ đề) thông báo kết quả 15/05/2027.

---

## 9. Checkpoint và artefact

| Đường dẫn (`runs/`) | Vai trò |
|---|---|
| **`gen_vn_{nfd,char}_v3/snap_350000.pt`** | **Bản chốt** (chỉ đọc) |
| `gen_vn_{nfd,char}_v3/snap_400000.pt` | Mốc cuối v3 |
| `gen_vn_{nfd,char}_v2/snap_300000.pt` | Bản chốt trước (chỉ đọc), [final_v2_300k.json](final_v2_300k.json) |
| `gen_vn_{nfd,char}_lt/snap_100000.pt` | Điểm khởi tạo v2 |
| `gen_vn_{nfd,char}/last.pt`, `gen_vn_*_v2a/last.pt` | Ablation (sampler, ghép lệch dòng) |
| `vae_vn32/final.pt`, `eval_ocr_vn/best.pt`, `eval_ocr_vn_local/best.pt` | VAE, OCR chấm CER, LocalOCR chấm `pair_ok` |
| `eval/vn_*` | `pair_ok` từng mốc (ảnh panel + bản ghi) |
| `std_vn/`, `std_pnl/`, `std_vn_sweep/` | Ảnh sinh, đặc trưng cache, `scores.json` của metric chuẩn / chữ hoa / quét cấu hình |
| `visual/` | Ảnh mẫu: `samples_nfd_v3_350k_final.png`, `showcase_v3_350k.png`, `best_vs_vatrpp_random.png`, `pnl32_samples.png` |

**Tái lập:**

```bash
bash scripts/pipeline.sh gen_nfd_v2 && bash scripts/pipeline.sh gen_nfd_v2_300k      # v2
python scripts/prepare_pnl.py --release <handwriting_word_v1.0> --out data/pnl32.npz
bash scripts/pipeline.sh gen_nfd_v3                                                   # v3 (tới 400k; chốt 350k)
python scripts/eval_holdout.py run --ckpt runs/gen_vn_nfd_v3/snap_350000.pt --vae runs/vae_vn32/final.pt \
    --ocr runs/eval_ocr_vn_local/best.pt --raw --pairs --out <out>.json
python scripts/eval_standard.py {jobs,partscribe,vatr,score,paired} [--pool pnl_test] --out <dir> --hwd-repo <HWD>
```
