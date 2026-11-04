# Từ nguyên nhân đến cách sửa: dynamics trong dòng của generator (31/10/2026)

Tiếp nối `docs/week3_progress.md` (H2: Emuru và TS-GAN thiếu drift baseline và thiếu biến thiên spacing). File này trả lời hai câu: **vì sao** chúng thiếu, và **sửa được không, rẻ đến mức nào**. Mọi số liệu đều trên val1; val2/test chưa đụng tới.

## 1. Nguyên nhân (`scripts/cause_analysis.py` → `reports/fix/cause_analysis.json`)

### 1a. Dynamics có phải thuộc tính của người viết không?
Đo tỷ lệ phương sai của chỉ số theo dòng nằm ở giữa các nhóm (ICC(1)), trên dòng thật train+val1:

| Chỉ số theo dòng | theo writer | theo form (trang) |
|---|---|---|
| độ dốc baseline | 0.31 | **0.47** |
| SD baseline | 0.23 | 0.29 |
| CV gap | 0.28 | 0.27 |

- **Độ dốc** gắn với trang mạnh hơn gắn với người viết. Một phần lớn drift là độ nghiêng của tờ giấy hoặc lúc scan, chứ không chỉ là hành vi viết. Điều này khớp với tuần 1.
- **Spacing (CV gap)** có độ ổn định theo writer tương đương theo form, nên đây là đặc tính phong cách thật.

### 1b. Đường style của generator có mang dynamics theo không?
Tính tương quan Spearman giữa chỉ số của dòng style tham chiếu và dòng đầu ra (537–589 dòng, CI bootstrap theo writer):

| | độ dốc | SD baseline | CV gap |
|---|---|---|---|
| real (dòng thật khác của cùng writer) | **+0.36** [0.14, 0.53] | +0.16 [0.02, 0.30] | +0.22 [0.03, 0.39] |
| Emuru | −0.19 [−0.33, −0.05] | +0.07 | **+0.32** [0.13, 0.45] |
| TS-GAN | −0.09 [−0.17, −0.01] | −0.06 | +0.17 [−0.00, 0.33] |

- Ở dữ liệu thật, một dòng khác của cùng người viết dự báo được độ dốc và spacing.
- Cả hai generator **không chuyển độ dốc** từ ảnh style. Emuru còn hơi ngược chiều.
- Emuru có chuyển *thứ hạng* spacing (người giãn hơn thì ra ảnh giãn hơn) nhưng nén mức biến thiên: SD 0.16 so với 0.39. TS-GAN gần như không chuyển gì.

### 1c. Vì sao (theo kiến trúc)?
- **TS-GAN.**
  - Discriminator là patch-GAN. Tính từ `model/discriminator_ap.py`, receptive field ngang là 64 px (đầu "med") và 216 px (đầu "low"), trong khi dòng rộng trung vị 984 px. Nó chỉ thấy khoảng **22% dòng** (1–2 từ).
  - Drift tuyến tính trên cả dòng vì thế *không thể bị phạt*. Loss CTC cũng không quan tâm vị trí dọc.
  - Style là một vector toàn cục; nhiễu spacing (`count_std`, `dup_std` trong config) gần như bằng 0, nên spacing tất định theo ký tự.
  - Training data **không bị deskew** (augmentation chỉ có shear/stretch), nên dữ liệu học có drift, nhưng mô hình không có động lực tái tạo nó.
- **Emuru.** Train trên font tổng hợp. Giả thuyết, chưa kiểm được vì không có dữ liệu train: dòng render từ font gần như không có drift, và mô hình sinh tiếp sau ảnh style nên giữ baseline thẳng.

**Tóm lại:** thiếu dynamics là hệ quả của (i) loss/discriminator cục bộ không quan sát được cấu trúc cấp dòng, và (ii) đường style toàn cục không mã hóa quỹ đạo dòng. Đây không phải do dữ liệu bị chuẩn hóa mất drift.

## 2. Cách sửa: re-layout hậu kỳ, không cần train (`hwline/relayout.py`, `scripts/relayout_lines.py`)

Tách dòng sinh thành từng từ (cùng bộ tách của metric). Mỗi từ giữ nguyên glyph; chỉ thay **độ lệch dọc** và **khoảng trống trước nó**. Mục tiêu tính theo core height, tương đối với baseline trung bình và gap trung bình của chính dòng đó, nên kích thước, mức giãn và phong cách của generator được giữ.

| Biến thể | Quỹ đạo mục tiêu |
|---|---|
| ident | không đổi gì, chỉ ghép lại (đối chứng artefact cắt/ghép) |
| jitter | nhiễu iid với độ lệch chuẩn bằng dữ liệu thật, không trend (đối chứng "augmentation ngây thơ") |
| **donor** | copy độ dốc, residual baseline và tỷ lệ gap từ một dòng thật train ngẫu nhiên có cùng số từ |
| **style** | độ dốc, SD residual và CV gap lấy từ dòng style tham chiếu (dòng thật của writer, vốn là input của generator) |

Độ phủ: 97–98% dòng (ident/jitter/donor); style 92% (dòng style cần ≥ 4 baseline hợp lệ).

## 3. Kết quả (537 dòng chung; `reports/fix/{profile,distances,cer}.csv`)

**Baseline** (lag-1 vượt null / trend R² vượt null / SD trong dòng):

| Nguồn | lag-1 | trend R² | SD |
|---|---|---|---|
| **real** | **0.27** | **0.15** | **0.18** |
| Emuru | 0.03 | 0.02 | 0.12 |
| Emuru + ident | −0.02 | 0.01 | 0.12 |
| Emuru + jitter | 0.03 | 0.04 | 0.19 |
| Emuru + donor | **0.27** | **0.18** | 0.20 |
| Emuru + style | 0.28 | 0.20 | 0.21 |
| TS-GAN | −0.04 | −0.01 | 0.04 |
| TS-GAN + ident | −0.03 | 0.00 | 0.05 |
| TS-GAN + jitter | 0.01 | 0.00 | 0.17 |
| TS-GAN + donor | **0.21** | **0.14** | 0.17 |
| TS-GAN + style | 0.29 | 0.22 | 0.18 |

**Khoảng cách phân phối theo dòng so với real** (Wasserstein-1, càng nhỏ càng tốt):

| Nguồn | trend R² | SD baseline | CV gap |
|---|---|---|---|
| Emuru → +donor | 0.111 → **0.050** | 0.076 → **0.019** | 0.119 → **0.013** |
| TS-GAN → +donor | 0.347 → **0.013** | 0.162 → **0.023** | 0.075 → **0.009** |
| TS-GAN + jitter | 0.141 | 0.043 | 0.038 |

**Nội dung:** CER TrOCR gần như không đổi, Δ trung bình ≤ 0.006 ở mọi biến thể. TS-GAN: tỷ lệ dòng có CER ≤ 0.2 là 99.9% → 99.9%.

Đọc kết quả:
1. **donor** đưa cả ba thống kê baseline và CV gap về mức thật trên cả hai generator, mà không làm hỏng nội dung.
2. **jitter** chỉ sửa được *độ lớn* (SD), không tạo được trend hay lag-1. Như vậy metric phân biệt được *cấu trúc* với *nhiễu*, và "thêm noise" không phải lời giải.
3. **ident** không tạo ra dynamics, nên artefact cắt/ghép không phải nguồn gốc của hiệu ứng.
4. **style** vượt quá mục tiêu: trend R² 0.20–0.22 so với 0.15. Độ dốc ước lượng từ một dòng thì nhiễu và còn mang độ nghiêng của trang. Donor tốt hơn trên mọi khoảng cách.
5. **Còn thiếu: lag-1 của gap.** Real 0.085; mọi biến thể sửa đều ≈ 0, kể cả Emuru gốc (0.07) cũng mất khi sửa. Tỷ lệ gap của donor không giữ được tương quan cục bộ sau khi khớp vào nội dung mới.

## 4. Giới hạn, cần nói thẳng trong bài

- **Vòng lặp đánh giá.** Cách sửa nhắm đúng vào các thống kê mà metric đo, và dùng cùng bộ tách từ. Vì vậy kết quả mục 3 chỉ chứng minh cách sửa *làm đúng thứ nó định làm*, chưa chứng minh ảnh *trông thật hơn*. Kiểm chứng độc lập phải là **khảo sát người** (2AFC: gốc vs đã sửa vs real), hoặc một discriminator cấp dòng train riêng.
- **Artefact thấy được:**
  - nền TS-GAN bị làm phẳng (vân giấy mất ở mọi biến thể, kể cả ident);
  - dấu câu đứng tách (ví dụ "construction ;") có thể bị tách thành từ riêng và dịch chuyển lạc chỗ.
  
  Xem `reports/fix/sheets/`.
- **Drift có thể đến từ giấy và máy scan:** form chiếm 0.47 phương sai độ dốc. Nếu reviewer coi đó là nhiễu thu nhận dữ liệu, phần "dynamics" thực chất của người viết chỉ còn spacing và phần residual cục bộ.
- Mới có IAM, 2 generator, val1.

## 5. Ý nghĩa cho bài

- Câu chuyện nguyên nhân → hệ quả → sửa đã khép kín về mặt *đo lường*: discriminator cục bộ và style toàn cục dẫn đến thiếu quỹ đạo dòng; một quỹ đạo lấy mẫu từ dữ liệu thật khôi phục được nó với chi phí gần 0.
- Hệ quả cho định vị bài: vì cách sửa rẻ như vậy đã khớp metric, **đóng góp chính phải là metric + audit + bằng chứng cảm nhận**. Cách sửa là baseline mạnh mà mô hình học sau này phải vượt, ví dụ ở lag-1 gap và tính nhất quán theo writer.
- Bước quyết định tiếp theo: thêm điều kiện "TS-GAN gốc vs TS-GAN + donor" vào khảo sát người. Nếu người chọn bản đã sửa là thật hơn, đó là kết quả chính của bài.

## Tái lập
```
PYTHONPATH=. .venv/bin/python scripts/relayout_lines.py --list data/dyn/list_tsgan_val1.csv --src tsgan
PYTHONPATH=. .venv/bin/python scripts/htr_cer.py --list data/gen/fix/list_tsgan_donor.csv --out data/dyn/fix/cer_tsgan_donor.csv
PYTHONPATH=. .venv/bin/python scripts/line_dynamics.py measure --list data/gen/fix/list_tsgan_donor.csv --source tsgan_donor --height 32 --out data/dyn/fix/words_tsgan_donor.csv.gz
PYTHONPATH=. .venv/bin/python scripts/eval_fix.py
PYTHONPATH=. .venv/bin/python scripts/cause_analysis.py
PYTHONPATH=. .venv/bin/python scripts/make_compare_sheets.py --fix --out reports/fix/sheets --n 30
```
