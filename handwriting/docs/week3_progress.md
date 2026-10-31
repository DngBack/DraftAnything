# Tuần 3 — pilot cảm nhận và audit Emuru (31/10/2026)

Định hướng đã chốt: bài theo khung *đo lường + metric dynamics được kiểm chứng bằng cảm nhận người + audit các generator SOTA + cách sửa tối giản*. Tuần này làm hai phép thử quyết định: (1) người có nhận ra dynamics trong dòng không; (2) một generator cấp dòng SOTA có tái tạo được nó không.

## 1. Pilot cảm nhận (chưa có dữ liệu người)

`scripts/make_perception_stimuli.py`: 160 dòng val1 (44 writer, tối đa 4 dòng/writer). Mỗi dòng được cắt thành từ (box XML; dấu câu gắn vào từ; pixel trong vùng box chồng nhau thuộc box có tâm gần nhất) rồi ghép lại trên nền trắng. **Mọi điều kiện, kể cả bản gốc, đều đi qua cùng pipeline ghép**, nên artefact cắt/ghép không thể làm lộ đáp án.

| Điều kiện | Thay đổi |
|---|---|
| orig | từ ở đúng baseline và gap đã đo |
| shuffle | xáo residual baseline và gap trong dòng: giữ marginal, phá phụ thuộc và drift |
| detrend | bỏ trend tuyến tính của baseline, giữ phần cục bộ và gap |
| flat | mọi từ trên baseline trung bình với gap trung bình (giống ghép từ) |
| catch | jitter dọc iid 0.6 core height (kiểm tra chú ý) |

Lỗi đã sửa: baseline của từ ngắn đôi khi đo sai. Trong shuffle/flat, lỗi đo này sẽ thành độ dịch thật và lộ đáp án. Vì vậy residual lệch khỏi trend dòng quá 0.45 core height được thay bằng giá trị trend.

Trang 2AFC (`scripts/build_perception_html.py` → `reports/perception/study.html`; artifact riêng tư slug `be7b384e-ada1-4bc9-98a7-16cc9e7e145b`):
- 66 lượt/phiên: 2 lượt thử, 30 shuffle, 14 detrend, 14 flat, 6 catch.
- Mỗi dòng xuất hiện tối đa 1 lần/người. Thứ tự và vị trí trên/dưới seed theo mã người tham gia.
- Kết quả lưu vào db của artifact (`responses/<viewer id>`).

Phân tích (`scripts/analyze_perception.py`): loại người có độ chính xác catch < 0.75; CI 95% bằng bootstrap chéo người × dòng. Kiểm tra trên dữ liệu giả lập 20 người: tìm lại đúng giá trị đặt (shuffle 0.60 → 0.63 [0.57, 0.70]; detrend 0.55 → 0.55 [0.46, 0.64]). Với ~20 người, hiệu ứng 0.60 đủ power; 0.55 thì không.

## 2. H2 trên Emuru (CVPR 2025)

- **Thiết lập.** HF `blowing-up-groundhogs/emuru` @ `c75b78d` (đã đọc code remote; cần transformers 4.49 / diffusers 0.32.2). Style: một dòng khác của cùng writer, ưu tiên khác form. Text: transcript của dòng đích. Emuru train trên font tổng hợp nên không rò rỉ writer IAM.
- **Đầu ra.** 744 dòng val1: 714 sinh được, 30 (4%) trả ảnh rỗng (`*.empty`, tính là thất bại). Tách từ ở 32 px thành công 97.5% (ảnh thật 97.7%).
- **Lọc nội dung là bắt buộc.** TrOCR-base-handwritten (`scripts/htr_cer.py`): CER trung vị ảnh thật 0.025 (lạc quan vì TrOCR được fine-tune trên IAM), Emuru 0.049; 16% dòng Emuru có CER > 0.2. Những dòng hỏng (chấm vô nghĩa, lặp từ) làm khung tách từ sai và **tạo phụ thuộc giả**: chưa lọc, Emuru cho baseline lag-1 +0.17 và gap lag-1 sau detrend +0.15; sau lọc, cả hai gần 0. Metric dynamics phải luôn đi kèm bộ lọc nội dung, và tỷ lệ bị loại phải được báo.

Kết quả trên 589 dòng có CER ≤ 0.2 ở cả hai nguồn (`reports/h2/profile_cer20.csv`; CI bootstrap theo writer):

| Đặc trưng | Nguồn | lag-1 | lag-1 sau detrend | trend R² vượt null | SD trong dòng |
|---|---|---|---|---|---|
| baseline | real | +0.28 [0.20, 0.35] | +0.04 [−0.01, 0.08] | **+0.14 [0.11, 0.18]** | 0.19 |
| baseline | Emuru | +0.04 [−0.03, 0.09] | +0.00 | **+0.02 [−0.00, 0.04]** | 0.12 |
| gap | real | +0.08 [0.04, 0.12] | +0.01 | +0.05 [0.02, 0.08] | **0.39** |
| gap | Emuru | +0.08 [0.01, 0.13] | +0.05 [0.00, 0.10] | +0.02 [0.01, 0.04] | **0.16** |
| slant_grad | real | +0.04 | 0.00 | +0.03 | 3.4 |
| slant_grad | Emuru | +0.04 | 0.00 | +0.03 | 3.0 |

Đọc kết quả:
- **Baseline:** Emuru gần như không có drift trong dòng (trend R² nhỏ hơn khoảng 7 lần so với ảnh thật) và ít biến thiên hơn. Đây là đúng thành phần mạnh nhất ở ảnh thật.
- **Gap:** cấu trúc lag-1 tương tự, nhưng độ phân tán khoảng cách giữa từ chỉ bằng ~40% ảnh thật: spacing quá đều.
- **Slant:** không khác biệt (và slant ở 32 px vốn đo kém, xem tuần 2).
- B0 (SpiS ghép từ) không có phụ thuộc nào (tuần 2; checkpoint rò rỉ writer nên chỉ mang tính tham khảo).

→ **H2 được ủng hộ trên một generator cấp dòng SOTA**: Emuru tái tạo tương quan gap nhưng không tái tạo drift baseline và độ biến thiên spacing. Chưa phải kết luận chung: Emuru zero-shot từ font, style từ 1 dòng, ảnh 64 px resize về 32 px. Cần thêm generator train trên IAM lines (TS-GAN, DiffBrush).

Kiểm tra độ bền: dùng real trên đúng 708 dòng Emuru sinh được (chưa lọc CER) cho cùng kết quả real (baseline trend R² 0.147, gap SD 0.39).

## 3. H2 trên TS-GAN (BMVC 2020, train trên IAM lines)

- **Thiết lập.** Người dùng tự chạy `scripts/gen_tsgan.py` (checkpoint `IAMslant_..._sMG` iteration 175000; style từ một dòng khác của cùng writer). Split của TS-GAN sạch với val1: train forms của nó ⊂ train + unassigned của ta.
- **Chất lượng.** 744/744 dòng sinh được, không dòng nào bị bỏ vì ký tự lạ. CER TrOCR trung vị 0.02, 99.9% số dòng có CER ≤ 0.2. Tách từ thành công 97.0%.

Kết quả trên 589 dòng chung (CER ≤ 0.2 ở cả real, Emuru, TS-GAN; `reports/h2/profile_3way.csv`). Riêng cặp real–TS-GAN trên 716 dòng chung cho cùng kết quả.

| Đặc trưng | Nguồn | lag-1 | trend R² vượt null | SD trong dòng |
|---|---|---|---|---|
| baseline | real | +0.28 [0.20, 0.35] | +0.14 [0.11, 0.18] | 0.19 |
| baseline | Emuru | +0.04 | +0.02 | 0.12 |
| baseline | TS-GAN | −0.04 [−0.08, 0.00] | −0.01 | **0.04** |
| gap | real | +0.08 [0.04, 0.12] | +0.05 | 0.39 |
| gap | Emuru | +0.08 | +0.02 | 0.16 |
| gap | TS-GAN | −0.00 | +0.00 | 0.16 |
| slant_grad | real | +0.04 | +0.03 | 3.4 |
| slant_grad | Emuru | +0.04 | +0.03 | 3.0 |
| slant_grad | TS-GAN | +0.00 | −0.00 | 2.7 |

- **TS-GAN** sinh dòng gần như thẳng tuyệt đối: SD baseline bằng ~1/5 ảnh thật, không có drift và không có phụ thuộc giữa các từ ở bất kỳ đặc trưng nào. Spacing đều như Emuru (SD 0.16 so với 0.39). Mô hình chỉ có một style vector toàn cục cộng noise iid; kết quả này khớp với kiến trúc đó.
- **Kết luận H2 (hiện tại):** cả hai generator cấp dòng đã thử, một train trên IAM lines và một zero-shot từ font, đều thiếu drift baseline và thiếu biến thiên spacing. Hai generator này lại có kiến trúc khác hẳn nhau (GAN, AR transformer). Đây là bằng chứng chính cho khoảng trống của bài.

Gợi ý "cách sửa tối giản", không cần train: tái bố cục hậu kỳ (post-hoc). Tách từ ảnh sinh (cùng `segment_words`), rồi đặt lại baseline/gap theo một mô hình drift + residual ước lượng từ ảnh thật (độ dốc theo dòng, SD gap theo writer). Pipeline ghép của thí nghiệm cảm nhận đã có sẵn. Cách này áp được cho mọi backbone và là đối chứng mạnh mà một mô hình học phải vượt.

## 4. Việc tiếp

- **DiffBrush:** chưa tải (checkpoint trên Google Drive), cần người dùng đồng ý.
- **Pilot cảm nhận:** chờ người tham gia.
- Thêm vào metric: bộ lọc CER cố định (≤ 0.2) và báo tỷ lệ loại; so khớp phân phối (không chỉ trung bình) của độ dốc drift và SD gap theo dòng.
