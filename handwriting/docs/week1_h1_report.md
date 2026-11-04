# Tuần 1 — chẩn đoán H1 trên IAM (bản chạy thử)

Ngày chạy: 30/10/2026. Phạm vi: §3.1–3.2 và §4.1–4.2 của `handwriting_research_implementation_plan_vi.md`. Chưa làm §4.3 (ghép từ thật) và chưa huấn luyện mô hình nào.

## 1. Dữ liệu

- Nguồn: Zenodo record 19003721 (bản IAM di chuyển có phép tác giả, license CC-BY-NC-4.0). Tải `lines.tar`, `xml.tar`, `ascii.tar`, `largeWriterIndependentTextLineRecognitionTask.zip`; cả 4 file khớp md5 của record. Lưu tại `/home/jovyan/bachdx2/datasets/iam/`.
- Split Large Writer Independent đúng như plan: train 6.161 dòng/283 writer, val1 900/46, val2 940/43, test 1.861/128.
- Audit (`reports/audit/split_audit.json`): không giao writer hoặc form giữa các split; không ảnh trùng md5; bbox dòng trong `lines.txt` khớp kích thước ảnh ở 100% dòng.
- 3.491 dòng IAM không thuộc split nào (497 form có cả dòng trong split lẫn dòng không gán) → không dùng.
- Loại dòng `segmentation=err` hoặc thiếu word box: train còn 5.408/6.161, val1 còn 745/900. Bộ lọc này có thể làm lệch mẫu về các dòng dễ phân đoạn hơn.
- Chỉ dùng train + val1. Val2/test chưa được đọc ở bước phân tích.

## 2. Extractor (ảnh gốc 300 dpi, word box từ XML)

| Descriptor | Cách đo | Tỷ lệ đo được (train/val1) |
|---|---|---|
| baseline, core height | band dài nhất của profile ngang ≥ 0.5·max; cần ≥2 chữ thường và ≥1 chữ chỉ có thân | 0.82 / 0.84 |
| slant (Vinciarelli) | góc shear tối đa hóa Σh² của cột nét liên tục | 0.81 / 0.84 |
| slant_grad (độc lập) | atan(gy/gx) có trọng số trên biên nét gần đứng | 0.86 / 0.87 |
| stroke width | 2·mean(distance transform trên skeleton) | 0.86 / 0.87 |
| gap | khoảng mực giữa hai từ chữ-số kề nhau / median core height của dòng | — |
| bg (đối chứng) | cường độ trung bình pixel không phải mực trong box | — |

Kiểm tra thủ công: tôi mới xem 50 word crop và 10 dòng (`reports/gallery/`), **chưa đủ mục tiêu 200 crop/50 dòng trong plan**. Baseline và box nhìn đúng. Slant Vinciarelli sai rõ ở vài từ có nét nối dài ("postpone", "they" ~42°), nên tôi thêm `slant_grad`. Hai extractor slant tương quan 0.88 trên giá trị thô nhưng chỉ **0.45 sau khi trừ trung bình dòng**: biến thiên slant trong dòng phần lớn là nhiễu đo. Stroke width chưa có kiểm định đo lường riêng; theo plan, chưa được dùng làm căn cứ kết luận.

## 3. Phương pháp kiểm định

- Residual nội dung: ridge cross-fit theo form trên train (GroupKFold 5), áp lên val1. Biến kiểm soát: độ dài, chữ hoa/số/dấu, ascender/descender, histogram 26 chữ cái, vị trí trong dòng. Residual được trừ trung bình theo dòng, nên mọi hằng số writer/dòng/scan đều bị loại.
- Thống kê: tương quan lag k = 1..3 theo chỉ số từ. Null là hoán vị giá trị hợp lệ trong cùng dòng (giữ marginal và pattern mask), 500 lần hoán vị. Hiệu ứng = obs − null. CI 95% bằng bootstrap theo writer (1.000 lần). p tối thiểu = 0.002.
- Các spec độ nhạy: raw (không residual), GBM thay ridge, bỏ biến vị trí, thêm covariate nền, null hoán vị trong nhóm độ dài từ, loại từ có box chồng lấn (1,7% cặp kề), detrend tuyến tính theo dòng (null áp lại detrend cho từng mẫu hoán vị).
- Lưu ý: bản đầu của spec detrend dùng null sai và cho kết quả âm giả ở mọi đặc trưng. Đã sửa và có unit test riêng (`tests/test_dependence.py::test_detrend_null_calibrated_for_trend_plus_iid`).

## 4. Kết quả (lag 1, spec chính; `reports/h1/lag_stats.csv`)

| Đặc trưng | train | val1 | Còn sau detrend tuyến tính? (train / val1) |
|---|---|---|---|
| baseline (raw) | +0.18 [0.14, 0.26] | +0.26 [0.17, 0.32] | +0.08 [0.04, 0.10] / +0.07 [0.00, 0.12] |
| baseline (bỏ skew theo form) | +0.08 [0.06, 0.10] | +0.13 [0.04, 0.18] | như trên |
| gap giữa từ | +0.05 [0.03, 0.09] | +0.10 [0.06, 0.13] | không: −0.01 / +0.03 (CI chứa 0) |
| slant_grad | +0.08 [0.06, 0.10] | +0.12 [0.06, 0.16] | nhỏ: +0.02 [0.01, 0.03] / +0.06 [0.02, 0.09] |
| slant (Vinciarelli) | +0.04 [0.03, 0.06] | +0.05 [0.00, 0.09] | không |
| log stroke width* | +0.09 [0.07, 0.11] | +0.09 [0.05, 0.14] | nhỏ: +0.04 / +0.03 |
| log core height | −0.01 [−0.03, 0.00] | 0.00 [−0.03, 0.03] | không |
| bg (đối chứng nền) | +0.03 [0.02, 0.04] | +0.02 [−0.01, 0.05] | không ổn định |

\* Chưa được kiểm định đo lường, chưa dùng để kết luận.

- Mọi hiệu ứng dương ở spec chính vẫn giữ dấu và CI > 0 trên val1 khi đổi sang GBM, null theo nhóm độ dài, thêm covariate nền hoặc loại box chồng lấn (`reports/h1/decision.json`).
- Ổn định: ở baseline/gap/slant_grad/stroke width, 63–84% writer có hiệu ứng lag-1 dương; hai nửa writer chia ngẫu nhiên cùng dấu; hiệu ứng có ở cả ba nhóm độ dài dòng (`stability.csv`).
- Hình dạng: hiệu ứng giảm nhanh theo lag, lag 3 ≈ 0. Variogram < 1 ở khoảng 8–16 core height và > 1 ở khoảng xa — dấu hiệu của drift tần số thấp trong từng dòng (`variogram.png`).
- Ở bin khoảng cách ngắn nhất (< 4 core height), variogram > 1 ở cả đặc trưng lẫn bg. Bin này chủ yếu là cặp từ ngắn hoặc dấu câu, đo kém tin cậy, nên không diễn giải.

## 5. Quyết định H1 theo quy tắc nội bộ của plan

**Tiếp tục, nhưng thu hẹp claim.** Ít nhất hai đặc trưng đo đáng tin cậy (baseline, gap, cùng slant khi tính theo cả hai extractor) có phụ thuộc trong dòng, tái xuất hiện trên val1 và sống qua các kiểm soát chính. Core height thì không có phụ thuộc. Nền chỉ có hiệu ứng nhỏ hơn nhiều và không tái lập trên val1.

Thu hẹp theo bằng chứng:

1. Phần lớn phụ thuộc là **drift riêng của từng dòng** (xu hướng tuyến tính theo x), không phải "trạng thái" thay đổi từ từ qua từng từ. Sau detrend, gap và slant Vinciarelli về 0. Chỉ baseline còn thành phần cục bộ rõ (~0.08); slant_grad và stroke width còn rất nhỏ.
2. Effect size khiêm tốn: tương quan dư 0.05–0.26.
3. Chưa phân biệt được drift do người viết với cong/nghiêng giấy cục bộ khi scan; bỏ skew theo form chỉ giảm baseline từ 0.18 xuống 0.08.

Hệ quả cho thiết kế mô hình (§5, §7):
- B3 (noise tương quan cố định) nên có thêm một biến thể **slope ngẫu nhiên theo dòng** — tức field là xu hướng tuyến tính ngẫu nhiên. Nếu dữ liệu chủ yếu là drift, baseline rẻ này có thể bắt gần hết hiệu ứng, nên M1 phải vượt được nó.
- Field nên có độ dài tương quan lớn: nút cách ~4 token như §7 là hợp lý; lag-1 là nơi có tín hiệu.
- Metric dynamics ở §10 nên báo riêng hai phần: trend theo dòng và phần dư sau detrend.

## 6. Chưa làm / việc tiếp theo

- Kiểm tra thủ công đủ 200 crop/50 dòng, ghi sai số extractor (đặc biệt là slant và stroke width).
- §4.3: thí nghiệm cắt-ráp (R0) và bản xáo descriptor.
- Phân tích theo khoảng cách ngang thay cho lag theo chỉ số từ làm kiểm định chính; thêm null hoán vị trong nhóm nội dung chi tiết hơn.
- Tuần 2: môi trường SpiS-GAN đã clone và pin tại `/home/jovyan/bachdx2/repos/spis-gan` (commit `6949fc2`), chưa cài dependency và chưa chạy. Các điểm đã thấy khi đọc code:
  - `iam_line` alphabet có space (index 1); index 0 `` ` `` là blank.
  - `CTCLoss(zero_infinity=True)` đúng là che lỗi như plan cảnh báo.
  - `get_lexicon` lọc từ dài ≥20 ký tự.
  - `strLabelConverter.encode` trả kiểu khác khi batch chỉ có 1 chuỗi (`len(text)==1`).

## 7. Tái lập

```bash
cd handwriting
python scripts/fetch_iam.py /home/jovyan/bachdx2/datasets/iam/raw largeWriterIndependentTextLineRecognitionTask.zip ascii.tar xml.tar lines.tar
# giải nén vào iam/{xml,lines,ascii,task}
PYTHONPATH=. python scripts/prepare_iam_lines.py
PYTHONPATH=. python scripts/extract_descriptors.py      # ~20 s, 32 tiến trình
PYTHONPATH=. python scripts/make_gallery.py
PYTHONPATH=. python scripts/analyze_line_dependence.py  # ~5 phút CPU
PYTHONPATH=. python -m pytest tests                     # 14 test
```
