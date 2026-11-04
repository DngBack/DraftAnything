# Từ chẩn đoán đến phương pháp (cập nhật 2026-10-29)

## Chẩn đoán đã có bằng chứng

1. **Định vị ngược (C2).** Decoder train bằng BCE trên nhãn tập đúng ở cấp ảnh nhưng đặt evidence lên tile sai. BBBC041: trophozoite 0.076. PanNuke: dead 0.115.
2. **Dựa vào ngữ cảnh (C1, F7).** Giữ cố định bằng chứng own, điểm vẫn dịch theo lớp đi kèm. AUROC khử tương quan thấp hơn AUROC in-dist 3–4 điểm.
3. **Bảng tra theo combination (PHOEBI).** Với 25 combination train, mỗi combination là một video: loss ở cấp tập cho phép nhớ "phiên → tập nhãn". Prior trơn theo nhãn p(y_k | y₋ₖ) chỉ dao động 0.3–0.7 logit giữa các combination, quá nhỏ để giải thích bảng tra này.
4. **Thông tin có ở cấp tile.** Tile vote của classifier chỉ train trên pure culture đúng chiều cho cả 5 species trên mixture chưa thấy (F2). ProtoMatch, phương pháp không học từ mixture, là baseline tốt nhất của PHOEBI (0.63).

**Mẫu số chung:** loss ở cấp tập thưởng cho mọi tín hiệu dự đoán được *vector nhãn của cả tập*, kể cả ngữ cảnh và danh tính combination. Tín hiệu đó không chuyển được sang combination mới.

## Ba hướng sửa và kết quả

### A. Contextual Logit Adjustment (CLA): sửa ở phía loss

- **Cơ chế:** train trên e_k + τ·logit p̂(y_k | y₋ₖ), với y₋ₖ là nhãn thật của các lớp khác; test bằng e_k. Đây là tổng quát hóa của logit adjustment (Menon 2021) từ prior biên sang prior có điều kiện.
- **Lý thuyết:** khi ngữ cảnh chỉ mang thông tin qua y₋ₖ, nghiệm tối ưu là e_k = LLR(o_k).
- **Kết quả:**
  - BBBC041, kết hợp với ABMIL + ràng buộc tile: AUROC khử tương quan **+0.021 [0.011, 0.034]**, in-dist +0.001.
  - PanNuke: +0.000 khử tương quan, −0.014 in-dist.
  - PHOEBI, linear: +0.006 / 0.000.
  - PHOEBI, ABMIL + CLA: 0.586 so với ABMIL 0.549, tức +0.037 (chưa có CI).
- **Diễn giải:** giả định bị vi phạm khi có biến gây nhiễu chung (loại mô trong PanNuke) hoặc khi failure mode là bảng tra combination (PHOEBI). CLA chỉ đúng một phần; nó là một ablation có giá trị, chưa phải phương pháp chính.

### B. CompMix: sửa ở phía dữ liệu

- **Cơ chế:** trộn tile của hai bag thuộc hai combination khác nhau, nhãn là hợp. Hợp lệ về nhãn và tạo combination mới.
- **Kết quả:**
  - Head tuyến tính trên PHOEBI: ΔAUROC −0.006 / −0.014.
  - Deep Sets + mix: 0.570, +0.011 [−0.036, +0.063].
  - **Không giúp.**

### C. Giám sát instance suy ra chính xác từ nhãn tập (PU learning ở cấp tile): bỏ hẳn loss ở cấp tập

- **Âm chính xác:** mọi tile của bag không chứa k. **Dương chính xác:** mọi tile của pure culture k. **Dương yếu:** tile ở top-quantile của bag chứa k.
- **Điểm ở cấp ảnh** = quantile của evidence tile, không có read-out nào được train. Loss không bao giờ thấy vector nhãn của cả combination, nên **không thể học bảng tra combination**.
- **Lý thuyết:** loss của mỗi species là một bài toán phân loại tile độc lập với nhãn đúng (âm, dương) hoặc nhãn PU. Rủi ro tối ưu chỉ phụ thuộc phân phối có điều kiện của tile theo nhãn tile, vốn bất biến khi phân phối combination thay đổi, miễn là "tile của k trông giống nhau trong mọi combination". Đây là phiên bản học được của ProtoMatch, có thêm âm chính xác từ mixture.
- **Ablation:** pure-only (gần ProtoMatch); q50 / q90 / mean để đọc điểm; tile_q (giả định MIL).
- **Kết quả PHOEBI (n = 1, gộp 26 dev combination, 3 seed):**
  - tileonly_q50: AUROC **0.609** [0.544, 0.666], mAP 0.558, F1 margin −0.046. Tốt nhất mọi run ở n = 1.
  - So cặp: so với meanstd_lin +0.028 [−0.041, +0.089] (P = 0.81); so với ABMIL +0.060 [+0.006, +0.115]; so với mean_lin +0.054 (P = 0.95).
  - Cải thiện đúng các species thưa: bs 0.44 → 0.51, pf 0.47 → 0.56, bt 0.60 → 0.65, fj 0.41 → 0.45. Giảm ở mx: 0.71 → 0.63.
  - Cách đọc điểm: q50 tốt, q90 kém (0.539). "Tỷ lệ tile có evidence" bền hơn "tile mạnh nhất".
  - Vẫn dưới mốc ProtoMatch 0.63 (đo trên test). Đang chạy ProtoMatch trên đúng 5 dev fold.
- **BBBC041:** tile-only q90 0.813 / khử tương quan 0.768, hơn tile + BCE (0.781 / 0.739) nhưng kém ABMIL + tile + CLA (0.825 / 0.808).
- **PanNuke:** tile-only q90 0.882 / 0.840, kém mean_lin (0.892 / 0.852) khoảng 0.01; định vị 0.76–0.89.

## Câu chuyện paper nếu C đứng vững

1. Chẩn đoán: loss ở cấp tập dạy ngữ cảnh và định vị sai (3 dataset, 3 họ decoder).
2. Nguyên lý: chỉ dùng phần giám sát mà nhãn tập xác định *chính xác* ở cấp instance; bỏ phần phụ thuộc vào combination.
3. Phương pháp: giám sát tile theo kiểu PU (âm chính xác, dương chính xác từ pure culture, dương MIL), cộng với đọc điểm bằng quantile.
4. Kết quả: vượt ProtoMatch trên combination chưa thấy, định vị đúng, bền với khử tương quan.
5. CLA và CompMix là các ablation có giải thích vì sao chúng chỉ giúp một phần.
