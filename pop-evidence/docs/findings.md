# Phát hiện đang tích lũy (hướng tới paper)

Mọi con số trong file này là development (5 dev fold gộp, 26 combination), trừ khi ghi khác. Test chưa được mở.

## F1. Decoder train trên nhãn cấp tập học shortcut ngữ cảnh, và shortcut đảo chiều khi gặp combination mới

- Nguồn: `results/analysis/errors_probe_cls_n4.md`.
- Hồi quy logit(điểm trung bình theo combination của k) lên vector nhãn của cả 6 species, trên 26 dev combination.
- Với `mean_lin`, hệ số của chính species là ka +8.96, mx +4.96, bt +0.42, pf −0.26, bs −1.73, fj −4.14.
  - **Ba species có hệ số âm**: khi species đó có mặt, điểm của nó lại giảm.
- Độ lớn trung bình của phần context (1.27) xấp xỉ phần own (1.37).
- Mẫu hình giữ nguyên ở `meanstd_lin` và `deepsets`.
- Hệ số ngoài đường chéo tương quan dương với hệ số phi đồng xuất hiện trong train ở phần lớn các hàng: fj +0.82, ka +0.83, bs +0.46.
- Cảnh báo: 26 điểm, 7 tham số mỗi hàng.
- **CI bootstrap theo combination (1000 resample), cập nhật 2026-10-29:** không sự đảo chiều nào phân giải được. Với mean_lin:
  - Không phân giải: bs −1.7 [−4.8, +1.9], fj −4.1 [−9.3, +1.4], pf −0.3 [−3.7, +3.1], bt +0.4 [−2.3, +3.1].
  - Dương có ý nghĩa: ka +9.0 [+4.3, +13.5], mx +5.0 [+0.5, +10.0].
  - meanstd và deepsets cho kết quả giống vậy.
- **Phát biểu được phép:** chỉ species đặc (ka, mx) có bằng chứng own phân giải được; own evidence của species thưa không phân biệt được với 0, dù ước lượng điểm âm. Chữ "đảo chiều" chỉ dùng như mô tả ước lượng điểm cho tới khi có thêm combination (3 partition, hoặc dataset thứ hai).

## F2. Thông tin species có trong tile và đúng chiều, nhưng decoder train trên mixture không dùng tới

- Nguồn: `results/analysis/pure_probe_cls.md`.
- Một classifier tuyến tính 5 lớp trên tile của pure culture (không thấy mixture nào) đạt độ chính xác tile 0.83 trên val.
  - Đây là cận trên: mỗi species là một video, nên species và phiên trùng nhau.
- Áp lên 26 mixture chưa thấy, tỷ lệ tile được gán cho k phân biệt được combination có và không có k, đúng chiều cho cả 5 species:

| species | pure-tile vote | mean_lin (mixture-trained) |
|---|---|---|
| bs | 0.62 | 0.45 |
| fj | 0.58 | 0.37 |
| pf | 0.57 | 0.50 |
| mx | 0.62 | 0.74 |
| ka | 0.88 | 0.83 |

**CI bootstrap theo combination (2000 resample):** bs 0.62 [0.32, 0.88], fj 0.57 [0.32, 0.80], mx 0.60 [0.35, 0.82], pf 0.57 [0.33, 0.80], ka 0.89 [0.74, 0.99]. Chỉ ka phân giải được. Phát biểu được phép: tile vote có ước lượng điểm đúng chiều cho cả 5 species, trong khi decoder train trên mixture có ước lượng điểm ngược chiều cho bs và fj. Với 26 combination, không cái nào trong hai mô tả đó tự nó có ý nghĩa thống kê.

Ý nghĩa: thất bại của decoder không chỉ do ảnh thiếu thông tin. Phương pháp học từ nhãn tập chuyển bằng chứng của tile thành bằng chứng ngữ cảnh. Một phần bằng chứng ở mức tile vẫn còn tồn tại khi chuyển sang combination mới.

Hạn chế:
- Tỷ lệ tile gán cho fj và pf khi vắng mặt vẫn cao (0.27–0.31). Có thể do tile ít tế bào, hoặc do fj và pf nhầm lẫn nhau (pf → fj 0.17).
- bt không có pure culture, nên phép thử này không áp dụng cho bt.

## F2b. Đối chứng nền, và cơ chế salience

- Nguồn: `scripts/tile_density.py`, mục (c) của `pure_probe_cls.md`.
- Chỉ số nội dung: std của mức xám sau hiệu chỉnh chiếu sáng cho mỗi tile.
- Median theo pure culture: ka 5.03 (6% tile dưới median toàn cục), mx 4.91 (32%), bs 3.22 (56%), pf 3.21 (56%), fj 2.73 (64%).
- Tile vote khi chỉ giữ tile trên median gần như không đổi (bs 0.62, fj 0.58, ka 0.89, mx 0.61, pf 0.63), nên F2 không phải do tile nền.
- Thứ tự "đặc" (ka > mx > bs ≈ pf > fj) trùng với thứ tự own evidence trong F1: ka và mx dương mạnh; fj, bs và pf âm hoặc gần 0.
- Giả thuyết cơ chế: khi pooling trên tập tile hỗn hợp, species đặc chiếm phần lớn khối lượng của biểu diễn tập. Bằng chứng của species thưa vì thế bị che, và decoder bù lại bằng tín hiệu ngữ cảnh (sự có mặt hoặc vắng mặt của species đặc).
- Hệ quả thiết kế: bằng chứng phải được tính trên từng tile trước khi gộp, và cách gộp không được để species đặc lấn át (vd. quantile hoặc tỷ lệ tile thay vì mean của embedding).

## F3. Thống kê ngoài mean thêm tín hiệu nhỏ nhưng nhất quán, chủ yếu cho bt

- Nguồn: `results/dev/probe_cls`.
- Ở n=4, Δ macro AP so với mean:
  - meanstd +0.037 [0.005, 0.067]
  - kme05 +0.035 [0.001, 0.066]
  - deepsets +0.027 [−0.019, 0.075]
  - meanquant +0.011 [0.002, 0.023]
- Ở n=1, meanstd +0.021 [0.001, 0.040].
- AUROC của bt tăng từ 0.52 lên 0.63–0.67 ở mọi biểu diễn phân phối.
- Mức tuyệt đối vẫn thấp: mean AUROC ≤ 0.60, và F1 thua cách đoán "có mặt tất cả".
- Gate 2 đạt theo các tiêu chí đã khóa, nhưng sát ngưỡng và chưa hiệu chỉnh so sánh bội.

## Hàm ý cho phương pháp

1. Bằng chứng cần được tính ở **mức tile và riêng cho từng species**. Điểm của k không được đọc từ vector chung của cả tập (F1).
2. Có thể neo evidence head bằng pure culture ở những species có pure culture (F2). bt cần cách khác: hybrid, hoặc học từ mixture với ràng buộc.
3. Thống kê phân phối của evidence (tỷ lệ tile, quantile) là cách gộp tự nhiên; biến "tile vote" của F2 chính là một quantile của evidence. Đây là chỗ PopEvidence gặp F3.
4. Metric chính cho paper nên có **phân rã own/context**. Đây có thể là đóng góp đánh giá, vì nó đo trực tiếp failure mode thay vì chỉ đo F1.

## F5. BBBC041: rò rỉ ngữ cảnh có ý nghĩa thống kê khi đơn vị là ảnh

- Nguồn: `scripts/analyze_instances.py`, `results/analysis/bbbc041.md`.
- Thiết lập: 866 ảnh, out-of-fold qua 5 fold, 3 seed. Hồi quy logit(score_k) lên y_k, log(1 + số tế bào k) và sự có mặt của các lớp khác. Bootstrap theo ảnh (500 resample).
- Số hệ số ngữ cảnh có CI loại trừ 0: mean_lin 11/20, meanstd_lin 15/20, deepsets 17/20. Decoder càng linh hoạt càng dựa vào ngữ cảnh.
- Ví dụ (deepsets):
  - leukocyte ← trophozoite +2.16 [1.95, 2.39]
  - gametocyte ← schizont +2.02 [1.67, 2.33]
  - schizont ← ring −1.68 [−2.06, −1.34]
- Lưu ý:
  - (i) Hệ số own bị chia giữa y_k và log-count nên không diễn giải riêng; chỉ hệ số ngữ cảnh được diễn giải.
  - (ii) Nếu chú thích bỏ sót tế bào k và việc bỏ sót tương quan với sự có mặt của lớp j, một phần hệ số có thể là nhiễu nhãn chứ không phải shortcut.
  - (iii) CV chia ở cấp ảnh; ảnh cùng bệnh nhân có thể nằm ở cả hai phía. Điều này không làm sai phép đo rò rỉ nhưng ảnh hưởng mức AUROC tuyệt đối.
- Ý nghĩa: claim C1 (decoder train trên nhãn tập học bằng chứng ngữ cảnh) **có power trên dataset thứ hai**, điều PHOEBI không cho phép. Phát biểu được: "giữ cố định bằng chứng thật của k, điểm của k vẫn dịch chuyển có ý nghĩa theo các lớp đi kèm".

## F6. Decoder train trên nhãn tập trả lời đúng ở cấp ảnh nhưng dựa trên tile sai; ràng buộc tile hợp lệ về nhãn sửa được định vị

- Nguồn: `results/dev/bbbc041_tile`, `scripts/analyze_instances.py`.
- Phép đo: AUROC của evidence từng tile (kênh 0) so với "tâm box của lớp k nằm trong tile", trên toàn bộ tile của 866 ảnh out-of-fold.

| Lớp | Không ràng buộc | λ_tile = 1 | λ_tile = 3 |
|---|---|---|---|
| gametocyte | 0.54 | 0.92 | 0.88 |
| leukocyte | 0.70 | 0.65 | 0.61 |
| ring | 0.48 | 0.61 | 0.63 |
| schizont | **0.21** | 0.87 | 0.86 |
| trophozoite | **0.17** | 0.87 | 0.86 |

- Không có ràng buộc, evidence của trophozoite và schizont **định vị ngược**: nằm trên tile không chứa chúng. Điểm ở cấp ảnh vẫn tốt (AUROC ảnh ≈ 0.82), nghĩa là decoder đúng vì lý do sai.
- Có ràng buộc thì định vị đúng, và AUROC ở cấp ảnh không giảm (0.824 so với 0.819).
- Rò rỉ ngữ cảnh ở cấp ảnh không giảm (16/20 so với 15/20). Giả thuyết: điểm số đọc cả m = 4 kênh evidence nhưng ràng buộc chỉ áp lên kênh 0, nên ngữ cảnh đi qua các kênh còn lại. Kiểm tra bằng m = 1 (`configs/bbbc041_m1.yaml`).
- **PHOEBI (n = 1), sửa lại 2026-10-29:**
  - AUROC: pe_quantile 0.563, +tile λ = 0.3: 0.558, +tile λ = 1: 0.565. Ràng buộc tile không đổi độ chính xác.
  - Mean |context|: 0.55 / 0.51 / 0.55. Mức giảm so với linear (≈ 1.27) đến từ kiến trúc PopEvidence (thang logit khác), **không** từ ràng buộc tile.
  - Quyết định: **no-go cho claim độ chính xác trên PHOEBI** (không vượt meanstd 0.581/0.601, và xa mốc ProtoMatch 0.63).
- **Kết quả m = 1 (bác bỏ giả thuyết kênh phụ):**
  - pe_quantile_m1: 15/20 hệ số ngữ cảnh có ý nghĩa; định vị schizont 0.09, trophozoite 0.08 (vẫn ngược).
  - pe_quantile_tile1_m1: 14/20; định vị 0.84–0.88.
  - AUROC ảnh 0.816–0.818, không đổi.
  - Vậy ngữ cảnh không đi qua các kênh không bị ràng buộc. Còn hai khả năng: (a) qua thống kê pooling trên các tile không chứa k; (b) nằm sẵn trong feature của tile hoặc trong sinh học/chú thích (giai đoạn đồng bộ, chú thích bỏ sót). Đang kiểm tra bằng `pe_max_tile1_m1` (score = max evidence của tile).
- Claim được phép lúc này: ràng buộc tile **khôi phục định vị** mà không mất độ chính xác. **Chưa** được claim là giảm rò rỉ ngữ cảnh ở cấp ảnh.
- Ứng viên claim chính: *"set-level supervision yields right-for-the-wrong-reasons decoders; label-valid tile constraints restore localisation at no accuracy cost"*. Cần thêm: CI cho AUROC định vị, và kết quả m = 1.

## F7. Mức dựa vào ngữ cảnh tăng theo độ linh hoạt của decoder, và ràng buộc tile không làm giảm nó

- Nguồn: BBBC041, mean |hệ số ngữ cảnh| chuẩn hóa theo độ lệch chuẩn của logit (để so các method có thang logit khác nhau).

| Method | Tỷ lệ rò rỉ | AUROC ảnh |
|---|---|---|
| mean_lin, meanstd_lin | 0.21 | 0.77 / 0.73 |
| deepsets | 0.39 | 0.82 |
| pe_quantile (m = 4), pe_quantile_m1 | 0.36, 0.35 | 0.82 |
| pe_quantile_tile{1,3}_m1 | 0.34, 0.33 | 0.82, 0.81 |
| pe_max_tile1_m1 | 0.30 | 0.78 |

- Max pooling trên evidence đã định vị vẫn để lại 13/20 hệ số ngữ cảnh có ý nghĩa. Vậy một phần sự phụ thuộc nằm sẵn trong feature của tile (DINOv2 đông cứng) hoặc trong sinh học/chú thích, không chỉ ở decoder.
- **Kết luận cho paper:**
  - (i) Đánh đổi: decoder linh hoạt hơn thì chính xác hơn nhưng dựa vào ngữ cảnh nhiều hơn (tỷ lệ gần gấp đôi).
  - (ii) Ràng buộc tile hợp lệ về nhãn khôi phục định vị (F6) nhưng **không** giảm mức dựa vào ngữ cảnh ở cấp ảnh.
  - (iii) Muốn giảm rò rỉ có lẽ phải can thiệp vào biểu diễn (fine-tune encoder dưới ràng buộc tile), hoặc chấp nhận rằng một phần là tín hiệu sinh học thật.

## F8. Tái hiện trên PanNuke (H&E, 7.901 ảnh, 5 loại nhân, instance mask)

- Thiết lập dev: train fold 1, đánh giá nửa còn lại của fold 2, và ngược lại. Fold 3 giữ cho test. Đơn vị là ảnh, 3 seed.
- AUROC ảnh: mean_lin 0.892, deepsets 0.902, pe_quantile_m1 0.880, pe_quantile_tile1_m1 0.875.
- **Định vị tile [95 % CI theo ảnh], không ràng buộc → có ràng buộc:**
  - connective 0.228 [0.218, 0.236] → 0.773 [0.766, 0.782]
  - dead 0.115 [0.091, 0.146] → 0.890 [0.860, 0.915]
  - epithelial 0.194 [0.179, 0.211] → 0.799 [0.783, 0.816]
  - inflammatory 0.751 → 0.895
  - neoplastic 0.850 → 0.879
- Định vị ngược ở 3/5 lớp khi không ràng buộc, cùng hiện tượng như BBBC041 (2/5).
- Cái giá về độ chính xác của ràng buộc (paired):
  - PanNuke: ΔAUROC −0.006 [−0.010, −0.002], ΔAP −0.009 [−0.017, −0.000]
  - BBBC041: ΔAUROC −0.002 [−0.008, +0.004], không có ý nghĩa
- Hệ số ngữ cảnh có ý nghĩa: 11–14/20 ở mọi method. Ràng buộc không làm giảm (giống F7).
- meanstd kém mean (ΔAP −0.030), lần thứ hai H1 không tổng quát.
- **Kết luận:** C2 (định vị ngược) và C3 (ràng buộc khôi phục định vị với cái giá ≤ 0.01 AUROC) tái hiện trên dataset thứ ba, modality khác.

## F9. Phương pháp: CLA, CompMix và giám sát tile chính xác (2026-10-29)

**CLA (BBBC041 / PanNuke, đơn vị ảnh; AUROC khử tương quan = trọng số đưa phân phối nhãn về tích các marginal):**
- ABMIL + tile + CLA so với ABMIL + tile:
  - BBBC041: Δ khử tương quan **+0.021 [0.011, 0.034]**, Δ in-dist +0.001.
  - PanNuke: Δ khử tương quan +0.000, Δ in-dist −0.014.
- PopEvidence + CLA: +0.006 (không có ý nghĩa) trên BBBC041; −0.043 trên PanNuke.
- Giả thuyết cho PanNuke: loại mô là biến gây nhiễu chung, vi phạm giả định của CLA.

**PHOEBI, n = 1, AUROC gộp 26 combination (3 seed):**

| Method | AUROC | macro AP | F1 margin |
|---|---|---|---|
| tileonly_q50 (giám sát tile chính xác, đọc median) | **0.609** [0.541, 0.670] | **0.558** | **−0.046** |
| abmil_cla1 | 0.586 | | |
| meanstd_lin | 0.581 | 0.530 | −0.111 |
| deepsets_mix (CompMix) | 0.570 | | |
| pe_quantile_tile1 | 0.565 | | |
| deepsets | 0.559 | | |
| mean_lin | 0.555 | 0.510 | −0.139 |
| abmil | 0.549 | | |
| tileonly_q90 | 0.539 | | |

- tileonly_q50 so với abmil: ΔAUROC +0.060 [0.006, 0.115], ΔAP +0.062 [0.020, 0.090], **có ý nghĩa**.
- So với pe_quantile_tile1: ΔAP +0.050 [0.003, 0.084], có ý nghĩa.
- So với mean_lin: +0.054 [−0.012, 0.120], P(Δ > 0) = 0.95.
- So với meanstd_lin: +0.028, không có ý nghĩa.
- **n = 4:** tileonly_q50 AUROC **0.617**, mAP **0.581**, cao nhất từ trước tới nay. bs lần đầu > 0.5 (0.57). So với meanstd_lin n = 4 (0.601): +0.016, không có ý nghĩa.
- **CLA giúp ABMIL trên PHOEBI:** 0.549 → 0.586.
- **Đọc điểm quan trọng:** median (q50) = 0.609, nhưng q90 = 0.539. Quantile cao bắt tile đặc/ngữ cảnh; median bền với cơ chế salience (F2b).
- Vẫn **chưa vượt rõ mốc ProtoMatch 0.63**. Mốc đó đo trên 27 test combination, chưa có số trên cùng dev fold vì devcv chưa chạy xong.

## F9. PHOEBI: giám sát tile chính xác (tile-only) là phương pháp tốt nhất; chuẩn hóa theo fold là proxy đúng cho test

- Nguồn: `results/analysis/phoebi_n{1,4}.txt`, `ensemble_phoebi*.txt`.
- AUROC gộp thô: tileonly_q50 0.609 (n = 1), 0.617 (n = 4); meanstd 0.581 / 0.601.
- **Chuẩn hóa thứ hạng trong từng fold** (không dùng nhãn) loại bỏ độ lệch thang điểm giữa 5 model của 5 fold. Độ lệch này không tồn tại lúc test, khi chỉ một model chấm điểm.
  - n = 1: tileonly 0.619, meanstd 0.600, ABMIL + CLA 0.608, mean 0.582.
  - n = 4: **tileonly 0.638 [0.565, 0.705]**, ABMIL + CLA 0.620, meanstd 0.615, mean 0.586.
  - Thứ hạng giữa các phương pháp giống với AUROC thô.
  - **Post-hoc** so với protocol §5: phải báo cáo cả hai cách.
- Ensemble trọng số bằng nhau (tileonly + meanstd [+ ABMIL + CLA]) đạt 0.640–0.643, gần như không thêm gì so với tileonly chuẩn hóa. Phần lớn "lợi ích ensemble" là do chuẩn hóa.
- tileonly so với meanstd: Δ ≈ +0.02 đến +0.04, P(Δ > 0) = 0.84–0.93, chưa có ý nghĩa. tileonly so với mean_lin và ABMIL: có ý nghĩa hoặc gần có ý nghĩa.
- Theo species, tileonly cải thiện species thưa: bs, pf. meanstd tốt hơn ở mx và bt.

## F10. Kết quả test đã khóa (đọc một lần)

- **Định vị (C2, C3) tái hiện trên cả hai test có instance:**
  - BBBC041 test (acquisition khác): không ràng buộc thì gametocyte 0.08, ring 0.19, schizont 0.04, trophozoite 0.07; có ràng buộc thì 0.86–0.98, trừ ring.
  - PanNuke fold 3: dead 0.08, epithelial 0.18, connective 0.24 → 0.76–0.92.
- **PHOEBI:** phương pháp chính tile-only **thất bại** (0.600 so với meanstd 0.664). Nguyên nhân là phụ thuộc pure culture; dev chỉ có partition 1337 nên không lộ điểm yếu này.
- **CLA:** cải thiện ABMIL có ý nghĩa trên PHOEBI test (+0.028 [+0.005, +0.050] với n = 4; +0.027, P = 0.974 với n = 1). ABMIL + CLA là method cao nhất (0.68 chuẩn hóa / 0.67 thô, so với ProtoMatch 0.63 được công bố). Tuy vậy, so với mean / meanstd chỉ +0.01, không có ý nghĩa.
  - BBBC041 test: ABMIL + tile + CLA cao nhất (0.577). PanNuke test: CLA gây hại (−0.026).
  - Diễn giải: CLA giúp khi failure mode là ngữ cảnh đồng xuất hiện hoặc combination (PHOEBI, BBBC041), và gây hại khi có biến gây nhiễu chung không nằm trong nhãn (loại mô ở PanNuke).
- **Bài học protocol:** dev fold của PHOEBI chỉ thuộc partition 1337 và luôn giữ đủ 5 pure culture, nên đánh giá quá cao các phương pháp dựa vào pure culture. Paper phải nêu rõ điều này.

## F10. Test đã khóa (đọc một lần): phương pháp chính thất bại trên PHOEBI; định vị và CLA đứng vững

- **PHOEBI** (27 combination, 3 partition):
  - tileonly_mean, đứng đầu dev, **đứng cuối test**: AUROC 0.561 (n = 1).
  - ABMIL + CLA đứng đầu: 0.671, so với ABMIL 0.632 và mean 0.654. Mức hơn ABMIL tái hiện kết quả dev.
  - Bài học: chọn phương pháp trên 26 dev combination không đáng tin cậy. CI dev ±0.07 đã cảnh báo điều này.
- **Cơ chế thất bại của tile-only:** đoán quá nhiều nhãn (FPR 0.60–0.65). Giả thuyết: ràng buộc "dương ở top-quantile" và "âm chính xác" được hiệu chỉnh cho các combination train. Test PHOEBI có nhiều combination bậc cao (6 species xuất hiện ở cả 3 partition), làm đổi tỷ lệ tile dương và lệch phân phối điểm. Chưa kiểm chứng.
- **BBBC041 test** (acquisition khác): định vị ngược và việc khôi phục bằng ràng buộc đều tái hiện, trừ ring. ABMIL + tile + CLA tốt nhất (0.577).
- **PanNuke test:** định vị ngược và việc khôi phục tái hiện. Độ chính xác: Deep Sets và mean tốt nhất; ràng buộc tốn 0.01–0.03; CLA tốn 0.026.

## Giới hạn thống kê (quyết định chiến lược)

PHOEBI chỉ có 40 combination; dev có 26, test 9 (27 khi gộp 3 partition). Ở cỡ này, chỉ hiệu ứng rất lớn (ka) mới đạt ý nghĩa ở mức species, và PHOEBI cũng gặp đúng giới hạn này. Một paper CVPR không thể dựa vào các claim per-species trên PHOEBI. Hệ quả:

1. Trên PHOEBI, claim phải là các thống kê gộp: mean AUROC/AP, số species có own > 0 phân giải được, paired Δ so với baseline. Không claim từng species.
2. **Dataset thứ hai có composition kiểm soát được** (bag tổng hợp từ crop đơn bào với hàng trăm combination, hoặc dữ liệu polyculture có nhiều combination) là cần thiết để kiểm tra C1–C3 với đủ power. Ưu tiên cao nhất sau Gate 3.

## Câu hỏi mở cần trả lời trước khi viết

- **Bias của tile vote:** F2 có đứng vững khi loại bỏ tile nền (ít tế bào) không? Cần một thước đo mật độ tế bào.
- **Context giảm có kéo theo AUROC tăng không?** Có kiểm tra được bằng consistency loss của PopEvidence.
- **Có giữ được trên các partition khác không?** Cần chạy trên LCO seed 1338/1339, vốn có các combination giữ lại khác.

## F11. CLA có biến kiểm soát loại mô không sửa được PanNuke (dev, 2026-10-29)

- Nguồn: `results/dev/cla_tissue_pannuke` (run do một phiên song song tạo). Phân tích AUROC khử tương quan được làm ở phiên này.
- AUROC in-dist / khử tương quan:
  - ABMIL 0.880 / 0.846
  - ABMIL + CLA theo nhãn 0.860 / 0.841
  - ABMIL + CLA nhãn + loại mô 0.846 / 0.839
  - mean_lin 0.892 / 0.852
  - mean_lin + CLA loại mô 0.868 / 0.847
- Kết luận: giả thuyết "loại mô là biến gây nhiễu làm CLA thất bại" **không được ủng hộ**. Trên PanNuke, đồng xuất hiện giữa các loại nhân có vẻ là tín hiệu thật (đúng sinh học), không phải shortcut. Bỏ prior thì mất thông tin mà không đổi lại được độ bền.
- Phát biểu đúng cho paper: CLA có ích khi phân phối combination thật sự dịch chuyển giữa train và test (PHOEBI LCO, test BBBC041). Nó không có ích khi đồng xuất hiện ổn định và mang thông tin (PanNuke).


## F12. Batch và phiên chụp lấn át loài trong embedding đông cứng (phoebi4 so với phoebi6)

- Nguồn: `results/analysis/phoebi4_checks.md`.
- phoebi4 là một batch nuôi cấy khác, chụp ở phiên khác, cùng thiết bị. Ánh xạ b, f, k, p → bs, fj, ka, pf được dataset card xác nhận.
- Embedding trung bình của pure culture phoebi4 gần nhất với **loài sai** trong phoebi6 ở 3/4 trường hợp: b→bt, f→mx, p→fj. Chỉ k→ka đúng; ka là loài "đặc" và khác biệt nhất (F2b).
- Cosine giữa cùng một loài ở hai batch: bs 0.883, fj 0.762, pf 0.919. Thấp hơn cosine giữa các loài khác nhau (ví dụ p→fj 0.951).
- Ý nghĩa: đây là bằng chứng độc lập cho confound "combination = phiên chụp". Đặc trưng DINOv2 đông cứng mã hóa batch và phiên mạnh hơn loài, nên decoder nào khớp theo phiên đều sẽ không chuyển sang batch mới. Nó cũng giải thích vì sao ProtoMatch và mọi decoder trên PHOEBI gần mức ngẫu nhiên cho các loài thưa.

## F13. Test ngoài phoebi4 (batch nuôi cấy và phiên mới; config khóa trước khi tải dữ liệu)

- Nguồn: `results/test/phoebi4_v1`, `results/analysis/phoebi4_paired.txt`. Train trên toàn bộ phoebi6, test trên 14 video phoebi4, đơn vị là video.
- AUROC n = 1 / n = 4:
  - mean_lin **0.682 / 0.680** (tốt nhất)
  - ABMIL + tile 0.664 / 0.669
  - meanstd 0.659 / 0.644
  - ProtoMatch 0.642 / 0.654
  - tile-only 0.638 / 0.629
  - Deep Sets 0.629 / 0.605
  - ABMIL 0.622 / 0.582
  - ABMIL + CLA 0.620 / 0.590
- **E1 (CLA giúp ABMIL): không tái hiện** (−0.002 / +0.008). Lợi ích +0.04 trên test phoebi6 (dịch chuyển combination, cùng batch) không chuyển sang dịch chuyển batch.
- **E2 (decoder linh hoạt không hơn linear): đúng chiều.** mean − ABMIL +0.06 / +0.10 (P = 0.81 / 0.96; ΔAP n = 4 có ý nghĩa). mean − Deep Sets +0.05 / +0.08.
- **E3 (tile-only không hơn mean): đúng chiều** (−0.04 / −0.05).
- **Thăm dò** (nằm trong config đã khóa nhưng không phải giả thuyết pre-register): ABMIL + tile − ABMIL +0.042 (P = 0.89) / **+0.087 [0.010, 0.169]**. Ràng buộc tile làm attention-MIL bền hơn khi đổi batch, đúng với cơ chế "nhìn vào tế bào thay vì ngữ cảnh". Cần xác nhận lại ở một tập khác.
- **Bức tranh tổng hợp sau 4 tập test:**
  - Khi đổi batch, head tuyến tính trên mean embedding là bền nhất. Decoder càng linh hoạt (Deep Sets, ABMIL) càng học theo phiên và mất nhiều nhất.
  - Ràng buộc tile giảm một phần mức mất này.
  - CLA chỉ giúp khi dịch chuyển là ở tổ hợp nhãn, không giúp khi dịch chuyển ở batch hay phiên.

## F14. Chuẩn hóa feature theo batch không sửa được dịch chuyển batch (thăm dò, post-hoc trên phoebi4)

- Nguồn: `results/test/phoebi4_bn_exploratory`, `results/analysis/phoebi4_bn_paired.txt`.
- Phương pháp: chuẩn hóa từng chiều feature bằng mean và std của cả batch (phoebi6 train; toàn bộ phoebi4, không dùng nhãn). Cùng method và cùng bag với `phoebi4_v1`.
- Δ AUROC so với feature gốc: từ −0.022 đến +0.023 ở mọi decoder, không cái nào có ý nghĩa. ProtoMatch giảm 0.065 / 0.075.
- fj vẫn ngược ở hầu hết method (AUROC 0.25–0.49), khớp với F12: fj của phoebi4 gần mx của phoebi6.
- **Kết luận:** dịch chuyển batch không phải là một phép dịch hay co giãn của feature. Nó làm đổi vị trí tương đối giữa các loài trong embedding đông cứng. Không cách sửa nào ở cấp decoder đã thử (CLA, ràng buộc tile, chuẩn hóa) bù được điều đó. Muốn tiến tiếp cần:
  - (a) dữ liệu train có nhiều batch cho mỗi loài, để học tính bất biến theo batch;
  - (b) can thiệp ở cấp biểu diễn (fine-tune hoặc encoder khác);
  - và trong cả hai trường hợp, một batch mới chưa dùng để test.

## F15. Sửa lỗi ràng buộc tile (code review 2026-10-30): "hợp lệ về nhãn" chỉ đủ khi lớp không quá thưa

- Lỗi: ràng buộc dùng trên BBBC041 và PanNuke có thành phần "pure" (mọi tile của ảnh một nhãn đều dương) và q = 0.9. Cả hai đều không suy ra được từ nhãn tập trên các dataset này (ảnh một nhãn vẫn có hồng cầu, nhân khác, nền). Xem `docs/code_review.md`.
- Phiên bản hợp lệ thật: âm chính xác cho lớp vắng mặt; dương = ít nhất một tile (max); không có "pure". Chạy lại trên dev, cùng fold và bag (`results/analysis/valid_tile_*.txt`).

| Dataset | Định vị: không ràng buộc → hợp lệ thật → ràng buộc cũ | AUROC ảnh (hợp lệ thật so với không ràng buộc) |
|---|---|---|
| PanNuke (PopEvidence) | con 0.23 → 0.77 → 0.77; dead 0.12 → 0.88 → 0.89; epi 0.19 → 0.80 → 0.80 | −0.016 |
| PanNuke (ABMIL) | 0.71–0.84 → 0.77–0.89 → 0.77–0.90 | −0.014 |
| BBBC041 (ABMIL) | gam 0.60, ring 0.30, sch 0.36 → **0.35, 0.35, 0.38** → 0.95, 0.61, 0.89 | **−0.030** |
| BBBC041 (PopEvidence) | sch 0.09, tro 0.08 → 0.71, 0.75 → 0.84, 0.88 | −0.032 |

- **Kết luận đã sửa:**
  - Ràng buộc thật sự hợp lệ về nhãn khôi phục định vị khi lớp chiếm một phần đáng kể số tile (PanNuke), nhưng không đủ khi lớp cực thưa (ký sinh trùng trong BBBC041: 0.2–3 % tile, hồng cầu áp đảo).
  - Ở đó cần thêm một giả định giám sát: coi mọi tile của ảnh một nhãn là dương có nhiễu, đối chiếu với âm chính xác.
  - Đây chính là biến thể đã chạy trên test BBBC041. Nó khôi phục định vị, nhưng **không** "hợp lệ về nhãn".
- Cơ chế giả định: tile chung (hồng cầu) bị ép âm chính xác ở mọi ảnh không có k và bị ép dương có nhiễu ở ảnh chỉ có k. Hai lực này triệt tiêu ở tile chung, nên chỉ tile đặc trưng của k còn giữ evidence dương.
- Hệ quả cho các test đã khóa: số liệu không đổi, nhưng phải gọi đúng tên phương pháp đã chạy ("tile constraints with single-label noisy positives"). Claim "label-valid" chỉ đúng cho PanNuke, ở dev.
