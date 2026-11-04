# Dàn ý paper và bản đồ claim → bằng chứng

Tên tạm: *When Set Labels Teach Context: Diagnosing and Fixing Shortcut Inversion in Compositional Microscopy Recognition*. Đặt lại tên theo kết quả cuối.

## Claim và trạng thái (cập nhật sau test đã khóa, 2026-10-29)

Trục: *Set labels train right-for-the-wrong-reasons decoders. Label-valid tile constraints restore localisation; contextual logit adjustment removes the label-conditional prior and improves attention-MIL under combination shift, failing exactly where its assumption fails.*

| # | Claim | Bằng chứng trên test (đã khóa, đọc một lần) | Trạng thái |
|---|---|---|---|
| C2 | Định vị ngược ("đúng vì lý do sai") | BBBC041 test (acquisition khác): ABMIL gametocyte 0.08, ring 0.19; PopEvidence schizont 0.04, trophozoite 0.07. PanNuke test: dead 0.08, epithelial 0.18, connective 0.24 | **Xác nhận trên test, 2 dataset** |
| C3 | Ràng buộc tile hợp lệ về nhãn khôi phục định vị | BBBC041 test 0.86–0.98 (trừ ring); PanNuke test 0.76–0.92. Cái giá AUROC: PanNuke 0.01–0.03; BBBC041 +0.003 | **Xác nhận trên test** |
| C9 | CLA cải thiện decoder attention khi combination dịch chuyển | **PHOEBI test: ABMIL + CLA so với ABMIL +0.039 [0.012, 0.066] (n = 1), +0.041 [0.015, 0.066] (n = 4)**. BBBC041 test: ABMIL + tile + CLA 0.577 so với ABMIL + tile 0.548. PanNuke test: −0.016 | **Xác nhận trên PHOEBI test**; thất bại trên PanNuke, nơi có biến gây nhiễu chung (loại mô), đúng như giả định của CLA dự báo |
| C9b | ABMIL + CLA vượt baseline đơn giản | so với mean_lin +0.013 đến +0.017 (CI chứa 0); tuyệt đối 0.671 / 0.673 > ProtoMatch 0.63 (đã công bố) | Không có ý nghĩa so với mean; vượt mốc PHOEBI |
| C8 | Giám sát chỉ ở cấp instance (tile-only) tổng quát tốt hơn | Dev: tốt nhất (0.635 / 0.648). **Test: kém nhất (0.561 / 0.551; so với mean −0.09)** | **Bác bỏ trên test.** Bài học về chọn phương pháp với ít combination |
| C1, C4 | Phụ thuộc ngữ cảnh; linh hoạt hơn thì phụ thuộc nhiều hơn | Chỉ trên dev (BBBC041, PanNuke) | Ủng hộ trên dev |
| C7 | Pooling phân phối | Không tổng quát | Âm |

### Đánh giá venue (sau test)

- **Đóng góp vững:**
  - (i) chẩn đoán định vị ngược, xác nhận trên test của 2 dataset;
  - (ii) ràng buộc khôi phục định vị, xác nhận trên test;
  - (iii) CLA, một tổng quát hóa có lý thuyết của logit adjustment, cải thiện có ý nghĩa trên test PHOEBI cho ABMIL, và có điều kiện thất bại được dự báo đúng;
  - (iv) các kết quả âm có giá trị: tile-only, pooling phân phối.
- **Điểm yếu cho CVPR main:**
  - lợi ích độ chính xác so với baseline đơn giản (mean_lin) không có ý nghĩa;
  - CLA chỉ giúp một họ decoder;
  - một dataset (PanNuke) cho kết quả âm.
- **Phù hợp nhất hiện tại:** MICCAI hoặc MIDL (chẩn đoán + phương pháp cho microscopy); workshop CVPR (CVMI). Muốn lên CVPR main cần: CLA có biến kiểm soát (ví dụ loại mô) sửa được PanNuke, và một lợi ích độ chính xác có ý nghĩa so với mọi baseline.

## Hình

1. Ma trận own/context (heatmap) của một decoder baseline và của PopEvidence, đặt cạnh nhau.
2. Pure-tile vote so với decoder train trên mixture, theo species (F2).
3. Pipeline của PopEvidence với ba ràng buộc tile.
4. Kết quả chính: mean AUROC/AP theo species, CI theo combination, cả 3 partition, đặt cạnh ProtoMatch/SimplexUnmix/ChannelGroup/MIL và mốc all-present.
5. Theo số field.
6. Trường hợp thất bại: bt (không có pure culture), cặp fj/pf.

## Việc còn lại theo thứ tự ưu tiên

1. Kết quả dev của PopEvidence và biến thể tile → Gate 3 trên dev.
2. Bổ sung `devcv` (ProtoMatch, SimplexUnmix, ChannelGroup trên 5 fold) vào cùng bảng và cùng phân rã own/context.
3. Khóa config → test 3 partition → báo cáo một lần.
4. Dataset thứ hai (C6).
5. CI cho phân rã own/context, đối chứng tile nền cho F2.
6. Viết bài.
