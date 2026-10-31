# Tuần 2 — tiến độ (31/10/2026)

## 1. Giao thức đo dynamics dùng chung cho mọi nguồn ảnh

`hwline/segment.py`, `hwline/profile.py`, `scripts/line_dynamics.py`.

**Tách từ.** Ảnh sinh không có word box, nên mọi nguồn — kể cả ảnh thật — đều được tách từ theo cùng một cách: nhị phân hóa Otsu, chọn n−1 khe cột rộng nhất, với n là số từ đọc từ transcript. Dấu câu đứng riêng được gộp vào từ trước.

Đối chiếu với box XML trên val1:

| Độ phân giải | Dòng tách được | Từ có IoU ≥ 0.7 | Dòng đúng hoàn toàn |
|---|---|---|---|
| Gốc | 97.5% | 90.0% | 78.5% |
| 32 px | 97.7% | 89.6% | 77.9% |

**Hồ sơ dynamics.** Cho mỗi đặc trưng:
- tương quan lag-1 vượt null hoán vị trong dòng;
- lag-1 sau khi bỏ xu hướng tuyến tính (null được detrend lại);
- tỷ lệ phương sai do xu hướng tuyến tính giải thích, trừ kỳ vọng dưới null;
- độ phân tán trong dòng.

CI 95% tính bằng bootstrap theo writer. Residual nội dung được cross-fit theo dòng, riêng cho từng nguồn.

## 2. Ảnh thật ở 32 px (`reports/h2/profile_real.csv`)

| Đặc trưng | lag-1 (val1) | trend R² vượt null (val1) |
|---|---|---|
| baseline | +0.26 [0.18, 0.32] | +0.15 [0.11, 0.18] |
| gap | +0.08 [0.04, 0.12] | +0.04 [0.02, 0.07] |
| slant_grad | +0.04 [0.01, 0.08] | +0.02 [−0.00, 0.04] |
| log core | −0.02 (không có) | — |

Kết quả trên train cùng dấu. Ở 32 px, giao thức dựa được chắc chắn vào **baseline** và **gap**. Slant gần như mất ở 32 px, nhưng vẫn đo được ở độ phân giải gốc.

## 3. B0: SpiS-GAN cấp từ ghép thành dòng (`reports/h2/profile_b0.csv`)

- Mọi chỉ số phụ thuộc và drift ≈ 0, với cả noise cố định theo dòng lẫn noise riêng từng từ.
- Độ phân tán gap chỉ bằng ~1/9 ảnh thật (gap cố định 16 px).
- Độ phân tán baseline lớn hơn ảnh thật: từ ngắn bị phóng to, vị trí dọc nhảy từ từ này sang từ kia (`reports/h2/b0_examples.png`).

Lý giải cơ chế: dữ liệu huấn luyện của SpiS crop sát từng từ rồi resize riêng về 32 px. Vị trí dọc và kích thước tương đối giữa các từ đã bị xóa khỏi dữ liệu, nên generator cấp từ không thể học được chúng.

**Chưa nên dùng các số B0 làm claim:** checkpoint B0 bị rò rỉ writer (xem mục 4).

## 4. Rò rỉ writer của checkpoint SpiS-GAN công khai

Ánh xạ writer trong HDF5 về writer IAM bằng cách khớp chuỗi từ (`reports/audit/spis_writer_map_*.csv`): 339/339 writer ánh xạ được. Trong số đó có **36 writer test, 11 writer val2 và 9 writer val1** của split Large Writer Independent.

→ Checkpoint `bestIAM.pth` không dùng được cho khởi tạo hay đánh giá chính. Đã tạo dữ liệu sạch theo định dạng SpiS tại `/home/jovyan/bachdx2/datasets/spis_clean`: 46.840 từ của 283 writer train, và 6.424 từ val1 dùng để chọn checkpoint. Chưa huấn luyện.

Ghi chú về môi trường tại commit đã pin:
- `networks/__init__.py` bị comment hết, nên `generate.py` không chạy.
- `requirements.txt` thiếu `timm`.
- `model32.py` import `wandb` và `fid_kid`.
- Đã lưu `reports/env_freeze.txt`.

## 5. Tổng quan novelty (`docs/novelty_audit.md`, do agent viết; có đánh dấu mức độ kiểm chứng)

- Chưa tìm thấy công trình offline nào đo phụ thuộc/drift trong dòng sau kiểm soát nội dung và writer, hoặc audit xem generator có tái tạo được nó không.
- Rủi ro:
  - drift tuyến tính gần với augmentation "warp cả dòng" (Varga & Bunke 2003);
  - field theo trục x gần với ALIS (ICCV 2021);
  - DiffBrush (ICCV 2025) đã có head căn dọc và khoảng cách;
  - SpiS-GAN là model cấp từ, trong khi đối thủ cấp dòng là Emuru (CVPR 2025), Eruku, DiffBrush, InkSpire.
- Framing được khuyến nghị: **nghiên cứu đo lường + benchmark dynamics audit các generator cấp dòng**, kèm variation field như plug-in nhỏ thử trên ≥2 backbone (ít nhất một backbone cấp dòng).

## 6. Bị chặn

Kiểm định H2 trên Emuru (`scripts/gen_emuru.py`, đã viết, chưa chạy). Model cần `trust_remote_code=True`, tức chạy code Python tải từ Hugging Face; lệnh này bị chặn chờ người dùng cho phép. Đối thủ cấp dòng khác (TS-GAN, DiffBrush, Eruku) cũng phải chạy code bên ngoài.
