# Experiment log

Mỗi lần chạy ghi: commit, dữ liệu, GPU, torch, lệnh, thời gian, metric, và mọi sai khác so với protocol.

## 2026-10-28 — khởi tạo

- PHOEBI clone tại `/home/jovyan/bachdx2/datasets/phoebi/repo`, commit `2a42be0707c0a94807610640d80a34a9a9dfc967`.
- Dataset `sochastic/PHOEBI` là gated repo: máy này chưa có HF token nên **chưa tải được** manifest hay ảnh. Cần `hf auth login` bằng tài khoản đã chấp nhận điều kiện trên dataset card.
- Môi trường: `.venv` (Python 3.13) dùng lại torch 2.14.0+cu130 của `../celltracking/.venv` qua `.pth`; cài thêm huggingface_hub, pyarrow, timm 1.0.30, scipy, matplotlib, pyyaml, pytest. Khác `environment.yml` của PHOEBI (Python 3.12, torch>=2.4); cần kiểm tra lại khi tái lập.
- GPU: 2× H200, cả hai đang bận ~99% bởi job khác (còn trống 1.5–3 GB). Smoke test chạy trên CPU.
- Smoke test trên synthetic (`scripts/make_synthetic.py`, cấu hình rút gọn `configs/smoke_*.yaml`, 2 seed, 300–400 step): pipeline chạy hết 5 fold, gộp, bootstrap và paired delta. Các con số **không phải bằng chứng** — xem `results/dev_synth/`.
- Kết quả smoke (delta=1.2, 5 fold, n=4): không method nào tách được bs/bt; AUROC của bt nằm trong khoảng 0.40–0.65 ở mọi method, kể cả PopEvidence. Oracle cho thấy phương sai trong bag bị chi phối bởi việc trộn các loài có mean khác nhau: độ phân tán tổng đo thành phần của bag chứ không đo bt.
- Positive control (delta=3.0, fold 0–1, n=4, 1 seed): AUROC của bt: mean_lin 0.56, pe_mean 0.44, meanstd_lin 0.93, pe_meanstd 1.00. Pipeline phát hiện được tín hiệu ngoài mean khi tín hiệu đủ mạnh, còn biểu diễn mean thì không. Đây là kiểm tra code, không phải bằng chứng cho H1.
- Hàm ý cho dữ liệu thật: nếu thông tin ngoài mean tồn tại nhưng yếu so với biến thiên do trộn loài và do phiên, probe tuyến tính trên std theo từng chiều có thể bỏ sót; ablation `pe_*` (chiếu theo từng species rồi mới pooling) là cần thiết, không chỉ là tùy chọn.

## 2026-10-28 — kiểm tra GPU

- Cả 2 H200 vẫn đang bận 99–100% (PID 1674423/1674424), còn trống 2.3 GB và 3.1 GB. Trên GPU 1 khi đang bị chia sẻ, DINOv2-S/14 bf16 đạt khoảng 13.6k tile/giây với đỉnh bộ nhớ 1.1 GB. Trích feature cho 1.92M tile chỉ mất khoảng 2–3 phút GPU, nên giới hạn thực tế là tốc độ đọc JPEG.
- Smoke `pe_mean`, `pe_meanstd` trên synthetic với `--device cuda` (GPU 1): khoảng 3 giây/fold khi huấn luyện, metric khớp với lần chạy CPU. Phần lớn thời gian 2 phút là bootstrap trên CPU.

## repro_phoebi fetch — 2026-10-29 01:50:50
- GPU: NVIDIA H200, 580.173.02
- torch: 2.14.0+cu130 13.0
- 2026-10-29 01:50:52 `/home/jovyan/bachdx2/DraftAnything/pop-evidence/.venv/bin/python tools/fetch_dataset.py` (phoebi@2a42be0)
  - done in 376 s

## repro_phoebi lco — 2026-10-29 01:57:29
- GPU: NVIDIA H200, 580.173.02
- torch: 2.14.0+cu130 13.0
- 2026-10-29 01:57:31 `/home/jovyan/bachdx2/DraftAnything/pop-evidence/.venv/bin/python experiments/run_phoebi_heldout.py --output_dir outputs/phoebi_heldout` (phoebi@2a42be0)
  - done in 822 s
- 2026-10-29 02:11:13 `/home/jovyan/bachdx2/DraftAnything/pop-evidence/.venv/bin/python experiments/run_phoebi_heldout.py --seed 1338 --output_dir outputs/phoebi_heldout_seed1338` (phoebi@2a42be0)
  - done in 620 s
- 2026-10-29 02:21:33 `/home/jovyan/bachdx2/DraftAnything/pop-evidence/.venv/bin/python experiments/run_phoebi_heldout.py --seed 1339 --output_dir outputs/phoebi_heldout_seed1339` (phoebi@2a42be0)
  - done in 707 s
- 2026-10-29 02:33:20 `/home/jovyan/bachdx2/DraftAnything/pop-evidence/.venv/bin/python experiments/aggregate_seeds.py` (phoebi@2a42be0)
  - done in 1 s

## 2026-10-29 — dữ liệu thật

- HF login DngBack; tải 120.000 ảnh (13 GB, 1024×1024). `results/audit.md`: không leakage; bs và bt chỉ cùng xuất hiện trong 1 trained combo nhưng có 2 test combo chứa cả hai; fold 4 không có dev combo chứa bt.
- Feature `cls`, `patchmean` (120000×16×384 fp16) trên GPU 0, khoảng 13 phút.
- Gate 1, LCO của PHOEBI (`repro_phoebi.sh lco`, GPU 1), F1 held-out theo seed 1337/1338/1339: SimplexUnmix 0.659/0.661/0.660, ProtoMatch 0.660/0.695/0.692, ChannelGroup 0.665/0.589/0.648. Seed 1337 của ProtoMatch khớp 0.6604 trong code của tác giả.

## repro_phoebi devcv — 2026-10-29 03:05:36
- GPU: NVIDIA H200, 580.173.02
- torch: 2.14.0+cu130 13.0
- 2026-10-29 03:05:38 `/home/jovyan/bachdx2/DraftAnything/pop-evidence/.venv/bin/python experiments/run_comp_stage0_granularity.py --output_dir outputs/compositional/stage0 --granularities cls224 patchmean224` (phoebi@2a42be0)

## repro_phoebi devcv — 2026-10-29 03:24:39
- GPU: NVIDIA H200, 580.173.02
- torch: 2.14.0+cu130 13.0
- 2026-10-29 03:24:41 `/home/jovyan/bachdx2/DraftAnything/pop-evidence/.venv/bin/python experiments/run_comp_stage0_granularity.py --output_dir outputs/compositional/stage0 --granularities cls224 patchmean224` (phoebi@2a42be0)

## repro_phoebi devcv — 2026-10-29 03:25:25
- GPU: NVIDIA H200, 580.173.02
- torch: 2.14.0+cu130 13.0
- 2026-10-29 03:25:29 `/home/jovyan/bachdx2/DraftAnything/pop-evidence/.venv/bin/python experiments/run_comp_stage0_granularity.py --output_dir outputs/compositional/stage0 --granularities cls224 patchmean224` (phoebi@2a42be0)

## 2026-10-29 03:5x — tạm dừng theo yêu cầu người dùng

- Đã dừng mọi job dev: probe cls report, popevidence_cls, popevidence_tile_cls, PHOEBI devcv (stage 0), BBBC041.
- Dự đoán đã lưu, có thể dùng lại (`run_dev.py` bỏ qua file đã có):
  - probe_cls: đủ 6 method × n = {1, 4}
  - popevidence_cls: pe_mean n = {1, 4}, pe_meanstd n = 1
  - popevidence_tile_cls: pe_quantile n = 1
  - bbbc041: mean_lin
- Chạy lại: `GPU=0 bash scripts/launch_all.sh`. BBBC041 chạy riêng bằng configs/bbbc041.yaml.

## 2026-10-29 06:0x — quyết định go/no-go

- PHOEBI, ràng buộc tile (λ = 0.3 và 1, n = 1): AUROC 0.558–0.565, không hơn pe_quantile (0.563) và không hơn meanstd_lin. **No-go cho hướng method theo độ chính xác.**
- BBBC041 (đơn vị ảnh): ràng buộc tile khôi phục định vị (trophozoite 0.076 → 0.878, schizont 0.090 → 0.837, CI không giao nhau) mà không mất AUROC ảnh. Rò rỉ ngữ cảnh tương đối không đổi (0.35 → 0.34).
- Hướng paper chuyển sang chẩn đoán "right for the wrong reasons" cộng khôi phục định vị. Bước tiếp theo: tái hiện trên PanNuke (dataset thứ ba, có instance mask), đang tải.

## repro_phoebi devcv — 2026-10-29 08:47:06
- GPU: NVIDIA H200, 580.173.02
- torch: 2.14.0+cu130 13.0
- 2026-10-29 08:47:14 `/home/jovyan/bachdx2/DraftAnything/pop-evidence/.venv/bin/python experiments/run_comp_stage0_granularity.py --output_dir outputs/compositional/stage0 --granularities cls224 patchmean224` (phoebi@2a42be0)

- 2026-10-29 09:23 LOCKED `locked_phoebi.yaml` sha256 bd71e2a6c882

- 2026-10-29 09:23 LOCKED `locked_bbbc041.yaml` sha256 d8a53196f14b

- 2026-10-29 09:23 LOCKED `locked_pannuke.yaml` sha256 c6fae0b11072
  - done in 2183 s
- 2026-10-29 09:23:37 `/home/jovyan/bachdx2/DraftAnything/pop-evidence/.venv/bin/python experiments/run_comp_nested_cv.py --granularities cls224 patchmean224` (phoebi@2a42be0)

- 2026-10-29 09:24 TEST RUN `bbbc041_v1` config@sha256:d8a53196f14b HEAD@15439af

- 2026-10-29 09:24 TEST RUN `phoebi_v1` config@sha256:bd71e2a6c882 HEAD@15439af

- 2026-10-29 09:38 TEST RUN `pannuke_v1` config@sha256:c6fae0b11072 HEAD@15439af

## 2026-10-29 ~10:00 — kết quả test đã khóa (đọc một lần)

- **BBBC041 test** (`bbbc041_v1`, acquisition khác):
  - AUROC ảnh 0.39–0.58. Tốt nhất là ABMIL + tile + CLA với 0.577.
  - Định vị ngược tái hiện (ABMIL gametocyte 0.08, ring 0.19; PopEvidence schizont 0.04, trophozoite 0.07).
  - Ràng buộc khôi phục định vị (0.86–0.98, trừ ring 0.19–0.24).
- **PanNuke test** (`pannuke_v1`, fold 3):
  - Deep Sets 0.893, mean 0.889 tốt nhất.
  - Ràng buộc tốn 0.01–0.03 AUROC, CLA tốn 0.026.
  - Định vị ngược tái hiện (dead 0.08, epithelial 0.18, connective 0.24); ràng buộc khôi phục (0.76–0.92).
- **ProtoMatch trên 5 dev fold của PHOEBI** (nested CV của PHOEBI): cosine/mean AUROC 0.545 [0.478, 0.614], cosine/q90 0.558.
  - done in 3758 s

- **PHOEBI test** (`phoebi_v1`, 3 partition, 27 combination, unit partition:combination). AUROC chuẩn hóa theo partition, n = 1 / n = 4:
  - abmil_cla1 0.679 / 0.683; mean_lin 0.667 / 0.677; meanstd 0.664 / 0.670; abmil 0.652 / 0.656; deepsets 0.644 / 0.630.
  - **tileonly_mean (phương pháp chính đã khóa): 0.600 / 0.600.** tileonly_q50 0.583 / 0.578.
  - AUROC gộp thô: abmil_cla1 0.671 / 0.673, mean_lin 0.654 / 0.659, tileonly_mean 0.561 / 0.551.
  - **Phương pháp chính thất bại:** so với meanstd −0.064 [−0.135, +0.004] (n = 1), P(Δ > 0) = 0.04. Sụp ở bt (0.35) và fj (0.29).
  - Theo partition (n = 1), tileonly / meanstd: 1337 0.650 / 0.667; 1338 (bs mất pure culture) 0.585 / 0.656; 1339 (mx mất pure culture) 0.589 / 0.708. Tile-only phụ thuộc pure culture. Dev fold (đều thuộc partition 1337, giữ đủ 5 singleton) không lộ ra điểm yếu này.
  - **CLA (so sánh cặp định trước, abmil_cla1 − abmil):** +0.027 [−0.000, +0.055], P = 0.974 (n = 1); **+0.028 [+0.005, +0.050], P = 0.994 (n = 4)**.
  - abmil_cla1 so với meanstd / mean: +0.007 đến +0.014, không có ý nghĩa.

- **PHOEBI test** (`phoebi_v1`, partition 1337/1338/1339, 27 combination, đọc một lần). AUROC n = 1 / n = 4 (chuẩn hóa thứ hạng theo partition):
  - ABMIL + CLA 0.671 / 0.673 (0.679 / 0.683): **tốt nhất**.
  - mean 0.654 / 0.659; meanstd 0.654 / 0.657; ABMIL 0.632 / 0.632; Deep Sets 0.634 / 0.618.
  - **tileonly_mean (phương pháp chính đã khóa) 0.561 / 0.551 (0.600 / 0.600): kém nhất**, đoán quá nhiều nhãn (4.1 so với 3.11; FPR ≈ 0.6).
  - **Phương pháp chính thất bại trên test; thứ hạng trên dev không chuyển sang test.** CLA trên ABMIL (+0.04) tái hiện mức đã thấy trên dev (+0.037).

- 2026-10-29 11:46 LOCKED `locked_phoebi4.yaml` sha256 696c617c7889
- 2026-10-29 11:47 phoebi4: ảnh là crop vuông 1003–1758 px (phoebi6: 1024 px). Giữ pipeline đã pre-register (tile 224 px ở độ phân giải gốc, lưới 4×4); độ phủ của lưới khác nhau giữa các ảnh.
- 2026-10-29 11:53 phoebi4 Amendment 1: check 1 thất bại do dịch chuyển batch (ánh xạ được dataset card xác nhận); tiếp tục với config đã khóa, chưa tính prediction nào.

- 2026-10-29 11:54 TEST RUN `phoebi4_v1` config@sha256:696c617c7889 HEAD@15439af
- 2026-10-29 12:29 TEST phoebi4_v1 hoàn tất. E1 không tái hiện; E2 và E3 đúng chiều; ABMIL + tile so với ABMIL +0.087 [0.010, 0.169] (thăm dò). Xem F13.

- 2026-10-29 12:30 LOCKED `explor_phoebi4_bn.yaml` sha256 75186cc64070

- 2026-10-29 12:30 TEST RUN `phoebi4_bn_exploratory` config@sha256:75186cc64070 HEAD@15439af
- 2026-10-29 12:47 THĂM DÒ phoebi4_bn_exploratory (post-hoc): chuẩn hóa theo batch, Δ ±0.02 không có ý nghĩa; ProtoMatch −0.07. Xem F14.
