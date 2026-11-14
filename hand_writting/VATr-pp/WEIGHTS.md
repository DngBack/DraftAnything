# Weights: đường dẫn để chủ sở hữu tự upload

Theo yêu cầu, checkpoint được để nguyên file để bạn tự upload. **Code không tuyên
bố các native weights đã được publish hoặc có URL công khai.**

## Thư mục upload trên máy nguồn

```text
/home/bachdx2/DraftAnything/hand_writting/VATr-pp/runs/weights_upload_261114/
```

Thư mục này chỉ chứa weights nguyên file, bundle đầu vào và metadata cần bàn giao,
không chứa các `.part` đã chia trước đó. Các file checkpoint là hardlink snapshot
ổn định; upload thư mục này không phụ thuộc việc `last.pth` của run dài tiếp tục
thay đổi. SHA-256 có trong `SHA256SUMS` và [artifact catalog](repro/artifacts.json).

| File | Vai trò |
|---|---|
| `baseline_175500.pth` | Recommended native baseline; nguồn pilot |
| `baseline_200000.pth` | Checkpoint cuối baseline 200k lịch sử |
| `baseline_100000.pth` | Nguồn các ablation sau 100k |
| `baseline_150000.pth` | Nguồn visual ablation và continuation lịch sử |
| `control_176500.pth` | Đối chứng sampling cũ |
| `component_176500.pth` | Sampling mới; nguồn continuation tới 250k |
| `component_176500_generator.pth` | Generator-only khoảng 152 MiB; chỉ inference |
| `component_snapshot_191000.pth` | Snapshot ổn định ở 191k, chưa phải final 250k |
| `ocr_best.pth` | OCR warmup tiếng Việt |
| `benchmark_ocr.pth` | OCR cố định của panel benchmark |
| `repro_inputs.tar.gz` | Manifest nguyên byte, charset/glyph/audit và fingerprint; không có HDF5 |

Các native full checkpoint khoảng 1,375 GB chứa G/D/W/R, Adam/RNG, config,
manifest SHA. Generator-only và benchmark-only không dùng để resume training.
[Danh sách đường dẫn đầy đủ](repro/UPLOAD_FILES.txt).

## Checkpoint đang tiếp tục cập nhật

```text
/home/bachdx2/DraftAnything/hand_writting/VATr-pp/runs/component_continuation_250k_v1/last.pth
/home/bachdx2/DraftAnything/hand_writting/VATr-pp/runs/component_continuation_250k_v1/best.pth
/home/bachdx2/DraftAnything/hand_writting/VATr-pp/runs/component_continuation_250k_v1/best_focus.pth
```

Các đường dẫn trên có thể thay file khi training; dùng snapshot/thư mục upload
ổn định để copy. Không có final 250k trước khi run thực sự hoàn thành. Mốc 98.500
lịch sử không giữ weights, nên không cung cấp link giả cho mốc đó.

## Phục hồi trên máy khác

Sau khi bạn upload/chuyển file, giữ nguyên tên, đưa vào một thư mục downloads:

```bash
python -m repro.fetch --list
python -m repro.fetch --id repro_inputs component_176500_generator \
  --downloads /path/to/downloaded-weights
```

Script kiểm tra size/SHA-256 rồi phục hồi vào đường dẫn canonical. Bạn có thể bổ
sung URL tải do mình quản lý vào catalog sau khi upload; hiện chỉ English upstream
có URL công khai pinned, lấy từ metadata gốc.

Một phần các part thử nghiệm đã được upload vào Drive trước khi người dùng chỉnh
hướng; hoạt động upload đã dừng. Chúng không phải một bản phân phối đầy đủ và
không được dùng làm mặc định trong catalog này.

Hướng dẫn môi trường, dữ liệu, đổi path và recipe: [REPRODUCE.md](REPRODUCE.md).
