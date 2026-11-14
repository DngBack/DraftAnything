# Tái lập VATr++ tiếng Việt

Tài liệu này là điểm bắt đầu cho một máy mới. Code, mô hình, trainer, test, report
và các script nghiên cứu đều nằm trong repo chính; không cần Git submodule.
[Provenance](../UPSTREAM.json) ghi commit nguồn. [WEIGHTS.md](WEIGHTS.md) và
[artifact catalog](repro/artifacts.json) ghi checkpoint, link thực, kích thước và
SHA-256, phân biệt full-resume với generator-only.

## 1. Phạm vi tái lập

Có thể tái lập inference với style folder riêng mà không cần dataset train; hoặc
resume các checkpoint native, chạy lại pilot đối chứng và continuation tới tổng
250.000 outer steps khi có đúng HDF5 nguồn. Code exploratory cũng được cung cấp,
nhưng các generator/recognizer nhỏ đó không phải model đang train 250k.

**Phân phối weights:** theo yêu cầu, chủ sở hữu tự upload checkpoint nguyên file.
Đường dẫn trên máy nguồn và SHA-256 nằm trong WEIGHTS.md / artifact catalog.
Chưa có URL công khai cho các native weights và không giả định chúng đã được
publish. Tải/nhận các file từ chủ sở hữu rồi dùng --downloads. Snapshot giữa run
không được gọi là final 250k.

## 2. Môi trường

Runtime gốc: Python 3.13.15, PyTorch 2.9.1+cu128, torchvision 0.24.1+cu128, GPU H200.
Các dependency khác được pin trong `requirements-repro.txt`. Không trộn file
`requirements.txt` cũ của đường legacy vào môi trường này.

```bash
git clone https://github.com/DngBack/DraftAnything.git
cd DraftAnything/hand_writting/VATr-pp
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-repro.txt
```

Pin commit Git được cung cấp cùng bản bàn giao hoặc commit chứa bộ artifact
catalog này thay vì dùng một `main` đã thay đổi. [environment.json](repro/environment.json)
ghi runtime; không cam kết bit-identical trên mọi GPU/driver. Khi resume, optimizer
và RNG được khôi phục nhưng iterator dữ liệu khởi động lại.

## 3. Tải và phục hồi artifact

Nhận/tải checkpoint nguyên file từ chủ sở hữu và giữ tên như trong WEIGHTS.md.
Các file hiện được chuẩn bị ở runs/weights_upload_261114 trên máy nguồn.
Nếu dùng các part thử nghiệm cũ thì không mở một part bằng torch.load; catalog
hiện tại ưu tiên nguyên file do chủ sở hữu tự upload.

```bash
python -m repro.fetch --list
python -m repro.fetch --id repro_inputs component_176500_generator \
  --downloads /path/to/Drive-download-folder
```

Script kiểm tra size/SHA-256 từng part, ghép đúng thứ tự, kiểm tra SHA-256 toàn
file rồi mới đặt tên cuối cùng. Bundle đầu vào được unpack có kiểm tra traversal;
không ghi đè file khác nội dung. Checkpoint full và inference-only không thay thế
cho nhau. Các file PyTorch/pickle trong workflow này phải đến từ artifact đáng
tin đã xác minh, không dùng file nguồn không rõ.

`repro_inputs` chứa manifest gốc, charset/glyph subset, audit, fingerprint dữ liệu
và glyph nguồn. **Không chứa ảnh HDF5 train/test.** Nó giữ nguyên byte manifest
để checkpoint cũ tiếp tục kiểm tra fingerprint đúng.

Public upstream English G dùng URL pinned và hash trong catalog, có thể tải bằng:

```bash
python -m repro.fetch --id english_generator --download
```

## 4. Inference trên style folder riêng — không cần HDF5

```bash
python -m repro.configure
export VATRPP_PATHS_JSON="$PWD/.local/repro_paths.json"
python generate_vietnamese.py \
  --checkpoint artifacts/component_176500_generator.pth \
  --manifest files/vietnamese/manifest.json \
  --style-folder /path/to/style-images \
  --text 'NGUYỄN' --output outputs/nguyen.png --device cuda
```

Folder style chứa PNG/JPG grayscale hoặc ảnh có thể chuyển grayscale. Script
chọn 15 support, resize cao 32 và pad/nén ngang về 192. Kết quả kèm ảnh reference
và JSON provenance. Ký tự ngoài charset báo lỗi thay vì âm thầm bỏ dấu.

[Smoke inference ở thư mục sạch](repro/fresh_inference_smoke.json) đã chạy CPU,
khôi phục artifact, dùng glyph ở vị trí mới và chặn mọi lần mở HDF5. Đây là kiểm
tra kỹ thuật bằng style tổng hợp, không phải chứng minh chất lượng viết tay.

## 5. Dữ liệu cho training / benchmark writer

Cần đúng bản packed HDF5 VNOnDB + ViHTGen + HAT2023 đã được cấp quyền sử dụng:

- `train.h5`: 212.597 mẫu thô.
- `test.h5`: 27.257 mẫu thô.
- Schema: `imgs`, `img_seek_idxs`, `img_lens`, `lbs`, `lb_seek_idxs`, `lb_lens`, `wids`.
- [SHA-256 và kích thước chính xác](repro/data_fingerprints.json).

Bản gộp HDF5 này không được redistribute trong code/weights release và không có
URL công khai được xác nhận ở đây. Dùng bản được cấp quyền trên máy nguồn hoặc
chuẩn bị dữ liệu tương đương bằng pipeline của nguồn; dataset khác hash không
được coi là tái lập đúng split/checkpoint này. Không dùng dữ liệu test để bổ sung
charset hay học OCR.

```bash
python -m repro.configure --data-dir /path/to/packed-hdf5
export VATRPP_PATHS_JSON="$PWD/.local/repro_paths.json"
```

Overlay thay đường dẫn trong bộ nhớ, kiểm tra content hash thay vì phụ thuộc
mtime của file đã copy. Nó **không sửa manifest gốc** nên raw manifest SHA khớp
checkpoint. Không sửa tay `source_train`, `source_test`, `glyph_path` trong
manifest rồi dùng checkpoint cũ với hash mới.

Nếu cần dựng lại audit từ đầu cho một dataset mới:

```bash
python scripts/prepare_vietnamese.py \
  --data /path/to/packed-hdf5 --glyphs files/unifont_source.pickle \
  --output files/vietnamese_rebuilt --seed 742
```

Manifest mới có path/mtime mới sẽ có SHA khác. Đây là một split/artifact mới;
resume checkpoint lịch sử phải dùng manifest gốc kèm overlay. HAT là nguồn OCR/
transcript, không phải writer thật cho style support.

## 6. Chạy lại pilot sampling cũ / mới từ cùng pretrained

Tải `baseline_175500`, `benchmark_ocr` và `repro_inputs` trước. Dùng output mới,
không overwrite checkpoint đã tải.

```bash
python -m repro.fetch --id baseline_175500 benchmark_ocr --downloads /path/to/parts
python -m repro.run --recipe pilot_legacy --output runs/reproduced_legacy
python -m repro.run --recipe pilot_component --output runs/reproduced_component
```

Hai lệnh trên in argv để kiểm tra. Thêm `--execute` để thực sự training:

```bash
python -m repro.run --recipe pilot_legacy --output runs/reproduced_legacy --execute
python -m repro.run --recipe pilot_component --output runs/reproduced_component --execute
```

Nguồn 175.500 → tổng 176.500; batch/support 8/15; LR encoder/G/D/writer =
5e-6/2e-5/1e-5/2e-5; OCR frozen, OCR/writer weight 4/0,1, HAT/rare fraction
0,35/0,3. Chỉ `--text-sampler` khác. [Recipes đầy đủ](repro/recipes.json).

Mốc tham chiếu của pilot gốc, không phải score được bảo đảm cho mọi máy:

| Mốc | CER supervisor | Exact |
|---|---:|---:|
| Nguồn 175.500 | 13,6257% | 62,5% |
| Legacy 176.500 | 17,9532% | 52,5% |
| Component 176.500 | 12,6901% | 65,0% |

OCR là supervisor proxy, không phải điểm đẹp/style độc lập. Ễ đứng riêng vẫn
0/30 tại pilot; không xem sampler như lời giải đã hoàn tất.

## 7. Resume nhánh component tới tổng 250k

```bash
python -m repro.fetch --id component_176500 benchmark_ocr --downloads /path/to/parts
python -m repro.run --recipe continuation_250k \
  --output runs/reproduced_component_250k --execute
```

Đây là thêm 73.500 outer steps từ 176.500, không phải thêm 250.000 bước mới.
G/D/W/R và Adam được restore; checkpoint/panel mỗi 1.000 bước. Chọn last/best
và review ảnh, không tự coi checkpoint cuối tốt hơn nguồn.

Các supervisor cũ trong `scripts/` giữ lịch sử/tên output cố định; trên máy mới
nên dùng `repro.run` đã thay path và output. Không chạy lại launcher cũ trỏ vào
thư mục đang chứa weights tải về. Cần chạy nền trên server thì redirect log và
dùng tmux/systemd/nohup theo môi trường của bạn.

## 8. Kiểm thử và khả năng kiểm toán

```bash
python -m unittest discover -s tests -v
```

[Smoke resume ở thư mục sạch](repro/fresh_resume_smoke.json) đã copy HDF5 sang đường
dẫn mới với mtime mới, khôi phục baseline CER 12,6901%, chạy thật 176.500→176.502
và xác nhận Adam tăng đúng G/D +4, writer +2.

Bộ test được chạy 44 tests pass sau khi thêm relocation, kiểm tra part assembly,
checksum corruption, traversal và inference không cần HDF5. Code vendored được
kiểm tra không có gitlink/submodule, không thiếu `.py`. [Log](repro/test_results.log).

Các report và model thử nghiệm: [mục lục](docs/README.md). Các link `runs/...`
trong report lịch sử trỏ tới artifact trên filesystem, không có nghĩa hàng chục
GB run outputs nằm trong Git. Weights quan trọng và input bundle có catalog;
những ảnh/metadata chưa đóng gói cần tải/generate lại theo run tương ứng.

Không gọi các checkpoint đã có exposure là zero-shot. Nhánh sạch trong
`research/longtail` có protocol/data fingerprint riêng; phải export và thực thi
lọc mọi image/OCR/corpus/support trước khi train mới, không đổi nhãn một run
pretrained thành run sạch.
