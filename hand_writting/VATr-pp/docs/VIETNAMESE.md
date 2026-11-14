# VATr++ cho tiếng Việt: audit dữ liệu và cách chạy

> **Cập nhật training:** OCR đã được huấn luyện 20.000 bước và đã khởi chạy fine-tune G pretrained. Xem [nhật ký training mới](261111_report_02_Nhat_ky_training_tieng_Viet.md). Các số 51/4 bước bên dưới là snapshot smoke trước đó.

Giải thích đầy đủ kiến trúc, công thức loss, lịch bốn pha và danh sách thay đổi:
[Thuật toán và báo cáo triển khai](261111_report_01_Thuat_toan_va_trien_khai_VATr_pp_tieng_Viet.md).

## Đây là VATr++, không phải SpiS-GAN

Mọi thay đổi nằm trong `hand_writting/VATr-pp`. `train_vietnamese.py` nhập trực tiếp
`Generator`, `Discriminator`, `WDiscriminator`, `CRNN` và `strLabelConverter` của
repo VATr-pp. Không import mạng hoặc trainer SpiS-GAN. Đường dẫn SpiS-GAN chỉ cung
cấp **dữ liệu HDF5**, và môi trường Python có đủ thư viện đã được dùng để kiểm thử.
Không sửa dữ liệu hoặc mã ở dự án nguồn.

Mặc định giữ kiến trúc VATr gốc (visual archetypes Unifont, ResNet18, Transformer,
CNN decoder, ảnh 32px), cùng lịch cập nhật trong `train.py`:

1. G: adversarial + OCR, cân bằng gradient.
2. D: hinge loss; OCR: CTC trên ảnh thật.
3. G: adversarial + writer loss, cân bằng gradient.
4. D: hinge loss; writer classifier: cross-entropy trên ảnh thật.

Mỗi pha sinh lại nội dung/ảnh như vòng train gốc. OCR warmup tiếng Việt là một giai
đoạn bổ sung để tránh dùng OCR chưa biết dấu làm bộ giám sát. Khi vào GAN, mặc định
OCR vẫn tiếp tục học ảnh thật; `--freeze-ocr` là ablation. Crop discriminator được
bật mặc định. Không bật cycle loss trong trainer Việt.

## Dữ liệu đã được kiểm tra

Nguồn: `/home/giaphk/SpiS-GAN/data/VNOnDB_ViHTGen_HAT2023`.
Đối chiếu ý nghĩa ID với tài liệu gộp dữ liệu của dự án nguồn, sau đó tự quét toàn
bộ 212.597 mẫu train và 27.257 mẫu test.

HDF5 gồm `imgs`, `img_seek_idxs`, `img_lens`, `lbs`, `lb_seek_idxs`, `lb_lens`,
`wids`. Ảnh grayscale uint8 được nối theo chiều rộng; nhãn là mảng Unicode code
point nối tiếp. Đã kiểm tra seek liên tiếp, số phần tử, độ dài dương và điểm cuối
mảng. Toàn bộ ảnh cao 32px. Test không chứa HAT2023; ID test có không gian riêng.

| Nguồn train | Số mẫu | Nhóm writer | Rộng min / median / max | Nhãn median / max | Ký tự khác nhau | Nhãn toàn chữ hoa |
|---|---:|---:|---|---|---:|---:|
| VNOnDB | 66.955 | 106 writer thật | 16 / 48 / 176 | 3 / 11 | 144 | 1.136 |
| ViHTGen | 42.642 | 189 writer thật | 3 / 36 / 543 | 3 / 8 | 157 | 648 |
| HAT2023 | 103.000 | 103.000 ID giả theo ảnh | 6 / 55 / 245 | 4 / 17 | 187 | 21.782 |

Test có 27.257 mẫu, 68 ID, rộng median 48px, max 338px và 147 ký tự khác nhau.
Không nhãn nào thay đổi khi chuẩn hóa NFC. Pipeline vẫn chuẩn hóa NFC ở mọi đầu
vào để không phụ thuộc cách một chuỗi có dấu được mã hóa.

### Những phát hiện ảnh hưởng trực tiếp đến cách train

- **Không được xem HAT2023 là 103.000 writer.** ID 295+i được cấp theo ảnh, mỗi ID
  chỉ có một mẫu. Không thể suy ra nhóm cùng người viết. Dùng trực tiếp sẽ làm
  writer head sai ý nghĩa, đồng thời support chỉ là bản lặp của một ảnh.
- VNOnDB–ViHTGen dùng ID train 0–294. Chia development theo 30 writer thật: 11
  VNOnDB và 19 ViHTGen, seed 742. Train còn 265 writer. Phân loại writer được remap
  về 0–264; ID HAT không đi vào writer head, support hay ảnh thật của GAN.
- Dữ liệu có **26 nhóm pixel giống nhau nhưng nhãn khác nhau**, tổng 53 mẫu, đều
  trong HAT. Cách ly cả nhóm; không tự chọn nhãn đúng. Danh sách được lưu tại
  `files/vietnamese/ambiguous_labels.json`.
- Sau cách ly nhãn mâu thuẫn, loại 75 bản ảnh lặp trong phần train thật và 406
  bản lặp trong HAT. So sánh theo kích thước + SHA-256 pixel, không phụ thuộc nhãn.
  Không tìm thấy ảnh trùng pixel giữa train/development/test trong lần quét này.
  Đây không phải kiểm tra near-duplicate hay bằng chứng người viết HAT độc lập.
- Theo đúng CNN OCR VATr 32px, số timestep là `floor(width/4)+1`. Độ dài CTC tối
  thiểu là số ký tự cộng số cặp ký tự lặp liền nhau. Có 2 mẫu ViHTGen, 8 HAT và
  1 test không đủ timestep. Số này khác audit của SpiS-GAN vì kiến trúc OCR khác.
- Charset lấy từ **train đã lọc**, không lấy từ development/test: 207 ký tự,
  gồm 186 chữ cái (52 ASCII + 134 chữ Việt ngoài ASCII), 10 chữ số và 11 ký tự
  còn lại gồm khoảng trắng/dấu câu. Có đủ glyph Unifont cho tất cả. Mọi ký tự
  trong development và test đều nằm trong charset này.
- Có **44 ký tự chỉ HAT mới cung cấp trong phần train**, chủ yếu chữ hoa có dấu,
  ví dụ `Ắ`, `Ễ`, `Ỡ`, `Ự`, `Ỹ`, cùng `Z` và `ỵ`. G có thể học chúng qua glyph và
  OCR, nhưng không có mẫu cùng writer thật để giám sát phong cách trực tiếp.
  Coverage OCR không bảo đảm sinh đúng phong cách cho các ký tự này.
- Độ lệch tần suất rất lớn: trong train thật, `Ố` và `Ồ` chỉ xuất hiện một lần;
  nhiều chữ hoa khác chỉ có 1–4 lượt. HAT bổ sung nhiều chữ hoa nhưng `Ỹ` vẫn chỉ
  có 13 lượt trước lọc. Giữ báo cáo tần suất theo nguồn và split để đánh giá lỗi
  theo nhóm chữ, thay vì chỉ dùng một CER tổng.
- Các dấu `! ? ; " '` không có trong charset train. Inference báo lỗi rõ ràng
  nếu được yêu cầu sinh, không âm thầm xóa. Muốn hỗ trợ cần thêm dữ liệu có nhãn
  tương ứng và tạo lại charset; chỉ thêm glyph không tạo ra bằng chứng đã học.
- Xem trực tiếp contact sheet: VNOnDB có nền trắng và nét đen rõ; ViHTGen có kẻ
  dòng/chấm và nền không đều; HAT có nhiều nền xám, crop sát. Tỷ lệ mực sát mép
  cao của HAT còn chịu ảnh hưởng nền tối, **không được xem là tỷ lệ mất dấu**.
- Ảnh đã giảm xuống 32px trước khi nhập. Đổi sang 64px bằng upsample không khôi
  phục dấu đã mất. Baseline giữ 32px; muốn nghiên cứu 64px nên lấy lại ảnh gốc.

### Manifest cuối

| Phần | Mẫu giữ lại | Cách dùng |
|---|---:|---|
| Train writer thật | 97.734 | G, D, writer; OCR và corpus |
| Development writer thật | 11.786 | 30 writer không thuộc train thật |
| HAT OCR | 102.533 | OCR warmup và corpus sinh, không dùng writer/style |
| Test | 27.256 | Tập đánh giá sau lọc CTC, không dùng chọn model |

Tổng OCR warmup là 200.267 mẫu. Chia nguồn đều xác suất (VNOnDB, ViHTGen, HAT)
để HAT không chiếm ưu thế chỉ vì số ảnh lớn. GAN chọn đều nguồn thật, sau đó chọn
đều writer trong nguồn. Support lấy cùng writer, ưu tiên transcript khác target
và không lấy lại ảnh target. Có thể lặp support nếu không đủ 15 mẫu phù hợp.

Giữ chiều rộng tự nhiên cho OCR warmup. GAN đưa ảnh thật về 16px/ký tự để khớp
quy ước ảnh sinh của VATr. Support vượt 192px được nén ngang toàn ảnh, không cắt
bỏ phần đuôi. Không thêm morphology, xoay/crop dọc hoặc blur mạnh trong cấu hình
ban đầu, tránh làm hỏng các chi tiết dấu.

Corpus chỉ lấy transcript train. Mặc định mỗi lần lấy nhãn sinh: 35% HAT,
65% writer thật; trong từng nguồn, 30% dùng trọng số ưu tiên ký tự hiếm, phần
còn lại dùng tần suất thực nghiệm. Không thay ký tự ngẫu nhiên thành tổ hợp dấu
không tự nhiên. Đây là thay đổi sampling cho miền tiếng Việt; không tuyên bố
đó là bản sao từng hyperparameter của thí nghiệm IAM trong paper.

## Những sửa đổi tính đúng đắn

- OCR có `K+1` lớp gồm CTC blank; nhận cả charset thống nhất.
- OCR trả `log_softmax`, đúng yêu cầu CTC; dùng input lengths theo từng ảnh thay
  vì chiều rộng padding cả batch. Các alignment không thể thực hiện được báo lỗi.
- Mỗi loss tổng chỉ backward một lần, sau khi tính hệ số cân bằng gradient.
- Giữ gradient qua OCR/writer khi học G nhưng khóa tham số các mạng giám sát.
- Lưu charset, cấu hình, manifest fingerprint, model, optimizer và RNG vào
  checkpoint. Kiểm tra nguồn dữ liệu qua size/mtime và checksum file glyph.
- Loader HDF5 mở handle theo process để dùng được với DataLoader workers; không
  chuyển cả ảnh thành pickle hoặc sửa file nguồn.
- Glyph là registered buffer không persistent, không gây sai device khi `.to()`.
- Inference chuẩn hóa NFC, kiểm tra charset và lưu support cùng ảnh để kiểm chứng.
- Sửa writer ID offset khi ghép nhiều dataset trong loader cũ; ngắt graph của
  fake history và sửa đường gradient cycle trong lớp VATr cũ.

`--adapt-generator-layout` là **thử nghiệm tùy chọn**: đổi cách sắp xếp feature
style và cộng positional encoding vào query glyph. Mặc định không bật, để giữ
hành vi kiến trúc gốc và thuận tiện nạp pretrained VATr. Không được so sánh hai
cấu hình như thể cùng kiến trúc; lưu flag trong checkpoint và kiểm tra khi resume.

Trainer gốc `train.py` vẫn tồn tại cho IAM/pickle. Dùng `train_vietnamese.py` cho
HDF5 vì cần tách writer thật và HAT. Checkpoint mới dùng `generate_vietnamese.py`;
không nạp qua loader legacy ánh xạ key theo thứ tự.

## Môi trường và lệnh chạy

Chạy từ thư mục `hand_writting/VATr-pp`.

Một môi trường riêng có thể được tạo như sau (chọn PyTorch phù hợp driver):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-vietnamese.txt
```

Trong phiên triển khai đã dùng Python có sẵn tại
`/home/giaphk/SpiS-GAN/.venv/bin/python` chỉ như interpreter chứa PyTorch 2.9.1,
torchvision 0.24.1, h5py/OpenCV. Không dùng checkpoint hay mã mô hình SpiS-GAN.
Python mặc định của máy thiếu OpenCV nên không chạy trực tiếp trước khi cài đủ.
Không cần wandb, streamlit hoặc tải Inception/ImageNet khi dùng trainer Việt.

### 1. Audit và chuẩn bị dữ liệu

```bash
python scripts/prepare_vietnamese.py \
  --data /home/giaphk/SpiS-GAN/data/VNOnDB_ViHTGen_HAT2023 \
  --glyphs ../VATr/files/unifont.pickle \
  --output files/vietnamese
```

`../VATr/files/unifont.pickle` là tài nguyên font đã có ở workspace, không phải
mô hình khác. Có thể truyền đường dẫn Unifont tương thích khác. Công cụ trích
riêng glyph cần thiết, ghi manifest, audit, CSV, corpus và contact sheet. File
nguồn chỉ được đọc. Glyph ngoài charset sẽ gây lỗi; không dùng glyph trống thay
thế một cách âm thầm.

### 2. Warmup OCR tiếng Việt

```bash
python train_vietnamese.py --stage ocr \
  --manifest files/vietnamese/manifest.json \
  --output runs/vatrpp_vn_ocr \
  --steps 20000 --batch-size 16 --workers 2 \
  --eval-every 1000 --save-every 1000
```

`validation_*.json` ghi CER có dấu, CER bỏ dấu và transcript error rate. Mặc định
đánh giá toàn bộ development; chỉ dùng `--eval-batches` để smoke test. `best.pth`
được chọn theo CER development. Cần xem lỗi dấu và ví dụ dự đoán trước khi cho
OCR hướng dẫn G; 20.000 bước là cấu hình xuất phát, không bảo đảm hội tụ.

### 3. Train VATr++

```bash
python train_vietnamese.py --stage gan \
  --manifest files/vietnamese/manifest.json \
  --ocr-checkpoint runs/vatrpp_vn_ocr/best.pth \
  --output runs/vatrpp_vn_gan \
  --steps 50000 --batch-size 8 --num-examples 15 --workers 2
```

Số bước ở đây là outer step: mỗi bước gồm 2 lần cập nhật G, 2 D, 1 OCR và 1
writer classifier. 50.000 bước là gợi ý điểm xuất phát, chưa được xác nhận đủ.
Checkpoint mặc định lưu `last.pth`; preview ảnh không phải đánh giá chất lượng.

Trong thư mục VATr-pp hiện không có checkpoint generator/Font Square pretrained.
Smoke test dùng G khởi tạo từ đầu. Để fine-tune pretrained **VATr/VATr++** thêm:

```bash
# Chọn một trong hai, không dùng đồng thời:
--generator-checkpoint /path/to/vatrpp.pth
--encoder-checkpoint /path/to/resnet_18_pretrained.pth
```

Nạp G theo tên key và shape, ghi rõ loaded/missing trong `config.json`; không
nạp OCR/writer head tiếng Anh. Nạp encoder grayscale hóa kernel RGB như train
code gốc. Mặc định kiến trúc/layout gốc phù hợp ablation pretrained. Learning
rate encoder 1e-5, phần G còn lại 5e-5, D 1e-5; đều có CLI để điều chỉnh.

Không dùng checkpoint OCR 51 bước smoke bên dưới cho một run chất lượng.
`--allow-untrained-ocr` chỉ dành cho kiểm tra kỹ thuật khi không có OCR checkpoint.

### 4. Resume

```bash
python train_vietnamese.py --stage gan \
  --output runs/vatrpp_vn_gan \
  --resume runs/vatrpp_vn_gan/last.pth \
  --steps 100000 --batch-size 8 --num-examples 15 --workers 2
```

`--steps` là tổng bước, gồm cả bước đã chạy. Stage, manifest và các tùy chọn
kiến trúc/loss chính phải khớp. Model, optimizer và RNG được khôi phục; iterator
của sampler khởi động lại, vì vậy không hứa hẹn khớp bit với một run không dừng.
Không ghi đè run đã có checkpoint nếu chưa chỉ định resume.

### 5. Sinh chữ tiếng Việt

Writer development đầu tiên của manifest hiện tại là ID 35:

```bash
python generate_vietnamese.py \
  --checkpoint runs/vatrpp_vn_gan/last.pth \
  --writer 35 --split val \
  --text 'Tiếng Việt rất đẹp' --output runs/vatrpp_vn_gan/example.png
```

Hoặc dùng `--style-folder /path/to/style_images` thay `--writer`. Dùng cả ảnh
người dùng cung cấp; không cần nhãn cho style folder. Văn bản sinh theo từng đơn
vị ngăn bằng khoảng trắng rồi ghép, đúng pipeline word-level hiện có. Ảnh đầu
ra cao 32px. Sidecar JSON và `.references.png` ghi chính xác support đã dùng.

### 6. Ablation cho giả thuyết distribution shift

Giữ cùng manifest, seeds, bước và panel khi so sánh:

- G scratch với G pretrained VATr++, cùng OCR Việt.
- Có HAT so với `--no-hat` ở **cả OCR warmup và GAN**. Không cho dùng OCR có HAT
  trong run ghi là `--no-hat`.
- `--hat-fraction 0` so với 0.35 để đo riêng corpus HAT; khác với bỏ HAT khỏi OCR.
- `--rare-fraction 0` so với 0.3 để đo sampling ký tự hiếm.
- `--freeze-ocr` để đo OCR cố định so với tiếp tục học ảnh thật.
- `--adapt-generator-layout` chỉ ở một thí nghiệm kiến trúc riêng.

Ưu tiên CER bằng **OCR tiếng Việt độc lập** để đánh giá ảnh G, kèm CER bỏ dấu,
nhóm chữ hoa, lỗi từng dấu và khả năng giữ phong cách của writer chưa thấy.
OCR nội bộ dùng huấn luyện không phải bộ đánh giá độc lập. Chưa có benchmark
định lượng chất lượng G được chạy trong phiên triển khai này. Chưa chứng minh
HAT và các split có writer thật độc lập vì không có metadata writer HAT.

## Kiểm chứng đã thực hiện

- `python -m unittest discover -s tests -v`: 4 test qua, gồm NFC/OOV, lớp blank,
  log-probabilities, CTC lặp ký tự, true widths, gradient và support writer thật.
- Quét đủ 239.854 mẫu, kiểm tra toàn bộ metadata, pixel hashes và coverage glyph.
- OCR 50 bước + resume lên 51 bước, batch 16, loss hữu hạn. CER sau 51 bước trên
  **64 mẫu đầu development** là 0.97156: OCR còn rất kém, chưa đủ dùng cho G chất
  lượng. Không diễn giải số đo trên 64 mẫu thành kết quả toàn development.
- VATr++ 3 outer step + resume lên 4, batch 2, **15 support**, đủ bốn pha, cập nhật
  OCR/writer, cân bằng gradient và discriminator crop; loss/gradient hữu hạn.
- Nạp checkpoint G rồi sinh `Tiếng Việt rất đẹp` với writer development 35; lưu
  ảnh tham chiếu và sidecar. Ảnh smoke là kiểm tra chạy được, chưa phải chữ đã học.
- GPU H200, PyTorch 2.9.1+cu128. Sau resume GAN một bước, peak allocated khoảng
  2.19 GiB, reserved 2.30 GiB. Không ngoại suy thành bảo đảm VRAM cho batch 8/ảnh
  dài/run dài. Profile lưu theo invocation, lần resume ghi đè profile trước.

Các artifact cuối nằm tại `runs/vatrpp_vn_ocr_smoke/` và
`runs/vatrpp_vn_gan_smoke/`. Các thư mục `vn_*` khác là thử nghiệm triển khai
trước khi chốt manifest/lịch train; không dùng chúng cho training tiếp hoặc báo cáo.

Dữ liệu sinh và checkpoint được gitignore. Mã, hướng dẫn và test chưa được commit;
không sửa các thay đổi có sẵn ở phần khác của workspace.
