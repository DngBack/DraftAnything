# Hướng dẫn dữ liệu: suy luận tổ tiên–hậu duệ từ ảnh tế bào chụp thưa

Ngày kiểm tra nguồn: 24/09/2026. Phục vụ đề xuất CVPR 2027 đã thống nhất.

Đây là hướng dẫn triển khai và protocol đề xuất, chưa phải báo cáo đã xử lý dữ liệu. Đã kiểm tra trang dữ liệu, trang preview ZIP và tài liệu định dạng; chưa tải toàn bộ ZIP, kiểm tra từng movie hoặc chạy benchmark. Các ví dụ số/ID bên dưới là ví dụ tự tạo, không phải mẫu trích từ dataset.

## 1. Chốt phạm vi dữ liệu đầu tiên

Mục tiêu phiên bản v0: nhận ảnh và các vị trí tế bào tại những thời điểm được quan sát; dự đoán mỗi tế bào ở mốc sau thuộc về tế bào nào tại mốc tham chiếu trước. Có thể bổ sung câu hỏi hai tế bào ở mốc sau có cùng mẹ trực tiếp hay không.

Đơn vị dữ liệu gồm ba cấp:

- **Movie/sequence:** một chuỗi ảnh của cùng một vùng quan sát. Đây là đơn vị chia train/validation/test tối thiểu.
- **Cell observation:** một tế bào xuất hiện trong một ảnh. Một tế bào tồn tại trong 20 ảnh tạo 20 observations, chưa phải 20 tế bào độc lập.
- **Track:** đoạn tồn tại của một tế bào trước khi phân chia hoặc kết thúc quan sát. Các track được nối bằng quan hệ mẹ–con thành lineage.

Dataset cung cấp ảnh và nhãn. Benchmark là cách chọn input, đáp án, split và phép chấm trên dataset. Chúng ta dùng dataset có sẵn để xây một benchmark nghiên cứu riêng; không gọi benchmark này là bài thi chính thức của Cell Tracking Challenge (CTC).

**Bắt đầu với One-in-a-Million.** Dùng SIM+ khi cần kiểm tra parser. Sau khi pipeline chạy đúng mới thêm HSC/MuSC; HeLa là kiểm tra bổ sung. Không cần tải tất cả trước khi biết bộ đầu tiên có đủ trường hợp nghiên cứu.

## 2. Tải đúng những gì cần thiết

| Ưu tiên | Bộ dữ liệu | Gói cần lấy | Dung lượng ZIP công bố | Khoảng chụp gốc |
|---|---|---|---:|---:|
| 1 | One-in-a-Million, C. glutamicum | ctc_format.zip | 2,2 GB | 1 phút |
| 2 | Fluo-N2DH-SIM+ | Training dataset | 91 MB | 29 phút |
| 3 | BF-C2DL-HSC | Training dataset | 1,6 GB | 5 phút |
| 4 | BF-C2DL-MuSC | Training dataset | 1,2 GB | 5 phút |
| 5 | Fluo-N2DL-HeLa | Training dataset | 182 MB | 30 phút |

Nguồn: [Zenodo One-in-a-Million](https://zenodo.org/records/7260137), [trang CTC 2D+Time](https://celltrackingchallenge.net/2d-datasets/). Các kích thước là số làm tròn trên website, không phải dung lượng giải nén. SIM+ là dữ liệu mô phỏng, không đại diện bằng chứng sinh học độc lập.

One-in-a-Million công bố 5 movies, hơn 1,4 triệu observations, khoảng 29.000 tracks và 14.000 divisions. Không dùng các counts này thay cho audit thực tế hoặc số mẫu sinh học độc lập. Preview ZIP hiện hiển thị thư mục `00` và tên ảnh dạng `t0000.tif`, nhưng cảnh báo không liệt kê hết file; không suy đoán tên toàn bộ năm thư mục từ preview.

Liên kết tải trực tiếp:

- [One-in-a-Million ctc_format.zip](https://zenodo.org/records/7260137/files/ctc_format.zip?download=1)
- [SIM+ train](https://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-SIM+.zip)
- [HSC train](https://data.celltrackingchallenge.net/training-datasets/BF-C2DL-HSC.zip)
- [MuSC train](https://data.celltrackingchallenge.net/training-datasets/BF-C2DL-MuSC.zip)
- [HeLa train](https://data.celltrackingchallenge.net/training-datasets/Fluo-N2DL-HeLa.zip)

`videos.zip` của One-in-a-Million là video minh họa annotation, không cần cho input. Test CTC công khai không kèm đáp án: v0 dùng một phần các movies có nhãn của gói train làm held-out test nội bộ. Ghi rõ đây là custom split trên public training data.

Ví dụ lệnh Bash chạy trên máy triển khai, chưa chạy trong báo cáo này:

```bash
mkdir -p data/downloads data/raw/one_in_a_million
curl -fL --retry 3 -C - \
  'https://zenodo.org/records/7260137/files/ctc_format.zip?download=1' \
  -o data/downloads/ctc_format.zip
python -m zipfile -t data/downloads/ctc_format.zip
```

MD5 công bố cho gói này: `f29fddadcee5c18c86716d3958c5e0da`. Sau khi tải, so checksum theo khối thay vì đọc toàn bộ file vào RAM:

```python
import hashlib
from pathlib import Path
from zipfile import ZipFile

p = Path('data/downloads/ctc_format.zip')
h = hashlib.md5()
with p.open('rb') as f:
    for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
        h.update(chunk)
assert h.hexdigest() == 'f29fddadcee5c18c86716d3958c5e0da'
with ZipFile(p) as z:
    infos = z.infolist()
    print('files:', sum(not i.is_dir() for i in infos))
    print('uncompressed GiB:', sum(i.file_size for i in infos) / 2**30)
    print('top-level entries:', sorted({i.filename.split('/')[0] for i in infos}))
```

Chỉ giải nén khi tổng bytes khai báo phù hợp ngân sách và còn dung lượng đĩa. Giữ archive gốc đến khi xác nhận extract thành công. Lưu URL, ngày tải, checksum và tên version trong manifest. Đọc điều kiện sử dụng tại nguồn trước khi phát hành lại dữ liệu; mặc định phát hành scripts, splits và labels phái sinh phù hợp điều kiện nguồn, không đóng gói lại toàn bộ ảnh.

Sau khi kiểm tra, lệnh giải nén:

```bash
python -m zipfile -e data/downloads/ctc_format.zip data/raw/one_in_a_million
```

## 3. Đọc ảnh và nhãn như thế nào?

Quy ước CTC: ảnh `0N/tT.tif`; tracking trong `0N_GT/TRA/man_trackT.tif` và `man_track.txt`; segmentation trong `0N_GT/SEG/man_segT.tif`. Thời gian có thể dùng 3 hoặc 4 chữ số. `man_track.txt` có bốn cột `L B E P`: ID track, frame bắt đầu, frame kết thúc, ID mẹ; `P=0` nghĩa là không có mẹ được định nghĩa. Nhãn dương trong TRA giữ ID theo thời gian; nền bằng 0. SEG ID không nhất thiết giữ theo thời gian. Nguồn: [CTC naming conventions](https://public.celltrackingchallenge.net/documents/Naming%20and%20file%20content%20conventions.pdf).

Ví dụ tự tạo:

```text
1 0 4 0
2 5 9 1
3 5 12 1
4 10 15 2
5 10 15 2
8 0 15 0
```

Diễn giải ví dụ: track 1 có hai con là 2 và 3; track 2 có hai con là 4 và 5. Track 8 là nhánh khác được ghi nhận. Track 1 tồn tại cả ở frame 0 và frame 4: hai đầu mút đều tính vào khoảng tồn tại. Không cộng thêm một frame sau E.

Nếu chọn mốc tham chiếu frame 2 và mốc cuối frame 12:

| Tế bào ở frame 12 | Tế bào tương ứng tại frame 2 | Quan hệ |
|---|---|---|
| 3 | 1 | Một lần chuyển mẹ–con |
| 4 | 1 | Hai lần chuyển mẹ–con |
| 5 | 1 | Hai lần chuyển mẹ–con |
| 8 | 8 | Cùng tế bào sống xuyên khoảng |

Tại frame 12, 4 và 5 là chị em trực tiếp. 3 và 4 cùng tổ tiên 1 tại frame 2 nhưng không phải chị em trực tiếp.

**Không đưa các ID 1,2,3,4,5,8 này vào model như đặc trưng.** Chính chúng chứa đáp án về temporal identity. Đổi nhãn về ID cục bộ từng frame, ví dụ bằng hoán vị có seed cố định; giữ bảng ánh xạ sang GT riêng cho evaluator. Không dùng ảnh label integer như ảnh intensity; mask input nếu cần phải là binary mask của từng instance. Không dùng màu overlay GT làm input.

## 4. Phân biệt marker và đường viền tế bào

CTC phân biệt tracking markers, gold segmentation và silver segmentation. Tracking thường bao phủ tốt các instances nhưng không mô tả đầy đủ hình dạng; gold segmentation tốt về vùng nhưng chỉ có nhãn hạn chế; silver segmentation có nguồn gốc thuật toán. Nhãn test được giữ kín. Nguồn: [CTC annotations](https://celltrackingchallenge.net/annotations/).

Thiết kế đầu vào của chúng ta:

| Chế độ | Input vị trí/vùng tế bào | Được kết luận gì? |
|---|---|---|
| Oracle-location | Tâm marker GT, crop từ ảnh gốc | Khả năng suy luận khi vị trí đã đúng |
| Oracle-mask, nếu có mask đầy đủ được xác minh | Mask đã bỏ temporal ID | Khả năng suy luận khi segmentation đã đúng |
| Predicted detections | Detector/segmenter chạy trên ảnh giữ lại | Hiệu quả pipeline thực tế |

Không tính cell area, eccentricity hoặc tốc độ tăng diện tích từ tracking marker rồi gọi đó là hình thái sinh học. Nếu chỉ có marker, lấy image crop quanh tâm và không dùng marker area làm phenotype. Với One-in-a-Million, nguồn công bố có segmentation đã sửa thủ công nhưng vẫn phải kiểm tra chúng nằm ở đâu, có đủ frame hay không và khớp tracking thế nào.

Nếu dùng SEG mask làm input, ánh xạ SEG instance sang TRA ID theo overlap/containment có kiểm tra một-một; không giả định hai loại ID bằng nhau. Ambiguous match hoặc thiếu match phải được ghi lại.

## 5. Tổ chức thư mục và bảng trung gian

Giữ ảnh gốc bất biến. Sử dụng đường dẫn tương đối trong bảng để dễ di chuyển máy.

| Thư mục đề xuất | Nội dung |
|---|---|
| `data/downloads/` | ZIP gốc |
| `data/raw/<dataset>/` | Dữ liệu đã giải nén |
| `data/index/` | Bảng sequences, frames, tracks, observations |
| `data/splits/` | Split manifests đã chốt |
| `data/benchmarks/v0/inputs/` | Các frame/detection được phép đưa vào model |
| `data/benchmarks/v0/labels/` | Targets; test labels chỉ evaluator đọc |
| `data/cache/` | Features/crops có giới hạn dung lượng |
| `data/audit/` | Thống kê, lỗi parse và overlay kiểm tra |

Schema đề xuất, do chúng ta thiết kế:

| Bảng | Các trường tối thiểu |
|---|---|
| `sequences` | dataset, sequence_id, experiment_id nếu có, modality, pixel_size_um, time_step_min, source_url, checksum, annotation_type |
| `frames` | sequence_id, frame_index, timestamp_min, image_path, shape, dtype, marker_path, mask_path, annotation_available |
| `tracks` | sequence_id, gt_track_id, start_frame, end_frame, parent_id, validity_status |
| `observations` | sequence_id, frame_index, local_detection_id, center_y, center_x, bbox nếu có, mask_source |
| `gt_mapping` — evaluator | sequence_id, frame_index, local_detection_id, gt_track_id, match_status |
| `windows` | window_id, split, sequence_id, anchor_frame, target_frame, observed_frames, timestamps, schedule_id, seed |
| `targets` — supervision/evaluator | window_id, target_detection_id, ancestor_detection_id, target_status, generation_depth, sibling_pair_labels |

Có thể dùng Parquet cho bảng lớn, JSON cho config và JSONL cho windows. Chưa cần export hàng triệu file crop. Timestamps dùng phút thực, không thay bằng số thứ tự của frame sau khi subsample. Khóa GT luôn gồm dataset + sequence + track ID vì ID có thể lặp giữa movies.

## 6. Audit trước khi tạo benchmark

Lập một hàng cho từng movie, gồm: số frame, kích thước ảnh, dtype, tổng observations, số tracks, số divisions hợp lệ, số root tracks, số annotation thiếu, số ID không khớp, số parent references không hợp lệ và dung lượng bytes.

Kiểm tra tự động:

1. Parse frame index từ tên file thành số; không trộn `t001` và `t0001` hoặc sắp theo chuỗi sai.
2. Ảnh và nhãn khớp shape; dtype nhãn nguyên không âm; IDs tồn tại trong bảng tracks.
3. Mỗi row có start ≤ end; parent tham chiếu được nếu khác 0; graph không có chu trình.
4. Parent kết thúc trước child bắt đầu. Nếu không liền frame, ghi gap thay vì tự sửa thành division bình thường.
5. Đếm children theo parent. Trường hợp khác hai con phải được xem xét, không tự biến thành binary division hoặc xóa âm thầm.
6. Kiểm tra marker thực tế có mặt tại frame được dùng. Khoảng B–E trong bảng không thay thế việc kiểm tra annotation còn đủ.
7. Xác định annotation region/field of interest (FOI); không mặc định toàn ảnh đều được chấm. Theo hướng dẫn của từng nguồn.
8. Phân biệt kết thúc track vì division, ra khỏi vùng, mất nhãn hoặc hết video. Nếu thiếu thông tin, ghi unknown/censored, không suy ra death.

Kiểm tra trực quan trên train/development: xem crop và ảnh toàn cảnh ở đoạn đầu/giữa/cuối; khoảng 20 division events và 20 trường hợp gần biên hoặc không khớp là điểm khởi đầu đề xuất. Nếu phát hiện lỗi hệ thống, mở rộng kiểm tra đúng nhóm lỗi đó. Không chỉnh protocol dựa trên failure cases của test đã mở.

Nguồn CTC FOI/annotation procedure: [Annotation procedure](https://public.celltrackingchallenge.net/documents/Annotation%20procedure.pdf). Phải đọc quy tắc bộ đang dùng trước khi chấm full-frame.

Tài liệu hiện quy định FOI lùi vào 25 pixel ở các cạnh x/y cho HSC, MuSC và Fluo-N2DL-HeLa; SIM+ dùng 0. Giữ nguyên ảnh input, áp dụng quy tắc vùng chấm thay vì tùy tiện crop và mất tọa độ. Không tự áp dụng mức 25 pixel này cho One-in-a-Million chỉ vì nó dùng định dạng CTC; phải theo annotation của nguồn đó.

## 7. Chia dữ liệu để tránh leakage

Với năm movies của One-in-a-Million, sau khi biết tên thật, gán alias M0…M4 trong manifest. Một split pilot minh họa:

```json
{
  "train": ["M0", "M1", "M2"],
  "validation": ["M3"],
  "test": ["M4"]
}
```

Đây chưa phải split chính thức của nguồn. Mọi ảnh, crop, schedule và window từ cùng movie phải ở cùng split. Nếu biết nhiều movies cùng thí nghiệm và muốn tuyên bố tổng quát hóa sang thí nghiệm mới, chia theo experiment; nếu chỉ có một experiment thì không thể chứng minh claim đó bằng cách chia movie.

Khi protocol đã khóa, có thể dùng năm folds: fold j test Mj, validation M((j+1) mod 5), train ba movies còn lại. Chọn siêu tham số bằng validation trong từng fold theo cùng một quy tắc đã chốt. Không dùng kết quả test các fold trước để tiếp tục thay đổi architecture rồi gọi đó là đánh giá không thiên lệch. Pilot vẫn nên có một movie giữ kín đến sau quyết định phương pháp.

Năm movies là ít đơn vị độc lập. Báo từng movie và trung bình macro; không lấy hàng triệu cells làm kích thước mẫu độc lập để tạo confidence interval rất hẹp. Nếu có group thí nghiệm, resampling theo group phù hợp.

**Pretraining:** kiểm tra dữ liệu của encoder, segmenter lẫn tracker. Trackastra pretrained metadata có các nguồn liên quan, cần đọc chính checkpoint định dùng: [metadata](https://github.com/weigertlab/trackastra/blob/main/trackastra/model/pretrained.json). Checkpoint frozen vẫn có thể đã thấy test. Không rõ provenance thì gắn nhãn diagnostic, không dùng làm bằng chứng clean held-out generalization. Áp dụng cùng nguyên tắc cho mọi baseline.

## 8. Tạo một mẫu ancestry đúng nghĩa

Chọn frame tham chiếu a và frame đích b, b>a. **Anchors là các tế bào có mặt tại a**, không nhất thiết là root ở đầu video. Hai anchors có thể có chung tổ tiên xa hơn trước a; điều đó không làm task vô nghĩa. Điều kiện cần là có nhiều anchors riêng biệt để phải phân biệt.

Input mẫu:

```json
{
  "window_id": "example_M0_a0002_b0012",
  "sequence_id": "M0",
  "anchor_frame": 2,
  "target_frame": 12,
  "observed_frames": [2, 7, 12],
  "timestamps_min": [2.0, 7.0, 12.0],
  "detections_source": "oracle_locations_relabelled",
  "task": "ancestor_at_reference_time"
}
```

Ví dụ này giả sử ảnh gốc cách nhau một phút và video bắt đầu ở mốc 0. File input không có gt_track_id, parent_id, division count hoặc ảnh của các frame bị ẩn.

Target cho mỗi cell ở b là một detection thuộc frame a, hoặc trạng thái nhãn riêng bên dưới. Không bắt mọi ancestor phải có hai con tại b: qua nhiều thế hệ có thể có nhiều hậu duệ. Một cell sống xuyên khoảng có chính nó ở a làm ancestor theo nghĩa ancestor-or-self của task; ghi rõ quy ước này trong paper.

Có thể dùng full field of view trước. Nếu cần crop, crop chỉ theo ảnh/detections được quan sát, với rule cố định; không dùng tương lai GT để chọn vùng chứa đúng descendants. Cell rời crop tạo censoring nhân tạo, phải được theo dõi riêng.

## 9. Sinh nhãn ancestry và sibling

Đối với mỗi GT track u có mặt ở b:

1. Kiểm tra u tồn tại tại a hay không. Nếu có, ancestor chính là u.
2. Nếu u bắt đầu sau a, đi ngược parent chain.
3. Dừng ở track v tồn tại tại a và có annotation hợp lệ tại a.
4. Ánh xạ v sang local detection ID của a bằng evaluator mapping.
5. Nếu không chứng minh được mapping, phân loại nguyên nhân; không tự gán nhãn âm.

Pseudocode logic lõi, giả sử bảng tracks đã qua audit:

```python
def trace_to_anchor(track_id, a, tracks, present_at_a):
    # tracks[id] = (start_frame, end_frame, parent_id)
    seen = set()
    depth = 0
    u = track_id
    while True:
        if u in seen or u not in tracks:
            return None, 'invalid_lineage', depth
        seen.add(u)
        start, end, parent = tracks[u]
        if start <= a <= end:
            if u in present_at_a:
                return u, 'valid_anchor', depth
            return None, 'missing_anchor_annotation', depth
        if end < a:
            return None, 'broken_temporal_path', depth
        if parent == 0:
            return None, 'unresolved_root_after_anchor', depth
        u = parent
        depth += 1
```

Hàm này **không tự kết luận root xuất hiện sau a là cell mới đi vào ảnh**. Nó có thể là entry thật, annotation bị cắt hoặc track bị đứt. Phải dùng metadata/kiểm tra riêng mới chuyển trạng thái thành known no-anchor.

Đã kiểm tra cú pháp hai đoạn Python trong hướng dẫn và chạy sáu kiểm tra logic trên ví dụ tự tạo: cùng tế bào, một thế hệ, hai thế hệ, root sau anchor, thiếu marker anchor và ID không tồn tại. Đây chỉ là kiểm tra logic minh họa, chưa kiểm thử parser/loader trên ZIP thực.

Ba loại khác nhau bắt buộc tách:

| Khái niệm | Ý nghĩa | Xử lý |
|---|---|---|
| `known_no_anchor` | Đủ bằng chứng không thuộc anchor nào trong tập tham chiếu quan sát | Có thể là lớp target riêng |
| `label_unknown` / `censored` | Nhãn không đủ để quyết định, hoặc anchor thiếu annotation | Mask khỏi supervised loss tương ứng; báo số lượng |
| `abstain` | Model từ chối đưa ra kết luận vì thiếu confidence | Quyết định của model, làm giảm coverage |

Missed detection ở frame a cũng khác biological no-anchor: cell gốc có thật nhưng candidate pool của detector thiếu nó. Ghi `anchor_missed` ở evaluation; không chấm như entry đúng. `P=0` không phân biệt được tất cả các trạng thái trên.

**Sibling target:** hai tracks khác nhau có cùng parent khác 0, quan hệ được audit là hợp lệ, và cả hai cell đang có mặt tại b. Parent bằng 0 ở cả hai không có nghĩa chúng là chị em. Nếu parent chưa biết thì label unknown, không mặc định negative. Khi một chị em đã phân chia, con của nó không còn là chị em trực tiếp với dì/cô của nó.

Trong train có thể subsample negative pairs bằng seed cố định, gồm cặp gần nhau khác lineage và cặp cùng anchored lineage nhưng không phải sibling. Test cần candidate rule cố định từ quan sát, báo tỷ lệ bao phủ các cặp positive thật; không chọn tập cặp test bằng GT rồi che giấu bước chọn đó.

## 10. Tạo ảnh thưa: hai thí nghiệm khác nhau

**Thí nghiệm A — giữ cùng bài toán, thay số ảnh cung cấp.** Giữ a,b cố định, chỉ thay tập thời điểm nội bộ được giữ. Ví dụ tự tạo a=20,b=60:

| Schedule | Frames được giữ |
|---|---|
| Dense | Tất cả frame 20…60 |
| Stride 2 | 20,22,…,60 |
| Stride 4 | 20,24,…,60 |
| Stride 8 | 20,28,36,44,52,60 |
| Endpoints | 20,60 |

Cùng anchors, targets và horizon cho phép so sánh lượng bằng chứng. Đây là thí nghiệm chính để kiểm tra câu chuyện research. Nếu interval không chia hết horizon, giữ endpoint và ghi delta-time thật của đoạn cuối.

**Thí nghiệm B — tăng thời gian giữa hai ảnh.** Dùng cặp (a,a+Δ) với nhiều Δ. Khi đó cả mức dịch chuyển, số thế hệ và tập cells ở mốc cuối có thể đổi. Báo riêng, không diễn giải mọi thay đổi metric là do số ảnh ít hơn trên cùng target.

Khởi đầu kiểm tra strides 1,2,4,8 nhưng audit train trước: với bacteria một phút/frame, stride 8 mới là tám phút, có thể chưa tạo nhiều trường hợp nhiều thế hệ. Chọn horizons theo phân bố thời gian phân chia quan sát được trên train và tính hợp lý của acquisition. Không lấy cùng stride để coi các datasets có cùng độ khó: HSC stride 8 là 40 phút; HeLa stride 8 là 240 phút.

Schedules không đều: giữ a,b, lấy một số frame nội bộ bằng seed cố định; lưu nguyên danh sách và timestamps. Không chọn bỏ frame dựa vào vị trí GT division ở main benchmark. Có thể có bộ stress test riêng chọn quanh division bằng GT, nhưng phải công khai đây là event-conditioned evaluation.

Tạo sparse views bằng indices; không copy ảnh. Áp dụng normalization theo rule đã chốt dùng train hoặc chỉ ảnh được quan sát. Không ước lượng normalization từ toàn bộ dense test movie rồi dùng thống kê đó cho sparse input. Causal/online experiment không dùng ảnh tương lai, kể cả để ổn định ảnh hoặc segmentation.

## 11. Đếm mức độ khó mà không làm lộ đáp án

Evaluator có thể lưu các strata từ dense labels:

- `generation_depth`: số lần đi qua parent link từ target về anchor; 0,1,≥2.
- Số track trung gian trên đường đó không xuất hiện ở bất kỳ thời điểm được giữ nào.
- Mật độ tế bào quan sát, độ dịch chuyển, số anchors và khoảng thời gian thực.
- Annotation gaps, border/FOI flags, detection failures.

`generation_depth≥2` không tự động đồng nghĩa mọi division đều bị ẩn. Một division trong dense data cũng chỉ được định vị trong khoảng giữa quan sát parent cuối và children đầu, không có timestamp liên tục chính xác. Lưu khoảng đó nếu cần.

Các trường sinh từ GT chỉ để supervision/stratification/evaluation theo protocol, không làm input cho inference. Nếu đề xuất đo hiệu quả khi thiếu trung gian, báo chính xác có bao nhiêu track trung gian hoàn toàn vắng trong schedule.

Giữ nhóm no-division có độ dịch chuyển tương tự để so sánh. Không chỉ giữ những windows nhiều phân chia làm kết quả chính; báo cả phân bố đầy đủ và từng nhóm. Window đủ nhiều anchors nhưng không nhất thiết độc lập về tổ tiên trước mốc tham chiếu.

## 12. Predicted detections và phép đo cuối cùng

Oracle experiment là bước chẩn đoán. Pipeline thực cần detector/segmenter không đọc test labels và không đọc hidden frames. Chạy cùng detector cho các tracking baselines khi muốn cô lập khác biệt về linking.

Evaluator nối predicted detections với GT bằng rule cố định và one-to-one assignment. Dùng mask overlap nếu có full GT masks; với markers dùng containment hoặc distance phù hợp và báo giới hạn. Ngưỡng lấy từ train/validation, không tune test. Split/merge, unmatched predictions và missed cells phải được thống kê riêng.

Chỉ báo ancestry trên detections matched sẽ che mất detection failures. Vì vậy báo hai tầng: conditional ancestry quality trên matched detections và kết quả đầy đủ của detection+ancestry, kèm tỷ lệ matching/detection recall. Không đổi unmatched predictions thành no-anchor thật.

Readout v0: với mỗi anchor i, đếm số descendants **còn quan sát được trong FOI tại b**. Đây không phải tổng số descendants từng được sinh ra, cũng không phải số division events. Cell đi ra khỏi ảnh làm giới hạn diễn giải.

Tính MAE trên cùng tập anchors và horizon đã chốt. Nếu chỉ cộng assignments được accept, count có thể thấp giả tạo. Có thể báo count từ toàn bộ predicted distribution trên tất cả candidates, hoặc count/interval theo policy được định nghĩa trước; luôn báo phần unresolved. Không tự bỏ anchors khó khỏi mẫu số.

## 13. Đánh giá khác domain

Hai chế độ phải báo tách:

- **Transfer không dùng nhãn domain đích:** train và calibrate trên source; đánh giá nguyên trạng trên mammalian movies. Báo rõ thay đổi modality, detection quality và sai số calibration.
- **Adaptation có giám sát:** dùng một phần movies có nhãn đích để fit/calibrate và movies khác để test. Công khai ngân sách nhãn; không gọi đó là zero-shot.

Không chia ngẫu nhiên cells trong cùng HSC movie cho validation và test. Nếu số movies nhỏ, coi đây là bằng chứng bổ sung, chưa phải kết luận phổ quát. SIM+ phục vụ sanity check, không thay thế kiểm chứng trên ảnh thật. LFCT chưa nằm trong v0 vì cần audit riêng về nhãn, completeness và dung lượng.

## 14. Giới hạn đĩa và tốc độ đọc

Mục tiêu kế hoạch: downloads + raw + cache + checkpoints + scratch tổng dưới 50 GB. Ví dụ phân bổ có dự phòng: ZIP 6 GB, raw/labels 22 GB, index/cache 6 GB, checkpoints 6 GB, scratch 4 GB; còn khoảng 6 GB dự phòng. Đây là trần thiết kế, không phải bytes đã đo. Nếu archive audit cho thấy raw vượt trần, chỉ triển khai nhóm ưu tiên trước hoặc xử lý từng bộ.

Đọc TIFF theo frame/window, không load toàn bộ movie vào RAM. Crop on-the-fly hoặc cache giới hạn. Giữ dtype gốc trên đĩa; normalization cho model ở loader; resize labels bằng nearest-neighbor và cập nhật tọa độ nhất quán. Không ghi đè TIFF gốc bằng PNG 8-bit. Với fluorescence nuclei, gọi feature là nuclear area nếu đó là đối tượng đã segment; không tự gọi là diện tích toàn tế bào.

Feature cache cần khóa theo dataset/version, split, model checkpoint, preprocessing và schedule nếu encoder dùng temporal context. Cache computed từ dense temporal context không được tái dùng cho sparse test như thể chỉ dùng ảnh được giữ.

## 15. Công việc và tiêu chí hoàn thành trong ba ngày đầu

| Ngày | Công việc | Đầu ra bắt buộc |
|---|---|---|
| 1 | Tải bộ chính; checksum; audit archive; parse movies/tracks; xem overlay | Manifest nguồn và bảng audit từng movie |
| 2 | Chốt split; local relabel; tạo ancestry/sibling targets; kiểm tra các trạng thái unknown/null | Split JSON, bảng targets, log lỗi nhãn |
| 3 | Sinh matched-horizon schedules; thống kê strata; chạy loader và baseline đơn giản | Input/label tách biệt, thống kê windows theo split/gap/depth, kết quả sanity baseline |

Chưa cần train mô hình lớn để hoàn thành phần dữ liệu. Điều kiện sẵn sàng:

- Mọi sample truy được về ảnh gốc, thời gian, movie và checksum.
- Test input không chứa temporal GT IDs, parent information hoặc hidden-frame features.
- Một vài ví dụ kiểm tra tay cho ancestor-or-self, một thế hệ, nhiều thế hệ, same-parent sibling và unknown đều đúng.
- Có đủ windows nhiều anchors và có phân chia trung gian để kiểm tra giả thuyết; số lượng phải đo, không mặc định từ tổng 14.000 divisions.
- Loại bỏ/mask nhãn nào đều có lý do và số lượng công khai theo split.
- Baselines nhận cùng input/candidate pool và cùng annotation budget.
- Có bảng dung lượng thực và loader không vượt RAM/đĩa dự kiến.

Nếu không có đủ multi-generation cases ở các khoảng chụp hợp lý, báo kết quả audit đó và điều chỉnh câu hỏi nghiên cứu; không chỉ tăng stride đến khi dữ liệu bị phá hỏng.

## 16. Những lỗi dễ làm bài mất giá trị

1. Chia random frames/cells rồi cùng movie xuất hiện trong cả train và test.
2. Giữ nguyên ID TRA hoặc GT overlay làm input.
3. Đồng nhất P=0 với tế bào mới xuất hiện, hoặc với negative sibling.
4. Dùng marker area làm cell area.
5. Dùng root ở đầu video làm anchor duy nhất, khiến mọi cell có cùng đáp án.
6. Dùng số thứ tự frame sau subsampling làm thời gian thật.
7. Cho dense teacher/temporal encoder nhìn hidden test frames rồi tái dùng features.
8. Chấm biological counts trên riêng các ca model đồng ý trả lời.
9. Dùng checkpoint đã train trên held-out movie mà gọi là chưa thấy dữ liệu.
10. Suy ra giảm phototoxicity từ việc bỏ ảnh sau khi thí nghiệm chụp đã diễn ra.

## 17. Tài liệu cần mở khi triển khai

- [One-in-a-Million: data và checksum](https://zenodo.org/records/7260137)
- [Preview ZIP — chỉ liệt kê một phần](https://zenodo.org/records/7260137/preview/ctc_format.zip?include_deleted=0)
- [CTC: tải dữ liệu và thông số acquisition](https://celltrackingchallenge.net/2d-datasets/)
- [CTC: loại annotations](https://celltrackingchallenge.net/annotations/)
- [CTC: quy ước tên file và format](https://public.celltrackingchallenge.net/documents/Naming%20and%20file%20content%20conventions.pdf)
- [CTC: annotation procedure và FOI](https://public.celltrackingchallenge.net/documents/Annotation%20procedure.pdf)
- [Trackastra: provenance checkpoint](https://github.com/weigertlab/trackastra/blob/main/trackastra/model/pretrained.json)

Các nguồn hỗ trợ định dạng/dataset; schema, split, nhãn mở rộng, lịch lấy mẫu và tiêu chí hoàn thành trong tài liệu này là thiết kế đề xuất của dự án, không phải protocol mặc định do tác giả nguồn cung cấp.
