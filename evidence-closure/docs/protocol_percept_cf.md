# Protocol pilot B: đọc hay đoán? Sửa chữ số trong ảnh bảng (khóa ngày 2026-11-06, trước mọi output của pilot này)

Bối cảnh: pilot A ([protocol_percept.md](protocol_percept.md)) cho thấy khi ảnh kém, model phát ra số sai một cách ổn định, gần như bịa, và không bao giờ từ chối. Dữ liệu quan sát không tách được hai nguồn: mất thông tin thị giác, hay đoán theo ngữ cảnh. Phải can thiệp vào pixel mới tách được.

## Kích thích

`data/percept_set.json` (`scripts/build_percept_set.py`): 80 câu số học, mỗi câu một tài liệu, thuộc TAT-DQA train và không trùng tài liệu với `h3_train_set` hay `attr_set`. Ô đích là một fact gold thỏa: có ≥ 3 chữ số, không xuất hiện ở chỗ khác trên trang, closure heuristic tìm được cả nhãn hàng lẫn nhãn cột.

Sửa ở độ phân giải gốc: phủ màu nền lên hộp OCR rồi vẽ lại bằng font Liberation khớp nhất, sau đó mới giảm chất lượng.

| điều kiện | nội dung |
|---|---|
| `orig` | ảnh không sửa |
| `null` | vẽ lại đúng các chữ số cũ; đo nhiễu do chính việc sửa ảnh |
| `cf` | đổi một chữ số (không phải đầu hay cuối) thành d_s |

Với mỗi ô, d_s và d_a (số đối chứng, **không bao giờ được vẽ**) được bốc ngẫu nhiên từ các chữ số ≠ chữ số gốc, seed cố định, nên lấy trung bình qua câu thì độ giống nét không thiên về bên nào.

Mức ảnh: s100, s60, s50, s40, s30. Model: Qwen2.5-VL-7B, Qwen3-VL-8B. Decode greedy.

## Prompt (pixel giữ nguyên, chỉ đổi văn bản)

| mã | nội dung | điều kiện chạy |
|---|---|---|
| `copy` | chỉ ô theo nhãn hàng và cột; chép số | orig, null, cf |
| `copyu` | như `copy`, thêm: nếu không đọc rõ thì trả lời UNREADABLE | orig |
| `addr` | cùng cách chỉ ô; viết biểu thức nhân số đó với 2, chép số nguyên văn | orig, cf |
| `program` | prompt câu hỏi gốc (E8) | orig, cf |

## Đại lượng (so chữ số, bỏ định dạng)

- **T:** output chứa số được vẽ. **O:** chứa số gốc, không có trong ảnh `cf`. **A:** chứa số đối chứng.
- Ngưỡng theo chiều cao chữ: hồi quy logistic của T theo log(chiều cao hộp OCR × tỷ lệ), lấy mức 50 %.
- Metacognition: AUROC của log-prob trung bình dự đoán `copy` đọc đúng; và P(UNREADABLE | `copy` sai) so với P(UNREADABLE | `copy` đúng), ghép cặp trên cùng ảnh `orig`.

CI 95 % bootstrap theo tài liệu.

## Tiêu chí (từng model)

**Điều kiện hợp lệ.** Ở s100 với `copy`: |T(null) − T(orig)| ≤ 5 điểm và T(cf) ≥ 80 %. Không đạt thì sửa kích thích, chưa diễn giải gì.

| giả thuyết | sống nếu | ý nghĩa nếu sống | nếu chết |
|---|---|---|---|
| **H-prior** | O − A > 0, CI không chứa 0, ở s40 hoặc s30, cả hai model | ngữ cảnh kéo bản đọc về giá trị hợp lý | bản đọc sai là nhiễu thị giác, không phải đoán theo tài liệu |
| **H-task** | T(`copy`) − T(`addr`) ≥ 5 điểm, CI không chứa 0, trên `orig` ở ít nhất một mức, cả hai model | đọc phụ thuộc nhiệm vụ dù pixel giữ nguyên; bài này là "copy vs compute" sạch hơn pilot A | H-instab chết hẳn; nhận thức là giai đoạn ổn định |
| **H-meta** | AUROC ≥ 0.75 ở s40 và s30 **và** tỷ lệ từ chối của `direct` ≈ 0 | model có tín hiệu biết mình không đọc được nhưng không nói ra | không có nền cho hướng tự biết |

H-prior chấm trên prompt `copy`; `addr` và `program` chỉ báo cáo thêm. H-meta tính trên `copy`, ảnh `orig`. Ngưỡng theo chiều cao chữ chỉ mô tả, không phải tiêu chí. Script phân tích `scripts/analyze_percept.py` được viết trước khi có output.

## Không được làm

Không đổi set, prompt, chấm điểm hay ngưỡng sau khi xem output. Phân tích thăm dò phải ghi rõ là thăm dò. Pilot 80 câu chỉ để quyết định có đầu tư tiếp không; không phải kết quả công bố.

## Giới hạn biết trước

- Font khác bản gốc; `null` đo phần này nhưng không loại hết.
- Prompt `copy` chỉ ô bằng nhãn từ heuristic, nhãn có thể sai nên T có thể thấp vì định vị.
- `O` có thể tăng nhờ thông tin dư trong bảng, cũng có thể do phần chữ số còn lại khiến model gợi nhớ một mẫu quen; đối chứng A chỉ loại độ giống nét.
- Câu trả lời `program` có thể không dùng ô đích, nên T(`program`) gồm cả lỗi định vị.
