# Protocol D: thí nghiệm quyết định cho luận đề "đọc thật, thất bại im lặng" (khóa ngày 2026-11-06, trước mọi output)

Bối cảnh: [acl2028_decision_memo.md](acl2028_decision_memo.md). Pilot B và C dùng 80 ô, đã xem. Protocol này là xác nhận trên tập mới, thêm can thiệp tách định vị khỏi đọc nét, và thêm một họ model ngoài Qwen.

## Dữ liệu

`data/decide_set.json` (`scripts/build_decide_set.py`): 500 ô, mỗi ô một tài liệu TAT-DQA train, không trùng tài liệu với `h3_train_set`, `attr_set`, `percept_set`. Bộ lọc ô giống pilot B: ≥ 3 chữ số, không lặp trên trang, có nhãn hàng và nhãn cột. Mọi loại câu. Mỗi ô có một ô mồi: một số khác trên cùng trang, cũng không lặp.

Model: Qwen2.5-VL-7B, Qwen3-VL-8B, InternVL3-8B (`-hf`, native). Decode greedy, không repetition penalty. Mỗi output lưu log-prob và margin top-1/top-2 theo token.

## Khối 1: ảnh giảm độ phân giải (s100, s50, s40, s30)

Như pilot B: ảnh `orig` với `copy`, `addr`, `copyu`, `copyu2`, `conf`, `program`; ảnh `cf` với `copy`, `addr`; ảnh `null` với `copy`.

- `copyu2`: như `copyu`, thêm rằng số sai tệ hơn nhiều so với UNREADABLE, và chỉ chép khi chắc chắn mọi chữ số.
- `conf`: chép số rồi ghi độ tự tin 0–100 rằng đã đọc đúng mọi chữ số.

Đại lượng T, O, A như [protocol_percept_cf.md](protocol_percept_cf.md).

## Khối 2: thí nghiệm 2×2 định vị × đọc nét (s50, s40, s30)

Mọi điều kiện dùng cùng khung: ảnh hạ xuống mức đó rồi phóng lại về kích thước gốc, để số token bằng nhau giữa các điều kiện. Prompt `copy`.

| | không có G | G: phục hồi pixel gốc trên **mọi** số OCR của trang |
|---|---|---|
| không có L | `base` | `G` |
| L: hộp đỏ quanh ô đích, prompt báo ô được khoanh đỏ | `L` | `LG` |

G không để lộ ô đích, vì mọi ô số đều sắc nét; nhãn hàng và nhãn cột vẫn mờ, và các từ dạng năm (như tiêu đề cột `2019`) không được phục hồi trừ khi là ô đích. L không thêm nét. Shapley hai người chơi trên T cho φ_L, φ_G và số hạng tương tác.

**Kiểm tra can thiệp:** hộp đỏ đặt trên ô mồi, prompt `boxonly` ("chép số trong hộp đỏ", không có nhãn), ở s100 và ở khung s40. Thêm `L` trên ảnh s100 để so với `copy`.

**Đối chứng thêm sau rà soát tài liệu (trước mọi output, chỉ mô tả):** hộp đỏ trên ô mồi với prompt vẫn nêu nhãn ô đích, ở s100 và khung s40, đo việc hộp "cướp" bản đọc; ảnh trắng với `copy`, `copyu2`, `conf`, làm mức sàn cho từ chối và độ tự tin; `program` trên `orig`, đối chiếu với "read-use gap" của arXiv 2611.01517; các cách gộp log-prob (token đầu, trung bình chữ số, min chữ số); khoảng cách log-prob và lời nói theo chiều cao chữ, kèm ECE.

## Tiêu chí (từng model; CI 95 % bootstrap theo tài liệu)

**Hợp lệ.** Ở s100, `copy`: |T(null) − T(orig)| ≤ 5 và T(cf) ≥ 80. Tỷ lệ chép đúng ô mồi với `boxonly` ở s100 ≥ 70 %. Model không đạt thì tách riêng và không dùng để kết luận phần liên quan.

| mã | câu hỏi | kết luận |
|---|---|---|
| **D1** | định vị và đọc nét có tách được không | **tách được** nếu ở ít nhất một mức, φ_L và φ_G đều có CI > 0. Báo cáo φ_G − φ_L và tương tác |
| **D2** | ngữ cảnh có kéo bản đọc không (`copy`, s40 và s30) | **có** nếu O − A có CI > 0 và ≥ 3 điểm; **không đáng kể** nếu cận trên CI < 3 điểm ở cả hai mức; còn lại chưa kết luận |
| **D3** | đọc có phụ thuộc nhiệm vụ không (T(copy) − T(addr), `orig`, s50, s40, s30) | **có** nếu ở một mức CI loại 0 và \|Δ\| ≥ 5; **không đáng kể** nếu CI nằm trong ±5 ở cả ba mức; còn lại chưa kết luận |
| **D4a** | có tín hiệu ẩn không | **có** nếu AUROC log-prob ≥ 0.75 ở s40 và s30, AUROC log-prob − AUROC cỡ chữ có CI > 0, và tỷ lệ cặp bất đồng trong cùng ô đúng > sai có CI > 50 % |
| **D4b** | model có tự nói ra được không | **có** nếu P(UNREADABLE \| sai) − P(UNREADABLE \| đúng) ≥ 20 điểm với CI > 0 ở s30 với `copyu` hoặc `copyu2` |
| **D4c** | có khoảng cách giữa tín hiệu ẩn và lời nói không | **có** nếu AUROC log-prob − AUROC độ tự tin ghi ra có CI > 0 ở s40 hoặc s30, trên prompt `conf` |

Cả D4a và D4c tính trên chính prompt tương ứng, ảnh `orig`.

**Quy tắc dừng** (từ bản ghi quyết định): đổi hướng phần liên quan nếu D1 không tách được ở cả ba model; nếu D4a không lặp lại ở InternVL, chỉ được nói về họ Qwen.

Báo cáo mô tả: ngưỡng chiều cao chữ theo prompt có CI; đường rủi ro theo độ phủ khi từ chối bằng log-prob (ngưỡng chọn trên một nửa tài liệu, đánh giá trên nửa kia) so với từ chối bằng lời. Có nhiều phép so sánh mà không hiệu chỉnh đa kiểm định; đọc CI sát biên một cách thận trọng.

## Không được làm

Không đổi set, prompt, kích thích, chấm điểm, ngưỡng sau khi xem output lượt chính. Chạy thử chỉ dùng ô của `percept_set`, không tính vào kết quả. Thăm dò phải ghi rõ là thăm dò.

## Nhật ký sai lệch

- 2026-11-06, 08:57 UTC: người dùng yêu cầu xem kết quả tạm khi lượt chính chưa xong (n = 178, 212, 98 ô). Không đổi gì trong lượt chạy, không dừng sớm; kết luận chỉ lấy từ đủ 500 ô.
- 2026-11-06, 09:16 UTC: thử `scripts/paper_tables.py` trên dữ liệu chưa đủ làm lộ phán quyết tạm (n = 212, 254, 123), trong đó D1 đổi so với lần xem trước. Không thay đổi gì.
- 2026-11-06, 09:20 UTC: dừng lượt chính trước khi đủ 500 ô vì người dùng cần GPU cho việc khác, không phải vì kết quả. Số ô đủ điều kiện: 218 (7B), 260 (Qwen3), 127 (InternVL). Mọi phán quyết trên mẫu này là tạm thời, và trái với cam kết "chỉ kết luận từ đủ 500 ô" ở mục trên, nên phải báo cáo như vậy. `run_decide.py` chạy tiếp được từ chỗ dừng.

## Giới hạn biết trước

- Khung phóng lại khác với ảnh hạ thật: số token lớn, nét đã mất. Báo cáo chênh lệch `base` so với `orig` cùng mức.
- Hộp đỏ và phần sắc nét là dấu hiệu nhân tạo; L có thể thu hút chú ý cả ở mức khác định vị.
- OCR bỏ sót số thì G không phục hồi ô đó; ô đích luôn có trong OCR.
- InternVL chia ảnh thành tile 448 px (`crop_to_patches=True`, tối đa 12 tile; mặc định của checkpoint ép cả trang vào một tile, đã bỏ sau chạy thử). Trang ở mọi mức đều thành 13 tile nên số token cố định; mức ảnh chỉ đổi lượng thông tin, khác với Qwen nơi số token giảm theo mức.
- Tập vẫn lọc theo fact khớp OCR; ảnh giảm chất lượng giả lập; một miền dữ liệu.
- Giấy phép của InternVL3 là `other`; cần kiểm tra trước khi công bố dữ liệu đầu ra.
