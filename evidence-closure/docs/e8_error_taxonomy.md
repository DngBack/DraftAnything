# E8 — Lỗi DocVQA trên bảng tài chính đến từ đâu? (dev, chẩn đoán)

Thiết lập: TAT-DQA dev; Qwen2.5-VL-7B chạy 600 câu / 233 tài liệu, Qwen3-VL-8B chạy 300 câu đầu. Model nhận trang đầy đủ, decode greedy. Với mỗi câu chạy 5 prompt:

| prompt | nội dung |
|---|---|
| `direct` | câu trả lời trực tiếp |
| `extract` | liệt kê các số cần dùng kèm nhãn |
| `program` | một biểu thức số học, được thực thi bằng Python |
| `hint` | `direct` kèm sẵn fact gold |
| `hintprog` | `program` kèm sẵn fact gold |

Script: `scripts/run_e8.py`, `scripts/analyze_e8.py`. Kết quả: `results/e8_{tag}.md` (chấm strict) và `results/e8_{tag}_pct.md` (percent-norm: chấp nhận ×100 khi scale gold là percent). Với percent-norm, thực thi `derivation` gold cho đúng đáp án ở 100 % câu arithmetic; với strict là 71–73 %, vì TAT-DQA ghi tỷ lệ phần trăm ở dạng ×100. Câu được coi là "đọc đủ" khi mọi fact gold xuất hiện trong output `extract` hoặc trong biểu thức của `program`.

## 1. Accuracy theo loại câu (percent-norm)

| | 7B direct | 7B hint | **7B program→Py** | 7B hint+prog | Qwen3 direct | Qwen3 hint | **Qwen3 program→Py** | Qwen3 hint+prog |
|---|---|---|---|---|---|---|---|---|
| span | 88.9 | 91.4 | – | – | 92.2 | 96.1 | – | – |
| multi-span | 94.9 | 97.5 | – | – | 97.2 | 100 | – | – |
| **arithmetic** | **39.9** | 53.5 | **81.0** | 90.5 | **31.6** | 55.1 | **88.6** | 91.8 |
| tất cả | 63.5 | 71.8 | | | 61.0 | 75.0 | | |
| đọc đủ fact | 90.0 % | | | | 93.7 % | | | |

## 2. Phân loại câu `direct` trả lời sai

| loại lỗi | 7B (219 câu) | Qwen3 (117 câu) |
|---|---|---|
| đọc đủ, biểu thức đúng → **chỉ tính nhầm** | **61.6 %** | **77.8 %** |
| đọc sai / chọn nhầm ô | 21.0 % | 11.1 % |
| đọc đủ, biểu thức sai (phép toán / thứ tự / điều kiện) | 10.0 % | 8.5 % |
| đọc đủ, chọn hoặc định dạng span sai | 7.3 % | 2.6 % |

## 3. Đòn bẩy và phần còn lại

| | 7B | Qwen3 |
|---|---|---|
| value acc, `direct` | 63.5 | 61.0 |
| value acc, **route thực tế**: dùng biểu thức nếu tính được, không thì `direct` | **78.0** | **85.0** |
| value acc, route oracle theo loại câu gold | 85.2 | 91.0 |
| value acc, hint + route (bằng chứng đầy đủ) | 91.3 | 94.3 |
| **chấm chính thức (giá trị + scale)**, `direct` | 46.3 | 49.0 |
| chấm chính thức, route thực tế (scale lấy từ `direct`) | 60.8 | 69.0 |
| tỷ lệ câu có scale mà `direct` ghi đúng scale (trong số câu giá trị đúng) | 63 % | 70 % |

Kết luận:

1. **Lỗi số một là tính nhầm** (62–78 % số câu sai). Model đọc đủ số ở ~90–94 % câu. Tách bước tính ra cho Python làm tăng câu arithmetic từ 32–40 % lên 81–89 %. Cách này hiệu quả nhưng đã được biết rõ (program-of-thought, TAT-QA TagOp), nên không mới.
2. **Lỗi số hai là scale/đơn vị.** Khi giá trị đã đúng, model vẫn bỏ hoặc ghi sai "thousand/million/percent" ở 30–37 % câu có scale. Riêng lỗi này làm mất **17 điểm** theo cách chấm chính thức ở 7B (route: 78.0 value so với 60.8 chính thức). Đây là chỗ mà **quan hệ** (giá trị ↔ caption đơn vị) thực sự quan trọng, tức là kết quả của H2 có ứng dụng ở đây.
3. **Bằng chứng đầy đủ (H1) chỉ giúp vừa phải**: +8 (7B) và +14 (Qwen3) trên mọi câu. Sau khi đã có Python, phần thêm chỉ còn +6 đến +9 điểm.
4. **Sau Python, lỗi còn lại ở 7B** (89 câu) gồm: đọc sai ô 47 câu (53 %), sai phép toán 26 câu, span 16 câu. Ba nhóm này cần **ba hành động khác nhau**.

## 4. Quyết định hướng: H3 thu hẹp — chọn hành động sửa theo loại lỗi

Dữ liệu ủng hộ đúng tiền đề của H3 (đề xuất mục 5): các lỗi có nguyên nhân khác nhau và cần hành động khác nhau. Ở đây chúng đã đo được trên dữ liệu thật:

| nguyên nhân | dấu hiệu | hành động |
|---|---|---|
| tính nhầm | `direct` ≠ giá trị biểu thức | thực thi biểu thức (Python) |
| scale | đơn vị thiếu hoặc mâu thuẫn với caption | đọc caption đơn vị (closure quan hệ từ OCR, hoặc hỏi lại VLM) |
| đọc sai ô | số trích ra không khớp nhãn hàng/cột trên OCR | đọc lại vùng được zoom / ground lại ô |
| sai phép toán | biểu thức không khớp loại câu hỏi (change, average, %) | kiểm tra lại biểu thức |

Bước tiếp theo (E9): đo **recovery rate theo loại lỗi** của từng hành động trên dev. Sau đó (E10): học **action-value predictor** trên train, gồm lợi ích kỳ vọng của từng hành động trừ chi phí. So sánh trên test với: pipeline cố định (luôn làm mọi hành động), trigger theo confidence, và phân loại lỗi rồi gán hành động cố định. Tất cả ở cùng ngân sách số lượt gọi model.

## 5. E9 — recovery rate của các hành động sửa lỗi (dev)

Chi tiết: `results/e9_qwen25vl7b.md` (600 câu) và `results/e9_qwen3vl8b.md` (300 câu). Base = route thực tế (có biểu thức thì dùng Python, không thì dùng `direct`). Chấm chính thức: giá trị + scale.

| | 7B | Qwen3 |
|---|---|---|
| direct | 39.5 | 42.7 |
| program→Py, scale từ direct | 54.8 | 62.3 |
| program→Py, scale OCR-closure (áp cho mọi câu) | 50.5 | 57.7 |
| program→Py, scale từ VLM | 41.7 | 55.3 |
| **oracle chọn scale đúng giữa {direct, OCR}** | **≈ 68** | **75.0** |
| trần: giá trị đúng + scale gold | 78.0 | 85.0 |

| hành động sửa giá trị | 7B: cứu / làm hỏng / ròng | Qwen3: cứu / làm hỏng / ròng |
|---|---|---|
| zoom (crop ×2) | 21/132 · 18/468 · +0.5 | 6/45 · 11/255 · −1.7 |
| verify (tự kiểm tra) | 3/132 · 4/468 · −0.2 | 0/45 · 2/255 · −0.7 |

- Hai hành động sửa giá trị **không đáng thêm một lượt gọi**: chúng cứu được rất ít và làm hỏng gần bằng số câu cứu được.
- Scale OCR-closure **đúng hơn trên câu có scale** (82–84 % so với 69–70 % của direct), nhưng **sai trên câu có scale gold rỗng**. Chọn đúng nguồn scale cho từng câu có dư địa **+11 đến +13 điểm**. Luật thủ công (OCR khi có phép toán hoặc khi direct ghi scale) **không** lấy được phần này (54.9 / 63.7).
- Hệ quả cho H3: giá trị của policy học được chủ yếu nằm ở **chọn nguồn scale và chọn nguồn giá trị**, không nằm ở việc gọi thêm hành động. Đây là ghi chú thăm dò trên dev; protocol H3 không đổi.
