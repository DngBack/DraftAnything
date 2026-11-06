# Protocol: quy lỗi bằng can thiệp theo chất lượng ảnh (khóa ngày 2026-11-06, trước khi chạy đủ)

Câu hỏi: khi ảnh tài liệu kém đi, phần lỗi do nhận thức có tăng không, so với tính toán và đơn vị?

## Dữ liệu

`data/attr_set.json`: 400 câu TAT-DQA train trên 400 tài liệu, không trùng tài liệu nào trong `h3_train_set.json`, cùng bộ lọc E1 (arithmetic 195, span 144, multi-span 53, count 8). Không dùng để học policy. 40 câu đầu chỉ dùng để hiệu chỉnh mức ảnh (`attrcal_*`). Phân tích chính chạy trên **360 câu còn lại** (`run_attr.py --skip 40`).

Hạn chế: bộ lọc E1 yêu cầu fact khớp OCR, nên tập này chưa đại diện cho toàn bộ TAT-DQA.

## Model và mức ảnh

Qwen2.5-VL-7B và Qwen3-VL-8B. Mức ảnh (`evc/degrade.py`): **s100, s60, s50, s40, s30** (tỷ lệ theo cạnh), **photo** (s60 + xoay ±2°, blur, JPEG 25). Mức kém nhất của tiêu chí chính là **s30**.

Hiệu chỉnh trên 40 câu chỉ xem tỷ lệ đọc đủ và EM `direct`, không tính Shapley. Đọc đủ, câu số học (21 câu), 7B / Qwen3: s100 86 / 90, s60 95 / 81, s40 57 / 48, s30 5 / 0, s20 5 / 0; photo 7B 71, Qwen3 chưa xong lúc khóa. Từ đó: bỏ s20 vì bão hòa cùng s30; thêm s50 vì đoạn s60→s40 giảm rất dốc; giữ photo bất kể số của Qwen3.

## Can thiệp

| người chơi | sửa bằng cách | ghi chú |
|---|---|---|
| P | thêm ô gold vào prompt | gồm cả định vị lẫn đọc |
| C | Python thực thi biểu thức | chỉ áp cho câu arithmetic |
| S | dùng scale gold, chấp nhận thêm quy ước ×100 của percent | hậu xử lý |
| O (phụ) | đưa công thức gold đã che số | chỉ ở s100 và s30; phân tích: `scripts/analyze_attr_op.py`, viết trước khi xem dữ liệu đủ |

Chấm bằng EM (`evc/official_metric.py`), tách đáp án bằng `direct_pred`. Giá trị Shapley tính trên trung bình theo câu. CI 95 % bootstrap theo tài liệu, ghép cặp giữa các mức (`scripts/analyze_attr.py`).

## Tiêu chí

1. **Kiểm tra can thiệp:** tỷ lệ đọc đủ phải giảm theo mức ảnh; nếu không, thí nghiệm không có giá trị.
2. **Chính:** tỷ phần P ở mức kém nhất − tỷ phần P ở s100 > 0, CI không chứa 0, ở cả hai model. Báo cáo thêm Δφ_P tuyệt đối.
3. **Lặp lại:** ở s100, φ_C + φ_S > φ_P, CI không chứa 0, ở cả hai model, trên tập mới này.

Nếu (2) không đạt: giả thuyết "chất lượng ảnh giải thích khác biệt với các benchmark" không được ủng hộ, và hướng này chỉ còn là phân tích trên một bộ dữ liệu.

## Giới hạn đã biết

- Can thiệp P không chắc sửa hết lỗi nhận thức, nhất là ở ảnh rất kém, vì model vẫn phải tìm ý nghĩa của các số. Ở câu span, P gần như đưa sẵn đáp án.
- Nhãn scale gold có nhiễu.
- Giảm độ phân giải giả lập chưa phải ảnh chụp thật; cần bộ dữ liệu thứ hai.

## Không được làm

Không đổi mức ảnh, prompt, can thiệp hay hàm chấm sau khi đã xem Shapley của dữ liệu đủ.
