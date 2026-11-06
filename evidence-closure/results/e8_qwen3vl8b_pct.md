# E8 — phân loại lỗi, qwen3vl8b — chấm program: percent-norm (×100 khi scale gold = percent)

n = 300 câu dev, 180 tài liệu.

Kiểm tra hàm chấm: thực thi `derivation` gold cho đúng đáp án ở 100.0 % câu arithmetic (n = 158).

## Accuracy theo loại câu

| loại | n | direct | hint (cho sẵn fact gold) | program→Python | hint+program→Python | đọc đủ fact |
|---|---|---|---|---|---|---|
| span | 102 | 92.2 | 96.1 | – | – | 93.1 |
| multi-span | 36 | 97.2 | 100.0 | – | – | 97.2 |
| arithmetic | 158 | 31.6 | 55.1 | 88.6 | 91.8 | 94.3 |
| count | 4 | 100.0 | 100.0 | – | – | 50.0 |
| all | 300 | 61.0 | 75.0 | – | – | 93.7 |

## Phân loại câu `direct` sai

Tổng câu sai: 117 / 300.

| loại lỗi | số câu | % lỗi |
|---|---|---|
| đọc đủ, biểu thức đúng → chỉ tính nhầm (calculation) | 91 | 77.8 |
| đọc sai / chọn nhầm ô (perception, localization) | 13 | 11.1 |
| đọc đủ, biểu thức sai (reasoning: phép toán / thứ tự / điều kiện) | 10 | 8.5 |
| đọc đủ, trả lời sai (chọn / định dạng span) | 3 | 2.6 |

## Mức cải thiện tối đa của từng cách sửa (CI 95 % bootstrap theo tài liệu)

- **Bằng chứng đầy đủ** (hint − direct, mọi câu): +14.0 [+9.5, +18.7]
- **Thực thi bằng Python** (program − direct, arithmetic): +57.0 [+49.0, +64.6]
- **Oracle chọn đúng giữa direct và program** (arithmetic): +57.6 [+50.0, +65.2]
- **Bằng chứng đầy đủ + Python** (hintprog − direct, arithmetic): +60.1 [+52.3, +67.7]
- Scale sai dù giá trị đúng: 36 câu / 97 câu có scale.

## Theo số fact

| n_facts | n | direct | hint | đọc đủ |
|---|---|---|---|---|
| 1 | 104 | 91.3 | 95.2 | 93.3 |
| 2 | 167 | 43.1 | 64.1 | 95.8 |
| 3 | 24 | 54.2 | 66.7 | 87.5 |
| 4+ | 5 | 60.0 | 60.0 | 60.0 |