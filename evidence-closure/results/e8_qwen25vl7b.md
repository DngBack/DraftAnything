# E8 — phân loại lỗi, qwen25vl7b — chấm program: strict

n = 600 câu dev, 233 tài liệu.

Kiểm tra hàm chấm: thực thi `derivation` gold cho đúng đáp án ở 70.9 % câu arithmetic (n = 316).

## Accuracy theo loại câu

| loại | n | direct | hint (cho sẵn fact gold) | program→Python | hint+program→Python | đọc đủ fact |
|---|---|---|---|---|---|---|
| span | 198 | 88.9 | 91.4 | – | – | 92.4 |
| multi-span | 79 | 94.9 | 97.5 | – | – | 96.2 |
| arithmetic | 316 | 39.9 | 53.5 | 79.7 | 87.0 | 87.3 |
| count | 7 | 57.1 | 57.1 | – | – | 71.4 |
| all | 600 | 63.5 | 71.8 | – | – | 90.0 |

## Phân loại câu `direct` sai

Tổng câu sai: 219 / 600.

| loại lỗi | số câu | % lỗi |
|---|---|---|
| đọc đủ, biểu thức đúng → chỉ tính nhầm (calculation) | 132 | 60.3 |
| đọc sai / chọn nhầm ô (perception, localization) | 46 | 21.0 |
| đọc đủ, biểu thức sai (reasoning: phép toán / thứ tự / điều kiện) | 25 | 11.4 |
| đọc đủ, trả lời sai (chọn / định dạng span) | 16 | 7.3 |

## Mức cải thiện tối đa của từng cách sửa (CI 95 % bootstrap theo tài liệu)

- **Bằng chứng đầy đủ** (hint − direct, mọi câu): +8.3 [+5.4, +11.2]
- **Thực thi bằng Python** (program − direct, arithmetic): +39.9 [+34.2, +45.4]
- **Oracle chọn đúng giữa direct và program** (arithmetic): +42.1 [+36.4, +47.4]
- **Bằng chứng đầy đủ + Python** (hintprog − direct, arithmetic): +47.2 [+40.8, +53.2]
- Scale sai dù giá trị đúng: 103 câu / 217 câu có scale.

## Theo số fact

| n_facts | n | direct | hint | đọc đủ |
|---|---|---|---|---|
| 1 | 202 | 88.1 | 91.1 | 92.6 |
| 2 | 341 | 50.4 | 63.0 | 89.7 |
| 3 | 47 | 55.3 | 59.6 | 89.4 |
| 4+ | 10 | 50.0 | 40.0 | 50.0 |