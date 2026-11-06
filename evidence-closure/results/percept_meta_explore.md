# Thăm dò sau pilot B: log-prob có hơn cỡ chữ và độ dài số không?

## qwen25vl7b

AUROC theo từng mức, ảnh orig (dự đoán `copy` đọc đúng):

| mức | log-prob TB | log-prob min | chiều cao chữ | số chữ số |
|---|---|---|---|---|
| s100 | 0.71 | 0.65 | 0.39 | 0.36 |
| s60 | 0.76 | 0.73 | 0.44 | 0.33 |
| s50 | 0.81 | 0.72 | 0.48 | 0.39 |
| s40 | 0.84 | 0.81 | 0.53 | 0.51 |
| s30 | 0.82 | 0.78 | 0.50 | 0.45 |

Cặp bất đồng trong cùng ô, cùng mức (orig/null/cf): bản đọc đúng có log-prob cao hơn bản đọc sai không?

| mức | số cặp | tỷ lệ đúng > sai, log-prob TB | log-prob min | chênh TB (nat) |
|---|---|---|---|---|
| s100 | 6 | 33 % | 17 % | -0.005 |
| s60 | 6 | 67 % | 83 % | +0.047 |
| s50 | 18 | 56 % | 56 % | +0.059 |
| s40 | 44 | 59 % | 55 % | +0.079 |
| s30 | 40 | 85 % | 80 % | +0.109 |
| all | 114 | 67 % | 63 % | +0.080 |

Trung bình theo câu, tỷ lệ đúng > sai: 67 % [57, 77], 40 câu có cặp bất đồng.

Ví dụ đọc sai, ảnh orig, s40/s30: $(11,061) ← (8,690) · $(11,863) ← (8,690) · $11.1 ← $11,436 · $11,535 ← $11,436 · 125.0 ← 20.0 · 100.0 ← 20.0 · 2,591 ← 2,597 · 542 ← (3,382)

## qwen3vl8b

AUROC theo từng mức, ảnh orig (dự đoán `copy` đọc đúng):

| mức | log-prob TB | log-prob min | chiều cao chữ | số chữ số |
|---|---|---|---|---|
| s100 | 0.83 | 0.83 | 0.35 | 0.32 |
| s60 | 0.74 | 0.71 | 0.45 | 0.33 |
| s50 | 0.75 | 0.74 | 0.50 | 0.32 |
| s40 | 0.85 | 0.84 | 0.62 | 0.35 |
| s30 | 0.82 | 0.80 | 0.74 | 0.51 |

Cặp bất đồng trong cùng ô, cùng mức (orig/null/cf): bản đọc đúng có log-prob cao hơn bản đọc sai không?

| mức | số cặp | tỷ lệ đúng > sai, log-prob TB | log-prob min | chênh TB (nat) |
|---|---|---|---|---|
| s100 | 8 | 50 % | 50 % | -0.006 |
| s60 | 8 | 50 % | 50 % | +0.035 |
| s50 | 26 | 88 % | 88 % | +0.041 |
| s40 | 44 | 70 % | 73 % | +0.057 |
| s30 | 26 | 81 % | 81 % | +0.061 |
| all | 112 | 74 % | 75 % | +0.048 |

Trung bình theo câu, tỷ lệ đúng > sai: 74 % [63, 85], 41 câu có cặp bất đồng.

Ví dụ đọc sai, ảnh orig, s40/s30: $(8,699) ← (8,690) · (2,665) ← (8,690) · $111,436 ← $11,436 · $11.4 million ← $11,436 · 2,767 ← 2,755 · 100.0 ← 20.0 · 125.0 ← 20.0 · 1,875 ← 4,468
