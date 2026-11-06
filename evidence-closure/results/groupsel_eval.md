# Learned group selector — recall@k của nhóm gold

Nhóm = số + closure heuristic. Học từ fact gold của TAT-DQA train (listwise). `lex` = số từ nhãn trùng câu hỏi.

| tập | scorer | n | @1 | @2 | @4 | @8 |
|---|---|---|---|---|---|---|
| train-val | lex | 935 | 31.0 | 52.1 | 63.3 | 74.0 |
| train-val | learned | 935 | 35.9 | 60.7 | 76.6 | 86.2 |
| dev E1 | lex | 600 | 32.2 | 53.6 | 64.4 | 73.8 |
| dev E1 | learned | 600 | 38.6 | 64.8 | 78.4 | 86.8 |
| test E3 | lex | 600 | 35.1 | 54.8 | 65.2 | 74.4 |
| test E3 | learned | 600 | 39.8 | 64.7 | 76.1 | 83.6 |