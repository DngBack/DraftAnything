# Learned closure vs heuristic (weak labels)

Train: 8823 câu / 1841 tài liệu; held-out train: 935 câu. Dev = tập E1 (600 câu). τ chọn trên held-out train để kích thước closure bằng heuristic (10.05 từ ngoài seed).

Nhãn yếu: word mà câu hỏi nhắc tới, hoặc từ scale gold. `query` thấy đặc trưng trùng-từ-với-câu-hỏi nên recall của nó gần như là trần (nhãn được định nghĩa từ chính câu hỏi), chỉ để tham khảo.

| Closure | split | size | precision | recall | F1 |
|---|---|---|---|---|---|
| heuristic | train-val | 10.0 | 0.382 | 0.259 | 0.309 |
| geo | train-val | 10.0 | 0.561 | 0.379 | 0.453 |
| query | train-val | 10.0 | 0.952 | 0.644 | 0.768 |
| heuristic | dev E1 | 9.6 | 0.401 | 0.243 | 0.303 |
| geo | dev E1 | 9.9 | 0.588 | 0.367 | 0.452 |
| query | dev E1 | 10.0 | 0.963 | 0.613 | 0.749 |

```
{
 "geo": {
  "tau": 0.8085651397705078,
  "pairs": 3439940,
  "pos_rate": 0.0716274157166481
 },
 "query": {
  "tau": 0.8038311004638672,
  "pairs": 3439940,
  "pos_rate": 0.0716274157166481
 }
}
```