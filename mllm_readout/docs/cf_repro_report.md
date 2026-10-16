# Tái lập setting 2604.09364 trên Visual CounterFact (2026-10-14)

Code: `cf_repro.py`. Số liệu: `results/cfrepro_*.npz`, `results/cfsteer_*.json`, logs `logs/cf_repro*.log`, `logs/cf_steer.log`.
Setting của họ: prompt "What color is the {object}?", đúng nếu logit màu nhìn thấy > logit màu prior ở vị trí cuối; học trên ~200 mẫu, đánh giá trên 293.

## 1. Calibration logit (20 lần chia 200/293, trung bình ± sd)

| Model | Gốc | Batch cal (0 nhãn) | Bias 14 số (200 nhãn) |
|---|---|---|---|
| LLaVA-1.5-7B ★ | 90.3±1.0 | 92.5±0.9 | 93.8±0.8 |
| Qwen2-VL-7B ★ | 82.1±1.4 | 94.4±0.9 | 92.8±1.2 |
| Qwen3-VL-4B | 93.3±0.9 | 93.9±0.7 | 94.2±1.2 |
| Qwen3-VL-8B | 91.9±1.1 | 91.4±1.0 | 94.0±1.3 |
| InternVL3.5-8B | 93.4±0.8 | 91.5±1.0 | 94.8±0.9 |
| Gemma-3-4B | 67.6±1.6 | 59.4±1.5 | 77.2±1.8 |
| Gemma-3-27B | 84.8±1.2 | 81.3±1.4 | 86.3±1.5 |

★ model của 2604.09364. Độ đúng gốc của ta lệch so với bài họ (LLaVA 90.3 vs 86.0; Qwen2-VL 82.1 vs 88.4); chạy ảnh gốc không thu nhỏ không đổi (90.5 / 83.6) → khác biệt do chi tiết pipeline (template/biến thể token), không phải độ phân giải.

## 2. Steering tuyến tính của họ, cài lại trong cùng pipeline (1 lần chia, cùng split)
Hướng = trung bình (h_counterfact − h_gốc) theo từng tầng trên 200 mẫu học, cộng ở mọi vị trí; quét tầng {0,4,…} × α {0.2…3}; chọn trên 100 mẫu học.

| Model | Gốc | Batch cal | Bias 14 số | Steering (chọn trên train) | Steering (oracle test) |
|---|---|---|---|---|---|
| LLaVA-1.5-7B | 88.1 | 90.4 | 92.2 | 92.5 (L0, α=1) | 94.5 |
| Qwen2-VL-7B | 82.6 | 94.2 | 92.5 | 91.5 (L0, α=3 — biên lưới) | 91.5 |

Steering cài lại mạnh hơn số họ báo (+4.4 vs +1.4 trên LLaVA). Cấu hình tốt nhất luôn ở tầng 0 → steering hoạt động như một phép dịch toàn cục.

## Kết luận trung thực
- Calibration **ngang** steering (LLaVA: 92.2 vs 92.5; Qwen2-VL: 92.5 vs 91.5), với 14 tham số trên logit đầu ra, không cần ảnh gốc ghép cặp, không chạm activation.
- Không được claim "calibration thắng steering"; claim đúng: phần lớn lợi ích của steering giải thích được bằng dịch ranh giới quyết định.
- Calibration không nhãn không ổn định (hại Gemma 3–8 điểm) → phương pháp chính cần vài nhãn.
- Chưa làm: lưới α rộng hơn cho Qwen2-VL; nhiều lần chia cho steering; InternVL2-8B (cần trust_remote_code).

## 3. Đối chứng dịch nhãn: calibration có chỉ học "tránh màu prior" không? (`cf_repro.py --control`)
Ảnh gốc (chưa đổi màu) có đáp án đúng = màu prior. Nếu bias học được trên ảnh counterfactual chỉ là "tránh màu prior", nó sẽ làm hỏng ảnh gốc.
Bias "trộn" = học trên cả hai ảnh của 200 vật thể học. Đánh giá trên vật thể kiểm tra, 20 lần chia.

| Model | Gốc CF | Gốc ảnh gốc | Bias chỉ CF: CF | → ảnh gốc | Bias trộn: CF | → ảnh gốc |
|---|---|---|---|---|---|---|
| LLaVA-1.5-7B | 90.5 | 98.5 | 93.8 | 98.0 | 93.3 | 98.9 |
| Qwen2-VL-7B | 82.9 | 93.0 | 93.4 | 96.5 | 91.9 | 97.5 |
| Qwen3-VL-4B | 93.5 | 99.8 | 94.6 | 99.8 | 94.0 | 99.8 |
| Qwen3-VL-8B | 92.4 | 96.3 | 94.1 | 96.1 | 93.7 | 97.1 |
| InternVL3.5-8B | 93.6 | 98.4 | 94.8 | 98.4 | 94.3 | 98.5 |
| Gemma-3-4B | 67.6 | 47.1 | 77.2 | 43.0 | 69.1 | 58.1 |
| Gemma-3-27B | 84.8 | 99.0 | 86.3 | 98.4 | 85.7 | 98.9 |

→ Bias chỉ-CF hầu như không hại ảnh gốc (−0.6 đến +3.5; riêng Gemma-4B −4.1). Bias trộn cải thiện CF ở 7/7 model và không hại ảnh gốc (−0.1 đến +11.0).
Calibration không khai thác "đáp án luôn khác prior"; nó sửa một lệch thật của ranh giới quyết định.
