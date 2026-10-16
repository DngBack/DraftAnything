# Tái lập setting 2607.09544 trên CountBench (2026-10-14)

Code: `count_repro.py`. Số liệu: `results/countrepro_*.npz` (logit), `results/countgen_*.npz` (sinh tự do). Logs `logs/count_repro*.log`, `logs/count_gen.log`.
Setting của họ: CountBench số đếm 1–9 (437 ảnh), chia 50/50 phân tầng; system "Output only the final count in the format <count>."; prompt "How many object(s) are in this image? Answer:"; sửa lỗi = hỏi lại "That's definitely wrong. The correct count should be:".

## 1. Sinh tự do + tách số (cách chấm của họ), toàn bộ 437 ảnh

| Model | Gốc (ta) | Gốc (họ) | Hỏi lại toàn bộ (ta) | Hỏi lại toàn bộ (họ) | Detector + hỏi lại (họ) |
|---|---|---|---|---|---|
| Qwen3-VL-2B ★ | 72.1 | 72.92 | 71.6 | 78.79 | 79.31 |
| Qwen3-VL-8B ★ | 78.9 | 79.92 | **72.1** | **89.02** | 88.59 |
| Qwen3-VL-4B | 74.4 | — | 72.8 | — | — |
| InternVL3.5-8B | 69.8 | — | 38.7 | — | — |
| Gemma-3-4B | 62.9 | — | 40.3 | — | — |

Độ đúng gốc khớp (±1 điểm) nhưng lợi ích của hỏi lại **không tái lập được**: với ta hỏi lại luôn làm giảm (−0.5 đến −31 điểm), như kỳ vọng khi bảo model "chắc chắn sai" — nó đổi cả câu đúng. Bản logit (một lượt / nhiều lượt) cho kết quả tương tự (Qwen-8B 78.9 / 73.4).

## 2. Calibration logit (nửa kiểm tra, 20 lần chia phân tầng 50/50)

| Model | Gốc | Calib (nửa học, ~218 nhãn) | Calib (5 nhãn/lớp) | Calib lượt 1+2 |
|---|---|---|---|---|
| Qwen3-VL-2B ★ | 73.0 | **85.7 (+12.7)** | 78.3 | 86.0 |
| Qwen3-VL-4B | 76.4 | 83.2 (+6.8) | 77.3 | 82.9 |
| Qwen3-VL-8B ★ | 80.4 | 83.3 (+2.9) | 78.5 | 85.2 |
| InternVL3.5-8B | 67.2 | 78.0 (+10.8) | 72.9 | 77.0 |
| Gemma-3-4B | 63.0 | 70.2 (+7.2) | 66.0 | 68.2 |

Probe/detector của họ cũng học trên nửa học → calibration dùng cùng lượng nhãn, không cần lượt suy luận thêm.

## Kết luận
- Calibration logit tăng 5/5 model (+2.9 đến +12.7); trên Qwen3-VL-2B vượt con số họ báo (85.7 vs 79.3).
- Trên Qwen3-VL-8B, con số họ báo (88.6) cao hơn calibration của ta (83.3), nhưng chính baseline "hỏi lại toàn bộ" của họ không tái lập được (ta: 72.1 vs họ: 89.0) → không thể so trực tiếp; cần liên hệ tác giả / code của họ.
- Với paper: báo cáo trung thực cả hai — calibration là baseline đơn giản mà các bài "encoded but unused" thiếu; lợi ích của re-prompting trong 2607.09544 không tái lập được trong pipeline độc lập.
