# G2 — kết quả (2026-10-14)

Protocol: `docs/g2_protocol.md`. Số liệu: `results/g2_table.csv` (`analyze_g2.py`), `results/fewshot_calib.csv` (`fewshot_calib.py`).

## Phân rã lỗi, ô có ≥ 30 lỗi

| Ô | n lỗi | Model | Calib (CV) | Probe | bias | loss | perc | Gated | Gated làm hỏng |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-4B · CounterFact | 56 | .886 | .957 | .972 | .70 | .14 | .16 | .953 | 0.7% |
| Qwen3-VL-8B · CounterFact | 65 | .868 | .941 | .970 | .65 | .17 | .18 | .945 | 0.5% |
| InternVL3.5-8B · CounterFact | 56 | .886 | .963 | .980 | .77 | .09 | .14 | .955 | 0.7% |
| Gemma-3-4B · CounterFact | 65 | .868 | .939 | .970 | .65 | .22 | .14 | .945 | 1.4% |
| Gemma-3-27B · CounterFact | 50 | .899 | .941 | .966 | .50 | .26 | .24 | .937 | 1.1% |
| Qwen3-VL-4B · What'sUp | 49 | .940 | .998 | .998 | .96 | 0 | .04 | .988 | 0.9% |
| Gemma-3-4B · What'sUp | 381 | .535 | .710 | .938 | .51 | .42 | .07 | .830 | 3.2% |
| Gemma-3-27B · What'sUp | 201 | .755 | .963 | .994 | .87 | .12 | .02 | .965 | 1.1% |
| Gemma-3-4B · attr | 33 | .960 | .963 | .971 | .21 | .12 | .67 | .959 | 1.4% |

attr (sau khi sửa bảng màu) và What'sUp quá dễ với Qwen-8B / InternVL (≤ 11 lỗi).

## Dự đoán đăng ký trước
- P1 (What'sUp có nhiều perc hơn spatial tổng hợp): **sai** — lỗi What'sUp chủ yếu là bias (51–96%), gần như không có perc.
- P2 (attr giống count, bias chiếm ưu thế): không kiểm được — attr quá dễ (chỉ 1 ô đủ lỗi, ô đó perc chiếm ưu thế).
- P3 (CounterFact: bias ≥ 30% ở ≥ 3/5 model): **đúng, 5/5** (bias 50–77%).

## Quyết định
Gated ≥ max(calib, steer-all) ở 4/9 ô; làm hỏng 3.2% ở Gemma-4B · What'sUp → **diagnostic-only**; repair có cổng chuyển sang phụ lục.

## Phân tích hậu kiểm (không đăng ký trước): calibration ít nhãn
Affine refit logit đáp án của chính model, chỉ K ảnh có nhãn / lớp (20 lần lấy mẫu):

| Ô | Model | Batch cal (0 nhãn) | K=5 | K=20 |
|---|---|---|---|---|
| CounterFact (5 model) | .868–.899 | .890–.913 | .919–.955 | .947–.971 |
| Gemma-27B · What'sUp | .755 | .754 | .918 | .956 |
| Qwen-8B · count | .803 | .833 | .902 | .918 |
| Qwen-4B · count | .725 | .815 | .853 | .890 |

40 ảnh có nhãn (5/lớp) cho +3.3 đến +6.9 điểm trên benchmark của 2604.09364, nơi steering báo cáo +1.4 đến +3.8 (cần tái lập cùng model/setting trước khi so trực tiếp).
Batch calibration không nhãn chỉ giúp được một phần → "bias" gồm cả dịch prior (không cần nhãn) lẫn lệch ranh giới (cần vài nhãn).

## Đánh giá
Paper khả dĩ: *"Seeing but not saying is mostly miscalibration."* Bằng chứng mạnh nhất: trên chính benchmark của bài cạnh tranh, 50–77% lỗi "prior thắng thị giác" của 5 model là lệch ranh giới logit, sửa bằng 40 nhãn.
Rủi ro: (i) gần với dòng calibration của LLM (Calibrate Before Use, Batch Calibration) → phải định vị là phản biện cách diễn giải cơ chế, không phải phương pháp calibration mới; (ii) chỉ forced-choice, token đầu; (iii) model lớn nhất 27B; (iv) chưa tái lập đúng setting của 2604.09364 / 2607.09544.
