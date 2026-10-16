# G1.5 — kết quả (2026-10-14)

Protocol: `docs/g15_protocol.md` (đăng ký trước). Số liệu: `results/g15_table.csv` (`analyze_g15.py`), `results/*_summary.json`.
16/16 ô xong (`gemma3_27b_count_real` chạy lại sau OOM do GPU bị job khác chiếm).

## Bảng phân rã lỗi (5-fold CV)

| Ô | n lỗi | Model | Calib | Probe | bias | loss | perc |
|---|---|---|---|---|---|---|---|
| Qwen3-VL-4B · count | 165 | .725 | .892 | .887 | .79 | .06 | .15 |
| Qwen3-VL-8B · count | 118 | .803 | .917 | .933 | .77 | .09 | .14 |
| InternVL3.5-8B · count | 168 | .720 | .837 | .892 | .58 | .19 | .24 |
| Gemma-3-4B · count | 305 | .492 | .485 | .627 | .18 | .32 | .50 |
| Gemma-3-27B · count | 162 | .730 | .788 | .780 | .43 | .20 | .38 |
| Qwen3-VL-4B · spatial | 61 | .898 | .967 | 1.00 | .74 | .26 | 0 |
| Qwen3-VL-8B · spatial | 8* | .987 | .997 | 1.00 | .88 | .13 | 0 |
| InternVL3.5-8B · spatial | 35 | .942 | .963 | .998 | .49 | .51 | 0 |
| Gemma-3-4B · spatial | 316 | .473 | .543 | .912 | .27 | .65 | .08 |
| Gemma-3-27B · spatial | 23* | .962 | .973 | .998 | .43 | .57 | 0 |
| Qwen3-VL-4B · CountBench | 50 | .886 | .908 | .911 | .36 | .18 | .46 |
| Qwen3-VL-8B · CountBench | 34 | .922 | .924 | .922 | .26 | .12 | .62 |
| InternVL3.5-8B · CountBench | 89 | .796 | .860 | .863 | .49 | .12 | .38 |
| Gemma-3-4B · CountBench | 104 | .762 | .780 | .824 | .17 | .29 | .54 |
| Gemma-3-27B · CountBench | 73 | .833 | .849 | .874 | .21 | .32 | .48 |

\* < 30 lỗi, không tính vào luật quyết định.

## Phán quyết (luật đăng ký trước)
13 ô đủ lỗi: calib sửa ≥50% khoảng cách ở 7/13 (0.54), loss chiếm ưu thế ở 4/13 (0.31) → **mixed → paper = chẩn đoán 4 nhánh**.

## Phát hiện
1. **Lỗi đếm tổng hợp = lệch calibration** (bias 58–79% ở Qwen/InternVL; Qwen đếm thiếu đúng 1).
2. **Lỗi đếm ảnh thật = perception** (perc 38–62%, cả Gemma-27B: 48%): probe không hơn model → thông tin không có ở bất kỳ tầng nào.
3. **Readout loss thật tập trung ở spatial và co lại theo scale**: Gemma 4B→27B spatial, lỗi loss 206 → 13 (acc .47 → .96).
4. **Bàn giao image→last token**: image token hết quyết định ở 44–56% độ sâu trong 10/10 ô (4 họ, 4B–27B); điểm last token tiếp quản dao động 61–90% (InternVL count: không vị trí đơn lẻ nào ≥50%).
5. **Repair khi lỗi là loss hoạt động tốt**: spatial sửa 74–90% lỗi, làm hỏng ≤1.4% (Qwen 4B/8B, InternVL, Gemma-27B); count chỉ sửa 1–55%.

## Hệ quả cho paper
"Encoded but unused" không phải một hiện tượng mà là 3 cơ chế khác nhau tuỳ (model, tác vụ, phân phối); claim đơn lẻ của các bài trước bị nhiễu bởi thiếu baseline calibration. Paper: chẩn đoán từng lỗi + repair có cổng (chỉ can thiệp khi chẩn đoán = loss).

## Việc tiếp theo (G2, hạn 27/10)
1. Thêm họ lỗi: attribute/màu, OCR, fine-grained; spatial ảnh thật (What'sUp).
2. Chạy calibration baseline trên benchmark của các bài "encoded but unused" cạnh tranh.
3. Repair có cổng: chẩn đoán → can thiệp → đo net accuracy so với calib-only và steering-all.
