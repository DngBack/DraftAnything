# Pilot G1 — kết quả (2026-10-13)

Protocol: `docs/pilot_g1_protocol.md` (đăng ký trước, có deviation log). Code: `stimuli.py`, `pilot.py`. Số liệu: `results/*_summary.json`, `*_probes/patch/repair.parquet`, `*_states.npz`.
Thiết lập: 600 ảnh tổng hợp / ô, 64 cặp counterfactual cho patching, forced-choice trên token đáp án, GPU 0 (H200).

## Bảng chính

| Ô | Model đúng | Calib logit của chính model | Probe tốt nhất | Probe đúng trên lỗi | Probe − max(model, calib) | Repair: sửa / làm hỏng → net | G1a (gốc) | G1a (sau calib) | G1b |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-4B · count | 72.5% | **89.2%** | 88.7% (L35, last) | 75.8% | −0.5 | 55% / 5.1% → 84.0% | ✅ | ❌ | ❌ |
| Qwen3-VL-4B · spatial | 89.8% | 96.7% | 100% (L24, last) | 100% | +3.3 | 90% / 0% → 99.0% | ✅ | ❌ | ✅ |
| Gemma-3-4B · count | 49.2% | 48.5% | 62.7% (L31, last) | 46.2% | +13.5 | 32% / 13% → 58.7% | ❌ | ❌ | ❌ |
| Gemma-3-4B · spatial | 47.3% | 54.3% | 91.2% (L17, last) | 89.6% | **+36.8** | 48% / 5.3% → 70.0% | ✅ | ✅ | ❌ |

Probe nhãn xáo trộn ≈ mức may rủi ở mọi ô (12–25%).

## Patching: dấu hiệu route-then-read xuất hiện ở 4/4 ô

| Ô | Image token còn quyết định (≥50%) đến | Token cuối bắt đầu quyết định (≥50%) từ |
|---|---|---|
| Qwen · count | L20 (56% độ sâu) | L24 (67%) |
| Qwen · spatial | L16 (44%) | L24 (67%) |
| Gemma · count | L18 (53%) | L24 (71%) |
| Gemma · spatial | L16 (47%) | L24 (71%) |

Qwen · spatial có "khoảng trống" L18–22 (không vị trí nào quyết định) → thông tin đi qua token câu hỏi (khớp 2607.03358).

## Phán quyết theo tiêu chí đăng ký trước

- **G1: TRƯỢT.** G1a đạt 3/4 theo định nghĩa gốc nhưng chỉ 1/4 sau đối chứng calibration; G1b đạt 1/4 (hai ô trượt sát ngưỡng làm-hỏng 5%: 5.1%, 5.3%).
- Claim "mọi lỗi đều là readout bottleneck" **không được ủng hộ** ở thang 4B trên dữ liệu tổng hợp.

## Điều dữ liệu thực sự nói

1. **Đa số "thấy mà không nói" là lệch calibration.** Qwen · count: model đếm thiếu đúng 1 một cách hệ thống (5→4, 6→5, 7→6, 8→7); hiệu chỉnh affine 9 logit của chính model đạt bằng probe. Thông tin nằm ngay trong logit đầu ra.
2. **Mất thông tin thật ở readout có tồn tại nhưng hiếm**: Gemma · spatial, probe tầng 17 đạt 91% trong khi calib chỉ 54%; probe giảm dần từ L18 (90.8%) đến tầng cuối (84%) và câu trả lời còn 47%.
3. **Có lỗi perception thật**: Gemma · count, probe image-token tầng 0 chỉ ~52% → encoder đã mất thông tin; repair không giúp.
4. **Bàn giao image→last-token ở ~45–55% → ~70% độ sâu là phổ quát** (4/4 ô, 2 họ LLM).

## Đề xuất pivot (cần người dùng quyết)

Headline mới khả dĩ (kiểu "negative/fair-comparison result"): *"Most of 'seeing but not saying' in MLLMs is miscalibration; genuine readout loss is rare and sits at a universal route→read handoff."* Đóng góp: (i) chẩn đoán 4 nhánh cho từng lỗi — perception / routing / readout-bias (calib sửa được) / readout-loss (chỉ probe sửa được); (ii) yêu cầu baseline calibration cho cả dòng bài "encoded but unused"; (iii) repair có cổng (chỉ can thiệp khi chẩn đoán = readout-loss) để giữ tỉ lệ làm-hỏng < 5%.

## Việc tiếp theo trước khi tin bất kỳ claim nào

1. Thêm model: Qwen3-VL-8B, InternVL3.5-8B (tải), Gemma-3-27B (đã có) — readout-loss có biến mất khi model lớn?
2. Thêm dữ liệu thật: CountBenchQA, What'sUp; thêm họ lỗi attribute (Visual CounterFact).
3. Chạy lại calibration baseline trên các thiết lập của 2604.09364 / 2607.09544 để kiểm tra phát biểu (1) trên chính benchmark của họ.
4. Đầu ra "10" của bài đếm bị chấm theo token đầu "1" — vẫn là đáp án sai, không ảnh hưởng kết luận, nhưng nên giới hạn đáp án hoặc chấm theo chuỗi khi mở rộng.
