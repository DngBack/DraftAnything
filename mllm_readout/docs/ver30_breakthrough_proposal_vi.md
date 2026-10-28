# Đề xuất ver30: "Ai đang nhìn — model hay probe?"

**Ngày:** 16/10/2026. **Trạng thái:** đề xuất + bằng chứng exploratory đã chạy hôm nay (chưa pre-register, chạy sau khi đã xem dữ liệu ver28/29). Paper ver28 và kết quả cũ không bị sửa.

## 1. Ý tưởng một câu

Mọi claim "MLLM thấy nhưng không nói" so sánh probe với output của model. Paper hiện tại mới kẹp claim đó **từ phía đầu ra** (calibration trên logits ứng viên). Đề xuất: kẹp thêm **từ phía đầu vào**, bằng probe trên chính thứ vision encoder đưa vào LLM (projector output, layer 0) và trên một encoder độc lập không thuộc MLLM (SigLIP2). Nếu một probe trên *ảnh* đã giải được bài, thì việc probe trên hidden state giải được bài không nói gì về việc MLLM "thấy" — **chính probe đang làm phần nhìn**.

Kẹp từ hai phía tách được "seeing but not saying" thành các sự kiện khác nhau trên pipeline, và mỗi loại dự đoán một cách sửa khác nhau. Đây là đóng góp khái niệm mới, vượt khỏi "calibration là baseline" (đã bị Koushik et al. chiếm một phần).

## 2. Bằng chứng đã chạy hôm nay

Script: `explore_flow.py` (→ `results/explore_flow.csv`, `explore_flow_layers.csv`), `explore_extenc.py`, `explore_hcfg.py` (→ `results/explore_hcfg/`). Probe = `pilot.probe_cv` của paper, cross-fit 5 fold, nhóm theo ảnh ở các cell generation.

### 2.1 Illusions (vùng LOSS chủ lực của paper) là tín hiệu có sẵn từ đầu vào

Trung bình qua 9 model; "added" = probe ở answer token trừ probe ở projector output (layer 0).

| Task (setting) | raw | probe answer token | probe projector L0 | added | LOSS errors | LOSS cũng giải được từ L0 |
|---|---:|---:|---:|---:|---:|---:|
| **illusions (gen)** | 52.6 | 86.2 | **85.9** | **+0.2** | 2144 | **1936 (90%)** |
| carlogos (gen) | 6.2 | 100 | 100 | 0.0 | 119 | 119 |
| dice (gen) | 14.7 | 96.4 | 85.9 | +10.5 | 179 | 107 |
| color_cf (closed) | 88.0 | 97.2 | 91.2 | +6.0 | 66 | 41 |
| countbench (gen) | 70.0 | 82.8 | 41.4 | **+41.4** | 264 | 120 (45%) |
| count synthetic (closed) | 73.6 | 85.2 | 52.4 | **+32.8** | 195 | 95 |
| count_real (closed) | 85.7 | 89.5 | 41.7 | **+47.7** | 75 | 38 |

Theo từng model, illusions: Qwen3-VL-32B có L0 90% so với answer 87%, 254/276 LOSS inherited. Gemma-3-4B: 93% so với 87%, 182/187. Gemma-3-27B: 90% so với 92%, 177/183. Qwen3-VL-4B: 84% so với 88%, 246/292.

**Đối chứng ngoài MLLM (SigLIP2-base, linear probe, cùng fold nhóm theo ảnh):**

| Task | SigLIP2 mean-pool | SigLIP2 lưới 4×4 | Probe trong MLLM |
|---|---:|---:|---:|
| illusions | 95.2 | **99.0** | 81–92 |
| countbench | 42.3 | 47.4 | 79–92 |

**Depth profile (Qwen3-VL-4B, illusions):** probe ở answer token đã đạt 88% từ layer 4, đỉnh 94% ở layer 20, cuối cùng 88%. Logit-lens Yes/No ở mức 50% cho đến layer 34/36. Model trả lời gần như hằng số theo từng họ illusion (VH và Poggendorff luôn "No"; Ponzo/Zöllner/Müller-Lyer hầu như "Yes"). CountBench thì ngược lại: probe answer token tăng đều từ 10% (L0) lên 84% (cuối), còn trung bình image token không bao giờ vượt 60%.

**Diễn giải:** trên illusions, Car Logos và Dice, ảnh đã tách được tuyến tính trước khi vào LLM. LLM không thêm thông tin giải mã được nào, và câu trả lời là prior theo họ ảnh. Claim của paper rằng "probe đọc được đáp án 81–92% trong khi model ở mức ngẫu nhiên" là đúng về mặt số, nhưng không phải bằng chứng model "thấy". Trên counting, LLM thực sự *tạo ra* bằng chứng mà ảnh thô không cho (+33 đến +48 điểm).

### 2.2 Sửa lỗi không cần nhãn thất bại đúng nơi dự đoán

`explore_hcfg.py`, Qwen3-VL-4B, illusions (264 item, 1 độ phân giải). Thay state ở layer ℓ bằng h_xám + γ(h_thật − h_xám), không nhãn, không probe:
- raw 57.2%; logit-contrast (kiểu VCD) tốt nhất 60.2%; hidden guidance tốt nhất 60.6% (L18, γ=2, mọi vị trí text sau ảnh); ở L9 với γ lớn giảm về 50%.
- Steering α=16 trước đây (52→79%) dùng hướng học từ **nhãn**. Hệ quả: nó có thể chỉ là tiêm quyết định của probe vào model ("probe-in-disguise"), không chứng minh model có dùng bằng chứng.
- Đã chạy đủ 2 model × 3 task. "Best" là cấu hình tốt nhất trong 32 cấu hình (layer × γ × scope), chọn trên chính tập test, nên lạc quan:

| Cell | raw | guidance tốt nhất | sửa / phá | logit-contrast tốt nhất |
|---|---:|---:|---:|---:|
| Qwen3-VL-4B illusions | 57.2 | 60.6 | 20 / 11 | 60.2 |
| Gemma-3-4B illusions | 52.3 | 54.2 | 8 / 3 | 51.9 |
| Qwen3-VL-4B MMVP | 78.3 | 80.3 | 7 / 1 | 79.0 |
| Gemma-3-4B MMVP | 64.0 | 64.3 | 7 / 6 | 64.3 |
| Qwen3-VL-4B CountBench | 76.7 | 78.5 | 11 / 3 | 78.3 |
| **Gemma-3-4B CountBench** | 63.6 | **70.5** | 43 / 13 | 65.2 |

  Hướng kết quả khớp dự đoán: guidance không nhãn gần như vô tác dụng trên illusions (INHERITED) và MMVP (UNSEEN). Gain duy nhất đáng kể là Gemma CountBench, nơi LLM thực sự tính ra bằng chứng (+6.9; logit-contrast chỉ +1.6, tức là guidance ở hidden state làm được điều logit-space không làm được). Đây mới là tín hiệu sơ bộ: cần chọn (layer, γ) trên split riêng và kiểm tra thêm model.

### 2.3 Giới hạn của bằng chứng hiện tại

- Tỷ lệ "inherited" chưa hiệu chỉnh cho mức đoán may. Với task nhị phân, probe ngẫu nhiên đúng khoảng 50%. Illusions 90% là vượt xa, nhưng POPE/MMVP/tally ~50% thì chỉ là chance.
- Input control chỉ có nghĩa khi **câu hỏi cố định theo task/họ**. POPE, MMVP và attr đổi câu hỏi theo item, nên probe chỉ-ảnh ở mức chance theo cấu trúc. Các cell đó phải dùng probe input có điều kiện theo câu hỏi, hoặc loại khỏi phép so sánh.
- Mean-pool làm mất bố cục không gian. Với spatial, cần probe lưới (như SigLIP 4×4) trên image token L0. Hiện "added" của spatial (+43) có thể bị thổi phồng.

## 3. Khung mới: phân loại lỗi kẹp hai phía

Với mỗi lỗi, dùng bốn readout cùng nhãn, cùng fold: (a) calibrator trên logits ứng viên, (b) probe ở answer token, (c) probe input (projector L0, lưới không gian), (d) probe encoder độc lập.

| Nhãn | Định nghĩa vận hành | Ý nghĩa | Dự đoán cách sửa |
|---|---|---|---|
| **BIAS** | (a) đúng | Ranh giới quyết định sai | Calibration |
| **UNSPOKEN** | (a) sai, (b) đúng, (c)(d) sai | LLM *tính* ra bằng chứng nhưng không nói | Guidance không nhãn **và** steering |
| **INHERITED** | (a) sai, (b) đúng, (c) hoặc (d) đúng | Bằng chứng có sẵn từ ảnh, LLM không tính | **Không** sửa được bằng guidance không nhãn; chỉ "compute" (thinking/decompose) hoặc steering có nhãn (probe-in-disguise) |
| **DROPPED** | (b) sai, (c)/(d) đúng | Đầu vào có, LLM làm mất | Không có readout nào trong LLM sửa được |
| **UNSEEN** | không readout nào đúng | Không nằm trong biểu diễn | Độ phân giải cao hơn / crop |

Đại lượng tóm tắt mỗi cell là **LLM value-added decodability** Δ = acc(answer probe) − max(acc input probe, acc external probe). Δ≈0 nghĩa là claim "encoded-but-unused" không nói gì về model.

## 4. Thí nghiệm then chốt để lên mức "đột phá": ma trận sửa lỗi

Nhãn vận hành (reviewer phàn nàn "không nhân quả") trở thành nhãn **có nghĩa nhân quả** nếu chúng dự đoán trước can thiệp nào sửa được lỗi nào.

- **Hàng:** loại lỗi (gán trên fold riêng, trước khi can thiệp).
- **Cột:**
  - R1 calibration.
  - R2 guidance thị giác không nhãn (`explore_hcfg.py`).
  - R3 steering có nhãn (đã có, `steer_g3.py`).
  - R4 "compute": Qwen3.5-2B/4B/9B với `enable_thinking=True`, và prompt phân rã ("Which line is longer?" rồi hỏi tiếp).
  - R5 "see more": độ phân giải ×2 / crop. VLMBias đã có sẵn 3 độ phân giải; CountBench có thể upscale.
- **Ô:** tỷ lệ lỗi được sửa và tỷ lệ câu đúng bị phá.
- **Pre-register:** đường chéo trội (BIAS→R1, UNSPOKEN→R2, INHERITED→R4 nhưng không R2, UNSEEN→R5), kiểm định hoán vị nhãn hàng.

**Đối chứng "probe-in-disguise":** steering với hướng học từ nhãn do **probe SigLIP dự đoán** (không dùng gì từ kiến thức của MLLM). Nếu nó sửa illusions ngang steering bằng nhãn thật, thì thành công của steering trên lỗi INHERITED không chứng minh model biết gì. Đây là một phản biện trực tiếp áp dụng được cho 2604.09364, 2610.17976 và 2610.05715.

## 5. Kiểm toán tài liệu ("bảng sống sót")

Áp hai đối chứng vào chính setting của các claim đã công bố:
- Visual CounterFact (2604.09364): màu nằm ngay trong pixel, input L0 = 91%. Dự đoán phần lớn là BIAS + INHERITED.
- CountBench re-prompting (2607.09544): Δ = +41. Đây là claim *sống sót* đối chứng đầu vào, nhưng phần lớn là BIAS.
- Nếu dữ liệu công khai: 2610.17976 (depth order), 2610.05715 (rotation), What'sUp (2610.06324 "Readout Blindness").

Một bảng kiểu "k trên N claim 'encoded-but-unused' sống sót cả hai đối chứng" là loại kết quả cấp lĩnh vực, giúp bài nổi bật.

## 6. Định vị so với nghiên cứu gần

Theo kiểm tra arXiv hôm nay của subagent: so sánh encoder với LLM ở mức tổng thể đã có (Fu et al. 2506.08008; Chandhok et al. 2507.10442; VisPer-LM 2412.09585). Chưa thấy ai dùng probe đầu vào làm **đối chứng theo từng lỗi** cho claim encoded-but-unused, chưa thấy ma trận sửa lỗi có đường chéo trội, và chưa thấy phản biện "probe-in-disguise" bằng nhãn từ encoder ngoài. Cần đọc toàn văn 2610.17976, 2610.05715, 2610.13527 (S-Space, có tách perception/computation) trước khi viết claim "đầu tiên". Có một bài OpenReview "Know but can't Say" (forum qUeA0BYyAk) chưa kiểm chứng được, cần đọc tay.

## 7. Cấu trúc paper mới

Tiêu đề đề xuất: **"Who Is Doing the Seeing? Bracketing 'Seeing but Not Saying' in Multimodal LLMs"**.

- C1. Hai đối chứng bắt buộc cho mọi claim probe-beats-model: đầu ra (calibration, đã có) và đầu vào (mới).
- C2. Áp lên 9 model × ~13 task: BIAS chiếm ưu thế trên counting/colour. Vùng "loss" nổi tiếng nhất (illusions) là INHERITED: LLM thêm 0 điểm decodability, câu trả lời là prior theo họ. Bằng chứng do LLM *tính ra* nhưng không nói tập trung ở counting.
- C3. Ma trận sửa lỗi xác nhận nhân quả cho nhãn. Thêm đối chứng probe-in-disguise cho steering.
- C4. Bảng sống sót cho các claim đã công bố + checklist (calibrate, REMAP, chia theo ảnh, probe đầu vào).

Giữ từ ver28: §4.2 (tái lập CounterFact/CountBench), calibration trong generation, cảnh báo VLMBias. Chuyển xuống supplement: một phần §4.5 (16 test cũ), gated repair âm.

## 8. Lịch đến 16/11

| Ngày | Việc | Gate |
|---|---|---|
| 17–19/10 | Probe đầu vào chuẩn: lưới không gian trên image token L0 (trích xuất lại, rẻ), hiệu chỉnh chance, lưu dự đoán từng item; thêm encoder ngoài thứ hai (DINOv2) | Illusions vẫn Δ≈0 và CountBench Δ≫0 với probe lưới |
| 20–21/10 | Viết protocol pre-register cho ma trận sửa lỗi + probe-in-disguise | Protocol có timestamp trước khi chạy |
| 22–29/10 | GPU: R2 (mở rộng), R4 Qwen3.5 thinking + phân rã, R5 độ phân giải/crop, steering bằng nhãn SigLIP | Đường chéo trội ≥3/4 hàng, p<0.05 hoán vị |
| 30/10–3/11 | Bảng sống sót (CounterFact, CountBench, + 1–2 claim mới nếu có dữ liệu) | ≥1 claim công bố bị đối chứng đầu vào giải thích |
| 4–14/11 | Viết lại paper (C1–C4), hình evidence-flow, ma trận | Body 8 trang |
| 10/11 | Đăng ký abstract | — |

**Nếu ma trận thất bại:** vẫn còn C1 + C2 + probe-in-disguise. Đó vẫn là phản biện mạnh hơn rõ so với ver28, chỉ thiếu phần "nhân quả".

## 9. Đánh giá thẳng về điểm số

Không thể hứa 9–10; điểm phụ thuộc kết quả và reviewer.

- ver28 được hai review mô phỏng chấm 7/10 và 5/10, chủ yếu vì novelty vừa phải và nhãn không nhân quả.
- Đề xuất này nhắm đúng hai điểm đó: (i) một đối chứng mới, đơn giản, áp được cho cả một dòng nghiên cứu; nó đảo ngược diễn giải ví dụ nổi bật nhất của chính paper và của 2604.09364. (ii) Nhãn được kiểm chứng bằng can thiệp.
- Nếu C1–C3 đều thành công, bài chuyển từ "paper phân tích cẩn thận" sang "paper đổi cách cả lĩnh vực đọc probe". Đó là loại bài có cơ hội vào mức cao nhất.
- Rủi ro chính: (a) probe lưới làm mờ ranh giới INHERITED/UNSPOKEN trên spatial; (b) reviewer cho rằng "probe trên ảnh giải được" là hiển nhiên với dataset tổng hợp. Cách phòng: đưa số trên ảnh thật (CountBench, What'sUp, CounterFact) lên trước, và nhấn mạnh rằng các bài đã công bố *không* làm đối chứng này.
