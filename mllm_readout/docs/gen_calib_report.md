# Calibration trong sinh tự do: CountBench và VLMBias (2026-10-14)

Code: `gen_calib.py` (`--task {countbench,vlmbias} --model M`, `--report`, `--stats`). Số liệu: `results/gencal_<task>_<model>.parquet`,
`results/gencal_stats.csv`. Logs `logs/gencal_v4{a,b}.log`. Bản chạy cũ (có lỗi, xem mục 4) ở `results/old/gencal_v{1,2,3}/`.

## 1. Cơ chế
Model sinh tự do, greedy, không đổi gì cho tới **bước quyết định** = bước đầu tiên mà argmax là một token đáp án
(bất kỳ chữ số nào với câu hỏi số; Yes/No với ảo ảnh; với VLMBias chỉ tính sau khi model đã viết "{", vì prompt yêu cầu đáp án trong ngoặc nhọn).
Tại bước đó: lấy điểm logit của các ứng viên (max trên các biến thể token *tự giải mã ra đúng ứng viên*), chọn lại lớp bằng
calibrator affine (hồi quy logistic trên điểm đã trừ trung bình, học trên nửa train, mỗi nhóm một calibrator), ép token của lớp đó,
chặn chữ số ở bước kế (để "1" không thành "12"), rồi để model sinh tiếp. Văn bản được tách đáp án như bình thường.

Đối chứng (đều học trên cùng nửa train):
- **const**: luôn trả lời nhãn phổ biến nhất của nhóm.
- **remap** (ánh xạ lại đầu ra): bảng "model nói X → đáp án thường là Y". Dùng nhãn nhưng không dùng gì ngoài câu trả lời của model;
  nếu calibration không vượt remap thì nó chỉ khai thác phân bố nhãn.
- **restricted**: argmax trong tập ứng viên tại bước quyết định (giới hạn miền đáp án, không calibration). Cao hơn raw tối đa +4.3 điểm, thấp hơn calibration nhiều → lợi ích không đến từ giới hạn miền.

Thống kê: "online" = độ đúng của văn bản sinh thật trên một lần chia 50/50 phân tầng, p = McNemar chính xác.
"offline 20 splits" = calibrator áp lên điểm bước quyết định, 20 lần chia (kiểm tra độ ổn định của lần chia).
Online khớp offline (CountBench: trùng hoàn toàn; VLMBias: lệch ≤ 7/1068 câu) → can thiệp lúc sinh làm đúng điều calibrator dự đoán.

## 2. CountBench (2607.09544 setting, đếm 1–9, nửa test n=220)

| Model | const | Gốc | Remap | **Calib (sinh thật)** | p vs gốc | p vs remap | Offline 20 splits: gốc / remap / calib |
|---|---|---|---|---|---|---|---|
| Qwen3-VL-2B | 13.6 | 70.5 | 72.3 | **83.2** | 8e-6 | 1e-4 | 72.7 / 74.1 / 85.1 |
| Qwen3-VL-4B | 13.6 | 72.7 | 73.6 | **83.6** | 8e-6 | 6e-5 | 74.7 / 76.7 / 83.3 |
| Qwen3-VL-8B | 13.6 | 77.7 | 80.0 | 82.3 | 0.02 | 0.30 | 79.0 / 81.2 / 84.2 |
| InternVL3.5-8B | 13.6 | 66.8 | 68.6 | **77.7** | 3e-4 | 2e-3 | 70.4 / 72.6 / 79.4 |
| Gemma-3-4B | 13.6 | 63.2 | 65.5 | 67.3 | 0.14 | 0.58 | 63.2 / 65.4 / 70.8 |

→ Calibration ở token đầu tăng độ đúng sinh tự do ở 5/5 model (+4.1 đến +12.7); có ý nghĩa và vượt remap ở 3/5 (Qwen-2B, Qwen-4B, InternVL).
So sánh với 2607.09544: "hỏi lại" của họ không tái lập được trong pipeline của ta (`docs/count_repro_report.md`); calibration không cần lượt suy luận thêm.

## 3. VLMBias (anvo25/vlms-are-biased, split "main", ảnh counterfactual; prompt gốc "… trả lời trong {}")
Chỉ giữ các nhóm có đáp án một token. Kiểm tra vai trò nhãn (`gold` và `expected_bias` của từng nhóm):
- **6 nhóm ảo ảnh** (Ebbinghaus, Müller-Lyer, Poggendorff, Ponzo, Vertical-Horizontal, Zöllner): Yes/No cân bằng, mỗi giá trị vừa là đáp án vừa là thiên kiến → **phép thử sạch duy nhất** (const = 50).
- Car Logos (gold 4/5, bias 3/4), Dice (gold 2/3, bias 3/4), Tally: một số giá trị **không bao giờ đúng** → calibrator dùng nhãn có thể học "đừng nói 3". Báo cáo cùng đối chứng remap.
- Animal Legs, Shoe Logos, Chess Grid: tập gold và tập bias rời nhau hoàn toàn → khai thác được trọn vẹn, chỉ ghi chú, không dùng.

### 3a. Ảo ảnh (sạch), nửa test n=396 / model

| Model | Gốc | Remap | **Calib (sinh thật)** | p vs gốc | p vs remap | Offline 20 splits: gốc / remap / calib |
|---|---|---|---|---|---|---|
| Qwen3-VL-2B | 49.0 | 50.0 | **60.1** | 0.004 | 0.004 | 49.1 / 50.2 / 58.9±2.7 |
| Qwen3-VL-4B | 55.8 | 57.3 | **62.4** | 0.03 | 0.15 | 54.8 / 55.5 / 60.9±1.8 |
| Qwen3-VL-8B | 51.3 | 51.3 | **58.6** | 0.047 | 0.06 | 51.8 / 50.9 / 57.8±2.3 |
| InternVL3.5-8B | 54.0 | 51.5 | 55.1 | 0.77 | 0.29 | 52.8 / 51.5 / 55.9±1.9 |
| Gemma-3-4B | 50.3 | 50.3 | 55.3 | 0.17 | 0.17 | 51.9 / 51.2 / 55.9±2.3 |

Model gốc ở mức đoán (49–56%). Calibration ở token Yes/No tăng 5/5 model (offline: +3.1 đến +9.8), có ý nghĩa ở Qwen (p<0.05 vs gốc ở 3/3);
mạnh nhất ở Zöllner (Qwen: 78–82%). Mức tăng **khiêm tốn**: model vẫn còn xa mức đúng trên ảo ảnh đã bị sửa — phần lớn lỗi ở đây là perception/thiên kiến thật, không chỉ lệch ranh giới.

### 3b. Nhóm đếm có nhãn bất đối xứng (đối chứng remap là thước đo đúng)

| Model | Car: remap → calib | Dice: remap → calib | Tally: remap → calib |
|---|---|---|---|
| Qwen3-VL-2B | 66.7 → 79.3 (p=.01) | 76.2 → 78.6 | 44.0 → 64.3 (p=.02) |
| Qwen3-VL-4B | 81.5 → 89.6 (p=.02) | 84.5 → 89.3 | 47.6 → 57.1 |
| Qwen3-VL-8B | 86.7 → 100.0 (p<1e-5) | 79.8 → 83.3 | 41.7 → 58.3 (p=.01) |
| InternVL3.5-8B | 75.6 → 100.0 (p<1e-9) | 88.1 → 86.9 | 40.5 → 20.2 (**tệ hơn**) |
| Gemma-3-4B | 80.7 → 85.2 | 73.8 → 79.8 | 46.4 → 47.6 |

Model gốc gần như luôn trả lời theo thiên kiến (đúng 0–18% trên Car/Dice). Remap — không nhìn ảnh, chỉ đổi nhãn — đã đạt 67–88%:
**bất kỳ phương pháp dùng nhãn nào trên các nhóm này phải so với remap, không phải với model gốc.** Calibration vượt remap có ý nghĩa ở Car (4/5) và Tally (3/5 Qwen), không ở Dice; InternVL Tally tệ hơn.

### Kết luận VLMBias
- Calibration ở token đầu áp dụng được cho sinh tự do trên benchmark khó, cơ chế khớp offline (lệch ≤7/1068 câu).
- Trên phần sạch (ảo ảnh) lợi ích thật nhưng nhỏ (+5 đến +11 điểm ở Qwen, không ý nghĩa ở Gemma/InternVL).
- Đóng góp phương pháp luận: **đối chứng remap** cho thấy phần lớn "lợi ích" trên các nhóm bất đối xứng của VLMBias đạt được mà không cần nhìn ảnh.
- InternVL suy luận dài: 70/1068 câu không có "{…}" trong 384 token → không quyết định, giữ đáp án gốc.

## 3c. MMVP (đối chứng đối xứng nhãn; 150 cặp ảnh, mỗi cặp hai đáp án ngược nhau; chia train/test theo cặp)

| Model | Raw | Calib. | 20 splits raw→calib | Cặp raw→calib |
|---|---|---|---|---|
| Qwen3-VL-2B | 72.7 | 74.7 | 73.2→73.7 | 48.0→50.7 |
| Qwen3-VL-4B | 75.3 | 76.7 | 77.9→78.5 | 52.0→54.7 |
| Qwen3-VL-8B | 76.0 | 76.7 | 78.1→79.3 | 53.3→56.0 |
| InternVL3.5-8B | 78.0 | 78.0 | 77.1→76.7 | 56.0→56.0 |
| Gemma-3-4B | 64.7 | 64.7 | 63.8→64.1 | 33.3→33.3 |

remap = raw ở mọi model (không có nhãn nào để khai thác). Không có thay đổi nào có ý nghĩa thống kê. Few-label (K=5/lớp) không vượt raw.
Đúng như dự đoán: lỗi MMVP là lỗi nhìn (perc), nên hiệu chỉnh ngưỡng không giúp. Đây là đối chứng âm cho thấy mức tăng ở CountBench/ảo ảnh không phải do bản thân can thiệp giải mã.
Lỗi đã sửa: tokenizer Qwen/InternVL gộp "(a" thành một token → thêm biến thể có ngoặc vào tập ứng viên (lần chạy đầu không có bước quyết định nào, đã chuyển vào results/old/mmvp_nodecision/).

## 3d. Thêm hai họ model (Qwen2-VL-7B, LLaVA-1.5-7B)

| Model | CountBench raw / remap / calib | Ảo ảnh raw → calib (p) | MMVP raw → calib | Car raw / remap / calib |
|---|---|---|---|---|
| Qwen2-VL-7B | 82.7 / 85.5 / 85.9 (p=.19) | 65.4 → 67.9 (.39) | 72.7 → 70.7 | 0.0 / 86.7 / 95.6* |
| LLaVA-1.5-7B | loại (27% không có bước quyết định) | 50.8 → 55.1 (.29) | loại (39%) | 0.0 / 85.9 / 100.0* |

Qwen2-VL đếm tốt nhất (82.7) và mức tăng của hiệu chỉnh không lớn hơn remap. Trên ảo ảnh, 7/7 model tăng nhưng chỉ ba Qwen3-VL có ý nghĩa thống kê.

## 3e. POPE (Yes/No cân bằng, 9000 câu, chia theo ảnh, test n=4500)

| Model | Raw | Calib | p | Tỉ lệ "Yes" raw→calib |
|---|---|---|---|---|
| Qwen3-VL-2B | 89.2 | 89.7 | .09 | 46→49 |
| Qwen3-VL-4B | 90.1 | 90.2 | .87 | 45→48 |
| Qwen3-VL-8B | 89.2 | 90.1 | 5e-5 | 46→48 |
| Qwen3-VL-32B | 89.8 | 91.1 | 2e-5 | 45→49 |
| InternVL3.5-8B | 88.8 | 90.1 | 2e-4 | 52→48 |
| Gemma-3-4B | 85.7 | 85.7 | 1 | 49→49 |
| Qwen2-VL-7B | 87.0 | 89.6 | 2e-10 | 40→47 |
| LLaVA-1.5-7B | 85.8 | 86.9 | 3e-3 | 43→48 |
| Gemma-3-27B | 84.0 | 84.2 | 0.48 | 56→54 |

Tỉ lệ "Yes" thô 40–56% (InternVL và Gemma-27B lệch về "Yes", còn lại lệch về "No"); hiệu chỉnh kéo 8 model về 47–49% và Gemma-27B về 54%, tăng +0.9 đến +2.6 ở 5/9 model (p<0.01), 4 model còn lại tăng ≤0.5 (n.s.). Mức tăng nhỏ vì POPE đã 84–90%.

## 3f. Qwen3-VL-32B (điểm quy mô)
- CountBench sinh tự do: 77.3 → 90.0 (p=6e-8; vượt remap 78.6, p=4e-7). Cùng ảnh, prompt "single digit" + chọn 2–9: 94.5 → prompt `<count>` làm lệch ngưỡng (đếm thừa tập nhỏ), hiệu chỉnh kéo lại.
- Ảo ảnh VLMBias: 50.3 → 57.1 (p=.04). MMVP: 82.0 → 80.7 (không giúp). POPE: 89.8 → 91.1.
- CounterFact: raw 77.0 (bám màu quen thuộc hơn 4B/8B: 92–93) → bias 90.6, batch-cal không nhãn 91.6.
- Chẩn đoán: count (33 lỗi, Rec. 80%, bias 58%), color_cf (63 lỗi, Rec. 75%, bias 67%); các ô khác <30 lỗi. Tổng 16/24 ô; kết luận tiền đăng ký giữ "mixed" (14/22) vì 32B thêm sau.

## 4. Lỗi đã sửa trong quá trình (deviations)
- v1 → v2: thêm đối chứng `restricted` và giữ điểm bước quyết định trong parquet (để tính offline/re-splits).
- v2 → v3: bước quyết định trước đó là "token đầu tiên thuộc tập ứng viên" → với "{12}" và ứng viên 2–5, quyết định rơi vào chữ số "2" → "{13}".
  Sửa: kích hoạt ở chữ số bất kỳ, chặn chữ số ở bước sau khi ép.
- v3 → v4: (a) biến thể " 2" có token đầu là dấu cách trong cả ba tokenizer → dấu cách lọt vào tập ứng viên/kích hoạt; chỉ giữ token tự giải mã ra ứng viên.
  (b) InternVL suy luận trước khi trả lời (23.5% câu VLMBias) → bị quyết định trên "C3" trong phần suy luận và bị cắt ở 24 token;
  VLMBias nay chỉ quyết định sau "{" và cho 384 token.
- Báo cáo trước (v1) "ALL valid: Qwen-8B 66.8 vs const 55.4" bị thổi phồng bởi Car Logos/Dice (xem mục 3) — đã rút lại.
- v5 (thêm họ model Qwen2-VL-7B, LLaVA-1.5-7B): tokenizer Llama (LLaVA) mã hóa "3" thành ["▁","3"]; code cũ chỉ lấy token đầu ("▁") nên LLaVA không có bước quyết định nào (437/437). Sửa: xét mọi token trong phần mã hóa. Đã kiểm tra tập id ứng viên/chữ số của Qwen3-VL, InternVL, Gemma, Qwen2-VL không đổi → kết quả cũ giữ nguyên. Parquet LLaVA lỗi chuyển vào results/old/llava_nodecision/.
- LLaVA-1.5-7B bị loại khỏi CountBench (sinh tự do): 118/437 câu trả lời viết số bằng chữ ("There are six object(s)"), không có chữ số để kích hoạt bước quyết định, raw parse chỉ 21.4. Giữ LLaVA cho ảo ảnh VLMBias (Yes/No) và MMVP.
- LLaVA-1.5 và Qwen2-VL (197/2136 câu) trả lời VLMBias trực tiếp, không dùng {}: cổng quyết định chung cho mọi model mở ở token đầu tiên hoặc sau "{". Không đổi gì với Qwen3-VL/InternVL/Gemma (0 câu của chúng bắt đầu bằng đáp án; Decide chỉ kích hoạt khi argmax là token đáp án). Chạy lại VLMBias cho Qwen2-VL và LLaVA.
- Quy tắc loại ô (áp dụng đồng nhất): ô model×tập có >10% câu không đạt bước quyết định thì không báo cáo trong bảng sinh tự do (LLaVA CountBench 27%, LLaVA MMVP 39%, InternVL Tally 73%).
- Báo cáo cf_repro/count_repro dùng chung một RNG cho mọi model → thêm 32B làm đổi split của các model xếp sau (lệch ≤1 điểm). Sửa: seed riêng cho từng model; bảng trong bài đã tính lại.

## 4. RÚT LẠI (2026-10-14): tăng trên VLMBias illusions là do rò rỉ split
Mỗi ảnh VLMBias xuất hiện 6 lần (3 độ phân giải × 2 cách hỏi, cùng đáp án). Split cũ phân tầng theo (subset, nhãn) nhưng
không gom theo ảnh → 64% ảnh có bản sao ở cả train và test. Với 20 split gom theo ảnh (vlmbias_grouped.py,
results/vlmbias_grouped.csv), mức tăng trên illusions biến mất: −9.6 đến +0.1 (trước đó +2.5 đến +9.8 offline,
+7 đến +11 online cho Qwen3-VL). Car Logos / Dice vẫn tăng khi gom theo ảnh (nhưng đó là subset bất đối xứng, \remap
đã đạt 63–90%). Few-label illusions bị bỏ. CountBench (9/437 item chung ảnh), MMVP, POPE (đã chia theo cặp/ảnh),
CounterFact, What'sUp (không trùng ảnh) không bị ảnh hưởng. Paper ver19: tab:gen_ill giữ cả hai cột (cảnh báo).
