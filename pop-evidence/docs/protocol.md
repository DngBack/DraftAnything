# PopEvidence — protocol khóa trước

Phiên bản 0 (2026-10-28), viết trước khi có dữ liệu thật. Mọi thay đổi sau ngày này ghi vào `reports/experiment_log.md` kèm lý do, và phải thực hiện **trước** khi mở test.

## 1. Câu hỏi và giả thuyết

Câu hỏi: cấu trúc phân phối của tile embeddings trong một quần thể có giúp nhận diện species khó và tổng quát hóa sang tổ hợp chưa thấy tốt hơn biểu diễn trung bình hay không?

| | Giả thuyết | Bác bỏ nếu (trên pooled dev, §5) |
|---|---|---|
| H1 | Thống kê ngoài mean chứa thông tin species | Không representation nào trong {meanstd, meanquant, kme, deepsets} có Δ macro AP so với `mean_lin` với cận dưới CI95 > 0 ở n=1 hoặc n=4 |
| H2 | Lợi ích gắn với compositional shift | Lợi ích chỉ thấy trên calibration (in-distribution) mà không thấy trên dev combos |
| H3 | Lợi ích là discriminability, không phải dự đoán nhiều nhãn hơn | Δ macro AP > 0 nhưng Δ mean AUROC ≤ 0, hoặc mean FPR tăng > 0.02, hoặc #pred tăng mà AP không tăng |
| H4 | Không đặc thù PHOEBI | Không tái hiện trên thiết lập độc lập (synthetic bags từ crop đơn bào, hoặc dữ liệu mới) |

## 2. Sự thật về dữ liệu ảnh hưởng tới thiết kế

Đọc từ code PHOEBI (commit `2a42be0`), cần xác minh lại trên dữ liệu bằng `scripts/audit.py`:

1. **Một video cho mỗi combination.** `combo` vừa là tập nhãn, vừa là culture, vừa là phiên thu nhận. Hệ quả:
   - Không có biological replicate trong cùng combination; "multi-field" nghĩa là nhiều frame/crop của cùng một video.
   - Mọi đặc trưng của phiên (nền, độ hội tụ, mật độ, chiếu sáng) trùng hoàn toàn với tập nhãn. Thống kê bậc hai (std, quantile) nhạy với các yếu tố này hơn mean. **Lợi ích từ variance có thể là chữ ký phiên chứ không phải sinh học**; chỉ dev combos chưa thấy mới kiểm tra được điều này, và dù vậy vẫn không tách được hoàn toàn (§8).
2. **Mỗi frame cho 3 crop; train/val chia ở cấp ảnh.** Crop cùng frame có thể rơi vào cả train và val → calibration bag là in-distribution và lạc quan. Chỉ dùng calibration cho threshold; không dùng để chọn model (theo `LCO_DEV_PROTOCOL.md` của PHOEBI).
3. **bt là singleton bị giữ lại**: không có pure culture của bt trong train hay dev. Prototype của bt (nếu dùng) chỉ có thể là hybrid.
4. **Nhãn chỉ ở cấp combination**, không có abundance. Một field có thể thiếu species có trong nhãn → BCE trên n=1 chịu label noise. Báo cáo n=1 và n=4 song song.
5. Kết quả PHOEBI liên quan đã biết (từ docstring code, chưa đọc paper): ProtoMatch seed 1337 held-out per-sample F1 0.660, mean AUROC 0.606; ChannelGroup mean AUROC 0.63 trên test gộp. AUROC held-out gần ngưỡng may rủi là xuất phát điểm, và PHOEBI đã thử aggregation mean / quantile 90 / max ở nhiều granularity (stage 0).

## 3. Split và quy tắc

- Development: 5 fold trong `lco_dev_folds.json` (26 trained combos bậc 2–4, mỗi combo được giữ lại đúng một lần; 5 singleton luôn ở train pool).
  - Train: `train_combos` ∩ partition train.
  - Calibrate (threshold): `train_combos` ∩ partition val.
  - Evaluate: mọi ảnh (train + val) của `dev_combos`.
  - Gộp 5 fold → 26 combination, tương tự test.
- Test: 9 held-out combinations, seed 1337. Chỉ `scripts/run_test.py` được dựng split này; script từ chối config chưa khóa và từ chối chạy lại.
- Mọi thứ được fit trên feature (PCA, bandwidth, random features, chuẩn hóa, prototype, threshold) chỉ fit trên train hoặc calibration của fold đang chạy. Feature extraction không dùng nhãn nên được chạy một lần cho toàn bộ ảnh.
- `Manifest.fold` assert fold không chạm test combos; `tests/test_core.py` kiểm tra lại.
- **Ba partition (thay đổi v1, 2026-10-29, trước khi mở test):** báo cáo test trên LCO partition 1337, 1338 và 1339 (27 combination, đơn vị bootstrap là `partition:combination`), giống Table 19 của PHOEBI. Split 1338 và 1339 được sinh bằng `tools/build_lco_splits.py --seed`. Hạn chế: dev fold chỉ tồn tại cho partition 1337, nên một số combination dùng để lựa chọn (dev của 1337) lại là test của 1338 hoặc 1339. PHOEBI cũng có cùng tình trạng. Kết quả riêng của partition 1337 là kết quả sạch nhất và sẽ được báo cáo tách riêng.

## 4. Bag

- Bag = n ảnh cùng combination → n×16 tile. n ∈ {1, 4} cho mọi method trong cùng bảng; mở rộng {1, 2, 4, 8} ở Experiment 3.
- Eval bag rời nhau, cố định theo `bag_seed=0`, độc lập với seed huấn luyện → mọi method và seed chấm trên **cùng bag**, cho phép so sánh cặp.
- Cap: 300 bag/combination khi eval, 150 khi calibrate, 600 khi train cho representation cố định. Train của model học được: lấy combination đều, rồi n ảnh không lặp.
- Chưa xử lý: tránh ghép 3 crop của cùng frame vào một bag (cần audit tên file trước).

## 5. Endpoint

- **Primary**: macro AP trên pooled dev (lựa chọn) và trên 9 test combos (báo cáo), trung bình theo species có cả combination dương và âm, trung bình theo 3 seed, CI95 bootstrap theo combination (2000 resample).
  - PHOEBI dùng mean per-species AUROC làm tiêu chí chính. Ta báo cáo cả hai; AUROC là secondary đầu tiên. AP phụ thuộc prevalence, nhưng prevalence là như nhau cho mọi method trên cùng bag, nên so sánh cặp vẫn hợp lệ.
- **Secondary**: mean AUROC; per-species AP/AUROC; per-sample F1 và margin so với all-present; macro F1; exact match; mean FPR; số nhãn dự đoán trung bình; context sensitivity (S_pos, gap, combination-level AUROC).
- **So sánh**: paired bootstrap trên cùng resample combination (`metrics.paired_delta`). Một khẳng định "A tốt hơn B" cần CI95 của Δ loại trừ 0.
- Đơn vị thống kê là combination (26 dev, 9 test). Không bao giờ dùng số ảnh hay số bag làm cỡ mẫu. Seed không phải replicate.
- Trước khi mở test: nếu số combination dương/âm của một species trên test < 2 thì species đó không có kết luận riêng (audit sẽ cho con số này).

## 6. Lựa chọn và khóa

1. Chọn representation, kiến trúc, λ_cons, n_fields và hyper-parameter chỉ trên pooled dev.
2. Ghi config cuối vào `configs/locked.yaml` với `locked: true`, `locked_at_commit: <sha>` và commit nó.
3. `scripts/run_test.py --config configs/locked.yaml` chạy test đúng một lần và ghi vào experiment log.
4. Bất kỳ điều chỉnh nào sau khi đọc test đều được báo cáo là post-hoc.

## 7. Gate

- **Gate 1 (tái lập)**: chạy được ProtoMatch và SimplexUnmix của PHOEBI ở random split và LCO seed 1337. Sai khác so với paper phải dưới 0.02 F1, hoặc giải thích được. `scripts/audit.py` không phát hiện leakage.
- **Gate 2 (probe phân phối)**, tất cả các điều kiện:
  - Ít nhất một representation ngoài mean có Δ macro AP > 0 với cận dưới CI95 > 0 so với `mean_lin` (cùng n).
  - Δ mean AUROC ≥ 0, Δ mean FPR ≤ 0.02.
  - Cải thiện xuất hiện ở ít nhất một trong hai species có AP thấp nhất dưới `mean_lin`.
  - Không đảo chiều giữa n=1 và n=4.
  - Nếu không đạt: ghi kết quả âm và đổi giả thuyết. Một ứng viên là thông tin nằm ở granularity mịn hơn (patch token) thay vì ở thống kê của CLS tile.
- **Gate 3 (PopEvidence)**: mốc ngoài bắt buộc là mean AUROC gộp 3 partition cao hơn ProtoMatch 0.63 [0.55, 0.71] (PHOEBI Table 19). Trên dev, phải vượt ProtoMatch chạy trên cùng 5 fold (`repro_phoebi.sh devcv`). Ngoài ra phải vượt `mean_lin`, baseline phân phối tốt nhất ở Gate 2, ProtoMatch, SimplexUnmix và ChannelGroup về macro AP (CI của Δ loại trừ 0 so với ít nhất `mean_lin` và baseline phân phối tốt nhất). Ablation phải chỉ ra thành phần nào tạo ra cải thiện (pooling, class-specific, consistency).

## 8. Confound và đối chứng

| Cách giải thích khác | Đối chứng |
|---|---|
| Thêm quan sát, không phải phương pháp | Mọi method nhận cùng n; báo cáo đường cong theo n |
| Dự đoán nhiều nhãn hơn | AUROC/AP không phụ thuộc threshold; FPR; #pred so với #true |
| Chữ ký phiên/acquisition | Dev combos là phiên chưa thấy; so sánh `cls` với `patchmean`; dùng lại bộ acquisition-robustness của PHOEBI cho model cuối |
| Đặc thù threshold | Primary endpoint không dùng threshold |
| Species không tách được bằng ảnh | Pure-culture discrimination trên train pool; confusion theo species |
| Đồng xuất hiện trong train | Ma trận incidence và co-occurrence (audit) |

## 9. Kiểm tra trên dữ liệu tổng hợp (không phải bằng chứng)

`scripts/make_synthetic.py` tạo dữ liệu có cùng cấu trúc với PHOEBI (40 combo, LCO, 5 fold), trong đó bs và bt có cùng mean nhưng khác độ phân tán theo một hướng. Bộ dữ liệu này chỉ dùng để kiểm tra code, không phải để xác nhận giả thuyết. Kết quả xem `results/dev_synth/`.
