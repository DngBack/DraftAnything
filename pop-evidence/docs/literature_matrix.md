# Novelty audit

Cập nhật 2026-10-29. Nội dung lấy từ một agent đọc trang arXiv, NeurIPS và CVPR. Mục ghi *unverified* là chưa đọc được trên trang gốc. Trước khi trích dẫn vẫn phải tự đọc bản PDF.

## Bài gần nhất và mức rủi ro

| Bài | Đã có gì | Khác biệt của ta | Rủi ro |
|---|---|---|---|
| **PHOEBI**, Baranwal, Hasan, Vyas, NeurIPS 2026 D&B, arXiv 2606.22890 | LCO (9 combo held out, 3 seed, 5 dev fold). §5: "a gradient-trained aggregator fits P(y \| combination)". App. C.3: mean vs q90 vs max cho AUROC held-out 0.59–0.63; 6 granularity cho 0.59–0.63; learned subspace đạt 0.83 trên dev nhưng chỉ 0.57–0.59 khi chuyển; mosaic pure-culture không giúp; "the frozen embedding of a mixture is not a mixture of its constituents' embeddings". Table 19 (27 combo gộp): chỉ ka trên ngưỡng may rủi với mọi decoder; fj chỉ với ProtoMatch (0.66); mx chỉ với ChannelGroup (0.73); bs, bt, pf không phân giải được. Best mean AUROC: **ProtoMatch 0.63 [0.55, 0.71]**. Attention-MIL: ΔF1 −0.33 | Chẩn đoán *đảo chiều* tường minh bằng phân rã own/context; ràng buộc tile hợp lệ về nhãn; multi-field | **Rất cao**. Phần âm tính gần như đã có. Phải vượt ProtoMatch 0.63 trên cùng 27 combo |
| Borowa et al., "Identifying Bacteria Species on Microscopic Polyculture Images Using Deep Learning", IEEE JBHI 27(1) 2023 | Multi-MIL cho polyculture với mọi combo của 4 species, ROC AUC > 0.9 (có giữ combo lại hay không: *unverified*) | Đánh giá trên combo chưa thấy | Cao: phải trích dẫn và đối chiếu protocol |
| Singh et al., "Don't Judge an Object by Its Context", CVPR 2020 | Tách đặc trưng của lớp khỏi context đồng xuất hiện | Nhãn chỉ ở cấp tập, không có box/CAM; đánh giá trên tổ hợp chưa thấy, không phải ảnh "ngoài context" | Trung bình: framing "context bias" đã quen thuộc |
| Liu et al., CCD, CVPR 2022; Chen et al., NeurIPS 2024 (causal label correlations) | De-bias context bằng can thiệp nhân quả trong multi-label | Như trên; không claim causal | Trung bình |
| Couture et al., MICCAI 2018, arXiv 1806.05083 | Quantile pooling của dự đoán instance trong MIL | Quantile pooling **không** phải đóng góp của ta | Pooling đơn thuần: cao |
| WILDCAT, Durand et al., CVPR 2017 | Pooling class-wise (chi tiết kmax/kmin *unverified*) | Evidence theo species là baseline kiến trúc | Trung bình |
| Set-DINO, Yao et al., arXiv 2406.05308 | Self-supervised consistency trên tập tế bào cùng perturbation | Ta làm multi-label decoding; consistency loss của ta liên quan lỏng | Thấp–trung bình với §13.3 |
| Svatko et al., "Deep Learning for BioImaging: What Are We Really Learning?", ICML 2026, arXiv 2603.13377 | Baseline đơn giản thường ngang SOTA trong microscopy | Trích dẫn làm động lực | Thấp |
| Zhang et al., CausalMIL, arXiv 2202.12570 (NeurIPS 2022 *unverified*) | MIL nhị phân, iVAE, OOD | Không claim causal | Thấp |
| Busa-Fekete et al., EasyLLP, NeurIPS 2023 | Học từ tỷ lệ nhãn | Ta không có tỷ lệ | Thấp |
| Chai et al., AAAI 2024 (text) | Compositional split cho multi-label text | Ảnh, nhãn tập | Thấp; là bài analogue duy nhất tìm được cho "held-out label combinations" |
| Ramaswamy et al., ICML 2016; Muandet et al. 2017; Szabó et al. JMLR 2016; Deep Sets; Set Transformer | KME, distribution regression, set functions | Chỉ là baseline | Không |

## Theo từng thành phần

| Thành phần | Đã có | Có thể là đóng góp? |
|---|---|---|
| Pooling phân phối (std, quantile, KME) | PHOEBI C.3, Couture 2018 | Không. Chỉ là probe (F3) |
| Evidence riêng theo species | WILDCAT, ChannelGroup | Không, khi đứng một mình |
| Ràng buộc tile hợp lệ về nhãn: absent → không tile nào, present → top-quantile, pure → mọi tile | Từng phần là quy tắc MIL chuẩn; MIML thường coi nhãn vắng là "missing", tức giả định ngược lại | **Có**, nếu là tổ hợp và gắn với cơ chế đảo chiều. Phải xử lý phát biểu "mosaic không giúp" của PHOEBI |
| Phân rã own/context trên combo chưa thấy, chẩn đoán đảo chiều | PHOEBI chỉ có một câu định tính | **Có**: đóng góp đánh giá/chẩn đoán |
| Consistency giữa các combo | Set-DINO (lỏng), invariance learning | Chỉ là phụ |

## Hệ quả cho kế hoạch

1. Thước đo thành công chính: mean AUROC gộp trên 27 combo held-out (3 partition), so trực tiếp với Table 19 của PHOEBI (ProtoMatch 0.63). Macro AP vẫn là primary đã khóa, AUROC báo cáo song song.
2. Cần chạy trên cả 3 LCO partition (1337/1338/1339) như PHOEBI, không chỉ seed 1337.
3. Câu chuyện của paper: (i) chẩn đoán đảo chiều, (ii) ràng buộc hợp lệ về nhãn sửa được đảo chiều, (iii) phân phối evidence thêm phần còn lại. Nếu (ii) không vượt 0.63, paper chỉ còn là phân tích.

## Novelty audit: Contextual Logit Adjustment (CLA), 2026-10-29

Agent đã đọc full text: DB loss, Clark et al., RUBi, CCD, CUE, BCP, Chen et al. NeurIPS 2024. Menon et al. mới chỉ grep text. LADE chưa xác minh.

| Bài | Cơ chế | Khác CLA |
|---|---|---|
| Menon et al., ICLR 2021 (logit adjustment) | +τ log π_y (prior biên) lúc train, multi-class | CLA thay π_y bằng prior có điều kiện p(y_k \| y₋ₖ) |
| CUE, arXiv 2605.01309 (BLA, Eq. 6) | Logit adjustment nhị phân cho BCE với prior biên π_c | Không phụ thuộc các nhãn khác |
| Wu et al., ECCV 2020 (DB loss) | Đồng xuất hiện chỉ dùng cho **trọng số mẫu** (Eq. 1–4); NTR dùng offset prior biên ν_i | Không có offset có điều kiện theo nhãn khác |
| **Clark et al., EMNLP 2019 (Bias Product / PoE)** | log p + log b từ một bias model đông cứng, chỉ dùng lúc train; lập luận độc lập có điều kiện tương tự | **Gần nhất.** Bias của họ là một view của x; của CLA là **nhãn thật y₋ₖ** (thông tin đặc quyền, chỉ có lúc train), sigmoid multi-label, và bao gồm cả prior biên |
| Cadene et al., NeurIPS 2019 (RUBi) | Mặt nạ nhân từ nhánh question-only, bỏ lúc test | Nhân và học chung; bias đến từ modality |
| He et al., 2019 (DRiFt) | Fit residual của bias model | Tương tự PoE (chưa xác minh chi tiết) |
| Liu et al., CVPR 2022 (CCD) | Trừ feature ngữ cảnh, giữ cả lúc test | Ở cấp feature |
| Chen et al., NeurIPS 2024 | Can thiệp nhân quả lên tương quan nhãn; split Exclusive/Co-occur | Baseline đánh giá quan trọng |
| **Li et al., ICML 2026 (BCP), arXiv 2606.12925** | **Lúc test cộng** PMI log P(c_k \| c_a) − log P(c_k) với anchor dự đoán | Ảnh gương của CLA: BCP bơm đồng xuất hiện vào lúc test, CLA tách nó ra lúc train |
| Lipton et al., ICML 2018 (BBSE) | Label shift đơn nhãn | Không tìm thấy công trình nào về dịch chuyển tổ hợp nhãn multi-label (3 lần tìm) |

**Kết luận:**
- Chưa thấy công trình nào làm "offset cộng lúc train từ một mô hình chỉ dùng nhãn thật của các lớp khác, bỏ đi lúc test".
- Rủi ro: reviewer coi đây là "PoE của Clark với bias model chỉ dùng nhãn", hoặc "PMI của BCP chuyển sang lúc train và đổi dấu".
- Phải trích dẫn và so sánh với Clark PoE, CUE/BLA, DB loss, BCP và Chen 2024.
- Phải xử lý phản biện "dùng nhãn thật thì suy biến", bằng regularization của p̂ và ablation theo τ và C.
- Đóng góp bảo vệ được: tổng quát hóa logit adjustment sang prior có điều kiện theo nhãn, kèm thiết lập dịch chuyển combination và bảo đảm bất biến.
