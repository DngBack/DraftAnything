# Stage6: xác nhận head đã khóa trên100 tài liệu mới

**Đã xác nhận gain so với MLP generic; chưa chứng minh superiority so với HGB hoặc gain riêng cho binding.**

100 tài liệu không trùng filename/OCR với800 tài liệu cũ. Qwen3.5-9B, schema rõ nghĩa, toàn bộ trang1280px;100 predictions/600 quyết định. Không fit/tune/calibrate lại head trên100 này. Head và scaler kiểm tra SHA256 trước khi chấm. Năm seed cố định lấy trung bình score. Primary official matcher, bootstrap2000 theo document; CI conditional trên head và threshold đã fit.

Extractor accuracy: **78.83%**; normalized secondary: 80.50%. Confidence extractor>0.99 vẫn sai25/342. Không output cap/missing key. Confidence pipeline dùng public annotation OCR; chưa phải image-only pipeline và chưa kiểm tra quality OCR thực tế.

| Head | AURC ↓ | Coverage | Lỗi/accept | Risk quan sát |
|---|---:|---:|---:|---:|
| MLP generic, cùng thông tin | 0.06588 | 34.17% | 7/205 | 3.41% |
| Learned relation | 0.05715 | 56.33% | 9/338 | 2.66% |
| HGB-fusion, scalar-only reference | 0.08604 | 27.83% | 7/167 | 4.19% |
| HGB-evidence, scalar-only reference | 0.06030 | 1.17% | 0/7 | 0.00% |
| HGB semantic, cùng thông tin; posthoc | 0.06085 | 7.33% | 2/44 | 4.55% |

Primary ΔAURC relation−generic: -0.00873, CI95% [-0.01715, -0.00049]. Coverage Δ+22.17 điểm %, CI [+20.00, +24.50]. Hai risk là point estimate; không khẳng định chênh risk có ý nghĩa hay certificate5%.

## Những đối chứng làm hẹp kết luận

1. **Coverage tăng phần lớn nhờ ABSENT.** Relation chấp nhận94/150 field absent (1 lỗi), generic0. Trên450 field có giá trị: relation244/450 (54.22%,8 lỗi), generic205/450 (45.56%,7 lỗi).94/133 quyết định tăng thêm là ABSENT. Không quy toàn bộ gain thành phát hiện binding.
2. **Normalized sensitivity chưa xác nhận ranking gain:** Δ-0.00303, CI [-0.00788, +0.00332]. Present-only AURC diagnostic cũng có CI chứa0: Δ-0.00655, CI [-0.01757, +0.00401].
3. **HGB cùng raw embeddings chưa bị vượt rõ ràng:** ΔAURC relation−HGB -0.00370, CI [-0.01652, +0.00853]. HGB có Brier tốt hơn; baseline này thêm sau primary confirmation, nên là diagnostic. HGB fit train200 với cấu hình cố định, còn hai neural head dùng thêm dev50 cho early stopping; không quy phép so này thành tuning/supervision budget hoàn toàn bằng nhau.
4. **Calibration là confounder lớn.** Với threshold riêng từng field, chọn chỉ trên calibration50 cũ: semantic-HGB coverage47.50%, risk12/285=4.21%; relation46.50%,10/279=3.58%; generic34.17%,9/205=4.39%. Đây là posthoc, không thay primary ngưỡng đã khóa.
5. **Không có per-field risk guarantee.** Ở primary pooled threshold, registrant_name của relation sai3/57=5.26%, signer_name1/12=8.33%, foreign principal chỉ accept1/100. Overall2.66% không chứng minh mọi field dưới5%.
6. **Chưa đủ novelty cho claim CVPR9/10.** Một shared projection và interaction MLP không tự xác lập đóng góp mới. Gold/span scope, reviewer audit, model family khác và dataset khác còn thiếu.

## Quyết định nghiên cứu tiếp

Giữ head này làm prototype. Ưu tiên audit200 quyết định đã chuẩn bị và thống nhất atomic-value/ABSENT/role ontology trước khi dùng binding supervision. Sau đó kiểm tra calibration theo field/ABSENT với cùng thông tin và nhãn cho mọi baseline; cần nguồn khác và model family khác.89 Short-Form còn chưa dùng, nhưng không đủ thay thế generalization ngoài dataset. Không tune thêm trên stage6 rồi tiếp tục gọi nó là confirmation mới.

[Protocol](stage6-protocol.md), [audit hướng dẫn](stage4-audit-guide.md), [PDF qualitative checks by AI](stage4-qualitative.md).
