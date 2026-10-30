# Hướng giải quyết: từ nguyên nhân đến phương án (research memo, 2026-10-30)

## 0. Nhắc lại chẩn đoán

Decoder train trên nhãn tập học **bối cảnh** thay vì tế bào. Có ba nguồn, theo thứ tự mức độ nghiêm trọng:

| # | Nguyên nhân | Bằng chứng | Đã có cách sửa? |
|---|---|---|---|
| N1 | **Batch / phiên chụp lấn át loài** trong embedding đông cứng | phoebi4: 3/4 loài gần loài *sai* của phoebi6 (F12); mọi decoder học được đều tụt khi đổi batch | **Chưa** (chuẩn hóa batch và TVN đều thất bại, F14) |
| N2 | **Đồng xuất hiện:** điểm của k dựa vào lớp đi kèm | 11–17/20 hệ số ngữ cảnh có ý nghĩa; AUROC khử tương quan giảm 3–4 điểm | Một phần: CLA (+0.04 trên test PHOEBI, không giúp khi đổi batch) |
| N3 | **Định vị sai:** evidence nằm trên tile không chứa lớp | Test BBBC041 0.04–0.19; PanNuke 0.08–0.24 | Có: ràng buộc tile (0.76–0.98) |

N1 là nguyên nhân gốc và chưa được giải quyết. Các hướng dưới đây ưu tiên N1.

## 1. Bằng chứng mới từ hai probe thăm dò (phoebi4 đã đọc, nên chỉ là thăm dò)

Nguồn: `results/analysis/reference_anchoring.md`, `reference_relative.md`. Đánh giá trên 10 video mixture của phoebi4.

| Cách nhận diện | AUROC | fj | Ghi chú |
|---|---|---|---|
| Prototype pure culture của **batch khác** (dạng đóng) | 0.589 | 0.40 | Tương đương ProtoMatch qua batch |
| **Prototype pure culture của cùng batch** (dạng đóng) | **0.764** [0.632, 0.894] | **0.73** | **+0.175** |
| Tỷ lệ tile gần prototype cùng batch nhất | 0.702 | 0.66 | |
| Classifier tile train trên pure culture cùng batch | 0.685 | 0.48 | |
| Head tuyến tính trên embedding thô, train trên phoebi6 | 0.619 | 0.35 | mean_lin |
| Head **học** trên độ tương tự với tham chiếu cùng batch (kể cả đã chuẩn hóa theo tham chiếu) | 0.53–0.55 | 0.28–0.30 | **Thất bại** |
| Làm trắng feature theo tham chiếu của batch (kiểu TVN) | 0.574 | 0.08 | Thất bại |

**Nguyên lý rút ra:**
1. **Tham chiếu cùng batch là đòn bẩy lớn nhất từ đầu dự án.**
2. **Mọi tham số học trên batch cũ đều mang theo shortcut** của batch đó, kể cả khi đầu vào đã là đặc trưng tương đối. Thứ chuyển được là *phép so sánh* với tham chiếu, không phải *ngưỡng* hay *trọng số* học trên một batch.
3. Căn chỉnh thống kê (mean/std, whitening) không đủ, vì batch làm đổi *vị trí tương đối* giữa các loài, không chỉ tịnh tiến hay co giãn.

## 2. Các phương án, xếp theo ưu tiên

### A. Reference-anchored learning (ưu tiên 1, giải quyết N1)

**Ý tưởng:** mỗi batch có vài mẫu pure culture làm chuẩn tham chiếu. Đây là thực hành rẻ, tương tự control well trong Cell Painting. Decoder không học "loài k nằm ở đâu trong embedding", mà học **cách so sánh** một tile với các prototype của *cùng batch*.

**Phương pháp đề xuất: Episodic Prototype MIL (EP-MIL).**
- Học một phép chiếu g(·) chung, dùng cho cả tile và tham chiếu.
- Điểm tile t với lớp k: s_{t,k} = τ·cos(g(z_t), g(p_k^{batch})), trong đó p_k^{batch} là trung bình g của các tile pure culture k **trong cùng batch, cùng episode**.
- Điểm ảnh: dạng đóng trên s, ví dụ mean hoặc tỷ lệ tile. **Không có head học ngưỡng.**
- **Train theo episode:** mỗi episode mô phỏng một "batch". Prototype được tính từ một tập con pure culture, mixture được chấm điểm theo prototype đó.
  - Loss = BCE trên điểm ảnh + ràng buộc tile (âm chính xác cho lớp vắng mặt).
- **Tạo đa dạng batch khi chỉ có một batch train:** augmentation ở cấp ảnh mô phỏng dịch chuyển (độ sáng, tương phản, gamma, blur, chiếu sáng nền) áp *đồng nhất* lên mọi ảnh của một episode, cả tham chiếu lẫn mixture. Như vậy g học được tính bất biến theo kiểu dịch chuyển đó, còn prototype luôn đi cùng batch của nó.
- **Vì sao kỳ vọng nó chuyển được:** điểm chỉ phụ thuộc vào so sánh tương đối với tham chiếu cùng batch. Khi dịch chuyển batch tác động đồng thời lên tham chiếu và mixture, phép so sánh vẫn giữ được.
- **Quan hệ với prior work:** Prototypical Networks (Snell 2017), meta-learning cho domain generalization; control-based normalization trong Cell Painting (TVN, sphering). Điểm mới: kết hợp với MIL nhãn tập và tổ hợp chưa thấy.
- **Đánh giá:** dev trên phoebi6. Test sạch cần một batch mới có cả pure culture và mixture; phoebi4 đã bị đọc nên chỉ dùng được cho thăm dò.

### B. Biểu diễn bất biến theo batch ở cấp encoder (ưu tiên 2, giải quyết N1, tốn kém)

- **B1. Fine-tune encoder:** dùng ràng buộc tile cùng objective đối nghịch domain (DANN) hoặc IRM, với domain là batch. **Cần ≥ 2 batch khi train.** Có thể dùng phoebi6 và phoebi4 cùng lúc, nhưng khi đó cần batch thứ ba để test.
- **B2. Test-time adaptation không nhãn:** tiếp tục self-supervised (DINO) trên ảnh chưa gán nhãn của batch mới, rồi áp phương án A.
- **B3. Augmentation mô phỏng batch khi trích feature**, như mô tả trong A; rẻ hơn B1.

### C. Đơn vị instance tốt hơn tile (ưu tiên 3, giải quyết N3, hỗ trợ N1)

- Phát hiện và crop từng tế bào thay vì tile 224 px. Phiên song song đang thử (`extract_objects.py`, 48 object mỗi ảnh, kèm feature hình thái).
- Đặc trưng hình thái tường minh (kích thước, độ dài, độ tương phản) có thể bền với batch hơn embedding.
- Kết hợp với A: prototype ở cấp tế bào.

### D. Giảm phụ thuộc đồng xuất hiện khi train (giải quyết N2)

- **D1. Reweighting khi train:** đánh trọng số mỗi mẫu bằng Π_k p(y_k)/p̂(y), giống thước đo khử tương quan, để phân phối nhãn khi train trở nên độc lập giữa các lớp. Rẻ, bổ trợ cho CLA.
- **D2. CLA** chỉ dùng khi dịch chuyển là ở tổ hợp nhãn (đã có).

### E. Dữ liệu và protocol (điều kiện cần cho mọi hướng)

- **Một batch mới** (pure culture + mixture) chưa dùng, để có test sạch.
- **Thiết kế thu dữ liệu:** mỗi batch kèm pure culture làm chuẩn; ít nhất 3 batch; nhiều tổ hợp hơn 40 để có power.

## 3. Kế hoạch đề xuất

1. **EP-MIL trên dev phoebi6:** episode, tham chiếu cùng batch, augmentation mô phỏng batch. Tiêu chí: không thua mean_lin trên dev LCO.
2. **Thăm dò trên phoebi4:** EP-MIL phải vượt prototype cùng batch dạng đóng (0.764). Nếu không vượt, phương án tốt nhất hiện có là chính prototype dạng đóng cùng batch, kèm thiết kế thu dữ liệu có tham chiếu.
3. **Xin hoặc thu một batch mới** để test sạch.
4. Song song: gộp kết quả feature cấp object của phiên kia vào A.

## 4. Câu chuyện paper nếu A thành công

"Set-labelled microscopy decoders learn context and batch, not cells. Diagnostics show anti-localisation and batch dominance. The fix is to *learn to compare against same-batch references* (EP-MIL) together with tile constraints. It transfers to unseen batches and unseen combinations, where every absolute decoder fails."


## 5. Cập nhật sau khi thử (2026-10-30)

- **Phương án A (EP-MIL) đã thử: thất bại** khi chỉ train trên một batch (0.63–0.66 so với 0.764 của dạng đóng). Xem F17.
- Các biến thể dạng đóng cùng batch đạt trần khoảng 0.76.
- **Khuyến nghị hiện tại:**
  - (1) Dùng **so sánh dạng đóng với mẫu chuẩn cùng batch** làm phương pháp thực tế khi đổi batch.
  - (2) **Thu hoặc xin dữ liệu ≥ 3 batch có mẫu chuẩn.** Đây là điều kiện cần cho mọi phương án có học (A, B1).
  - (3) Phương án C (đơn vị cấp tế bào) vẫn có thể thử với dữ liệu hiện có.
