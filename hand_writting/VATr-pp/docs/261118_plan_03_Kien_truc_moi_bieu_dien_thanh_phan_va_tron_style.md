# Plan: Kiến trúc mới cho HTG, biểu diễn thành phần nét và trộn style hai tầng

- Ngày: 18/11/2026. Trạng thái: đề xuất, chưa triển khai.
- Kế thừa: [261118_report_02](261118_report_02_Nguyen_nhan_nham_dau_chu_hoa_resolution_hay_kien_thuc_G.md) (lỗi dấu do prior tần suất), [relation_coverage/261118_report_01](relation_coverage/261118_report_01_Tong_hop_ba_mo_hinh_H1_dose_selfm.md) (H1 đã đóng, writer-ID của writer chưa thấy thấp), [261117_plan_03](261117_plan_03_Glyph_128_va_CNN_glyph_encoder.md) (GlyphCNN).
- Ràng buộc: đây là kiến trúc mới, không phải bản nâng cấp VATr. Mô hình phải mạnh trên IAM tiếng Anh. Tiếng Việt có dấu chồng là phép thử khó.

## 1. Chẩn đoán từ bằng chứng đã có

| Triệu chứng | Nguyên nhân trong VATr++ (đọc từ code/report) | Hệ quả thiết kế |
|---|---|---|
| Sai loại dấu ở chữ hoa (È→Ê) | Mỗi ký tự là một nguyên tử: `Linear(256→512)` trên bitmap. Exposure dự báo lỗi với AUROC 0,92. 42/66 chữ hoa có dấu có 0 ảnh thật | Cần chia sẻ tri thức **theo thành phần**: cách writer viết dấu hỏi trên ê thường phải dùng lại được cho Ể hoa |
| Ảnh mờ, dấu dính | Ảnh ra cao 32 px, feature map 4 hàng, `linear_q` reshape thẳng ra lưới. GAN + OCR đóng băng | Tăng chiều cao ảnh, tăng số hàng latent, dùng generator không lấy trung bình mode |
| Writer chưa thấy chỉ đúng 29% (ảnh thật 81%) | Style là một bộ nhớ patch phẳng, không có mục tiêu danh tính | Tách style **toàn cục** (danh tính, học tương phản) và style **cục bộ** (cách vẽ từng thành phần) |
| Danh tính ký tự chỉ đi vào q/k | Ký tự "vẽ" bằng cách chọn patch style, value không mang nội dung | Nội dung phải là token riêng mang hình học, không chỉ là chỉ số tra cứu |

Bài học đã có: tách dấu theo Unicode bằng tay (VietCompose) kém hơn checkpoint gốc, và sampling lại không giúp. Không lặp lại hai điều này.

## 2. Câu hỏi nghiên cứu

> Nếu nội dung được biểu diễn bằng các **thành phần nét học được, có vị trí**, và style được truy xuất **theo thành phần** thay vì theo ký tự, thì mô hình có sinh đúng và đúng phong cách những tổ hợp ký tự **chưa từng có ảnh thật** hay không, trong khi vẫn giữ chất lượng trên IAM?

Đây là câu hỏi chung cho HTG: chữ hoa và ký tự hiếm trong IAM, dấu chồng trong tiếng Việt.

## 3. Kiến trúc (tên tạm: PartScribe)

```
 văn bản ─► [A] Part tokenizer ─► token thành phần (what, where) ─► [B] Context encoder ─┐
                                                                                          ├─► [D] Denoiser DiT (flow matching, latent f=4) ─► VAE decoder ─► ảnh 64 px
 support ─► [C] Style encoder ─► g (danh tính toàn cục) + M (bộ nhớ patch cục bộ) ────────┘
```

### [A] Part tokenizer, tiền huấn luyện riêng, không cần nhãn thành phần
- Đầu vào: mỗi ký tự render bằng K font (serif, sans, script, viết tay) ở 64×64. Đa font buộc biểu diễn trừu tượng khỏi một kiểu chữ.
- Encoder: CNN nhẹ cộng **slot attention** cho ra S slot (S≈4–6). Mỗi slot gồm:
  - `what`: lượng tử hóa vào codebook dùng chung (≈128–256 mã, VQ hoặc FSQ);
  - `where`: hộp vị trí/tỉ lệ liên tục.
- Mục tiêu:
  1. Tái tạo glyph từ các slot (decoder ghép slot, như object-centric learning). Mục tiêu này buộc slot phải là thành phần.
  2. Tương phản giữa các font: cùng ký tự thì gần nhau.
  3. Tương phản ký tự với "hard negative" là cặp anh em cùng thân (È/Ê, ẻ/ế) để nhấn mạnh khác biệt nhỏ.
- Probe, không dùng làm loss: `z(ể) − z(e) ≈ z(Ể) − z(E)`, độ sạch của các cụm mã cho dấu mũ, dấu hỏi, dấu chấm i/j.
- Ablation: dùng phân rã NFD làm giám sát yếu, so với hoàn toàn không giám sát. Bản không giám sát là bản chính vì áp dụng được cho mọi chữ viết.

### [B] Context encoder: quan hệ giữa các ký tự
- Transformer hai chiều trên toàn bộ token thành phần của từ (hoặc dòng). Mỗi token được gắn chỉ số ký tự và vị trí trong ký tự.
- Mục tiêu: hình dạng một ký tự phụ thuộc hàng xóm (nét nối, dấu dịch chỗ khi cạnh chữ cao, cỡ chữ hoa đầu từ).
- Một **bộ dự đoán độ rộng** cho từng ký tự, có điều kiện theo `g`, tương tự duration predictor trong TTS. Bộ này đặt bố cục ngang tường minh, giúp DiT không phải tự học alignment, và giảm mờ do vị trí không chắc chắn.

### [C] Style hai tầng
- `g` (một vài token): encoder trên support, huấn luyện tương phản theo writer (SupCon trên writer train của IAM và các writer tiếng Việt). Thêm **gradient reversal từ một head OCR** để `g` không mang nội dung, chống sao chép từ như Emuru.
- `M` (bộ nhớ patch): feature map của support có độ phân giải cao. Token thành phần cross-attend vào `M` để lấy "cách writer này vẽ thành phần này". Vì mã `what` dùng chung giữa các ký tự, dấu mũ học từ ê thường sẽ được truy xuất khi vẽ Ê hoa. **Đây là cơ chế chuyển giao, và là đóng góp chính.**
- Khi train, loại khỏi support các từ trùng với từ đích để tránh sao chép.

### [D] Trộn các không gian vector
Ba nguồn (nội dung, style toàn cục, style cục bộ) có thống kê khác nhau. Nguyên tắc trộn:
1. Mỗi nguồn đi qua projection riêng, LayerNorm, rồi cộng embedding loại nguồn trước khi vào không gian chung d.
2. **Vai trò khác nhau thì kênh khác nhau**, không concat:
   - `g` cùng timestep đi qua **AdaLN-zero** (điều biến toàn cục: độ nghiêng, độ dày, cỡ);
   - nội dung đi qua cross-attention thứ nhất ("vẽ cái gì, ở đâu"), có bias đơn điệu theo bố cục từ [B];
   - `M` đi qua cross-attention thứ hai ("vẽ thế nào"), dùng query là token thành phần đã căn chỉnh.
3. Mọi nhánh điều kiện được khởi tạo cổng bằng 0 để train ổn định.
4. Classifier-free guidance **tách riêng** cho nội dung và style, cho phép chỉnh độ đúng chữ và độ giống writer độc lập.

### [E] Độ nét
- Ảnh cao 64 px (thử 96 px cho tiếng Việt). Latent của VAE dùng f=4, tức 16 hàng thay vì 4.
- VAE huấn luyện riêng cho chữ viết tay, với loss L1 + perceptual + PatchGAN + OCR trên ảnh tái tạo. Chặn trên chất lượng: CER và độ nét của ảnh tái tạo phải gần ảnh thật trước khi train DiT.
- Rectified flow / flow matching để tránh lấy trung bình mode. Distill về vài bước sau khi chốt kết quả.

## 4. Định vị so với tài liệu (cần đọc kỹ trước khi chốt claim)

| Công trình | Đã làm | Khác biệt cần chứng minh |
|---|---|---|
| VATr / VATr++ | Glyph Unifont làm query | Token thành phần học được, có vị trí, thay cho nguyên tử ký tự |
| LF-Font, MX-Font, MX-Font++ (font chữ Hán/Hàn) | Style cục bộ theo thành phần, MX-Font không cần nhãn | Áp dụng cho chữ viết tay **liền nét, cả dòng**, có writer, không có lưới ký tự cố định |
| SDT, CASHG (online), DiffInk (online) | Style writer + ký tự; CASHG có bộ nhớ ngữ cảnh bigram | Ảnh offline; style **theo thành phần**, không theo ký tự hay bigram |
| InkSpire (ICLR 2026) | DiT trộn style/nội dung/nhiễu trong một latent | So trực tiếp làm baseline mạnh; claim là chuyển giao thành phần sang tổ hợp không có ảnh |
| One-DM, DiffusionPen, Emuru | Diffusion/AR few-shot | Đã đo: identity writer chưa thấy 3%, Emuru chỉ tái dùng từ trùng khớp |

Rủi ro novelty: InkSpire và CASHG rất gần về cơ chế trộn. Đóng góp phải nằm ở **biểu diễn thành phần và phép thử chuyển giao**, không nằm ở việc dùng DiT.

## 5. Thí nghiệm quyết định

1. **Phép thử chuyển giao (chính):**
   - Tiếng Việt: chữ hoa có dấu có 0 ảnh thật, so với chữ thường tương ứng có nhiều ảnh.
   - IAM: giữ lại toàn bộ ảnh chứa một số chữ hoa hoặc ký tự hiếm (chọn trước, ghi vào protocol).
   - Đo CER, tỉ lệ thay bằng anh em cùng thân (thước đo mới từ report_02), HWD theo ký tự và writer-ID.
2. **IAM parity:** FID/HWD/CER/writer-ID trên split chuẩn, so với VATr++, DiffusionPen, Emuru, One-DM, InkSpire.
3. **Độ nét:** độ rộng biên nét, năng lượng gradient, tỉ lệ dấu dính (component analysis trên vùng dấu), đánh giá mù bằng người.
4. **Ablation:** bỏ slot (về token ký tự); bỏ `M` (chỉ `g`); bỏ gradient reversal; 32 px so với 64 px; bỏ bộ dự đoán độ rộng.

## 6. Lộ trình và điểm dừng

| Giai đoạn | Việc | Điều kiện đi tiếp |
|---|---|---|
| 0 | Đọc kỹ InkSpire, CASHG, MX-Font++; chốt protocol giữ lại tổ hợp | Không có công trình nào đã làm chuyển giao thành phần cho HTG offline — **Đạt có điều kiện, xem [report_04](261118_report_04_Tong_quan_tai_lieu_va_protocol_giu_lai_to_hop.md)** |
| 1 | Part tokenizer trên font | Probe phép tính dấu đúng ≥ chuẩn đặt trước; mã dấu mũ/dấu thanh tách cụm |
| 2 | VAE f=4, 64 px | CER ảnh tái tạo ≈ ảnh thật |
| 3 | DiT nhỏ chỉ có `g` cùng token ký tự (baseline nội bộ) | Đạt mức IAM gần DiffusionPen |
| 4 | Thêm token thành phần và `M` | Thắng baseline nội bộ ở phép thử chuyển giao, không thua trên IAM |

Nếu giai đoạn 4 không hơn giai đoạn 3 ở phép thử chuyển giao, giả thuyết chính bị bác. Khi đó ghi kết quả null như với H1, không tinh chỉnh tiếp.
