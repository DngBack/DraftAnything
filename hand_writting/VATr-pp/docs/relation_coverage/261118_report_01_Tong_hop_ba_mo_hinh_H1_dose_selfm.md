# Tổng hợp ba mô hình: nét nối, sao chép bằng chứng và danh tính người viết

Ngày 18/11/2026. Inference-only. Tiếp nối các report [02](261117_report_02_Thi_nghiem_cong_H1_relation_coverage_IAM.md) và [03](261117_report_03_Lap_lai_H1_tren_Emuru_va_thuoc_do_writer_ID.md).

- Code: `research/relation_coverage/`.
- Số liệu: `h1_assets/summary_three_models.json` và `h1_assets/dose_*.json`.

## Ba mô hình

| Mô hình | Họ | Support | Có thấy writer IAM khi train? |
|---|---|---|---|
| VATr++ (HF, legacy layout) | GAN + Transformer | 15 ảnh từ | Split 339 writer: 16/49 writer H1 đã thấy |
| Emuru (CVPR 2025) | Tự hồi quy T5 trên latent VAE | 1 dòng thật + transcript | Không (train trên font tổng hợp) |
| DiffusionPen (ECCV 2024) | Latent diffusion | 5 ảnh từ, trung bình 1 vector style | Split 339 writer, như VATr++ |

Thông tin trùng writer lấy từ `writers_dict_train.json` của DiffusionPen. VATr++ dùng cùng split này (`num_writers = 339`). Danh sách lưu ở `research/relation_coverage/iam339_train_writers.json`.

Thước đo:
- **HWD:** VGG16 train trên Font².
- **Writer-ID:** ResNet18 train trên 283 writer train chính thức của IAM, không trùng với writer test. Trên ảnh thật của writer test, chỉ từ một ảnh từ, mạng nhận đúng người viết trong 62,9% trường hợp (trên 128 writer).

## 1. Nét nối trong support không ảnh hưởng, ở cả ba mô hình

**Hồi quy liều–đáp ứng có hiệu ứng cố định theo từ đích** (cùng một từ, nhiều support có độ phủ bigram khác nhau):

| Mô hình | Số ảnh sinh / từ đích | β HWD (độ phủ 0→1) | β writer-ID | Lợi ích tối đa còn tương thích với CI, tính theo % hiệu ứng sai người viết |
|---|---|---|---|---|
| VATr++ | 4.572 / 762 | +0,027 [−0,014; +0,068] | +0,001 [−0,004; +0,006] | ≈ 2% / 2% |
| Emuru | 2.267 / 762 | −0,042 [−0,122; +0,047] | +0,000 [−0,012; +0,012] | ≈ 16% / 18% |
| DiffusionPen | 4.542 / 757 | −0,026 [−0,073; +0,020] | +0,001 [−0,003; +0,005] | ≈ 15% / 10% |

β âm nghĩa là độ phủ cao thì ảnh gần thật hơn. Không có β nào khác 0 có ý nghĩa thống kê. Tương phản UNC − COV cũng bằng 0 ở cả hai nhóm writer (đã thấy / chưa thấy) và ở cả ba mô hình.

## 2. Chỉ Emuru sao chép được bằng chứng, và chỉ khi trùng khớp nguyên từ

| Mô hình | Có bản sao: HWD | Có bản sao: writer-ID | Tận dụng thông tin riêng theo từ (HWD / writer-ID) |
|---|---|---|---|
| VATr++ | +0,032 [+0,019; +0,045] | +0,009 [+0,007; +0,010] | 3,4% / 6,9% |
| DiffusionPen | −0,002 [−0,048; +0,041] | +0,005 [+0,001; +0,010] | ≈ 0% / 4,3% |
| Emuru: SELFM − SELF (chỉ phần do chính từ đó) | +0,363 [+0,283; +0,440] | +0,040 [+0,029; +0,053] | 39% / 32% |
| Emuru: COV − SELFM (cùng lần viết) | −0,028 [−0,112; +0,060] | −0,025 [−0,039; −0,011] | — |

Điều kiện SELFM (cùng dòng, đã xóa từ đích khỏi ảnh và text) bác bỏ giả thuyết "lợi ích của Emuru đến từ cùng mực, cùng lần viết". Phần lợi ích đến từ việc có **chính từ đó** trong ngữ cảnh. Một dấu hiệu khác cùng chiều: trong hồi quy liều–đáp ứng, độ phủ **chữ cái** của dòng style giúp Emuru rõ theo HWD (−0,62 khi độ phủ 0→1), trong khi độ phủ bigram thì không.

## 3. Danh tính người viết: mọi mô hình còn rất xa ảnh thật

Đây là tỉ lệ nhận đúng writer (top-1 trong 49 writer) ở điều kiện COV:

| | Writer đã thấy (16) | Writer chưa thấy (33) |
|---|---|---|
| Ảnh thật | 84,9% | 80,6% |
| VATr++ | 57,4% | **29,0%** |
| Emuru | 23,3% | 2,9% |
| DiffusionPen | 17,1% | 3,2% |

Emuru chưa từng thấy IAM nhưng vẫn chênh mạnh giữa hai nhóm. Vì vậy chênh lệch "đã thấy / chưa thấy" không thể quy hoàn toàn cho việc học thuộc. So sánh công bằng là cột writer chưa thấy. Nhóm "đã thấy" chỉ có 16 writer, nên các con số ở cột này nhiễu.

## Kết luận

1. **Bỏ giả thuyết "quan hệ chưa thấy trong support là nút thắt".** Kết quả null lặp lại ở ba họ mô hình (GAN, tự hồi quy, diffusion), ở cả writer đã thấy lẫn chưa thấy, và ở hai thước đo. Với VATr++, CI đủ hẹp để loại trừ mọi hiệu ứng lớn hơn khoảng 2% hiệu ứng sai người viết.
2. **Phát hiện chính là một đánh đổi về cách dùng support.**
   - Mô hình nén style thành vector toàn cục (VATr++, DiffusionPen) không dùng được bằng chứng cụ thể.
   - Mô hình điều kiện trong ngữ cảnh (Emuru) dùng được bằng chứng, nhưng chủ yếu khi trùng khớp nguyên từ hoặc trùng chữ cái, không khai thác được trùng một phần như bigram. Đồng thời Emuru giữ danh tính kém.
3. **Với writer chưa thấy, mô hình tốt nhất (VATr++, 29%) mới đạt khoảng 1/3 mức ảnh thật (81%).** Đây là khoảng trống lớn và đo được.

## Giới hạn

- Mỗi điều kiện chỉ sinh một lần.
- Định dạng support khác nhau giữa các mô hình, nên chỉ so được tương phản bên trong từng mô hình. So sánh tuyệt đối (bảng 3) chỉ mang tính mô tả.
- Hai thước đo đều là proxy; chưa có chấm mù bởi người.
- Hai trục trặc khi dùng Emuru đã sửa (dòng style ghép từ các từ rời; điều kiện dừng gốc) được mô tả trong report 03.
