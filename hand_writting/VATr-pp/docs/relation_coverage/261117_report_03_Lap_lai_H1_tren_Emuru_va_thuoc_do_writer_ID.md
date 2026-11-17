# Lặp lại H1 trên Emuru và thêm thước đo nhận dạng writer

Ngày 17/11/2026. Inference-only. Tiếp nối [report 02](261117_report_02_Thi_nghiem_cong_H1_relation_coverage_IAM.md). Code: `research/relation_coverage/` (`gen_saved.py`, `run_emuru_lines.py`, `writer_id.py`, `score_models.py`). Tổng hợp: [h1_assets/two_model_summary.json](h1_assets/two_model_summary.json).

## Thiết lập bổ sung

**Thước đo thứ hai: mạng nhận dạng writer.**
- ResNet18 train trên 283 writer *train* của IAM, không trùng với 128 writer test.
- Embedding là feature lớp áp chót.
- Kiểm định trên ảnh thật: chỉ từ một ảnh từ, mạng nhận đúng người viết trong 62,9% trường hợp, trên 128 writer chưa từng thấy (ngẫu nhiên 0,8%).
- Trong số 49 writer của H1, ảnh từ thật được nhận đúng người viết trong 82,0% trường hợp.

Mọi ảnh, cả thật lẫn sinh, đều được cắt sát mực và đưa về cao 32px trước khi chấm.

**Emuru** (CVPR 2025, revision ghim `c75b78dd`) chỉ được train trên font tổng hợp, nên không thể đã thấy writer IAM.

Hai trục trặc đã gặp và cách xử lý:
1. **Style ghép từ các từ rời:** Emuru sinh ra ảnh rỗng hoặc chữ sai, kể cả khi đã cắt ảnh theo chiều cao dòng. Vì vậy với Emuru, support là **một dòng IAM thật**, đúng cách dùng gốc của mô hình.
2. **Điều kiện dừng gốc:** nó quét cả vùng ảnh style. Đã thay bằng cách tìm điểm dừng sau phần style và sau khi đã bắt đầu có mực.

Các điều kiện của Emuru:

| Điều kiện | Dòng style |
|---|---|
| SELF | Dòng chứa từ đích |
| COV | Dòng khác của cùng writer, phủ mọi bigram của từ đích, không chứa từ đích |
| UNC | Dòng khác của cùng writer, không phủ bigram nào của từ đích |
| OTHER | Dòng của writer khác |

Kết quả có 220/762 từ đích đủ điều kiện và 8 lần sinh lỗi, còn lại 212 cặp (writer, từ) thuộc 45 writer.

## Kết quả

Bảng dưới là các tương phản ghép cặp, CI 95% bootstrap theo writer. Với HWD và writer-ID (1−cos), số dương nghĩa là kém hơn.

| Tương phản | VATr++: HWD | VATr++: writer-ID | Emuru: HWD | Emuru: writer-ID |
|---|---|---|---|---|
| Thiếu nét nối (UNC − COV) | −0,019 [−0,048; +0,013] | −0,002 [−0,007; +0,002] | +0,064 [−0,051; +0,173] | +0,008 [−0,004; +0,020] |
| … chỉ giữ từ mà UNC đủ chữ cái | — | — | +0,079 [−0,039; +0,201] | +0,011 [−0,003; +0,023] |
| Sai người viết (OTHER − COV) | +0,667 [+0,564; +0,780] | +0,174 [+0,154; +0,194] | +0,754 [+0,565; +0,943] | +0,070 [+0,048; +0,094] |
| Có bản sao (COV − SELF) | +0,032 [+0,019; +0,045] | +0,009 [+0,007; +0,010] | +0,344 [+0,238; +0,449] | +0,014 [+0,002; +0,026] |
| **Tận dụng thông tin riêng theo từ** | **3,4%** | **6,9%** | **36,7%** | **11,2%** |
| Nhận đúng writer (COV), ảnh thật đạt 82% | 38,6% | | 9,9% | |

"Tận dụng" = hiệu ứng có bản sao chia cho khoảng chênh "cùng từ / khác từ" đo trên ảnh thật (HWD 0,94; writer-ID 0,12).

## Diễn giải

1. **Thiếu nét nối không gây suy giảm có ý nghĩa thống kê ở cả hai mô hình.**
   - VATr++: ước lượng gần 0 và khoảng tin cậy hẹp.
   - Emuru: ước lượng đi đúng chiều H1 nhưng khoảng tin cậy chứa 0 (n = 212, thiếu power).
   - Hiện chưa có bằng chứng rằng "quan hệ chưa thấy" là nút thắt đo được.
2. **Hai mô hình nằm ở hai đầu của một đánh đổi.**
   - VATr++ giữ danh tính người viết tốt hơn nhiều (38,6% so với 9,9% nhận đúng writer), nhưng gần như bỏ qua bằng chứng cụ thể (3–7%).
   - Emuru đọc ngữ cảnh dòng nối tiếp, nên dùng bằng chứng cục bộ nhiều hơn (11–37%), nhưng giữ danh tính kém.
   - Cả hai còn rất xa ảnh thật (82%).
3. **Mức 36,7% (HWD) của Emuru là cận trên.** Dòng SELF là cùng một lần viết với từ đích (cùng mực, cùng nét), và HWD nhạy với yếu tố này. Thước đo writer-ID ít nhạy hơn và cho 11,2%.

## Giới hạn

- Support của hai mô hình khác định dạng (15 từ so với 1 dòng), nên chỉ so được tương phản *bên trong* từng mô hình.
- Ở Emuru, độ phủ chữ cái giữa COV và UNC không luôn khớp nhau. Đã có kiểm định trên tập con UNC đủ chữ cái.
- Mỗi điều kiện chỉ sinh một lần.
- Chưa có mô hình diffusion (DiffusionPen, One-DM).

## Bước tiếp theo đề xuất

1. **Gỡ nhiễu "cùng lần viết" ở SELF của Emuru:** thêm điều kiện SELF-masked, tức cùng dòng nhưng xóa vùng từ đích khỏi cả ảnh lẫn text.
2. **Tăng power cho hiệu ứng nét nối:** dùng hồi quy liều–đáp ứng theo độ phủ bigram trên toàn bộ 762 từ, thay vì chỉ giữ hai cực.
3. **Thêm một mô hình diffusion** để kết luận về đánh đổi có tính tổng quát.
4. **Thiết kế kiến trúc:** kết hợp token danh tính toàn cục (thế mạnh của VATr) với bộ nhớ truy xuất cục bộ theo ngữ cảnh (thế mạnh của Emuru). Mục tiêu đo được là tăng *cả hai* chỉ số: nhận đúng writer và mức tận dụng thông tin riêng theo từ.
