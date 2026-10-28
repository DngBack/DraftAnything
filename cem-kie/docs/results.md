# CEM-KIE — nhận định kết quả khả thi (26/10/2026)

Số liệu đầy đủ ở [results/main/report.md](../results/main/report.md), [results/audit.md](../results/audit.md) và `results/llm/`. Test là Kleister-Charity dev-0 (440 tài liệu), 3 seed, protocol khóa trước ở [protocol.md](protocol.md).

## Tiêu chí khóa trước

| tiêu chí | kết quả | đạt? |
|---|---|---|
| G-A: hiện tượng competing evidence tồn tại | income/spending: heuristic ≤ 2,5 F1, oracle ứng viên 88–89, median ~250 giá trị ứng viên/tài liệu | **đạt** (mô tả) |
| G-B: cem_r4 − set2 trên money ≥ +2 pp, CI > 0 | +0,97 [−1,47, +3,25]; so với set4 (khớp tham số): −0,22 [−2,61, +2,19] | **không đạt** |
| G-C: cem_r4 > cem_r1 trên money | −0,40 [−1,98, +1,21] | **không đạt** |
| G-D: resolver giúp | cem_r4 − nores = −1,62 [−2,93, −0,41]: resolver **làm hại** | **không đạt** |
| G-E: lợi thế giữ ở tài liệu dài | không có lợi thế để giữ | – |

## Số chính (F1 %, test)

| hệ thống | money | other | micro |
|---|---|---|---|
| heuristic tốt nhất (frequent) | 1,2 | – | – |
| point (pointer độc lập) | 21,7 ± 1,7 | 81,8 | 60,1 |
| set4 (Transformer, 3,7M) | 25,6 ± 0,9 | 81,1 | 60,5 |
| cem_r4 (3,4M) | 25,3 ± 0,5 | 82,9 | 62,3 |
| cem_r4_nores (1,5M) | 26,9 ± 1,2 | 83,0 | 62,2 |
| Qwen3-VL-4B zero-shot, toàn văn (1 lượt) | **55,3** | **91,4** | **76,9** |
| oracle ứng viên | ~88,5 | ~95 | – |

LLM − cem_r4: money +30,0 pp [26,5, 33,6], other +8,4 pp [6,8, 10,1].

## Nhận định

1. **H1 bị bác trên dữ liệu chuẩn ở thiết lập này.** Ở cùng ứng viên và cùng feature, competing slots không hơn Transformer dùng chung, số slot không tạo xu hướng (r1 ≈ r2 ≈ r4 ≈ r8), và resolver quan hệ ẩn làm kém đi. Lợi thế nhỏ của CEM trên `other` (+1,9 pp so với set) có ý nghĩa thống kê. Nhưng nó nằm ở các field dễ (date/number/postcode), không phải chỗ proposal nhắm tới, và cem_r1 cũng đạt được.
2. **Nút thắt là biểu diễn bằng chứng, không phải cơ chế phân giải.** Trên money mọi mô hình học kém đều nhau (21–27 F1, so với oracle 88), và train loss về 0,04 trong khi val không tăng. Vector context đông cứng quanh một con số trong bảng bị làm phẳng không mang đủ thông tin cột/năm/dòng. Cơ chế memory nào đặt trên nó cũng không bù được.
3. **Baseline đọc toàn văn mạnh hơn hẳn**: zero-shot, không train, nhưng gấp đôi money F1. Bất kỳ đóng góp kiến trúc nào cũng phải đo so với baseline này, không phải so với point/set.
4. **Failure mode còn lại là có thật và đo được.** Khoảng 23–26% dự đoán money của mọi hệ thống, **kể cả LLM (24%)**, là một con số cách gold dưới 30%, nhiều khả năng là cột năm trước hoặc một subtotal. Đây là competing evidence đúng nghĩa và chưa ai giải được. Nhưng nó là bài toán cấu trúc bảng (cột năm hiện tại so với năm trước, total so với subtotal), không phải bài toán giữ giả thuyết xuyên trang.

## Giới hạn

- Text không có ngắt trang, nên "xuyên trang" chỉ được xấp xỉ bằng vị trí ký tự. Chưa dùng layout/ảnh (PDF nằm trong git-annex, 12 GB).
- Encoder đông cứng, mỗi mention một vector CLS. CEM với encoder fine-tune, hoặc đặt trên hidden state của LLM, chưa được thử.
- Chỉ 5/8 field. Không có nhãn quan hệ nên relation loss không được huấn luyện. Val 259 tài liệu nên chọn epoch có nhiễu.
- Phân loại lỗi "gần đúng 30%" là heuristic, chưa kiểm tra bằng tay.

## Khuyến nghị

Không đầu tư full model CEM-KIE theo hướng slot memory trên đặc trưng đông cứng. Nếu tiếp tục, câu hỏi đáng hỏi là: *vì sao cả LLM vẫn nhầm năm trước/subtotal ở ~24% số trường tiền, và có thể sửa bằng biểu diễn cấu trúc bảng hoặc tự kiểm (ví dụ ràng buộc income − spending = net movement) hay không?* Ràng buộc đó gần với CoSolve-KIE hơn, và nên được kiểm tra trước bằng oracle: chọn lại trong top-M ứng viên của LLM theo luật số học.
