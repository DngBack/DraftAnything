# CoSolve-KIE oracle — nhận định (26/10/2026)

Protocol khóa trước: [protocol-cosolve.md](protocol-cosolve.md). Số liệu: [results/cosolve/report.md](../results/cosolve/report.md). Test dev-0 gồm 433 tài liệu có income/spending, × 3 seed unary set4. Solver học trên val (259 tài liệu).

## Kết quả (money F1 %, test)

| hệ thống | money F1 | precision |
|---|---|---|
| LLM greedy (Qwen3-VL-4B, tham chiếu) | 55,3 | 55,5 |
| indep (LLM + set4 unary, không ràng buộc) | 49,7 ± 1,0 | 58,9 |
| fixed (ràng buộc cứng) | 54,3 ± 0,8 | 54,5 |
| learned (λ học được) | 56,0 ± 1,2 | 65,8 |
| evcond (λ·g(D)) | 56,1 ± 1,2 | 65,8 |
| oracle tập ứng viên | 88,2 | 100 |

## Tiêu chí

| tiêu chí | kết quả | đạt? |
|---|---|---|
| B-1: learned − indep ≥ +2 pp, CI > 0 | +6,33 [+4,48, +8,49] | **đạt** |
| B-2: evcond − learned, CI > 0 | +0,02 [+0,00, +0,06] | **không đạt** (về thực chất bằng 0) |
| B-3: precision không giảm quá 2 pp | 58,9 → 65,8 | **đạt** |

## Nhận định

1. **Suy luận đồng thời có giúp, nhưng chủ yếu là bù lại phần unary bị yếu đi.** Learned hơn indep 6,3 pp và precision tăng 7 pp, tức không ép ra giá trị sai. Tuy vậy, so với LLM greedy chỉ +0,7 pp, CI [−1,8, +3,1]. Indep thấp hơn greedy 5,6 pp vì spending được chấm như key đầu tiên (lệch phân phối) và mất điều kiện vào income mà LLM tự có khi sinh. Ràng buộc chủ yếu khôi phục lại sự phụ thuộc đó. **Chưa có bằng chứng solver vượt baseline thực tế.**
2. **Điều kiện kích hoạt theo bằng chứng (đóng góp novelty của B) không có tác dụng.** Gate g(D) học về gần hằng số. Ở dữ liệu này, ràng buộc net đúng ở ~87% tài liệu nên không có gì để học về "khi nào ràng buộc có hiệu lực".
3. **Ràng buộc số học không gỡ được lỗi chính.** 90% lỗi của learned là cặp **thỏa** ràng buộc net nhưng sai. Phân loại 653 lỗi (3 seed):
   - 38%: gold không có trong text (lỗi OCR hoặc bộ sinh ứng viên)
   - 38%: giá trị dự đoán nằm **cách một mention của gold ≤ 40 ký tự**
   - 10%: cách ≤ 200 ký tự
   - 13%: xa

   Trong số lỗi có thể sửa được, ~61% là con số nằm ngay cạnh gold. Điều này khớp với nhầm cột trong cùng một dòng bảng (năm nay so với năm trước, hoặc restricted/unrestricted/total). Cặp (I′, S′) của cột bên cạnh cũng thỏa I′ − S′ = N′, nên ràng buộc không phân biệt được hai cặp này.

## Kết luận chung cho hai proposal

- CEM-KIE (slot memory) không đạt H1 ([results.md](results.md)).
- CoSolve-KIE đạt B-1 nhưng không vượt LLM greedy, và không đạt phần novelty (B-2).
- Failure mode thực sự, có thể đo được và còn mở, là **gắn giá trị với cột (năm báo cáo) trong bảng tài chính bị làm phẳng**. Cả LLM, mô hình candidate và solver số học đều mắc. Đây là bài toán cấu trúc bảng/layout, không phải memory xuyên trang hay ràng buộc giữa field.

## Giới hạn

- Unary của LLM chấm từng field độc lập. Chấm có điều kiện p(S | I) có thể làm baseline indep mạnh hơn và thu hẹp khoảng cách.
- Ràng buộc chỉ có một loại (net). Chưa thử ràng buộc theo cột (I và S cùng cột năm) hay theo report_date, vì cần layout hoặc nhận diện header năm.
- Phân loại "cạnh nhau ≤ 40 ký tự" là proxy cho nhầm cột, chưa kiểm tra bằng tay. Val nhỏ (259 tài liệu).
