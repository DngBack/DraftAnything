# CoSolve-KIE oracle — protocol (khóa 26/10/2026, trước khi chạy solver)

Đây là tiếp nối của [CEM-KIE](results.md). Ở đó, cả mô hình candidate lẫn LLM đều nhầm khoảng 24% số trường tiền sang một con số gần đúng (cột năm trước, subtotal). Câu hỏi: ràng buộc liên field có sửa được lỗi này khi suy luận đồng thời hay không, và việc học *khi nào* ràng buộc có hiệu lực có thêm giá trị hay không?

## Ràng buộc

`net(I, S) = 1` nếu income I và spending S đều khác NULL và |I − S| xuất hiện như một con số tiền trong text (sai lệch ≤ 1,5, tính cả ×1000). Kiểm tra trước trên nhãn gold: ràng buộc đúng ở 89% (train) / 86% (dev-0) số tài liệu có cả hai gold trong text. Rủi ro biết trước: cặp năm trước cũng thỏa ràng buộc.

## Ứng viên và unary

- Ứng viên: hợp top-10 giá trị của set4 qua 3 seed, cộng câu trả lời greedy của LLM, cộng NULL.
- Unary 1: log p_set4 (theo từng seed; giá trị ngoài top-10 của seed đó nhận log 1e-6).
- Unary 2: log p_LLM('{"field": value,' | tài liệu), Qwen3-VL-4B, chấm đúng nhánh trên KV cache dùng chung.

## Solver

Liệt kê đầy đủ mọi cặp (I, S). Điểm của cặp:
`a·(u_llm(I) + u_llm(S)) + b·(u_set(I) + u_set(S)) + λ·g(D)·net(I, S)`, với bias NULL riêng.

Tham số được học bằng NLL của cặp gold trên softmax toàn bộ cặp, tức suy luận chính xác và khả vi. Huấn luyện **chỉ trên val** (259 tài liệu), báo cáo trên dev-0.

| biến thể | mô tả |
|---|---|
| llm_greedy | câu trả lời greedy của LLM (tham chiếu) |
| indep | λ = 0: hai field độc lập, cùng unary |
| fixed | ràng buộc cứng: nếu có cặp thỏa, chỉ chọn trong các cặp thỏa |
| learned | λ học được, g = 1 |
| evcond | g(D) = σ(w·φ(D)); φ: có từ khóa net/surplus/deficit, tỉ lệ cặp thỏa trong tập ứng viên, cặp unary-top có thỏa không, margin unary |
| oracle | gold nếu nằm trong tập ứng viên |

## Tiêu chí

- **B-1:** learned − indep trên money ≥ +2 pp, CI 95% bootstrap > 0 (cộng dồn 3 seed unary set4).
- **B-2:** evcond − learned có CI > 0.
- **B-3:** precision của learned không giảm quá 2 pp so với indep, tức không ép ra giá trị chỉ để thỏa ràng buộc.
- Thăm dò: tỉ lệ lỗi của learned mà cặp dự đoán vẫn thỏa ràng buộc ("nhất quán nhưng sai", nghi là năm trước).
