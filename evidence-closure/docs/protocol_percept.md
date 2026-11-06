# Protocol thăm dò: "nhận thức" có phải một giai đoạn ổn định không? (viết 2026-11-06, trước khi tính các đại lượng dưới đây)

Khung Shapley trong [protocol_attr.md](protocol_attr.md) giả định có một giai đoạn nhận thức tách khỏi tính toán: ảnh cho ra một "bản đọc", rồi các bước sau dùng bản đọc đó. Nếu đọc phụ thuộc vào prompt khi pixel giữ nguyên, giả định này sai ở mức đáng kể.

Đã xem trước khi viết: 12 dòng ví dụ của 7B (6 câu số học, s100 và s30), là nguồn của giả thuyết. Chưa xem thống kê nào.

## Pilot A: phân tích lại dữ liệu sẵn có, không dùng GPU

Dữ liệu: `results/attr_{tag}_{level}.jsonl`, câu số học. So sánh `program` và `opprog`: cùng ảnh, decode greedy, nên khác biệt do prompt (`opprog` có thêm công thức đã che số). Bổ sung: `extract` và `program` ở `results/e8_*.jsonl` (dev, s100).

Một "bản đọc" của fact gold f trong một output: số trong output có khoảng cách edit nhỏ nhất với f (chỉ xét chữ số), với khoảng cách ≤ nửa độ dài f. Không có số như vậy thì coi như không đọc.

| mã | đại lượng |
|---|---|
| M1 | bất ổn định: trong các fact được cả hai prompt đọc, tỷ lệ hai bản đọc khác nhau |
| M2 | P(`program` đọc đúng \| hai prompt đồng ý) so với \| không đồng ý |
| M3 | nguồn gốc của mỗi số `program` phát ra: fact gold / số khác trên trang (OCR) / cách một số trên trang đúng 1 chữ số / bịa; so với mức ngẫu nhiên từ chuỗi chữ số cùng độ dài |
| M4 | tỷ lệ model từ chối hoặc nói không đọc được, theo mức ảnh |

## Tiêu chí loại (áp ở s30, từng model)

- **H-instab sống** nếu M1 ≥ 25 % ở s30 và CI của M1 ở s100 không chứa 0. Dưới mức đó, lỗi đọc chủ yếu ổn định theo prompt và khung giai đoạn vẫn dùng được.
- **Đồng thuận có ích** nếu M2 chênh ≥ 20 điểm. Đây là tín hiệu tin cậy không cần gold.
- **M3 có thông tin** nếu tỷ lệ "bịa" tăng ≥ 15 điểm từ s100 tới s30 và cao hơn hẳn mức ngẫu nhiên giải thích được.
- **M4 đáng nêu** nếu tỷ lệ từ chối ở s30 < 5 % trong khi tỷ lệ đọc đủ của `program` ở s30 chỉ khoảng 20 %.

## Giới hạn biết trước

`opprog` có thể đổi ô được chọn chứ không chỉ đổi cách đọc, nên phép gióng theo khoảng cách edit có thể gán nhầm. Vì vậy M1 chỉ là cận thô; pilot GPU có prompt trỏ thẳng vào một ô (protocol riêng) mới là phép đo sạch.
