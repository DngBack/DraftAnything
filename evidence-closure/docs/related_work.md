# Công trình liên quan (rà soát 2026-11-06, đã kiểm chứng từ bản gốc)

Chi tiết, độ sâu đọc của từng bài và tài liệu trích dẫn: [paper/verified_refs.md](../paper/verified_refs.md), [paper/refs.bib](../paper/refs.bib). Đọc toàn văn 2611.01517, Hwang et al. (DocHallu), CADET, ConfBench, nhưng qua văn bản trích từ PDF nên phải kiểm lại số từ PDF trước khi trích. Phần lớn bài còn lại chỉ đọc abstract và đoạn tìm theo từ khóa. Không tìm thấy không có nghĩa là không có; lĩnh vực ra vài preprint liên quan mỗi tháng, nên phải rà lại trước khi nộp.

Claim của bài: (a) Shapley theo giai đoạn và chất lượng ảnh; (b) kiểm tra nhân quả bằng sửa chữ số trong ô bảng; (c) tách token, nét và định vị ở cùng ngân sách token; (d) tín hiệu ẩn so với lời nói.

| claim | đánh giá | bài gần nhất |
|---|---|---|
| (a) | **mở** trong những gì đã đọc: bốn bài đọc toàn văn đều không dùng Shapley hay trục chất lượng ảnh | CADET 2609.38851 (necessity/sufficiency, can thiệp trong prompt); 2605.05716 (Shapley trên thành phần agent, chỉ là tiền lệ phương pháp) |
| (b) | **mở về thiết kế**: chưa thấy sửa một chữ số thành chữ số khác trong ô bảng | Hwang et al. (xóa chữ số, nên đáp đúng là từ chối, không phải đọc theo phản thực); 2605.27750; 2607.21617; 2609.13745 (chỉ qua abstract) |
| (c) | **một phần**: "phóng to cứu việc đọc" và "InternVL chia tile cố định nên phóng ít tác dụng" **đã được công bố**. Phần chưa thấy: so ảnh phóng với pixel gốc ở cùng số token, kết hợp với định vị trong thiết kế giai thừa, trên ô bảng thật | VLMs Read Aloud 2611.01517; Res-Bench 2510.16926v3 (padding và SR tách thông tin khỏi độ dài token); Found but Not Read 2610.02880 và 2611.01517 (hộp đỏ gần như không giúp ở ngữ cảnh dài); 2608.26949 (không phóng ảnh dưới ngân sách) |
| (d) | **đã có như phát hiện chung, và không nhất quán**: 2607.22034, 2607.24440 cho token thắng lời nói; ConfBench cho hai model trái chiều nhau, và hai nhánh có kích thước ảnh khác nhau. Không dùng làm điểm mới | 2607.22034; 2607.24440; ConfBench 2608.01792; 2608.06532; 2610.22981; 2609.32653; 2604.17274 |

## Nguy cơ trùng lớn nhất

1. **VLMs Read Aloud (2611.01517).** Đã có phóng ảnh bằng bicubic trên Qwen3-VL-8B (nhận dạng 49 % lên 86 % từ 9 đến 15 px, văn xuôi), và giải thích về InternVL3.5. Paper của ta không được trình bày việc này như phát hiện mới. Giao thức của họ chỉ dùng bằng chứng gold, nên khác biệt giữa khoảng cách đọc-dùng của họ và kết quả chép ≈ biểu thức của ta chưa có giải thích; ứng viên chưa kiểm định là hỏi đáp đa bước trên văn xuôi so với chép một ô.
2. **Res-Bench.** Đã tách lượng thông tin và độ dài token, nhưng bằng padding và SR học được, trên benchmark tổng quát. Cần đọc toàn văn trước khi trích.
3. **Hộp đỏ không giúp** đã được công bố hai lần, ở ngữ cảnh dài. Kết quả của ta phải so sánh rõ với hai bài này, không trình bày như điều bất ngờ.

## Lưu ý khi trích

- Hwang et al. (ACL 2026 Findings): tên bài không phải DocHallu; không dùng số 84.8/61.0 vì mâu thuẫn với Hình 4; tác giả nói patching "không xác lập một nguồn gốc nhân quả duy nhất".
- ReadBench: kết luận "độ phân giải ảnh hưởng không đáng kể" chỉ từ Gemini 2.0 Flash.
- 2611.01517 dùng InternVL3.5, không phải InternVL3-8B. Từ "preregistered" là của tác giả, và các đăng ký chưa công khai.
