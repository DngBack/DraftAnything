# Stage 3: xác nhận trên holdout mới — ảnh/OCR × semantic schema

Ghi trước khi sinh prediction hay đọc gold holdout stage 3, ngày 28/09/2026. Motivation: smoke train và hai mẫu development VRDU cho thấy `registration_num`/`registrant_name` khó hiểu với schema ngắn; field date gold có thể chứa label text. Stage 2 đã dùng test300, nên mọi kết luận trên đó là pilot.

Holdout mới: 100 filename Short-Form có OCR và PDF, không nằm trong bất kỳ train/valid/test của stage 2 (600 docs). Chọn bằng SHA256 filename, lấy 100 đầu. Không lọc theo chất lượng ảnh, giá trị gold hay correctness. Lưu manifest trước inference. Chưa có bảo đảm độc lập theo registrant/person; chỉ xác nhận theo document. Không fit hay tune trên 100 này.

Thí nghiệm 2×2, cùng Qwen3.5-2B, greedy và max_new_tokens320:

- OCR thật vs ảnh PDF, cả hai dùng cùng schema baseline stage 2.
- Schema ngắn vs schema làm rõ semantic role, chỉ phát triển từ train và hai mẫu development đã xem: số đăng ký là số cạnh Registration No, registrant là tổ chức đăng ký FARA (trong Short Form không phải individual ở đầu form), date là date of signature chứ không phải received stamp/revision date. Signer name/title là người ký, không suy đoán title khi không ghi.

Ảnh render mọi page, cạnh dài tối đa1280px. Không dùng annotation bbox để crop. Mọi doc nhiều trang/failed download/truncation phải báo; không âm thầm drop. Primary value error theo matcher official, strict schema single-value/ABSENT; subtype `on-page-wrong` chỉ chứng minh giá trị có trong OCR, không mặc nhiên là binding. Paired document bootstrap B=2000 cho chênh accuracy và từng nhóm. Báo high-confidence error (mean token prob>0.99), field-present vs absent, latency/memory.

Kiểm tra sensitivity label-contaminated date riêng bằng quy tắc cố định từ development (tách phần `Date of signature` khỏi gold), nhưng không thay primary matcher hoặc fitting labels. Không lấy improvement extractor làm improvement selection. Không gọi schema clarification hoặc two-read fusion là novelty; đối chiếu ExtractConf/PLV trước method claim.

Decision: nếu lợi ích selection stage 2 biến mất khi semantic schema đúng, chuyển claim từ kiến trúc binding sang audit failure mode và xác định hypothesis mới. Nếu còn lỗi natural binding giữa field rõ nghĩa, đó là nhóm cần learned relational head tiếp theo.

## Bổ sung trước khi mở metrics stage 3

Áp dụng nguyên HGB-fusion/HGB-evidence của Qwen3.5 stage 2 và nguyên threshold từ calibration50 vào bốn điều kiện mới. Không refit head, không đổi threshold, không dùng nhãn holdout cho inference feature. Báo transfer khi đổi modality/schema và accuracy gốc để không nhầm extraction gain với selection gain. Tính secondary normalized risk trên đúng tập accept ban đầu, không chọn lại threshold theo nhãn normalized.
