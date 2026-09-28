# Stage 2: evidence support và chuyển template — protocol

Ngày 28/09/2026, ghi trước khi chấm kết quả stage 2. Mục tiêu nghiên cứu là tìm tín hiệu hữu ích vượt baseline HGB mạnh, không đặt trước kết quả dương hay điểm reviewer.

## A. CORD evidence features

Giữ prediction của 2 extractor, thêm tín hiệu từ OCR text và boxes không dùng category/is_key/gold: sự có mặt của key mô tả field, khoảng cách từ value đến key, candidate cạnh tranh cùng field và key cạnh tranh khác field. Alias Indonesia/Anh là rules khai báo theo schema, không học từ test; không gọi rules này là phương pháp mới. Head HGB nhận cùng nhãn, hyperparameters và tuning budget cho mỗi cấu hình.

Cấu hình cố định: fusion; fusion+binding; fusion+evidence; tất cả; evidence không geometry. Hyperparameters giữ nguyên pilot v2 (100 cây, 7 leaves, min leaf 40, L2=1). Huấn luyện train800, calibration validation100, đánh giá test100. Đây tiếp tục là exploratory trên test đã xem. Thêm 5-fold document cross-validation trong train để kiểm tra tín hiệu có tái hiện, không dùng fold score làm risk certificate. Alpha chính 5%, secondary 2%,10%; AURC/Brier và accepted risk báo cùng coverage.

Bootstrap theo document B=1000, resample calibration và test; head giữ nguyên. Kiểm tra sensitivity subtotal=total như pilot cũ. Không tự động chọn cấu hình thắng để gọi là confirmation.

## B. VRDU FARA unseen-template pilot độc lập

Nguồn: https://github.com/google-research-datasets/vrdu . Split `FARA-lv3-unk_Short-Form-train_200-test_300-valid_100-SD_0.json`: train200, valid100, test300. Không đổi split theo kết quả. Valid chia deterministically 50 development và 50 calibration bằng hash filename; khóa cấu hình head trước inference. Giữ test template Short-Form chưa thấy; không tuyên bố source independence nếu chưa kiểm tra output completeness.

Đầu tiên dùng OCR thật đã cung cấp trong VRDU + extractor text Qwen2.5-1.5B-Instruct, một lượt JSON, greedy, tối đa256 output tokens. Đây **không phải** thí nghiệm VLM/image trên VRDU. Mục đích là kiểm chứng protocol nguồn mới và giá trị key/value/layout support, trước khi tiêu tốn GPU cho image model. Sáu schema field gốc, chỉ single-value hoặc ABSENT; field nhiều giá trị khác nhau đánh giá riêng/exclude. Không lấy annotation span làm inference candidate hay input prompt. Giữ tất cả OCR text trong budget; mọi truncation phải báo.

Matcher lấy từ toolkit Google Research, báo riêng strict value match và matcher gốc khi cần; không sửa matcher để tăng điểm. ABSENT đúng khi prediction null và gold không có. Tách value xuất hiện trên trang, absent hallucination và candidate-not-grounded; automatic taxonomy chưa là audit người.

Baseline logprob, LR/HGB extractor+page-presence, HGB thêm OCR text/layout support. Cùng nhãn train, fixed hyperparameters. Ngưỡng target5% chọn calibration50; test300 không dùng fitting/tuning. Báo AURC, Brier, coverage/risk, CI theo tài liệu, count errors/accepted. Calibration nhỏ có thể cho zero coverage hoặc vượt risk; giữ nguyên và báo thật.

## C. Giới hạn novelty và tiêu chí đầu tư

Đã đọc abstract nguồn chính của ExtractConf (arXiv:2606.24420), PLV (2609.20110), Valid Per-Field Selective Risk Control (2608.14639). Geometry/OCR fusion hoặc calibration không được claim đầu tiên. Chỉ tiếp tục thành method paper nếu signal gain tồn tại sau head mạnh và trên unseen-template tự nhiên, rồi có architecture và ablation cùng thông tin giải thích cơ chế. Tín hiệu âm cũng phải ghi, không nâng complexity chỉ để tìm test gain.

## Bổ sung robustness trước khi đánh giá VRDU

Chạy thêm Qwen3.5-2B từ cache, cùng OCR prompt, disable thinking, cùng greedy budget. Cả hai smoke chỉ dùng 4 train docs. Chưa thay prompt theo test.

Official valid và test đều là Short-Form, train là Amendment/Dissemination Report. Vì vậy primary có 50 nhãn calibration từ template mới, không gọi là zero-shot threshold transfer. Secondary: giữ cùng head fit trên 150 train docs; 50 train docs còn lại (khóa bằng hash filename) làm source-template calibration. So với 50 target-template calibration đã khóa. Cả hai threshold chạy cùng score/head trên test300 để cô lập tác động calibration domain. Endpoint này ghi trước khi mở metrics VRDU.
