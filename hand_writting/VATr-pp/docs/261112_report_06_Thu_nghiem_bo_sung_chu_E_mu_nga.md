# Thử nghiệm bổ sung giám sát ảnh thật cho chữ Ễ

## Bằng chứng và mục tiêu

Sau lọc, train writer thật và development writer thật không có ảnh chứa `Ễ`. HAT có 100 ảnh chứa chữ này. HAT chỉ cung cấp OCR/content trong run cũ, không trực tiếp cung cấp mục tiêu ảnh cho generator.

OCR giám sát hiện tại được kiểm tra trên 100 ảnh HAT đó: CER toàn từ 15,25%, đúng toàn từ 48/100; 66/100 dự đoán chứa `Ễ`. Đây là dữ liệu OCR từng có thể dùng train, không phải kiểm tra độc lập. Kết quả cho thấy OCR có biết ký tự nhưng chưa ổn định. Chi tiết: [audit OCR ảnh thật](samples_E_tilde/ocr_real_audit.json); [sample ảnh thật](samples_E_tilde/real_samples_Ễ.png).

Cuối A/B/C, prompt riêng `Ễ` chưa được OCR đọc đúng ở cả 30 writer mỗi nhánh. Chưa thể quy tất cả lỗi về ảnh sinh mà không xem ảnh.

## Phần đã triển khai

Thêm auxiliary **single-image self-reconstruction** cho GAN:

1. Chọn ảnh HAT có transcript chứa `Ễ` từ manifest đã lọc.
2. Chuẩn hóa tương phản theo percentile 5–95 để giảm nền xám; giữ nguyên crop dọc. Target resize về chiều rộng 16 px/ký tự như đầu ra G.
3. Dùng transcript làm query và chính ảnh đó làm tham chiếu đơn. Lặp ảnh tham chiếu theo chiều support chỉ để khớp input 15 ảnh của kiến trúc.
4. Tính L1 có trọng số nét mực (5 lần vùng nền), mask phần padding; cập nhật **chỉ G**.
5. **Không đưa ID HAT vào writer head, không gom HAT thành nhóm writer, không tuyên bố 15 bản lặp là 15 mẫu cùng người viết.** D, writer và OCR không cập nhật trong bước auxiliary.

Đây là cách đưa hình chữ HAT trực tiếp vào loss của generator. Tự tái tạo có thể học lối tắt sao chép từ reference; thành công ở tác vụ này không chứng minh sinh được `Ễ` theo phong cách writer khác. Chỉ số chuyển phong cách phải xem riêng trên panel genuine writer.

## Dữ liệu auxiliary và đánh giá

Chia theo transcript bằng hash cố định seed 742: **69 ảnh train / 31 ảnh eval**. Không transcript nào xuất hiện ở cả hai phía auxiliary. Eval này chỉ giữ riêng khỏi loss auxiliary mới; ảnh và transcript có thể đã được dùng ở OCR warmup/corpus trước đây. Không gọi đó là test chưa từng thấy.

Hai tầng đánh giá được lưu riêng:

- `rare_content_eval/`: tái tạo 31 ảnh với chính ảnh làm reference; target trái / reconstruction phải, L1 và OCR CER. Chỉ nhánh auxiliary có bài kiểm tra này.
- `expanded/`: panel cũ 600 mẫu, 30 writer thật giữ riêng; theo dõi tổng CER và riêng `Ễ`, `NGUYỄN`. Đây mới là kiểm tra khả năng áp dụng chữ sang reference của writer thật, vẫn chấm bằng OCR nội bộ.

Thêm `per_prompt` vào JSON benchmark. Lưu `best_focus.pth` và `best_focus_selection.json` theo CER trung bình các prompt chứa ký tự mục tiêu (hiện `Ễ` và `NGUYỄN`); vẫn giữ best tổng thể, last và 3 checkpoint gần nhất. Best focus không tự động được chọn vào production/run 150k.

## Pilot đang triển khai

Nguồn chung: `runs/ablation_100k_v1/A_control/best.pth`, bước **107.500**. Chạy hai nhánh tuần tự, mỗi nhánh đến **112.500**, tức thêm 5.000 outer steps:

| Nhánh | Auxiliary | Cấu hình |
| --- | --- | --- |
| auxiliary | Bật | weight 0,2; batch 2; mỗi 4 outer steps |
| control | Tắt | Cùng baseline và lịch đánh giá |

LR G encoder 5e-6, G còn lại 2e-5, D 1e-5, writer 2e-5; OCR weight 4, writer weight 0,1; clip 5 và OCR frozen. Không đổi kiến trúc.

Auxiliary có thêm một cập nhật G mỗi 4 outer steps; so sánh cùng số outer steps nhưng **không phải cùng số cập nhật G hay cùng chi phí tính toán**. Các chuỗi ngẫu nhiên cũng có thể khác sau bước auxiliary.

Run đến 150k hiện tại vẫn chạy riêng theo lựa chọn A/B/C trước đó. Pilot không ghi đè trọng số hoặc tự đổi cấu hình của run này. Queue pilot dừng nếu có nhánh lỗi và không tự động promote.

## Kiểm chứng trước khi chạy pilot

- **8 unit test pass**, gồm chia theo transcript, không tạo target writer, shape/range của batch, mask padding và gradient loss.
- Smoke chạy thật từ 107.500 đến 107.504, đủ bốn pha GAN và một auxiliary update, lưu checkpoint/profile thành công.
- Ở smoke, loss tái tạo = **0,79344**, norm gradient G auxiliary trước clip = **0,68610**, hữu hạn. Đây là kiểm tra chạy được, không phải bằng chứng cải thiện chất lượng.

## Cách theo dõi và tiêu chí quyết định

- [Trạng thái hai nhánh](../runs/rare_hat_Ễ_pilot_v1/status.json).
- Log auxiliary: `runs/rare_hat_Ễ_pilot_v1/auxiliary/train.log`.
- Ảnh tái tạo: `auxiliary/rare_content_eval/reconstruction_<step>.png`.
- Ảnh chuyển sang writer thật: `auxiliary/expanded/benchmark_<step>.png`.
- `comparison.json` cập nhật sau khi một nhánh hoàn tất, gồm tổng CER, exact và chỉ số riêng `Ễ`/`NGUYỄN`.

Chỉ cân nhắc mở rộng sau khi đối chiếu với control: `Ễ` cải thiện trên genuine writer references, sample có đủ mũ/ngã, tổng CER và style không suy giảm rõ. L1 tự tái tạo thấp hơn một mình không đủ. Nếu tự tái tạo tốt nhưng chuyển writer vẫn thất bại, cần thay cơ chế content/style hoặc bổ sung dữ liệu writer thật; không kéo dài bước tự động.
