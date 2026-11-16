# Inference few-shot từ 12 ảnh ngoài tập dữ liệu

Ngày 16/11/2026. Inference hoàn tất; không train/fine-tune.

## Thiết lập

- Input: `/home/bachdx2/DraftAnything/few-shot`, 12 PNG RGBA chữ viết tay bút xanh. Nội dung tham chiếu có tên riêng, từ và chữ số viết bằng chữ.
- Checkpoint: `runs/weights_upload_261114/component_176500_generator.pth`, bước 176.500, chọn theo benchmark trước đó; bản 250k không vượt nguồn. Chưa có đối chứng chọn checkpoint riêng cho writer ngoài tập này.
- FP32 eval, native generator, grayscale 32px, 15 support slots. Dùng đủ 12 ảnh theo thứ tự số và lặp thêm 3 ảnh theo seed. Không lấy mẫu có hoàn lại toàn bộ 15 slot để tránh bỏ sót ảnh đầu vào.
- `clean_742`: tách nét bút xanh bằng chênh lệch kênh màu, crop và chuẩn hóa; `clean_743`: cùng tiền xử lý, đổi 3 ảnh lặp; `raw_742`: grayscale gốc để đối chiếu ảnh hưởng của nền. Không sửa ảnh sinh để làm đẹp.

## Kết quả xuất

806 prompt × 3 cấu hình = **2.418 ảnh PNG**. Gồm ký tự Latin/số; nguyên âm và thanh thường/hoa; cặp đối chiếu dấu; tên riêng; chữ hoa; từ khó; tổ hợp stress tổng hợp; 24 dòng nhiều từ. Một số chuỗi stress không phải từ tiếng Việt có nghĩa và được đánh dấu trong metadata.

Có gallery HTML tìm kiếm/lọc nhóm và cấu hình, 29 contact sheets, ảnh tổng quan so sánh và bản ZIP. Mỗi dòng nhiều từ được ghép từ ảnh từng từ với khoảng cách 16px, không phải mô hình sinh liên tục cả câu. Đây không phải 2.418 mẫu writer/seed độc lập; từ đã cache trong từng cấu hình và được dùng lại trong các dòng ghép.

## Quan sát trực quan

Trên panel tổng quan, một số từ thông dụng/tên riêng như Lan, Anh, Thị giữ được dáng nét tham chiếu; ảnh gốc còn ảnh hưởng của nền và đường kẻ. Tiền xử lý sạch giảm một phần nền nhưng có thể làm nét đậm hơn. Một số dấu và cấu trúc chữ vẫn sai; chữ hoa hiếm và dấu kép như Ễ, Ỡ vẫn không ổn định. Đây là nhận xét trực quan có giới hạn, chưa chấm người đọc hoặc OCR độc lập và không báo cáo CER/accuracy cho bộ này.

Một số target trùng từ xuất hiện trong ảnh tham chiếu, nên không dùng chúng làm bằng chứng tổng quát hóa nội dung chưa thấy. Các seed chỉ đổi phần ảnh lặp trong support, không biểu diễn ba người viết hoặc ba lần huấn luyện độc lập. Màu bút xanh không được tái tạo vì mô hình native xuất grayscale.

## Artifact và tái lập

- [Gallery HTML](../../runs/fewshot_external_261116_component176500/index.html).
- [Ảnh tổng quan so sánh](../../runs/fewshot_external_261116_component176500/overview_comparison.png).
- [Ảnh đầu vào gốc](../../runs/fewshot_external_261116_component176500/references_original.png).
- [ZIP toàn bộ](../../runs/fewshot_external_261116_component176500.zip).
- `run.json`: checksum checkpoint/input, support đã chọn, seed, thời gian, bộ nhớ GPU và kiểm tra artifact.
- `prompts.json`, `samples.json`: toàn bộ prompt, nhóm và ánh xạ file ảnh; `SHA256SUMS`: checksum toàn bộ bundle.
- Script: `scripts/infer_fewshot_gallery.py`. Lệnh tái lập nằm trong README của output. Ảnh tham chiếu gốc được copy vào `reference_inputs` trong bundle.

Đã kiểm tra decode và kích thước đủ 2.418 PNG, xác nhận toàn bộ prompt hợp alphabet checkpoint, kiểm tra CRC toàn bộ ZIP. Training không được khởi động. Artifact inference nằm trong `runs`, chưa upload lên GitHub.
