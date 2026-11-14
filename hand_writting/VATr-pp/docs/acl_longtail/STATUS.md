# Trạng thái nghiên cứu và hướng training hiện hành

Cập nhật 14/11/2026. Người dùng đã chỉ định tiếp tục từ checkpoint VATr++ có
sẵn theo `261114_report_01_Tong_hop_trien_khai_va_ket_qua_200k.md`.

## Run hiện hành theo yêu cầu mới: 250.000 steps

Đã khởi chạy continuation **176.500 → 250.000** từ nhánh component, giữ nguyên
cấu hình và optimizer. Run nền: `runs/component_continuation_250k_v1/status.json`.
[Báo cáo tự cập nhật](261114_report_05_Continuation_component_den_250k.md) có link panel 600 mẫu mới nhất.
Checkpoint/benchmark mỗi 1.000 bước, giữ last/best/best_focus và 3 checkpoint
định kỳ gần nhất. Không coi việc khởi chạy là đã hoàn thành 250k.

## Mainline: fine-tune pretrained, không xây lại generator

**Pilot đã hoàn thành**, cả hai nhánh exit code 0. Kết quả và xác nhận optimizer/OCR tại [261114_report_04_Finetune_pretrained_175500_den_176500.md](261114_report_04_Finetune_pretrained_175500_den_176500.md). Không có run mới tự động nối tiếp pilot này.

- Nguồn: `runs/continuation_200k_v1/best.pth`, bước thực tế **175.500**.
- Báo cáo 200k khuyến nghị bản best này; nó khác `last.pth` ở 200.000.
- Khôi phục G, D, writer, OCR, Adam và RNG qua trainer native.
- Runtime gốc: `/home/giaphk/SpiS-GAN/.venv/bin/python`, torch 2.9.1+cu128.
- Pilot: 175.500 → 176.500 cho hai nhánh `legacy` / `component`.
- Chỉ thay trọng số lấy transcript ở lượt ưu tiên rare; kiến trúc, nguồn dữ
  liệu, batch/support, LR và các loss giữ theo nguồn.
- Bản component trộn 50% trọng số ký tự cũ và 50% trọng số thành phần/cặp
  tương tác có xét hoa/thường, bên trong rare fraction 0,3; HAT fraction 0,35.
- HAT tiếp tục không được dùng làm writer thật hoặc nhóm style.
- Baseline khôi phục của cả hai nhánh khớp artifact nguồn: CER 13,6257%, exact
  62,5% trên cùng panel 600 mẫu. Vì vậy đây thực sự là continuation pretrained.

Trạng thái thực tế: `runs/pretrained_longtail_v1/status.json`.
Log từng nhánh: `runs/pretrained_longtail_v1/{legacy,component}/train.log`.
Checkpoint cũ được giữ nguyên; không tự động promote model mới.

Đây là thích nghi có exposure cũ, **không gọi là zero-shot**. Sampling thành phần
là ablation phân bổ lượt học, không tạo ra ảnh writer thật còn thiếu hoặc chứng
minh generator đã học tách dấu. Module query/memory mới phải được thử riêng từ
cùng pretrained source, sau khi có đối chứng ổn định; chưa ghép kiến trúc mới vào
native checkpoint trong pilot này.

## Nhánh exploratory trước đó đã lưu riêng

`runs/acl_longtail_v1/steering.json` đánh dấu nhánh nghiên cứu sạch trước đó là
archived; các tiến trình còn chạy đã dừng khi người dùng chỉnh hướng. Không tự
khởi chạy lại các generator nhỏ từ đầu. Giữ artifact và code để tái sử dụng audit,
phân tích đối chứng, thống kê và kiểm thử, không gộp chúng vào kết quả VATr++.
`261114_report_03_Ket_qua_pilot_long_tail.md` là kết quả nhánh exploratory, không phải điểm của pretrained mainline.

## Phát hiện dữ liệu mới có thể kiểm tra

`character_context_audit.json` đếm từ metadata đã xuất trực tiếp từ HDF5:

- HAT có 100 ảnh chứa Ễ nhưng **0 ảnh transcript đúng một ký tự Ễ**.
- HAT có 103 ảnh chứa Ỡ nhưng **0 ảnh Ỡ đứng riêng**.
- Writer thật có 166 ảnh chứa ễ, 57 ảnh chứa ỡ, nhưng không có ảnh các ký tự
  này đứng riêng trong train.

Do đó failure của prompt ký tự đơn còn liên quan đến độ phủ ngữ cảnh/độ dài,
không chỉ tần suất Unicode. Không kết luận nhân quả chỉ từ các số đếm này. Cần
so ký tự đơn, trong từ, crop ký tự có kiểm tra alignment, và chất lượng OCR độc
lập trước khi gán mọi lỗi cho thiếu ảnh hoặc cách ghép dấu.

## Khoảng trống trước khi gọi là bài ACL hoàn chỉnh

1. Đối chứng pretrained, nhiều seed và ngân sách cập nhật/compute có kiểm soát.
2. Thử cơ chế chia sẻ thành phần trên pretrained G; sampling riêng chưa đủ.
3. Ảnh chữ hoa hiếm cùng writer/support để giám sát hoặc kiểm tra phong cách;
   prompt-only OCR vẫn đánh giá được độ đọc đúng nhưng không thay thế paired fidelity.
4. Đánh giá độc lập về nội dung, dấu, hình học và writer; chấm mù bằng người đọc.
5. Đối chứng VATr++ chuẩn, WriteViT và CONSTANT; không chỉ generator nhỏ tự viết.
6. Chốt protocol cuối trước khi mở test, xác minh nguồn/license và hoàn thiện
   phần kết quả/giới hạn. Không tự điền số đo chưa chạy.
