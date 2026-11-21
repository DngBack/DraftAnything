# Mục lục tài liệu VATr++

## Tái lập và weights

[Hướng dẫn từ clone sạch](../REPRODUCE.md) · [Weights và checksums](../WEIGHTS.md).

## Quy tắc tên report

`YYMMDD_report_XX_Ten_report.md` — ví dụ `261114_report_05_Continuation_component_den_250k.md`.

- `YYMMDD`: năm, tháng, ngày; `261114` là 14/11/2026.
- `XX`: số thứ tự hai chữ số trong ngày, dùng chung cho các thư mục con và bắt đầu từ `01` mỗi ngày. Giữ số đã cấp khi cập nhật report.
- Tên mô tả dùng chữ không dấu và dấu gạch dưới để dễ tìm kiếm.
- Ngày lấy từ nội dung/tiêu đề report; khi không có ngày rõ ràng thì dùng ngày sửa file gốc. Thứ tự trong ngày sắp theo mạch đọc, không khẳng định giờ tạo.
- Giữ thư mục chủ đề và các ảnh/JSON/CSV đi kèm. Hướng dẫn, đề cương, protocol, bản thảo và trạng thái vận hành giữ tên chức năng.

## Theo dõi training hiện tại

[Continuation component đến 250k](acl_longtail/261114_report_05_Continuation_component_den_250k.md) · [Trạng thái nghiên cứu](acl_longtail/STATUS.md).

Training 250k đã hoàn tất; xem report tổng hợp ngày 15/11/2026 và các artifact benchmark được lưu cùng tài liệu.

## Report theo ngày

### 18/11/2026

| Số | Report |
|---:|---|
| 03 | [Plan: Kiến trúc mới, biểu diễn thành phần nét và trộn style hai tầng](../../PartScribe/docs/261118_plan_03_Kien_truc_moi_bieu_dien_thanh_phan_va_tron_style.md) |
| 04 | [Giai đoạn 0 PartScribe: tổng quan tài liệu và protocol giữ lại tổ hợp](../../PartScribe/docs/261118_report_04_Tong_quan_tai_lieu_va_protocol_giu_lai_to_hop.md) |

### 16/11/2026

| Số | Report |
|---:|---|
| 01 | [Inference few-shot: 12 ảnh, 806 prompt](acl_longtail/261116_report_01_Inference_few_shot_12_anh_806_prompt.md) |

### 15/11/2026

| Số | Report |
|---:|---|
| 01 | [Tổng hợp kết quả sau 250.000 steps](acl_longtail/261115_report_01_Tong_hop_ket_qua_sau_250k.md) |

### 14/11/2026

| Số | Report |
|---:|---|
| 01 | [Báo cáo tổng hợp triển khai và kết quả cải tiến VATr++ tiếng Việt đến 200.000 steps](261114_report_01_Tong_hop_trien_khai_va_ket_qua_200k.md) |
| 02 | [VietCompose: measured pilot results and research status](vietcompose/261114_report_02_Ket_qua_pilot_VietCompose.md) |
| 03 | [Measured long-tail research pilot](acl_longtail/261114_report_03_Ket_qua_pilot_long_tail.md) |
| 04 | [Fine-tune VATr++ từ checkpoint pretrained](acl_longtail/261114_report_04_Finetune_pretrained_175500_den_176500.md) |
| 05 | [Tiếp tục component sampling đến 250.000 steps](acl_longtail/261114_report_05_Continuation_component_den_250k.md) |
| 06 | [Báo cáo các cải tiến đã triển khai và thí nghiệm đang chạy trên VATr++](acl_longtail/261114_report_06_Cai_tien_da_trien_khai_va_training_dang_chay.md) |
| 07 | [Giải thích chi tiết thuật toán bổ sung và các ý tưởng nghiên cứu VATr++ tiếng Việt](acl_longtail/261114_report_07_Giai_thich_chi_tiet_thuat_toan_bo_sung_va_y_tuong.md) |

### 12/11/2026

| Số | Report |
|---:|---|
| 01 | [Training debug — 2026-11-12](261112_report_01_Debug_training.md) |
| 02 | [Báo cáo training VATr++ tiếng Việt — 100.000 steps](training_report_100k/261112_report_02_Chi_tiet_training_100k.md) |
| 03 | [Báo cáo kết quả fine-tuning VATr++ tiếng Việt — 100.000 steps](261112_report_03_Ket_qua_finetune_100k.md) |
| 04 | [Báo cáo cải tiến đánh giá và tiến độ triển khai sau fine-tuning 100k](261112_report_04_Cai_tien_va_tien_do_sau_100k.md) |
| 05 | [Thử nghiệm cải thiện VATr++ sau 100k](261112_report_05_Thiet_ke_ablation_sau_100k.md) |
| 06 | [Thử nghiệm bổ sung giám sát ảnh thật cho chữ Ễ](261112_report_06_Thu_nghiem_bo_sung_chu_E_mu_nga.md) |
| 07 | [OCR đọc đúng nhưng chữ sinh chưa đẹp: phân tích và thử nghiệm cải thiện](261112_report_07_OCR_dung_nhung_chu_xau_va_cai_tien.md) |

### 11/11/2026

| Số | Report |
|---:|---|
| 01 | [VATr-pp cho tiếng Việt: thuật toán, thay đổi và nhật ký triển khai](261111_report_01_Thuat_toan_va_trien_khai_VATr_pp_tieng_Viet.md) |
| 02 | [Nhật ký huấn luyện VATr++ tiếng Việt — 11/11/2026](261111_report_02_Nhat_ky_training_tieng_Viet.md) |

## Tài liệu tham khảo và nghiên cứu

- [Hướng dẫn chạy tiếng Việt](VIETNAMESE.md).
- [Đề xuất VietCompose](vietcompose/RESEARCH_PROPOSAL.md).
- [Thiết kế nghiên cứu long-tail](vietcompose/LONG_TAIL_RESEARCH_VI.md).
- [Protocol nghiên cứu ACL](acl_longtail/STUDY_PROTOCOL.md).
- [Bản thảo bài nghiên cứu](acl_longtail/PAPER_DRAFT.md).

Bảng đối chiếu tên cũ → tên mới: [report_catalog.json](report_catalog.json).
