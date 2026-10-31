# handwriting — biến thiên có cấu trúc trong dòng chữ viết tay

Hiện thực kế hoạch `../handwriting_research_implementation_plan_vi.md`. Trạng thái: đã xong bản chạy thử Tuần 1 (chẩn đoán H1 trên IAM); chưa huấn luyện GAN.

- `hwline/` — parse IAM, descriptor theo từ, thống kê phụ thuộc trong dòng (null hoán vị, bootstrap theo writer)
- `scripts/` — fetch_iam, prepare_iam_lines (manifest + audit), extract_descriptors, make_gallery, analyze_line_dependence
- `reports/` — audit, gallery kiểm tra extractor, kết quả H1
- `docs/week1_h1_report.md` — kết quả và quyết định H1

Dữ liệu IAM: `/home/jovyan/bachdx2/datasets/iam` (Zenodo 19003721, CC-BY-NC-4.0). SpiS-GAN pin tại `/home/jovyan/bachdx2/repos/spis-gan` @ 6949fc2.
