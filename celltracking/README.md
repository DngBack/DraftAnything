# celltracking

Dữ liệu và benchmark v1 cho bài toán suy luận tổ tiên–hậu duệ từ ảnh tế bào chụp thưa.

- Hướng dẫn dữ liệu (protocol): [docs/260924_cell_ancestry_data_guide.md](docs/260924_cell_ancestry_data_guide.md)
- Báo cáo triển khai v0 (audit, quyết định thiết kế): [docs/260924_cell_ancestry_v0_report.md](docs/260924_cell_ancestry_v0_report.md)
- Báo cáo nghiên cứu (dữ liệu + phân tích baseline, có PDF): [docs/output_report/](docs/output_report/260924_cell_ancestry_baseline_study.md)
- Rà soát hướng nghiên cứu và thí nghiệm quyết định cho CVPR 2028: [docs/260930_cvpr2028_research_angles.md](docs/260930_cvpr2028_research_angles.md)
- Cập nhật 02/10/2026: đối chứng pairwise năm fold, T98G/LFCT/FAST và recA/rpsM; đã phát hiện và sửa lỗi centroid nằm trong crop thay đổi theo frame. Điểm recA/rpsM khóa trước ban đầu không hợp lệ; bản chạy lại cho thấy nở hướng tâm hơn nearest một ít nhưng chưa đủ thành phương pháp CVPR ([báo cáo Markdown](docs/output_report/261002_cvpr_celltracking_research_update.md), [PDF](docs/output_report/261002_cvpr_celltracking_research_update.pdf), [kiểm toán recA](docs/261002_reca_future_holdout_protocol.md), [kiểm toán rpsM](docs/261002_rpsm_radial_future_holdout_protocol.md)).

Các pilot đầu tiên của hướng suy tập hậu duệ và biến dạng (dùng dữ liệu đã index, benchmark v1 và checkpoint `family_set_h132.pt` hiện có):

```bash
python experiments/structure_flow_pilot.py topology
python experiments/structure_flow_pilot.py flow --seqs M3
python experiments/family_shape_ablation.py
python experiments/family_topology_pilot.py
python experiments/family_flow_joint_pilot.py
python experiments/family_flow_shape_factorial.py  # flow × hình dạng; báo cáo trong docs/output_report/
python experiments/coarse_grain_pilot.py   # hướng A: nén genealogy theo ancestry (mục 7 của tài liệu trên)
python experiments/leave_one_movie_center_context.py  # thông tin frame đích ở H=132
python experiments/leave_one_movie_multihorizon.py     # năm fold, H=33/66/132
python experiments/nearest_center_horizon_control.py   # motion prior không dùng family decoder
python experiments/leave_one_movie_pairwise_control.py  # đối chứng học từng cặp, năm movie
python experiments/growth_clock_transfer.py            # đối chứng chéo miền CTC
python experiments/tui_t98g_external_test.py            # T98G test theo protocol khóa trước; cần ZIP Zenodo
python experiments/lfct_example_external_test.py         # một field LFCT; cần XML GitHub
python experiments/fast_microcolony_external_test.py     # bảy colony FAST; cần ZIP Figshare
python experiments/reca_microcolony_external_test.py     # chín colony recA; cần files Zenodo theo protocol
python experiments/summarize_reca_external.py            # tóm tắt theo colony và gap
python experiments/growth_density_regime_probe.py        # probe hậu nghiệm, không là test blind
python experiments/radial_sector_probe.py                # đối chứng đăng ký/nở hướng tâm trên development sets
python experiments/rpsm_radial_external_test.py          # chín colony rpsM theo protocol đã khóa
python experiments/summarize_rpsm_external.py            # tóm tắt rpsM theo colony/gap
python experiments/unsupervised_prior_gate_probe.py     # probe chọn prior theo năng lượng đám điểm
```

```bash
DS="one_in_a_million sim_plus hsc musc hela"
python main.py fetch $DS                           # -> data/ (chỉ dữ liệu đầu vào)
for d in $DS; do python main.py index $d; done     # -> outputs/index, outputs/audit
python main.py build && python main.py baseline    # -> outputs/benchmarks/v1, outputs/results
python main.py stats && python main.py figures     # -> outputs/tables, outputs/figures
PYTHONPATH=. python -m pytest tests
```

Build lại PDF (cần pandoc + xelatex):

```bash
cd docs/output_report && pandoc 260924_cell_ancestry_baseline_study.md -o 260924_cell_ancestry_baseline_study.pdf \
  --pdf-engine=xelatex --resource-path=. -V fontfamily=fontspec -V mainfont="DejaVu Serif" \
  -V sansfont="DejaVu Sans" -V monofont="DejaVu Sans Mono" -V geometry:margin=2cm -V fontsize=10pt \
  -V colorlinks=true --toc --no-highlight --columns=50
```

Nếu máy không có font DejaVu hệ thống, trỏ tới bản đi kèm matplotlib, ví dụ `-V mainfont=DejaVuSerif -V mainfontoptions="Path=<site-packages>/matplotlib/mpl-data/fonts/ttf/,Extension=.ttf,BoldFont=*-Bold,ItalicFont=*-Italic,BoldItalicFont=*-BoldItalic"` (tương tự cho `sansfont`/`monofont` với `*-Oblique`).

`data/` (≈30 GB) và phần nặng của `outputs/` (index, benchmarks, overlays) không nằm trong git; tái lập được bằng các lệnh trên.
