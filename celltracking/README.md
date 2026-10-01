# celltracking

Dữ liệu và benchmark v1 cho bài toán suy luận tổ tiên–hậu duệ từ ảnh tế bào chụp thưa.

- Hướng dẫn dữ liệu (protocol): [docs/260924_cell_ancestry_data_guide.md](docs/260924_cell_ancestry_data_guide.md)
- Báo cáo triển khai v0 (audit, quyết định thiết kế): [docs/260924_cell_ancestry_v0_report.md](docs/260924_cell_ancestry_v0_report.md)
- Báo cáo nghiên cứu (dữ liệu + phân tích baseline, có PDF): [docs/output_report/](docs/output_report/260924_cell_ancestry_baseline_study.md)
- Rà soát hướng nghiên cứu và thí nghiệm quyết định cho CVPR 2028: [docs/260930_cvpr2028_research_angles.md](docs/260930_cvpr2028_research_angles.md)

Các pilot đầu tiên của hướng suy tập hậu duệ và biến dạng (dùng dữ liệu đã index, benchmark v1 và checkpoint `family_set_h132.pt` hiện có):

```bash
python experiments/structure_flow_pilot.py topology
python experiments/structure_flow_pilot.py flow --seqs M3
python experiments/family_shape_ablation.py
python experiments/family_topology_pilot.py
python experiments/family_flow_joint_pilot.py
python experiments/coarse_grain_pilot.py   # hướng A: nén genealogy theo ancestry (mục 7 của tài liệu trên)
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
