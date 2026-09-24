# celltracking

Dữ liệu và benchmark v0 cho bài toán suy luận tổ tiên–hậu duệ từ ảnh tế bào chụp thưa.

- Hướng dẫn dữ liệu (protocol): [docs/260924_cell_ancestry_data_guide.md](docs/260924_cell_ancestry_data_guide.md)
- Báo cáo triển khai v0 (audit, quyết định thiết kế): [docs/260924_cell_ancestry_v0_report.md](docs/260924_cell_ancestry_v0_report.md)
- Báo cáo nghiên cứu (dữ liệu + phân tích baseline, có PDF): [docs/output_report/](docs/output_report/260924_cell_ancestry_baseline_study.md)

```bash
DS="one_in_a_million sim_plus hsc musc hela"
python main.py fetch $DS                           # -> data/ (chỉ dữ liệu đầu vào)
for d in $DS; do python main.py index $d; done     # -> outputs/index, outputs/audit
python main.py build && python main.py baseline    # -> outputs/benchmarks/v0, outputs/results
python main.py stats && python main.py figures     # -> outputs/tables, outputs/figures
python -m pytest tests
```

Build lại PDF (cần pandoc + xelatex):

```bash
cd docs/output_report && pandoc 260924_cell_ancestry_baseline_study.md -o 260924_cell_ancestry_baseline_study.pdf \
  --pdf-engine=xelatex --resource-path=. -V fontfamily=fontspec -V mainfont="DejaVu Serif" \
  -V sansfont="DejaVu Sans" -V monofont="DejaVu Sans Mono" -V geometry:margin=2cm -V fontsize=10pt \
  -V colorlinks=true --toc --no-highlight --columns=50
```

`data/` (≈30 GB) và phần nặng của `outputs/` (index, benchmarks, overlays) không nằm trong git; tái lập được bằng các lệnh trên.
