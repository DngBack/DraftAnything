# CEM-KIE — kiểm tra khả thi competing evidence memory trên Kleister-Charity

Kiểm tra giả thuyết H1 của proposal CEM-KIE trên một benchmark KIE tài liệu dài chuẩn: giữ nhiều giả thuyết bằng chứng riêng cho từng field có giúp phân giải bằng chứng cạnh tranh tốt hơn chấm điểm độc lập, vector gộp, hay Transformer dùng chung hay không. So sánh ở cùng encoder đông cứng, cùng feature và cùng router.

- [Protocol và tiêu chí khóa trước](docs/protocol.md)
- [Audit dữ liệu](results/audit.md): hiện tượng competing evidence và các heuristic.
- [Kết quả](results/main/report.md) và [nhận định](docs/results.md).
- Stage B, CoSolve oracle (solver đồng thời income/spending với ràng buộc net): [protocol](docs/protocol-cosolve.md), [kết quả](results/cosolve/report.md), [nhận định](docs/results-cosolve.md).

## Cấu trúc

- `cemkie/data.py`: đọc Kleister-Charity, sinh ứng viên bằng regex (tiền, ngày, số đăng ký, postcode) và chuẩn hóa về format expected.tsv.
- `cemkie/model.py`: trunk + router dùng chung; các biến thể `point`, `pool`, `set2/4`, `cem_r{1,2,4,8}`, `cem_r4_nores`.
- `scripts/audit.py`: thống kê ứng viên, oracle, heuristic (CPU).
- `scripts/embed.py`: mã hóa context mọi mention bằng `BAAI/bge-base-en-v1.5`, lưu vào `cache/`.
- `scripts/train.py`, `scripts/report.py`, `scripts/run_all.sh`.

## Chạy lại

Venv dùng lại torch của `../celltracking/.venv` qua file `.pth`, chỉ cài thêm `transformers`.

```bash
git clone --depth 1 https://github.com/applicaai/kleister-charity.git /home/jovyan/bachdx2/datasets/kleister/kleister-charity
cd cem-kie
.venv/bin/python scripts/audit.py
.venv/bin/python scripts/embed.py          # GPU, tạo cache/train.pt và cache/dev-0.pt
bash scripts/run_all.sh                    # 9 variant × 3 seed
.venv/bin/python scripts/report.py --tag main
.venv/bin/python scripts/llm_baseline.py                          # Qwen3-VL-4B zero-shot, dev-0
TAG=topm VARIANTS="set4 point" bash scripts/run_all.sh            # lưu top-M cho solver
(cd scripts && ../.venv/bin/python llm_score.py val && ../.venv/bin/python llm_score.py test)
(cd scripts && ../.venv/bin/python solver.py)
```
