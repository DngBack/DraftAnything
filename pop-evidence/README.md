# PopEvidence — population evidence cho nhận diện thành phần trong ảnh vi sinh

Kiểm tra xem cấu trúc phân phối của tile embeddings trong một quần thể có giúp nhận diện species khó và tổng quát hóa sang tổ hợp chưa thấy tốt hơn biểu diễn trung bình hay không. Dữ liệu là PHOEBI (6 species, 40 combination, nhãn cấp combination, leave-combinations-out). Encoder đông cứng (DINOv2-S/14) và tiền xử lý lấy đúng từ pipeline PHOEBI.

- [Protocol khóa trước](docs/protocol.md): giả thuyết H1–H4, split, endpoint, gate, confound.
- [Novelty audit](docs/literature_matrix.md).
- [Experiment log](reports/experiment_log.md).
- Audit dữ liệu: `results/audit.md` (sinh khi đã có manifest).

## Trạng thái

| Giai đoạn | Trạng thái |
|---|---|
| A. Tái lập PHOEBI | Đã clone và đọc code; **dataset gated, chưa có HF token** |
| B. Phân tích lỗi | Có sẵn công cụ: per-species, per-combination, context sensitivity (`popev/metrics.py`) |
| C. Probe phân phối | Đã cài và test trên synthetic: mean, mean+std, mean+quantile, KME, Deep Sets (`configs/probe.yaml`) |
| D. PopEvidence | Đã cài: evidence theo species, pooling mean/std/quantile, consistency loss, ablation class-specific (`configs/popevidence.yaml`). Chỉ chạy sau Gate 2 |
| Test | `scripts/run_test.py`, khóa bằng commit, không chạy lại được |

## Cấu trúc

- `popev/phoebi.py`: manifest LCO, 5 dev fold → chỉ số hàng của feature store; assert fold không chạm test.
- `popev/features.py`: trích CLS / patch-mean của 16 tile mỗi ảnh bằng `UnitExtractor` của PHOEBI → `cache/features_<kind>.npy` (N, 16, 384) fp16.
- `popev/bags.py`: bag n ảnh cùng combination; eval bag cố định theo `bag_seed`, nên mọi method chấm trên cùng bag.
- `popev/pooling.py`: biểu diễn cố định (fit chỉ trên train).
- `popev/models.py`: `LinearProbe`, `DeepSets`, `PopEvidence`, `context_consistency`.
- `popev/train.py`: train một method trên một split, threshold argmax-F1 trên calibration bag.
- `popev/metrics.py`: AP/AUROC/F1/FPR, bootstrap theo combination, paired delta, context sensitivity, bảng theo combination.
- `popev/experiment.py`: chạy method × fold × n × seed, gộp dev, viết `report.md` và `summary.json`.
- `scripts/`: `audit.py`, `extract.py`, `run_dev.py`, `run_test.py`, `make_synthetic.py`, `repro_phoebi.sh`.

## Chạy

```bash
# 0. quyền truy cập dataset (một lần): chấp nhận điều kiện trên huggingface.co/datasets/sochastic/PHOEBI, rồi
.venv/bin/hf auth login
bash scripts/repro_phoebi.sh fetch                  # ~13 GB vào $PHOEBI_ROOT/data/images

# 1. audit + tái lập (Gate 1)
.venv/bin/python scripts/audit.py                   # results/audit.md
bash scripts/repro_phoebi.sh random
bash scripts/repro_phoebi.sh lco
bash scripts/repro_phoebi.sh devcv                  # ProtoMatch/SimplexUnmix/ChannelGroup trên cùng 5 fold

# 2. feature (GPU, một lần, ~2.4 GB mỗi kind)
.venv/bin/python scripts/extract.py --kinds cls patchmean

# 3. probe phân phối (Gate 2), rồi PopEvidence (Gate 3)
.venv/bin/python scripts/run_dev.py --config configs/probe.yaml
.venv/bin/python scripts/run_dev.py --config configs/probe.yaml --kind patchmean
.venv/bin/python scripts/run_dev.py --config configs/popevidence.yaml

# 4. test, một lần, sau khi khóa configs/locked.yaml
.venv/bin/python scripts/run_test.py --config configs/locked.yaml

# kiểm thử
.venv/bin/python scripts/make_synthetic.py && .venv/bin/python -m pytest -q tests
.venv/bin/python scripts/run_dev.py --config configs/smoke_probe.yaml --data_dir data/synth \
    --feature_tag _synth --out results/dev_synth/probe --device cpu --n_boot 300
```

Venv dùng lại torch của `../celltracking/.venv` qua file `.pth`. PHOEBI nằm ở `$PHOEBI_ROOT` (mặc định `/home/jovyan/bachdx2/datasets/phoebi/repo`) và không bị sửa.
