# Binding-aware selective KIE — pilot v2

**Kết quả mới nhất:** [tiến độ nghiên cứu CVPR](docs/research-progress.md), [stage 2](docs/stage2-results.md), [holdout mới trên ảnh/OCR](docs/stage3-results.md).

Thử nghiệm theo [ý tưởng](docs/idea.md), tập trung E0, correctness baselines cùng thông tin (E1) và ablation (E3). Dùng lại prediction thật của hai VLM trong `../selective-kie/outputs`, không sinh thêm kết quả extractor.

## Chạy lại

```bash
# Từ workspace root
uv venv select-kie-v2/.venv
uv pip install --python select-kie-v2/.venv/bin/python -r select-kie-v2/requirements.lock.txt
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 select-kie-v2/.venv/bin/python select-kie-v2/scripts/pilot.py
select-kie-v2/.venv/bin/python select-kie-v2/scripts/make_report.py
```

Python được dùng trong lượt đầu: 3.14.4. Cần tải CORD-v2 (~2.3 GB parquet, thêm cache Arrow); phân tích/head chạy CPU. Cache tự đặt trong thư mục dự án để tránh lỗi permission ở HF cache hệ thống.

## Artifact

- [Protocol](docs/pilot-protocol.md): thiết lập, endpoint và giới hạn trước lượt phân tích mới.
- [Báo cáo số liệu](docs/pilot-results.md) và [nhận định](docs/pilot-findings.md).
- `results/metrics.json`: AURC, Brier, coverage/risk, bootstrap và sensitivity.
- `results/predictions-*.csv`: score của từng head cho từng quyết định test.
- `results/manifest.json`: hash input, kiểm tra trùng ảnh và seed.
- `results/audit.csv`: mẫu ngẫu nhiên và mẫu lỗi bổ sung; các cột human còn trống.
- `results/risk-coverage.png`: đường ranking trên test; khác với threshold chọn từ validation.
- `results/run.log`: output thực tế của lượt chạy.
- `results/heads/*.joblib`: các correctness head đã fit, kèm thứ tự features và field.

`scripts/pilot.py` tái sử dụng loader/normalization/taxonomy của pilot cũ. Không đưa category, gold hay nhãn error vào feature huấn luyện; nhãn correctness chỉ là target. Baseline HGB và LR fusion+binding được cấp cùng feature matrix, cùng nhãn train. Không có tuning trên test.

Riêng pilot v2 là phân tích thăm dò trên test đã từng được sử dụng, chưa có unseen-template evaluation. Các bootstrap CI có điều kiện trên head đã fit, chỉ resample validation/test theo tài liệu. Không có bảo đảm distribution-free risk hoặc kết quả fine-tune VLM.

## Stage 2 / 3

Stage2 mở rộng bằng official VRDU unseen-template split; stage3 dùng100 Short Form mới chưa dùng ở stage2. Không fine-tune backbone. Các protocol và báo cáo ghi rõ primary/secondary, calibration domain và normalization.

```bash
# CPU: CORD evidence ablation
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  select-kie-v2/.venv/bin/python select-kie-v2/scripts/cord_stage2.py

# GPU: sau khi đặt fara.jsonl.gz và official split trong data/vrdu
CUDA_VISIBLE_DEVICES=0 select-kie-v2/.venv-gpu/bin/python select-kie-v2/scripts/vrdu_extract.py
CUDA_VISIBLE_DEVICES=1 select-kie-v2/.venv-gpu/bin/python select-kie-v2/scripts/vrdu_extract.py \
  --model /home/bachdx2/hf_cache/hub/models--Qwen--Qwen3.5-2B \
  --output vrdu-qwen35-predictions.jsonl

# CPU evaluation (require600 complete docs)
select-kie-v2/.venv/bin/python select-kie-v2/scripts/vrdu_evaluate.py
select-kie-v2/.venv/bin/python select-kie-v2/scripts/vrdu_evaluate.py --predictions vrdu-qwen35-predictions.jsonl
select-kie-v2/.venv/bin/python select-kie-v2/scripts/vrdu_label_sensitivity.py
select-kie-v2/.venv/bin/python select-kie-v2/scripts/stage2_report.py

# Fresh holdout manifest/PDFs, then2x2 experiment (resumable)
select-kie-v2/.venv/bin/python select-kie-v2/scripts/prepare_holdout.py
CUDA_VISIBLE_DEVICES=0 select-kie-v2/.venv-gpu/bin/python select-kie-v2/scripts/holdout_extract.py --modality image --batch-size 4
CUDA_VISIBLE_DEVICES=1 select-kie-v2/.venv-gpu/bin/python select-kie-v2/scripts/holdout_extract.py --modality ocr --batch-size 8
select-kie-v2/.venv/bin/python select-kie-v2/scripts/holdout_evaluate.py
select-kie-v2/.venv/bin/python select-kie-v2/scripts/stage3_report.py
```

GPU env lượt chạy: Python3.10.12, torch2.6.0+cu124, transformers5.17.0, A30. Torch được reuse read-only từ `/home/bachdx2/CS-WAE/.venv/lib/python3.10/site-packages` qua `.pth` trong env mới. Khi chuyển máy, cài torch tương thích riêng rồi dùng `requirements-gpu.lock.txt`; không phụ thuộc `.pth` máy này. Kernel DeltaNet/causal-conv dùng reference PyTorch, chưa tối ưu throughput.

Dữ liệu raw/PDF và env bị gitignore. Tải `registration-form/main/dataset.jsonl.gz` và official split từ https://github.com/google-research-datasets/vrdu ; revision/hashes nằm trong manifests. File matcher Apache2.0 vendored tại `vendor/vrdu_match_utils.py` từ Google Research, giữ copyright. Script inference chứa default cache paths của máy hiện tại; `vrdu_extract.py` hỗ trợ `--model`, các script holdout dùng snapshot Qwen3.5 cố định.

## Stage 4: scale and length control

Protocol: [stage4-protocol.md](docs/stage4-protocol.md). Separate fresh100-document manifest excludes stages2/3; all-page image extraction, three prompts per model. `stage4_extract.py` supports an explicit local snapshot path and resumes by document/schema. Download Qwen3.5-9B from its [official model repository](https://huggingface.co/Qwen/Qwen3.5-9B), preserving the snapshot revision.

```bash
select-kie-v2/.venv-gpu/bin/python select-kie-v2/scripts/prepare_stage4.py
select-kie-v2/.venv-gpu/bin/python select-kie-v2/scripts/verify_stage4_inputs.py
CUDA_VISIBLE_DEVICES=1 select-kie-v2/.venv-gpu/bin/python select-kie-v2/scripts/stage4_extract.py \
  --model /absolute/path/to/Qwen3.5-2B/snapshot --tag qwen2b --modality image --batch-size 4
CUDA_VISIBLE_DEVICES=0 select-kie-v2/.venv-gpu/bin/python select-kie-v2/scripts/stage4_extract.py \
  --model /absolute/path/to/Qwen3.5-9B/snapshot --tag qwen9b --modality image --batch-size 1
PYTHONDONTWRITEBYTECODE=1 select-kie-v2/.venv-gpu/bin/python select-kie-v2/scripts/stage4_evaluate.py
select-kie-v2/.venv/bin/python select-kie-v2/scripts/stage4_report.py
```

Human audit is pending. Use [audit guide](docs/stage4-audit-guide.md) and the blinded viewer/CSV generated after evaluation; do not infer binding prevalence from the targeted error sample.

Actual9B run also supports two disjoint GPU workers using `stage4_extract_shard.py --shard 0` and `--shard 1`, with otherwise the same arguments. Shards select documents by SHA256 parity and retain any predictions completed by the single-worker runner. Run `stage4_merge.py` after both finish; it requires exactly300 unique tasks. Use batch2 for9B on A30; multi-page batch4 can exceed24GB. Runtime retries are documented in `results/stage4/runtime-events.json` and excluded from generate-only latency.

After human reviewers fill their CSV files, run `stage4_audit_summary.py`; it reports pending labels, agreement on completed overlap and binding prevalence on the random9B-clarified sample only. Keep the original independent labels before adjudication.

## Learned-head pilot and fresh confirmation (stages5/6)

[Full outcome and limitations](docs/stage6-results.md). This is a prototype: gains against an equal-information MLP do not establish superiority to semantic HGB, binding-specific gains or a CVPR score. Primary confirmation remains fixed; stronger HGB, field-threshold and present-only diagnostics are marked posthoc.

Use the existing GPU environment for torch/MiniLM and the original CPU environment (sklearn1.9.1) for archived HGB estimators. Set these task-specific variables to downloaded local snapshot directories:

```bash
KIE_ENCODER_PATH=/home/bachdx2/hf_cache/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41
KIE_MODEL_9B=/home/bachdx2/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 select-kie-v2/.venv-gpu/bin/python select-kie-v2/scripts/relation_features.py --encoder "$KIE_ENCODER_PATH"
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 select-kie-v2/.venv/bin/python select-kie-v2/scripts/relation_references.py
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 select-kie-v2/.venv-gpu/bin/python select-kie-v2/scripts/relation_train.py

# Freeze before any confirmation predictions; existing manifest is verified, not overwritten.
select-kie-v2/.venv-gpu/bin/python select-kie-v2/scripts/freeze_relation_heads.py
select-kie-v2/.venv-gpu/bin/python select-kie-v2/scripts/prepare_stage6.py
select-kie-v2/.venv-gpu/bin/python select-kie-v2/scripts/verify_stage6_inputs.py
# Run these two disjoint workers concurrently on the two GPUs.
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 select-kie-v2/.venv-gpu/bin/python select-kie-v2/scripts/stage6_extract.py --model "$KIE_MODEL_9B" --tag qwen9b --modality image --batch-size 2 --shard 0
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 select-kie-v2/.venv-gpu/bin/python select-kie-v2/scripts/stage6_extract.py --model "$KIE_MODEL_9B" --tag qwen9b --modality image --batch-size 2 --shard 1
select-kie-v2/.venv-gpu/bin/python select-kie-v2/scripts/stage6_merge.py
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 select-kie-v2/.venv-gpu/bin/python select-kie-v2/scripts/stage6_evaluate.py
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 select-kie-v2/.venv/bin/python select-kie-v2/scripts/stage6_references.py

# Additional diagnostics; do not substitute them for the registered primary comparison.
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 select-kie-v2/.venv/bin/python select-kie-v2/scripts/semantic_hgb.py
PYTHONDONTWRITEBYTECODE=1 select-kie-v2/.venv/bin/python select-kie-v2/scripts/stage6_sensitivity.py
PYTHONDONTWRITEBYTECODE=1 select-kie-v2/.venv/bin/python select-kie-v2/scripts/field_threshold_diagnostic.py
select-kie-v2/.venv/bin/python select-kie-v2/scripts/latest_report.py
```

The encoder uses [official MiniLM mean-pooling instructions](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). All confidence features use annotation OCR; its production cost/quality is unmeasured. Inference `batch_seconds` includes generation and token log-probability extraction, and excludes PDF rendering/model loading. Encoder-related timers cover their surrounding processing blocks and are not end-to-end latency. Scalar clipping/standardization fit train only. Fresh confirmation is document-independent, while the target template was already present in development/calibration; registrants/persons may recur.
