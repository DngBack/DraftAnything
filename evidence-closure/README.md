# Evidence Closure — relational visual evidence dưới ngân sách token (TAT-DQA)

Kiểm tra H2 của đề xuất "Relational Evidence Closure": ở cùng ngân sách visual token, giữ *nhóm* bằng chứng có quan hệ (giá trị + nhãn hàng + header cột + đơn vị) có tốt hơn giữ token độc lập theo relevance không. Chưa fine-tune VLM.

- **Kết quả cuối và quyết định go/no-go:** [docs/findings.md](docs/findings.md) — kết quả trước lỗi chấm attention: [docs/findings_superseded.md](docs/findings_superseded.md)
- Protocol khóa trước: [docs/protocol.md](docs/protocol.md) · sai lệch: [docs/deviations.md](docs/deviations.md)
- Báo cáo số: `results/e1_qwen25vl7b.md`, `results/e1_qwen25vl3b.md`, `results/e2_qwen25vl7b.md`, `results/closure_eval.md`, đồ thị `results/e1_*_budget.png`

## Trạng thái

| Thí nghiệm | Nội dung | Trạng thái |
|---|---|---|
| E1–E7 | *mọi so sánh với B_attn ở E1–E7 dùng bước chấm attention có lỗi, xem deviations #12* | |
| E1 | A/B/C/D, seed attention / gold, 5 ngân sách, đối chứng locality; 7B và 3B; 600 câu dev | xong |
| Learned closure | MLP layout (geo) và query-conditioned, nhãn yếu từ TAT-DQA train | xong — không hơn heuristic |
| E2 | learned closure + group scoring trong VLM (7B, dev) | xong (thăm dò) |
| E3 | xác nhận trên TAT-DQA test, điều kiện khóa | xong — lặp lại toàn bộ claim |
| E4 | chấm nhóm bằng lex / VLM tự chấm (dev) | xong — VLM tự chấm kém |
| E5 | learned group selector (`evc/groupsel.py`), dev + test | xong — 3B +25, 7B −5 |
| E6 | lai H50 = attention + nhóm học được, dev → test khóa | xong — **bị thay thế** (baseline lỗi) |
| E7 | Qwen3-VL-8B (DeepStack) | xong — **bị thay thế** (baseline lỗi) |
| **FINAL** | sửa 2 lỗi chấm attention, layer (16,20,24), 3 model trên test | **xong — structural gap đạt; H50 − B_attn ≈ 0 ở cả 3 model (no-go cho H2 như phương pháp nén)** |

## Cấu trúc

- `evc/data.py`: TAT-DQA → word OCR, fact word theo `block_mapping`, render trang từ PDF đúng tọa độ OCR.
- `evc/closure.py`: tách dòng; heuristic closure (nhãn hàng, header cột, đơn vị); ánh xạ word → ô token 28 px; các bộ chọn `fill`, `attn_closure`, `group_closure`.
- `evc/learned.py`: đặc trưng cặp (seed, ứng viên), nhãn yếu, `PairMLP`, `LearnedClosure`.
- `evc/groupsel.py`: learned group selector (đặc trưng nhóm từ OCR/layout, listwise loss) — dùng trong E5/E6.
- `evc/vlm.py`: Qwen2.5-VL, attention câu hỏi → visual token (hook layer 4/8/12/16), decode greedy theo batch với tập token con, giữ M-RoPE gốc.
- `evc/metrics.py`: chấm value / scale dùng chung cho mọi điều kiện.
- `scripts/`: `build_set.py`, `build_test_set.py`, `run_e1.py`, `run_e2.py`, `run_e3.py`, `run_e4.py`, `run_e5.py`, `run_e6.py`, `train_closure.py`, `train_groupsel.py`, `analyze_e1.py`, `analyze_e2.py`, `analyze_e34.py`, `viz_closure.py`, `viz_tokens.py`.

## Chạy lại

```bash
python3 -m venv .venv && .venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 \
  && .venv/bin/pip install "transformers>=4.57,<5" accelerate pillow numpy pandas scipy matplotlib pymupdf
# dữ liệu: next-tat/TAT-DQA (dataset json + docs zip) vào data/, model Qwen2.5-VL-{7B,3B}-Instruct vào hf/
.venv/bin/python scripts/build_set.py                     # data/e1_set.json + render trang
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_e1.py --tag qwen25vl7b
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_e1.py --model hf/Qwen2.5-VL-3B-Instruct --tag qwen25vl3b
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/train_closure.py   # cần data/docs_train
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_e2.py
.venv/bin/python scripts/analyze_e1.py --tag qwen25vl7b && .venv/bin/python scripts/analyze_e2.py
.venv/bin/python scripts/build_test_set.py && CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_e3.py
```

Trên máy này pip bị đứng khi tải từ PyPI/pytorch, nên wheel được tải bằng curl có resume rồi cài với `--no-index`. Mỗi lượt 600 câu × 30 điều kiện mất ~45–90 phút trên một H200 dùng chung.

> Đính chính: các số ở hai mục H3 và H3b dưới đây dùng thước đo dễ dãi. Số theo EM chính thức: [docs/official_metric_correction.md](docs/official_metric_correction.md).

## H3 (kết quả test, 2026-11-05)

Learned answer-source selection hơn baseline tốt nhất ở cùng ngân sách: 7B **+8.7 / +9.3** (B = 2 / B = 3), Qwen3 **+7.0** (B = 3). CI không chứa 0. Chi tiết và hạn chế: [docs/h3_results.md](docs/h3_results.md).

## H3b (test2, 2026-11-05)

Bộ chọn tách đôi v2 (nguồn giá trị + bộ phân loại scale) đạt 71.9–73.4 (7B) và 70.9–72.4 (Qwen3). Nó hơn luật tra bảng +6 đến +8 điểm và hơn H3-learned +4 đến +6 điểm; mọi CI đều không chứa 0. Chi tiết: [docs/h3b_results.md](docs/h3b_results.md).

## H3c (dev2, EM, 2026-11-06)

Học lại trên nhãn EM không giúp. v2 hơn Fixed-2 ở cả hai model (+5.7 đến +14.8); chỉ hơn H3-learned ở 7B. Chi tiết: [docs/h3c_results.md](docs/h3c_results.md).

## Hướng ACL 2028 (pilot, 2026-11-06)

Các pilot sửa chữ số trong ảnh và bản ghi quyết định: [docs/acl2028_decision_memo.md](docs/acl2028_decision_memo.md).

## Paper draft

[paper/](paper/README.md): ACL-format draft on an incomplete sample. See its README for status and how to rebuild.
