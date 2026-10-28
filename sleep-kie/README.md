# SLEEP-KIE — recurrent consolidation cho single-read long-document KIE

Thử nghiệm theo [ý tưởng](docs/idea.md). Stage 0 kiểm tra trực tiếp **Go 1** (recurrence có tốt hơn N=1 và tốt hơn updater sâu hơn ở cùng compute không) trên Controlled Long-KIE dạng token với encoder đông cứng, trước khi đầu tư vào VLM/OCR thật.

- [Protocol stage 0](docs/stage0-protocol.md): dữ liệu, variant A0–A5, endpoint, tiêu chí go/no-go.
- [Nhật ký pilot stage 0](docs/stage0-pilot-log.md): **trạng thái hiện tại**. Pipeline đã chạy, N=1 học được lookup/revision và ngoại suy tới 200 trang; N=4 tied / deep4 chưa train ổn định nên ma trận khóa chưa chạy.
- Kết quả stage 0: `docs/stage0-results.md` sẽ được sinh bởi `scripts/report.py` sau khi chạy ma trận.

## Cấu trúc

- `sleepkie/data.py`: bộ sinh tài liệu (lookup, revision, link theo thứ tự bất kỳ, aggregation, missing; có distractor), encoder ngẫu nhiên đông cứng chạy trên GPU.
- `sleepkie/model.py`: `SleepKIE` (schema-conditioned slots, top-K selection, write → N block tied/untied → gate), `FullContext` (tham chiếu không streaming), `build()` cho bảng A0–A5.
- `scripts/train.py`: train một variant/seed, đánh giá trên lưới test cố định, đo latency/peak memory module memory.
- `scripts/report.py`: bảng theo T, loại suy luận, khoảng cách bằng chứng; paired bootstrap và quyết định Go 1.
- `scripts/run_stage0.sh`: toàn bộ ma trận 7 variant × 3 seed.

## Chạy lại

Chỉ cần `torch` và `numpy` (đã test với `torch 2.14 + cu130`, dùng chung `../celltracking/.venv`).

```bash
cd sleep-kie
python scripts/train.py --model A2_N4 --seed 0 --dev --tag pilot   # tuning: chỉ dev seed
bash scripts/run_stage0.sh                                          # ma trận khóa trước
python scripts/report.py --tag stage0
```

Stage 0 là synthetic. Nó chỉ trả lời câu hỏi về cơ chế memory, không phải Go 2–4 (grounding bbox, Pareto với VLM/retrieval, domain thật). Kleister-NDA, DocILE và MMLongBench-Doc thuộc stage sau, nếu Go 1 đạt.
