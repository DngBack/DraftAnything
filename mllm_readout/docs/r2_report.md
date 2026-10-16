# R2 report (protocol: docs/r2_protocol.md) — 2026-10-15

## E. Real-image counterfactual partners, What'sUp (whatsup_pairs.py, results/r2_whatsup.csv)
205 object pairs x 4 relations, same prompt within a pair; 5 folds grouped by object pair. 13 models; 4 cells reach
>= 20 loss errors.
| cell | model acc | loss errors | probe right on partners | calib. | model |
|---|---|---|---|---|---|
| Gemma-3-27B | 75.5 | 22 | 0.97 | 0.97 | 0.82 |
| Gemma-3-4B | 53.5 | 153 | 0.92 | 0.75 | 0.61 |
| LLaVA-1.5-7B | 61.1 | 83 | 0.96 | 0.87 | 0.74 |
| Qwen2-VL-2B | 67.2 | 85 | 1.00 | 0.85 | 0.56 |
- E1 median 0.96 (bar 0.70) -> **holds**. E2 4/4 -> **holds**.
- E3: on correctly answered items the probe is right on partners 0.94-0.99, on bias errors 0.93-1.00: on real photos
  the probe tracks the image equally on loss errors. The probe is near ceiling on What'sUp (93-99%), so E1 mainly
  shows that the signal is image-driven for unseen object pairs; it does not separate loss from bias errors.
- Reading together with B1 (synthetic, 0.65): the synthetic miss may reflect few loss errors on near-solved synthetic
  tasks (attr 0-3 per cell); on real photos with many loss errors the visual-dependence test passes clearly.

## F. Steering strength chosen per model on validation images (alpha_select.py, results/r2_alpha.csv) — INCOMPLETE
Finished 7/13 models (dev: Qwen3-VL-2B/4B, Gemma-3-4B, Qwen2-VL-7B; held-out: Qwen3.5-2B/4B, Qwen2-VL-2B).
Missing: Qwen3.5-9B (held-out), Qwen3-VL-8B/32B, Gemma-3-27B, InternVL3.5-8B, LLaVA-1.5-7B — runs were killed
three times without an error (session teardown / shared node), GPUs then full. Sanity: the grid reproduces the
earlier alpha=1 and alpha=16 runs on 99.7-100% of items.
| model | alpha* | illusions raw / calib / a16 / a* (test) | CountBench raw / calib / a* |
|---|---|---|---|
| Qwen3-VL-2B | 32 | 49.9 / 39.0 / 74.6 / 75.0 | 72.5 / 84.2 / 71.4 |
| Qwen3-VL-4B | 8 | 55.0 / 48.8 / 86.7 / 86.2 | 76.4 / 82.6 / 70.9 |
| Gemma-3-4B | 64 | 53.7 / 37.6 / 61.4 / 73.9 | 64.0 / 71.3 / 68.1 |
| Qwen2-VL-7B | 4 | 62.0 / 54.4 / 80.5 / 80.3 | 81.6 / 85.6 / 76.6 |
| Qwen3.5-2B (ho) | 16 | 58.1 / 43.6 / 76.2 / 74.8 | 75.2 / 82.0 / 67.8 |
| Qwen3.5-4B (ho) | 16 | 59.0 / 47.5 / 66.7 / 64.9 | 74.3 / 81.0 / 74.1 |
| Qwen2-VL-2B (ho) | 64 | 50.2 / 33.5 / 50.2 / 54.9 | 73.9 / 80.6 / 52.7 |
Interim: F1 1/3 held-out (cannot reach 3/4 only if 9B passes -> max 2/4: **F1 fails** regardless of the missing run).
F2 5/7, F3 7/7 (interim). Accuracy still rises at alpha=64 for Gemma-3-4B and Qwen2-VL-2B (grid ceiling).
Reading: selecting alpha per model helps where the needed strength is far from 16 (Gemma-3-4B +12.5), but does not
rescue Qwen3.5-4B or Qwen2-VL-2B; the strength bar fails on held-out models with either protocol.
