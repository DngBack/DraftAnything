# Training debug — 2026-11-12

Resumed `runs/vatrpp_vn_finetune_100k/last.pth` at step 54000 toward total step 100000.
Process metadata: `runs/vatrpp_vn_finetune_100k/process.json`; its `log` field identifies the current invocation log.
Checkpoints save every 500 steps. The unsaved 54001–54200 steps from the stopped invocation are replayed.

## Findings

- Actual optimizer learning rates after resume: G encoder 5e-6, remaining G 2e-5, D 1e-5, W 2e-5. No resume learning-rate mismatch observed.
- Historical averages for steps 10000–15000 / 25000–30000 / 45000–50000:
  - OCR fake CTC: 0.4338 / 0.2660 / 0.2296.
  - Writer fake CE: 5.3473 / 4.6171 / 4.1578.
  - Writer real CE: 4.3455 / 3.6790 / 3.1347.
  Losses still decrease, but more slowly. Total generator loss combines an adversarial term with a dynamically balanced auxiliary term, and is not a monotonic quality metric.
- Fixed-panel CER improves from 50.74% initially to 11.03% at step 41500, then fluctuates; 13.24% at step 54000. Only 48 samples and a supervisor OCR, not independent evaluation.
- Frozen OCR is in eval mode with frozen parameters as intended. Its warmup validation CER is 16.19% on 11786 real samples. It is an imperfect supervisor, particularly for diacritics; the size of its effect on generated-image errors remains unproven.
- The first resumed step has G OCR gradient norm 54.27 and G writer norm 14.85 before clipping at 5. This confirms clipping at that step, not chronic saturation. New logs record pre-clip norms and effective auxiliary weights to investigate frequency.
- Legacy style feature layout is active. `compute_style` directly views [B*N,512,...] as [B,512,...] without first permuting the support axis. This mixes the channel/support interpretation and is a plausible style-conditioning limitation. The corrected layout already exists behind `--adapt-generator-layout`; enabling it also changes query positional encoding. Do not silently switch a pretrained/resumed run: test separately with matched evaluation.
- No loss NaN/Inf was found in historical logged scalar values. Four existing Vietnamese unit tests pass after diagnostic instrumentation.

## Changes for this invocation

Only scalar diagnostics were added to the trainer: actual optimizer learning rates, separate generator adversarial terms, effective auxiliary weights, and gradient norms before clipping. Optimizer, sampling, architecture and loss formulas are unchanged.

## Next diagnostic decisions

Review repeated pre-clip norms and fixed-panel CER across checkpoints. If clipping is persistent, compare a controlled learning-rate/loss-balancing ablation; do not raise clipping thresholds on the strength of one batch. Use a larger held-out set and an independent OCR plus writer/style metrics before declaring convergence or changing architecture. This run remains targeted at 100000 total steps.
