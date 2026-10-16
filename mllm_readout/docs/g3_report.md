# G3 report: the decomposition at the generation decision step (protocol: docs/g3_protocol.md)

Run 2026-10-14, 9 models x 7 sets = 63 cells (diag_gen.py; results/g3/, results/g3_table.csv).
CV grouped by base image; text-only control; exclusion: <30 errors, >10% no decision, text-only >= 75%.
Excluded: all Car Logos and Dice cells (text-only 85-100%: the gold is a function of the question), LLaVA-1.5 on
CountBench/MMVP/Tally and InternVL3.5/Qwen3-VL-32B on Tally (>10% no decision). 40 cells remain.

## Pre-registered predictions
1. G1.5 rule on the 40 cells: Rec >= 50% in 12/40 (30%), loss-dominant in 20/40 -> **mixed** (as G1.5: 14/22).
2. MMVP perception-dominated (perc > bias, loss): **8/8, holds**.
3. POPE bias < 30%: **8/9, holds** (Qwen2-VL 33%).

## Per set (medians over models)
| set | cells | Rec>=50% | model acc | probe@answer pos | bias | loss | loss beyond prompt | perc | text-only |
|---|---|---|---|---|---|---|---|---|---|
| CountBench | 8 | 6 | 64-82 | 78-92 | 40 | 29 | 28 | 39 | 9 |
| Illusions | 9 | 0 | 49-62 | 81-92 | 34 | 58 | 34 | 7 | 36 |
| MMVP | 8 | 0 | 64-83 | 60-82 | 10 | 28 | 10 | 66 | 50 |
| POPE (3000) | 9 | 3 | 83-90 | 87-91 | 23 | 32 | 6 | 47 | 68 |
| Tally | 6 | 3 | 20-40 | 39-69 | 44 | 28 | 23 | 26 | 11 |

## Post hoc (not pre-registered; g3_posthoc.py, results/g3_posthoc.csv)
- "loss beyond prompt" = loss errors on which the text-only control is wrong (a lower bound on image-attributable loss).
  POPE: the prompt alone (which object is asked about) predicts 68%; almost all apparent POPE loss (32%) is this prior
  (6% remains). Illusions: the text-only control is anti-informative (36%, an artefact of grouped splits over balanced
  prompts), so the bound is very conservative; still 25-44%.
- Probe restricted to the answer position (last token) on illusions: 85-92% (best layer), loss 42-61%. The information
  reaches the position where the answer is produced; it is not only in the image tokens.
- G1.5/G2 real-image cells: exact-prompt text-only control gives color_cf 18% (majority 20%), What'sUp 0%, CountBench
  18% (majority 14%): no prompt leak. Text leakage could only raise loss there, i.e. the 16/24 is conservative.

## Reading
Three regimes, consistent with what generation-time calibration does:
- CountBench: bias (Rec >= 50% in 6/8) -> calibrated generation helps 8/8.
- MMVP: perception (perc 55-72%) -> calibration does nothing.
- Edited illusions: genuine readout loss — the answer is linearly readable at the answer position (85-92%) while models
  answer at chance, and no calibrator helps once the split is grouped by image. The one place where "seeing but not
  saying" holds as stated in the literature.
- POPE: small bias, mostly perception once the prompt prior is controlled.
Implication for the paper: "most encoded-but-unused failures are miscalibration" is not supported over all 64 cells
(28/64 with Rec >= 50%). The claim becomes: the diagnostic separates the regimes; miscalibration dominates
counting/colour; loss is real for edited illusions and spatial relations.

## Addendum 2 result: blank-image text probe (diag_blank.py, results/g3_blank.csv)
Medians over cells: text-only probe (same prompt, grey image) CountBench 20%, illusions 54%, MMVP 50%, POPE 65%, Tally 29%.
loss_txt (loss errors the text probe gets wrong): CountBench 25%, illusions 26.6%, MMVP 13%, POPE 8.4%, Tally 20%.
- POPE criterion (median <= 10%): **stands** (8.4%). Apparent POPE loss is the prompt prior.
- Illusions criterion (median >= 30%): **fails** (26.6%). Post hoc reason: on a binary task a chance-level text probe
  (54%) is right on about half of the items by luck, so the criterion discards ~half of the loss errors regardless.
  Cleaner comparison (also post hoc): probe with image 89%, text-only probe 54%, model 52%. Paper wording must not call
  the illusion errors "genuine loss" without this qualification.

## Addendum 3 result: probe-guided steering at the decision step (steer_g3.py, results/g3_steer.csv)
Pre-registered (alpha=1): illusions steered >= calib + 10 in **7/9 -> holds**, but driven by grouped calibration falling
below raw (median calib 41.8 vs raw 52.0); alpha=1 itself moves raw by +0 to +8 (median +1.5). Broken <= 1.7%.
Post hoc (alpha=16, chosen after seeing alpha=1; 2B dose check alpha=4 -> 56.7):
| set | cells | raw | calib | a=1 | a=16 | fixed (a16, % errors) | broken (a16, % correct) |
|---|---|---|---|---|---|---|---|
| Illusions | 9 | 52.0 | 41.8 | 53.5 | 79.2 | 70.0 | 12.4 |
| CountBench | 8 | 76.3 | 82.6 | 78.1 | 72.0 | 14.0 | 10.1 |
Double dissociation: steering repairs illusions (near probe_L 78-87) but hurts CountBench; calibration the opposite.
Caveat: strength is post hoc -> confirmatory run on held-out models with alpha fixed at 16 (Addendum 4) if possible.
Paper: tab:steer (main), tab:supp_steer; abstract/intro/§4.3/conclusion softened ("mostly loss", blank criterion failed). ver25.
