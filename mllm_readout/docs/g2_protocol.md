# G2 protocol: does the per-error diagnostic generalise? (pre-registered 2026-10-14, before any G2 output)

Follows G1.5 (docs/g15_report.md, verdict "mixed" -> paper = 4-way per-error diagnostic + gated repair).

## New cells (same pilot.py pipeline: probes, calibration control, patching where pairs exist, repair)
- attr (synthetic): 6-10 small shapes; exactly one of the queried kind; "What color is the {kind}?"; 8 colour classes.
  Partner = same layout, target recoloured.
- spatial_real: What'sUp Controlled_Images A+B (820 real photos; ServiceNow/whatsup_all); first caption is gold.
  Classes {on, under, left, right, front, behind}; question lists the 4 options of that image's subset. No partners (no patching).
- Models: qwen3vl4b, qwen3vl8b, internvl35_8b, gemma3_4b, gemma3_27b (27B only if a GPU has >= 70 GB free).

## Predictions (from G1.5, stated before running)
- P1: spatial_real error mix differs from synthetic spatial: perc share higher (real images), loss share lower.
- P2: attr behaves like count (bias-dominant) rather than spatial (loss-dominant) for Qwen/InternVL.

## Decision (gated repair, all cells with >= 30 errors)
Gated repair (label-free) = if the CV probe agrees with the calibrated answer, output it; otherwise steer toward the probe class (G1 repair, primary config).
Compare net accuracy of: model, calib-only, steer-all (G1 repair), gated.
- Go (paper headline holds): gated >= max(calib-only, steer-all) in >= 2/3 of cells AND gated break-rate <= 2%.
- Else: report the diagnostic alone (analysis paper); repair moves to the appendix.

## Addendum (2026-10-14, before any G2 output was inspected)
- color_cf: Visual CounterFact colour split (493 real photos, object recoloured against its prior; 2604.09364's benchmark).
  Gold = visible colour; 8 classes. Same models. Prediction P3: calibration explains a large part of the "prior wins" errors
  that 2604.09364 attributes to arbitration failure (bias share >= 30% in >= 3/5 models). Included in the gated-repair decision.

## Deviations log
- 2026-10-14, after qwen3vl4b_attr: 93% of its errors were brown->red and magenta->purple (PIL named colours look like the
  other class), i.e. a colour-naming artefact that calibration trivially remaps. attr now uses prototypical RGBs
  (stimuli.ATTR_RGB: red, blue, green, yellow, orange, purple, black, pink); the old run is in results/old/ and is re-run.
