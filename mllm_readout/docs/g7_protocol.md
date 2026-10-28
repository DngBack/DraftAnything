# G7 protocol: geometric variable vs. equality decision

Written 2026-10-19, before any G7 computation. This protocol answers an external review of ver33. The objection is
that failing to transfer an equal/unequal readout across nuisance regimes does not show that no component forms the
judgement. The representation may contain the geometric variable while a binary decision boundary is not invariant.

Already seen:
- All G5/G6 results.
- In particular, the "which line is longer" readouts across regimes (illusynC: SigLIP2 86%, own LLM input 66%, decision state 69%).

Not yet computed:
- Any regression of the length ratio.
- Any fixed-threshold rule.
- Any text-only run.
- Any psychometric fit.
- The new stimulus set illusynP, including regime C.
- Any intervention along a geometric direction.

Model set: the eight G5 models. "In ≥2/3" means ≥ 6 of 8, or 2/3 of those evaluated.

## G — geometric readout with a fixed decision rule (confirmatory)

Target variable: r = log(L_ref / L_other).
- Müller-Lyer: L_ref is the top shaft.
- Ponzo: L_ref is the upper bar.
- Vertical–horizontal: L_ref is the horizontal line.
- r = 0 for equal items.

The lengths are recovered exactly from the seeded generator. illusyn.py records them without changing the random stream, which is checked by byte-equality of the images against the committed generator.

**Readout.** Ridge regression of r on standardised features. Alpha is chosen by 5-fold CV inside the source regime from {1, 10, 100, 1000, 10000}. Training uses all source items, equal and unequal.

**Rule.**
- Predict "equal" iff |r̂| ≤ τ.
- τ is chosen on the source regime only: it maximises equal/unequal accuracy of 5-fold cross-validated predictions on the source.
- The rule is applied unchanged to the target regime.
- Accuracy is the mean of A→B and B→A.

**Comparator.** The direct equal/unequal linear classifier across regimes, as in G5/G6 (fit_probe).

Primary set: illusynC (fresh, 8 models with G3 states). Secondary sets: illusynL and illusyn (reported, not tested).

| Test | Criterion |
|---|---|
| G1 | own LLM input (l0): rule accuracy across ≥ binary classifier across + 0.05, in ≥2/3 models |
| G2 | decision state (75%): rule accuracy across ≥ binary across + 0.05, in ≥2/3 models |
| G3 | own LLM input: rule accuracy across ≥ 0.60, in ≥2/3 models |
| G4 | SigLIP2 (model-free): rule accuracy across ≥ binary across + 0.05 |

**Interpretation, fixed in advance.**
- If G1–G3 hold: the geometric variable is represented and transfers, and the model's own policy (S6: model ≤ input − 10) does not apply an invariant comparison rule to it. The central claim becomes "the variable is represented; the decision rule is not".
- If they fail: the absence claim is strengthened. Even a structured readout with a source-fixed rule does not yield a transferable equality decision.

Also reported:
- Correlation of r̂ with r on the target.
- Signed accuracy on unequal target items.
- An oracle τ chosen on the target (labelled as using target labels).

## T — the comparison rule in text (confirmatory)

For every item of illusynP (below), the model receives the two lengths as integers in pixels, with no image, and the same question. Example: "In an image, the top horizontal line is 182 pixels long and the bottom horizontal line is 175 pixels long. Are the two horizontal lines equal in length? Answer in curly brackets, e.g., {Yes} or {No}."

The decision is forced: the prompt is followed by the assistant turn opening "{", and the higher-scoring of the Yes/No first tokens is taken. This is the same decision step as G3.

| Test | Criterion |
|---|---|
| T1 | text accuracy ≥ 0.90 on equal items and on items with \|Δ\| ≥ 0.10, in ≥2/3 models |
| T2 | text accuracy ≥ image accuracy (same items, G3 decision step) + 0.25, in ≥2/3 models |

If T1 and T2 hold, the comparison rule itself is available to the LLM. The failure lies in obtaining or using the measurement from the image.

## Y — psychometric curves and an unseen regime C (confirmatory)

**illusynP.** illusyn.items with n_per=140, diffs (0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 0.80), seed 2029, tag "P_", regimes A, B and C.
- Per family and regime: 140 equal items and 20 per difference level.
- 2520 images in total.
- Regime C is new and is rendered only for this set:
  - dashed lines (dash 10–14 px, gap 5–7 px), width 3–5 px, dark random hue;
  - a light diagonal-stripe background (period 16–24 px);
  - the whole figure scaled by 0.70–0.90 about the centre.

**Psychometric model.** Logistic regression of the model's "unequal" (No) answer at the G3 decision step on |Δ|, with and without regime indicators. Likelihood-ratio tests use scipy chi2 and unpenalised fits.

| Test | Criterion |
|---|---|
| Y1 | slope on \|Δ\| > 0 with LR p < 0.01, in ≥2/3 models (the model has a psychometric function at all) |
| Y2 | regime indicators improve the fit with LR p < 0.01, in ≥2/3 models (the nuisance shifts the model's criterion) |
| Y3 | unseen regime C, own LLM input: the geometric rule (ridge and τ fitted on A ∪ B as above) on C ≥ the binary classifier trained on A ∪ B tested on C + 0.05, in ≥2/3 models |

Reported:
- Curves per model and regime, for the model and for the readouts.
- The |Δ| at which the model reaches 75% "unequal", if any.
- The false-alarm rate at Δ = 0.

## I — intervention along the geometric direction (exploratory, 4 models)

Models: qwen3vl4b, qwen3vl8b, internvl35_8b and llava15_7b, on illusynC.
- Image-position states at the input of the layer at 25% depth are captured for every item (mean over image positions).
- Two directions are computed on regimes A ∪ B: d+ = mean(r ≥ 0.25) − mean(equal) and d− = mean(r ≤ −0.25) − mean(equal).
- On 240 equal items (balanced over family and regime), the model generates to the decision step with α·d added to every image position at that layer's input (prompt pass).
- Conditions: α ∈ {0.5, 1, 2} for d+ and d−, plus a random direction with the norm of d+ (α = 2).
- Read-out: the share of "No" answers.

Signature of use: "No" increases under both d+ and d− relative to random, a U-shape in signed r. This part is exploratory and reported as such.

## Deviations

Added below with a timestamp before the affected result is read.
- 2026-10-19, before any G7 result:
  - Text prompt construction. The assistant turn is opened with " {" when the chat template ends in a non-space (LLaVA's "ASSISTANT:" would otherwise merge ":{" into one token).
  - Yes/No candidates are the tokens that decode to the word, as in diag_gen.cand_ids. pilot.Runner's first-token rule picks the bare "▁" token for Llama tokenisers.
  - Smoke check: one hand-made pair (182 vs 175) and one equal pair (180 vs 180) were scored on llava15_7b and qwen35_4b to check tokenisation. Seen: LLaVA answered No to both; Qwen3.5-4B answered No and Yes. No illusynP item was scored.
- 2026-10-19, before any G7-I run:
  - The directions d+ and d− are estimated on all decided illusynC items except the 240 test items.
  - An unsteered condition (α = 0) is added as a reference.
  - Analysis code for Y3 is g7_geom.py illusynP (source A ∪ B, target C).
- 2026-10-19 02:35 UTC, bug fix after the first G-run:
  - In the vertical–horizontal family, equal items carried |log(L_ref/L_other)| ≈ 1e-16 from float rounding. g7_geom.py therefore labelled 61–79% of them "unequal" (r ≠ 0), for both the rule and the binary comparator.
  - That run's G1–G4 output was seen: G1 2/8, G2 1/8, G3 2/8, G4 fail, all on corrupted labels.
  - Fix: illusyn.items sets logr = 0 exactly for equal items. The analysis was rerun unchanged, and the rerun is the registered result.
  - The g7_steer smoke test failed for the same reason (too few "equal" items to sample); no steering result was produced.

## Addendum A — spatial (grid) own-input readout (exploratory; written 2026-10-19 ~03:00 UTC, before computing it)

**Seen.**
- The registered G-run: G1–G4 fail. Own-input rule across 51–56%, while DINOv2 reaches 72% and SigLIP2 67%; both frozen encoders use the 4×4 grid plus mean features.
- An exploratory check: SigLIP2 and DINOv2 mean-pooled to one vector fall to rule 52% / 58%, with corr(r̂, r) 0.51 / 0.45.
- So the own-input readouts of G (mean over image positions) may lose the geometry by pooling, not because the model lacks it.

**Analysis.**
- Features: own LLM input as the 4×4 grid plus mean of image-token states entering layer 0 (explore_input.py --src <model>, layer index 0), 17 × d.
- Same ridge rule, τ and binary comparator as G, on illusynC (A↔B) and illusynP (A ∪ B → C).

**Fixed reading.**
- If the grid rule across is ≥ binary + 0.05 and ≥ 0.60 in ≥2/3 models, the absence claim is restricted to pooled readouts, and the paper says the geometric variable is available in the model's own input.
- Otherwise the G conclusion stands for spatial readouts too.
- Labelled exploratory either way, since the motivating comparison was seen.
