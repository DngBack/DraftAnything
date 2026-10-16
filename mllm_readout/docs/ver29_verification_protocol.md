Verification protocol — written 2026-10-16, before the new prototype results

Objective: determine whether the proposed residual/consistent readout improves
held-out correction, and identify simpler explanations if it does not. This is
an internal timestamped plan, not an externally registered preregistration.
Existing ver29 pilot data have already been inspected. Do not call these images
untouched test data. Preserve the original paper and result files.

Stage A: cached-state prototypes (CPU).
- Development models: Qwen2-VL-2B, Qwen3-VL-4B, Gemma-3-4B.
- Additional evaluation models: Qwen3.5-2B/4B/9B, InternVL3.5-8B,
  Qwen3-VL-8B/32B, Gemma-3-27B, Qwen2-VL-7B, LLaVA-1.5-7B, as available.
  They appeared in ver28; this is transfer of the new algorithm, not unseen models.
- All six views of a base image stay in the same outer fold. The classification
  label budget is the same unique training images for all methods.
- Primary setting: train on wording A, test wording B on different images, both
  directions. Same-wording predictions are matched controls.
- Baselines: raw, candidate-margin calibration, standardized full-state LR,
  nullspace LR where native rows are available, supervised both-wording LR.
- Prototypes: template mean subtraction estimated on training images only;
  projection of paired prompt-change directions (rank 1/4, fixed in advance);
  paired-invariance regularized linear readout (fixed lambda=1,10) if needed.
- Access distinction: mean/projection methods can see TARGET WORDING states on
  TRAINING images, without extra label annotation. This is prompt adaptation,
  not zero-shot transfer to an unseen prompt. Both-wording LR has the same unique
  image labels and is an essential baseline. No test-image centering in Stage A.
- Report fixed C=1 initially; if tuning is needed, use nested training-only CV.
  Do not select a method separately per model using test accuracy.

Stage B: actual VLM extraction (GPU).
- Start with Qwen3-VL-4B and Qwen2-VL-2B. Add Gemma-3-4B if feasible.
- Use one resolution per base image × wording (264 examples/model).
- Obtain exact real-image decision prefixes, record their token IDs. Measure
  real and grey-image post-normalization states at THE SAME prefix. Also use a
  fixed answer-marker prefix, shared across real/blank and across images.
- Save final states, candidate logits, prompts, image keys, raw prefix/text,
  sequence lengths, no-decision counts and numerical reconstruction checks.
- Contrast features h(real,prefix)-h(blank,prefix) are tested against raw states,
  blank states, and score contrast. Candidate-null features are an ablation.
- Fixed-prefix and generated-prefix experiments are different scopes and must
  remain distinct. No decision-token state may include the gold answer.

Stage C: intervention and honesty checks.
- Fit classifiers on outer training images only. Any gating/strength selection
  uses a disjoint inner validation subset; all estimators are then frozen.
- Run generated answers on held-out examples for promising candidates and the
  strongest simple baseline. Compare offline predictions with parsed text.
- Erase/restore selected evidence and run blank/shuffled-image controls where
  an improvement survives. Native and repaired causal effects are separate.
- Do not present a forced probe answer as evidence of native model mediation.
- Report accuracy, fixed errors, broken correct answers, image-cluster paired
  intervals and the label/forward-pass budget. Inspect family-holdout failures.

Success criteria for an encouraging prototype: >=10 percentage points above raw
and output-only calibration on the illusion repair task, AND demonstrable gain
over simple probe routing on shifted prompts (target >=5 points) without choosing
hyperparameters on that test. A method that only matches the both-wording probe
is useful prompt adaptation, not yet a novel breakthrough. Family transfer and
damage <=2% are separate stronger claims and require independent evidence.

If the sophisticated method loses to mean centering, report that result and
investigate nuisance prompt offsets. If all cross-family repairs fail, retain
the negative result and do not claim a universal visual repair. New adaptive
follow-ups must be timestamped and labelled exploratory before running them.

Exploratory addendum 2026-10-16, after Stage A development results:
Mean centering improves transfer but remains below both-wording LR; removing
4 paired-change directions helps more than rank 1. Test fixed ranks 16 and 64
without per-model test selection, save separately. This follows observed results
and is exploratory. A positive rank choice will require separate confirmation.

Exploratory addendum 2026-10-16, after rank-16/64 development outputs:
Higher-rank projection approaches but has not exceeded the both-wording probe.
Test label efficiency at 2/5/10/20 labels per class and full labels, ranks 64 and
PCA32 fixed. All variants receive the same unique labelled training images;
unlabelled outer-training prompt-pair states may estimate preprocessing. Add
both-wording PCA32 and both-wording projected readouts. Do not use target test
features for transformations. This analysis is exploratory and cannot establish
novelty from performance on the same development dataset.
