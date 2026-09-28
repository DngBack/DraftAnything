# Stage8 plan: independent KILE/LIR confirmation on DocILE

## Why this source

FARA stage7 did not pass the relation-vs-HGB gate. AdBuy labels were traversed during reconnaissance and are not a valid blind test for this project. DocILE is the next scientifically useful target because its official tasks cover both field localization (KILE) and binding fields into line items (LIR), with annotated real, synthetic and unlabeled documents and many layouts. Its official evaluation includes confidence-ranked predictions and line-item grouping, which tests the research hypothesis more directly than a string-only score.

## Access and eligibility

The official maintainers require a dataset access request and token before download. Obtain access at the [official DocILE site](https://docile.rossum.ai/) and follow the [dataset repository](https://github.com/rossumai/docile). Until access is granted, do not download copies from unofficial mirrors. This experiment is research, not a DocILE competition submission; re-check dataset terms and model-use restrictions before training/evaluation.

## Locked design after access

1. Preserve the official annotated-trainval/test split and source checksums. Do not inspect test labels or preview test PDFs during development. Use the official evaluation package unchanged.
2. Develop the extraction-to-localization adapter using train/validation only. Generate value text plus a grounded evidence span; align that span to precomputed OCR words and their boxes. Reject ambiguous/non-aligning spans with an explicit status instead of silently choosing a nearby value.
3. For KILE, produce field type, page, box and confidence. For LIR, additionally assign a row/line-item id. Keep the existing relation head, an equal-information generic MLP, and a semantic non-neural/GBDT baseline. Give each model the same candidate evidence, training labels, tuning budget and calibration labels.
4. Freeze all models, calibration, thresholds, postprocessing and output serialization before running on the official test split. Score with official AP and F1/precision/recall; report KILE and LIR separately, coverage-risk curves for abstention, and error counts by field/layout. Bootstrap by document/layout cluster rather than individual field.
5. If no head improves over matched baselines on held-out KILE and LIR, stop this architecture claim. If the gain appears only in FARA, frame it as domain-specific and do not present it as a general confidence method.

## Immediate blocker

No DocILE access token or authorized data files are present in this workspace. The official site says access is granted through its dataset request form. Data-dependent implementation must start after that grant; the current workspace has no legitimate path to the private download URLs.
