# Stage7: fair target-domain calibration and present-value confirmation

Registered2026-09-28 before new extraction/scoring or gold inspection. Use all89 remaining Short-Form files, excluding stage2 all600 and stages3/4/6 all100 each (900 prior). Select by filename hash, no outcome filtering. No fresh FARA Short-Forms remain afterward; stop reusing this source as fresh confirmation.

Qwen3.5-9B clarified schema and frozen learned heads unchanged. Generic MLP and relation5-seed ensembles; semantic HGB fixed stage5 config and an additional HGB using the same source200 training + targetdev50 early-stopping labels, maximum100 trees, patience15, tol1e-5, no test-based selection. HGB uses fewer parameters; information/supervision parity is tested, not exact tree/MLP capacity equivalence.

Calibration uses stage4 image9B-clarified100 documents, now explicitly development data already examined for method design. SHA-sort filename: first50 fit logistic score maps, last50 choose empirical alpha5% thresholds. No stage6 or stage7 labels used to fit/select maps or thresholds. Same map family/label budget for all heads: raw pooled, pooled Platt, field/null offsets, and raw per-field thresholds. Field/null logistic input: logit(raw score), six field indicators, predicted-null indicator and six field×predicted-null interactions. StandardScaler fit first50; LogisticRegression C1, max_iter1000, lbfgs, fixed seed. Predicted-null is inference data; gold presence must never enter scoring/calibration features. Save maps, thresholds, heads and SHA before new inference.

Co-primary challenge: official AND predefined-normalized PRESENT-only AURC, relation versus development-stopped semantic HGB with field/null calibration. Paired document bootstrap2000,97.5% intervals each (Bonferroni over two metrics). A positive architecture claim requires both intervals strictly negative; this still does not establish binding-specific gain. Report all heads/maps, all-field metrics, observed risk/error counts, present/absent split, per-field, latency and failures. No best map/head selection on test. Other contrasts descriptive.

Empirical alpha5% does not certify risk; calibration50 with clustered fields is small. HGB/MLP design choices informed by prior stages; these89 docs are fresh only for this fixed comparison. Annotation OCR, gold span conventions, repeated registrants and pending human audit remain limitations. No binding auxiliary loss or simulated human labels.

Separate InternVL architecture-transfer diagnostic uses previously evaluated stage6 documents, retaining Qwen-domain heads/maps without InternVL fitting. It is not a fresh cross-family confirmation. A new source is needed for an independent second-dataset result.

## AdBuy data reconnaissance: holdout invalidated

The official `DeepForm-unk_template-train_200-test_159-valid_100-SD_0.json` split and Google Research evaluator were recovered. During schema reconnaissance, annotations from the full corpus were traversed for counts and example values. Therefore this split's test cannot be treated as a blind holdout in this study. No model inference or test score should be reported as confirmation on this split. Use a genuinely unseen source/holdout and freeze its protocol before reading labels. The vendored evaluator and split file are retained for reproducibility only.
