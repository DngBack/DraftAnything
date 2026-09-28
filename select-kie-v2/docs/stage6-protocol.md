# Stage6: frozen-head fresh-document confirmation

Registered after stage5 pilot results, before new predictions and gold evaluation. Select100 previously unused Short-Form files by SHA256 filename, excluding stage2 all600, stage3 all100 and stage4 all100. Selection uses filename only, no filtering on gold/quality/correctness.189 documents available before selection. Report exact-text duplication checks and any failures. Still same dataset and model family, not person-independent.

Frozen Qwen3.5-9B, clarified schema from stage3 unchanged, all pages1280px, greedy, thinking disabled,320 output token cap. Inference two disjoint hash-parity shards, batch2; do not omit multi-page documents.

Frozen generic and relation heads (all five seeds), scalar scaler, MiniLM snapshot and feature-builder from stage5. No refit, early stopping, seed selection, new calibration or threshold adjustment. Mean the five correctness probabilities. Keep the exact alpha5% empirical thresholds saved in stage5 metrics.

Primary: paired document bootstrap2000 for relation-minus-generic official AURC on all600 decisions. Secondary: coverage, risk, normalized matcher, Brier, risk among high-confidence extractor outputs, and per-field. Report frozen HGB references separately; they have less information. High confidence is mean value-token prob>0.99. No distribution-free certificate from an empirical threshold or small error count.

All confidence features use public annotation OCR plus model predictions, not gold annotation spans. This is a hybrid image-extractor/OCR-confidence pipeline. Dataset gold noise remains; primary official unchanged and secondary normalized predefined. Human200-decision audit still pending. No claim of binding-specific superiority until independent subtype labels.

Commitment: publish all primary results even if confidence gain disappears or risk exceeds5%. No additional architecture tuning within this stage.
