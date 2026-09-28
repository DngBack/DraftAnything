# Stage5: learned relation pilot

Exploratory after viewing stage4; five fixed seeds per architecture. Generic38353 parameters versus relation37601 (generic2% larger). Frozen MiniLM encoder and identical raw information; BCE correctness only, no binding labels or auxiliary binding loss.

| Dataset / extractor | Generic AURC | Relation AURC | Generic coverage/risk | Relation coverage/risk |
|---|---:|---:|---:|---:|
| Stage2 OCR2B test300 | 0.22829 | 0.22709 | 11.56%/5.77% | 21.28%/6.27% |
| Stage4 image2B clarified100 | 0.09058 | 0.08593 | 29.33%/1.70% | 41.33%/3.23% |
| Stage4 image9B clarified100 | 0.06966 | 0.05960 | 32.50%/2.56% | 52.83%/2.21% |

These thresholds are selected empirically at alpha5% on50 old calibration documents. Risk exceeds5% on the old OCR test; no risk-control guarantee. All seed AURCs and development stopping traces are retained in metrics/training-traces. Test300 has already been used, and stage4 was inspected before designing this head.

Equal-information semantic HGB is an additional post-confirmation diagnostic, fixed prior HGB config without target fitting or tuning. The architecture result must be read with [fresh confirmation](stage6-results.md), not as proof of a novel binding method.

[Protocol](stage5-protocol.md). Encoder: [official MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
