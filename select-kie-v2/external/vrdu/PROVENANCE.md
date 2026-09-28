# Google Research VRDU evaluator provenance

These files are copied from the official Google Research repository:

- `evaluate.py`: https://github.com/google-research/google-research/blob/master/vrdu/evaluate.py
- `evaluate_utils.py`: https://github.com/google-research/google-research/blob/master/vrdu/evaluate_utils.py
- `match_utils.py`: https://github.com/google-research/google-research/blob/master/vrdu/match_utils.py
- `benchmark_utils.py`: https://github.com/google-research/google-research/blob/master/vrdu/benchmark_utils.py

The source files carry the Apache License 2.0 header. The upstream repository's license is at https://github.com/google-research/google-research/blob/master/LICENSE. The evaluator is preserved unchanged for benchmark compatibility; this project has not yet run a valid DeepForm test evaluation.

The VRDU dataset and split are maintained separately at https://github.com/google-research-datasets/vrdu. The local dataset source revision is recorded in `data/vrdu/revision.json`. The local official split is `data/vrdu/adbuy-DeepForm-unk_template-train_200-test_159-valid_100-SD_0.json`.

Integrity hashes for the vendored evaluator at import time:

| File | SHA-256 |
|---|---|
| `benchmark_utils.py` | `9a598cb28dc7c48a57fc0348dc3a3eafd1596a74f85c40f1420eae3f0f0fc7f8` |
| `evaluate.py` | `8b65d3ccd17facdbbb58d280ba093bba6527e97bf8a304eeee0204b37cbd8112` |
| `evaluate_utils.py` | `d972628071031e1918adb7d66da8c6c2a3349b20bd8a96f8674a79c306100bd4` |
| `match_utils.py` | `ee49bf28b18096196ee700870062b8fc308cd4ace6fc7164cdcaddb93f3b830b` |
