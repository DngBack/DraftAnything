# Vietnamese orthographic long-tail experiments

**Current mainline:** the user selected continuation from native pretrained VATr++,
not the scratch controls below. See `docs/acl_longtail/STATUS.md` and
`scripts/run_pretrained_longtail_pilot.py`. Native training now has an opt-in
`--text-sampler component`; its default remains `legacy`.

Run commands from `hand_writting/VATr-pp`. This package implements a clean
metadata/exposure layer, scratch paired generation controls, compositional CTC,
and evaluation tooling. It does not turn the legacy 100k/200k checkpoints into
zero-shot models. See `docs/acl_longtail/` for experiment status and paper gaps.

## Environment

Install the appropriate PyTorch build for your machine separately. Core tools
use torch, numpy, Pillow and h5py; the separate legacy OCR evaluator additionally
needs the repository's `requirements-vietnamese.txt`. No global packages were
changed for this work: missing opencv/safetensors were placed in
`/tmp/vatrpp-longtail-test-deps` for legacy test imports. That temporary path is
not a permanent dependency installation or a reproducible environment lock.

## Real data preparation

```bash
python -m research.longtail.prepare preflight \
  --manifest files/vietnamese/manifest.json --output docs/acl_longtail/preflight.json
python -m research.longtail.prepare export \
  --manifest files/vietnamese/manifest.json \
  --output runs/acl_longtail_v1/records.jsonl
python -m research.longtail.prepare protocol \
  --records runs/acl_longtail_v1/records.jsonl \
  --output runs/acl_longtail_v1/zero --holdout ỄễỠỡẴẵ --budget 0
```

The exporter hashes every actual image, rejects duplicate pixels/IDs and writer
key overlap across splits, and preserves HAT as writer=null. Writer keys inherit
the source metadata: namespace separation of unknown test IDs is not evidence
that physical writers are disjoint. Hashing detects exact duplicates only.
Records refer to source images; pixel hashes are rechecked when images are read.
The reader caches at most four HDF5 handles per process, not a copy of the data.

All new training consumers use `ExposurePolicy`: image, OCR, corpus, support and
memory permissions. This does not retrofit the legacy trainer. Support is always
clean, same writer and different transcript; acquired target images cannot enter
support. Per-glyph image budgets 0/1/5/20 are nested, use unique images, exclude
HAT target images and report unavailable requested examples rather than inventing
them. Canonical target glyph access is explicitly allowed for the glyph control.

## Scratch controls and clean recognizer

```bash
python -m research.longtail.train_recognizer \
  --records runs/acl_longtail_v1/records.jsonl \
  --protocol runs/acl_longtail_v1/zero/protocol.json \
  --output runs/acl_longtail_v1/clean_recognizer \
  --mode factorized --steps 5000 --eval-every 500 --device cuda
```

The recognizer sums base/modifier/tone/case scores into legal complete-character
scores, appends a separate blank, normalizes, and applies ONE CTC. Thus semantic
none is nonblank and all roles use the same alignment. `--mode whole` provides a
same-backbone conventional character-head control. Neither model is initialized
from legacy OCR; target glyph images/transcripts are filtered from both. CTC
length is floor(width/4); images too narrow for repeated labels are explicitly
expanded, and repairs are logged.

```bash
python -m research.longtail.train \
  --records runs/acl_longtail_v1/records.jsonl \
  --protocol runs/acl_longtail_v1/zero/protocol.json \
  --output runs/acl_longtail_v1/example_new_arm \
  --mode factorized --sampler interaction --hidden 128 --batch-size 32 \
  --reconstruction balanced-bce --foreground-weight 4 \
  --ocr-checkpoint runs/acl_longtail_v1/clean_recognizer/best_seen_cer.pth \
  --ctc-weight 0.2 --steps 1000 --eval-every 250 --device cuda
```

Use `--mode glyph` for the canonical full-glyph control. These small paired
models share a support encoder but differ in decoder architecture and parameter
count: they are NOT native VATr reproductions or a parameter-matched superiority
test. The layered model uses the existing explicit renderer. Component-local
writer memory, annotated geometry supervision, and episodic component masking
remain unimplemented; a global support vector must not be called a mark memory.

The original default L1 reconstruction remains available as a diagnostic; a
100-update real-data profile collapsed toward white backgrounds. Balanced BCE
and clean CTC guidance are explicitly new configurations, not a successful result
inferred from lower L1. Both representations must receive the same losses.

Development defaults to two fixed targets per writer plus ALL eligible holdout
words. Supports and control targets are fixed across optimization seeds. This is
a targeted panel, NOT full-development CER. Checkpoints are saved with optimizer,
exposure fingerprints and RNG; exact iterator resume is not implemented. Use a
fresh output for every run. `best_pixel_proxy.pth` is only a proxy, not a promoted
model. Recognition checkpoint selection uses seen-combination dev CER; unseen
combination scores are diagnostic. Test data are not selected or evaluated here.

## Evaluation

`evaluate.py` joins external transcription predictions to references by sample
ID, rejects missing/extra predictions and computes CER, word exact, macro glyph
F1, role errors and writer-cluster percentile intervals. Unsupported glyphs are
reported; a single writer is not a population CI. F1 penalizes insertions as well
as deletions. The conditional tail CI must be inspected for valid bootstrap draws.

```bash
python -m research.longtail.transcribe \
  --references runs/acl_longtail_v1/example_new_arm/dev_001000/references.jsonl \
  --checkpoint runs/vatrpp_vn_ocr_train/best.pth \
  --output runs/acl_longtail_v1/example_new_arm/ocr_eval_1000 --device cuda
```

That legacy OCR is separate from new G training and may know target glyphs. It
was selected using legacy development, so it remains a screening metric, not a
blind human quality/style evaluation. Do not add its gradients to a clean run.
For human annotations use `evaluate --evaluator-id ...` with a separate
`predictions.jsonl` (sample_id, prediction).

`suite.py` plans 72 scratch reconstruction controls across 4 budgets, 3 seeds,
2 representations and 3 samplers. Its execution gate requires all requested real
evaluation glyphs and budgets. It intentionally refuses unavailable uppercase
paired-image comparisons. Single-arm training remains usable for the lower-case
clean transfer diagnostic; absent uppercase paired targets stay explicit. The
suite defaults are reconstruction-only: OCR/BCE pilots are separate configurations.

`collection.py` emits 144 vowel glyph prompts (12 bases × 6 tones × 2 cases) plus
NGUYỄN and a blank annotation sheet. Templates are not collected handwriting.
Assign splits by writer before collection and retain source resolution. Legal
individual letters do not imply arbitrary tone-swapped words are valid syllables.

## Tests

```bash
python -m unittest discover -s tests -p test_longtail.py -v
# Full legacy suite requires the additional Vietnamese dependencies.
python -m unittest discover -s tests -v
```

Core tests use synthetic images/HDF5 and real optimizer steps on both model
branches. These are engineering validation, not evidence of handwriting quality.
