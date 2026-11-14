# VietCompose research prototype

Read [the research proposal](../../docs/vietcompose/RESEARCH_PROPOSAL.md) before interpreting any results. This is an experimental role-aware query adapter; it is not a validated paper. `renderer.py` separately implements an explicit ink compositor that has not been trained on handwriting.

## Implemented

- Unicode/NFD factorization of base/modifier/tone/case, including explicit Đ/đ handling.
- Shared modifier/tone embeddings, role interaction, writer-memory cross-attention and zero-initialized residual queries.
- Whole-glyph shared-backbone control.
- Frozen-OCR component-marginal CTC, paired image/feature supervision, component error diagnostics.
- Warm-start adapter-only training, checkpoint saving/resume, fixed 600-prompt/60-target evaluation.
- A separate leakage-aware holdout protocol for future fresh training.

## Completed warm-start pilot

Run from the VATr-pp repository root. Original runtime: `/home/giaphk/SpiS-GAN/.venv/bin/python`.

```bash
PYTHONPATH=. /home/giaphk/SpiS-GAN/.venv/bin/python -m research.vietcompose.run_pilot
```

This queue has already completed in `runs/vietcompose_pilot_v1`. Do not start another copy. It runs factorized, then glyph control, each for **1,000 new adapter updates** from source legacy checkpoint 175,500. Original G/D/OCR remain frozen; trainable adapter parameters: 580,864. No automatic promotion. Both arms were worse than the unchanged source CER; see the measured results report.

```bash
tail -f runs/vietcompose_pilot_v1/factorized/train.log
cat runs/vietcompose_pilot_v1/status.json
```

`comparison.json` updates when each trial finishes. Intermediate component diagnostics are in `components_<adapter_step>.json`; sample panels are under `expanded/` and `visual/`.

Loss: pixel + 0.2 fixed-feature + 0.05 character CTC + 0.05 component CTC. Coefficients are pilot defaults. Semantic “none” is nonblank in component CTC. OCR proxy is still the source supervisor, not an independent evaluator.

## Checkpoints

`adapter_last.pth` contains only new adapter parameters, optimizer/RNG and an immutable source-checkpoint hash/path. It is **not** compatible with the native `generate_vietnamese.py` loader. Recreate the original generator from `source`, freeze its weights, call `install_composition`, then load the adapter state with only the missing `glyph.*` keys permitted. The source file must still exist and match its recorded hash.

To resume training in the same output:

```bash
PYTHONPATH=. /home/giaphk/SpiS-GAN/.venv/bin/python -m research.vietcompose.train_pilot \
  --mode factorized --output runs/vietcompose_pilot_v1/factorized \
  --resume runs/vietcompose_pilot_v1/factorized/adapter_last.pth --steps 1000
```

`--steps` is the total adapter-update target for that trial. DataLoader sampling restarts on resume, so bitwise equivalence to an uninterrupted run is not promised. Pilot version 1 intentionally leaves native model generation unchanged unless an adapter is explicitly installed.

## Future clean protocol

```bash
PYTHONPATH=. /home/giaphk/SpiS-GAN/.venv/bin/python -m research.vietcompose.prepare_protocol
```

This creates split indices and coverage checks. **It does not train or enforce a clean zero-shot experiment.** Existing Vietnamese G/OCR checkpoints have held-out-character exposure. A fresh controlled training pipeline must apply the split to every data, corpus, OCR and style-support path before making a zero-shot claim.

## Tests

```bash
/home/giaphk/SpiS-GAN/.venv/bin/python -m unittest discover -s tests -v
```

The prototype is opt-in. Existing checkpoints, run outputs and legacy trainer defaults were not replaced.

## V2 renderer engineering smoke

`LayeredInkComposer` separates body/modifier/tone ink and differentiable affine placement. It is a separate architecture, not the adapter trained above.

```bash
PYTHONPATH=. /home/giaphk/SpiS-GAN/.venv/bin/python -m research.vietcompose.synthetic_smoke --steps 300
```

This run uses one canonical font, not handwriting or writer styles. It is only an engineering check; it cannot establish superiority over VATr. Outputs are in `runs/vietcompose_renderer_synthetic_smoke_v1`. Full tests now total 18 passing tests.

[Measured pilot results](../../docs/vietcompose/261114_report_02_Ket_qua_pilot_VietCompose.md).
