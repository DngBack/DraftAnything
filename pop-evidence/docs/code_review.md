# Code review (2026-10-30)

Scope: `popev/` (train, models, metrics, bags, phoebi, experiment, pooling, features, cla) and the runners. The review reads the code against what the experiments claim.

## Issues found

| # | Where | Problem | Effect on results | Fix |
|---|---|---|---|---|
| 1 | `models.tile_evidence_loss`, default `use_pure=True` | The "pure sample" term (every tile of a single-label bag carries the class) is label-valid only for **pure cultures** (PHOEBI). In BBBC041 and PanNuke a single-label image still contains red blood cells, other nuclei and background, so the term forces evidence on tiles without the class. | Every BBBC041 and PanNuke run with `lambda_tile > 0` (dev and the locked tests) used a constraint stronger than the labels license. The localisation claims (C3) there must be re-checked with the valid form. | Corrected configs set `tile_pure: false`. Results: `results/dev/valid_tile_{bbbc041,pannuke}`. |
| 2 | Same function, `q = 0.9` | "Present ⇒ the 0.9-quantile tile carries the class" requires about 10 % of tiles to be positive. In BBBC041 a class occupies 0.2–3 % of tiles. The label-valid multiple-instance assumption is "at least one tile" (q = 1.0, max). | Same runs as #1. On PHOEBI, sparse species in mixtures are affected too. | Corrected configs set `tile_q: 1.0`. |
| 3 | `train.run`, fixed type with `mix` and a CLA covariate | Mixed bags are appended to H, but covariate rows are taken from the unmixed bags only, so the shapes differ. | None: no config used the combination. | Raises a clear error. |
| 4 | `train.run`, set models | Training tiles were indexed on the CPU and copied to the GPU every step. | Speed only. | Training tiles are kept on the GPU (fp16, about 1 GB for PHOEBI). `gather` works on either device. |

Tests: `tests/test_core.py::test_tile_loss_label_valid_variant` asserts that the corrected form is satisfied by a single strong tile, and that the old forms are not.

## Checked and found correct

- **`metrics`:** weighted bootstrap equals resampling by duplication for AUROC; AP differs by < 0.001; both are tested.
- **Paired deltas:** use identical resamples.
- **`decorrelation_weights`:** reproduces the product of marginals.
- **`bags`:** evaluation bags depend only on `bag_seed`, so every method scores the same bags. The sampler draws combinations uniformly, which matches the CLA prior estimated with one row per unit.
- **`phoebi.Manifest`:** folds assert no overlap with test combinations. The instance-dataset calibration split is a hash of the unit id. `store_paths` remapping is verified on partitions 1338/1339 and phoebi4.
- **`experiment`:** split index `G` and store rows `R` are saved per bag. Multi-partition units are prefixed `partition:`.
- **`run_test`:**
  - refuses configs that are not locked and configs changed after locking;
  - refuses a second read of an existing test directory;
  - logs every read.

## Outcome of the re-runs (dev)

- **PanNuke:** the label-valid form restores localisation as well as the old form, at −0.014 to −0.016 image AUROC.
- **BBBC041:** it does not. ABMIL stays at 0.35–0.38 for the anti-localised stages, and accuracy drops 0.03.
- The old form's pure term, read as "all tiles of a single-label image are noisy positives", is what restores localisation there.
- The locked BBBC041 and PanNuke tests ran the old form. Their numbers stand, but the method must be named for what it is (findings F15).
