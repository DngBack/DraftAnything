"""Prospective OIAM-to-rpsM endpoint ancestry test; see frozen protocol.

Run from celltracking/: python experiments/rpsm_radial_external_test.py
No rpsM lineage ID is used by any inference method.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.loader import Windows  # noqa: E402
from experiments.family_set import BENCH, RESULTS, ROOT, examples  # noqa: E402
from experiments.fast_microcolony_external_test import family_f1  # noqa: E402
from experiments.fetch_rpsm import HELD_OUT  # noqa: E402
from experiments.multihorizon_center_context import fit_multi, multi_cost  # noqa: E402
from experiments.pairwise_geometry_baseline import (  # noqa: E402
    HORIZONS, pair_assignment, train_classifier)
from experiments.radial_sector_probe import geometry_assignments  # noqa: E402
from experiments.reca_microcolony_external_test import make_window  # noqa: E402
from experiments.reca_schnitzcells import load_colony  # noqa: E402

GAPS = (10, 20, 40)
MINUTES_PER_FRAME = 3


def main():
    roots = [ROOT / "data/raw/rpsm_holdout" / name for name in HELD_OUT]
    missing = [str(r) for r in roots if not (r / "data" / f"{r.name}_lin.mat").exists()]
    if missing:
        raise FileNotFoundError(f"rpsM holdout files absent: {missing}")
    windows = Windows(BENCH, ROOT / "data").windows
    train = [e for alias in ("M0", "M1", "M2") for h in HORIZONS
             for e in examples(alias, windows, compensate=False, horizon=h)]
    pair_model = train_classifier(train)
    center_model = fit_multi(train)
    rng = np.random.default_rng(20261002)
    rows, audits = [], []
    for root in roots:
        frames, mapping, tracks, invalid_parents, audit = load_colony(root)
        audits.append(audit)
        print("audit", audit, flush=True)
        for gap in GAPS:
            for start in sorted(frames):
                if start + gap not in frames or not len(frames[start]) or not len(frames[start+gap]):
                    continue
                e, valid, truth = make_window(start, gap, frames, mapping,
                                               tracks, invalid_parents, rng)
                if not len(truth):
                    continue
                e["window"]["horizon_frames"] = gap * MINUTES_PER_FRAME
                assignments = geometry_assignments(e)
                assignments["pairwise_geometry"] = pair_assignment(e, pair_model)
                assignments["family_center"] = multi_cost(e, center_model).argmin(axis=1)
                for method, prediction in assignments.items():
                    guess = prediction[valid]
                    rows.append({"colony": root.name,
                                 "window_id": f"{start}_{start+gap}",
                                 "source_frame": start, "target_frame": start + gap,
                                 "gap_frames": gap,
                                 "gap_minutes": gap * MINUTES_PER_FRAME,
                                 "n_anchors": len(e["anchor"]),
                                 "n_targets": len(e["target"]),
                                 "n_scored": len(truth),
                                 "n_excluded": len(e["target"]) - len(truth),
                                 "growth_ratio": len(e["target"]) / len(e["anchor"]),
                                 "method": method,
                                 "accuracy": float(np.mean(guess == truth)),
                                 "set_f1": family_f1(truth, guess)})
            print(root.name, gap, "complete", flush=True)
    out = pd.DataFrame(rows)
    path = RESULTS / "rpsm_radial_external_test.parquet"
    out.to_parquet(path, index=False)
    pd.DataFrame(audits).to_csv(RESULTS / "rpsm_radial_external_audit.csv", index=False)
    per_colony = out.groupby(["colony", "gap_frames", "method"])[
        ["accuracy", "set_f1"]].mean()
    print("equal-colony macro", per_colony.groupby(["gap_frames", "method"])
          .mean().round(4).to_string(), flush=True)
    print("per-colony", per_colony.round(4).to_string(), flush=True)
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
