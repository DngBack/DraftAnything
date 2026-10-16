"""Family-centre error of LineageRAFT checkpoints on OIAM M3/M4 (hypothesis C3, docs/coupled_refinement_protocol.md).

For every scored target: distance between the final centre of its true ancestor and the centroid of that ancestor's
true family in the target snapshot, in median anchor spacings (canonical units, as error_decomposition.py). Windows,
labels and features come from the genealogy_constrained cache, so every tag is scored on the same windows.

  python experiments/centre_error.py raft_v3 raft_v3_s1 raft_v3_s2 raft_cpl raft_cpl_s1 raft_cpl_s2
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.loader import Windows  # noqa: E402
from experiments.family_set import BENCH, RESULTS, ROOT, examples  # noqa: E402
from experiments.genealogy_constrained_decode import CACHE, HS, family_centres, featurized, raft_scores  # noqa: E402
from experiments.lineage_glue import image_shape_of  # noqa: E402
from experiments.lineage_raft import load  # noqa: E402

OUT = RESULTS / "centre_error.parquet"


def main(tags):
    with open(CACHE, "rb") as f:
        items = pickle.load(f)
    windows = Windows(BENCH, ROOT / "data").windows
    rows = []
    for tag in tags:
        model = load(tag)[0]
        for alias in ("M3", "M4"):
            shape = image_shape_of(alias)
            for h in HS:
                ex = {e["window"]["window_id"]: e for e in examples(alias, windows, compensate=False, horizon=h)}
                for w in (x for x in items if x["sequence"] == alias and x["H"] == h):
                    n_a, la, t = len(w["pa"]), w["lab_all"], w["truth"]
                    _, _, centre = raft_scores(model, *featurized(ex[w["window_id"]], h, shape))
                    true_c = family_centres(w["cb"][la >= 0], la[la >= 0], n_a, centre)
                    sp = float(np.median(cKDTree(w["ca"]).query(w["ca"], k=2)[0][:, 1]))
                    err = np.linalg.norm(centre[t] - true_c[t], axis=1) / sp
                    rows.append(pd.DataFrame({"tag": tag, "sequence": alias, "H": h, "window_id": w["window_id"],
                                              "n_anchors": n_a, "err": err.astype(np.float32)}))
        print(tag, flush=True)
    d = pd.concat(rows, ignore_index=True)
    if OUT.exists():
        d = pd.concat([pd.read_parquet(OUT).query("tag not in @tags"), d], ignore_index=True)
    d.to_parquet(OUT, index=False)
    d["dense"] = d.n_anchors > 200
    print("median target centre error (spacings)")
    print(d.groupby(["dense", "H", "tag"]).err.median().unstack("tag").round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1:])
