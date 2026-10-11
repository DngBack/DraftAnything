"""How many snapshots does lineage need? Accuracy of ancestry over H minutes when the
population is imaged every D minutes (H/D + 1 snapshots), composing per-interval links.

Linkers: capacity-2 Hungarian with the fission prior (chained_pseudo_lineage.link), and
LineageRAFT argmax (raft_v3, OIAM-trained) at gap D. OIAM M3/M4, GT tracks for scoring.

  python experiments/sparsity_curve.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.chained_pseudo_lineage import link  # noqa: E402
from experiments.family_set import RESULTS  # noqa: E402
from experiments.lineage_glue import Movie  # noqa: E402
from experiments.lineage_raft import load, predict  # noqa: E402

STRIDES = {66: (1, 2, 3, 6, 11, 22, 33, 66), 132: (1, 2, 4, 11, 22, 33, 44, 66, 132)}


def chain(m, s, H, D, linker):
    """Ancestor index in frame s for every cell of frame s+H, composing links at stride D."""
    ts = list(range(s, s + H + 1, D))
    idx = np.arange(len(m.frames[s + H]["pos"]))
    for a, b in reversed(list(zip(ts[:-1], ts[1:]))):
        lk = linker(m.frames[a], m.frames[b], b - a)
        idx = np.where(idx >= 0, lk[np.maximum(idx, 0)], -1)
    return idx


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["raft_v3"])
    ap.add_argument("--no-hungarian", action="store_true")
    ap.add_argument("--out", default="sparsity_curve")
    a = ap.parse_args()
    models = {t: load(t) for t in a.tags}
    rng = np.random.default_rng(0)
    rows = []
    for alias in ("M3", "M4"):
        m = Movie(alias)

        def raft_of(model, margs, m=m):
            return lambda fa, fb, d: predict(model, fa, fb, d, m.image_shape, margs.get("no_shape", False),
                                             init=margs.get("init", "radial"))[0][:, :-1].argmax(1)
        linkers = {} if a.no_hungarian else {"hungarian": None}
        linkers.update({t: raft_of(*mm) for t, mm in models.items()})

        def hung(fa, fb, d):
            return link(fa, fb, div_cost=10.0, max_px=1e5)

        for H in (66, 132):
            starts = [k for k in sorted(m.frames) if k + H in m.frames and len(m.frames[k]["pos"]) >= 2]
            for s in rng.choice(starts, 16, replace=False):
                s = int(s)
                _, _, label = m.window(s, s + H)
                ok = label >= 0
                for D in STRIDES[H]:
                    for name, fn in linkers.items():
                        fn = fn or hung
                        p = chain(m, s, H, D, fn)
                        rows.append({"movie": alias, "H": H, "start": s, "D": D, "snapshots": H // D + 1,
                                     "method": name, "n": int(ok.sum()), "n_anchors": len(m.frames[s]["pos"]),
                                     "accuracy": float(np.mean(p[ok] == label[ok]))})
            print(alias, H, flush=True)
    d = pd.DataFrame(rows)
    d.to_parquet(RESULTS / f"{a.out}.parquet", index=False)
    for H, g in d.groupby("H"):
        w = g.assign(c=g.accuracy * g.n).groupby(["method", "snapshots"])[["c", "n"]].sum()
        print("H", H, "\n", (w.c / w.n).unstack("snapshots").round(3).to_string())


if __name__ == "__main__":
    main()
