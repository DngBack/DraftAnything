"""One extra snapshot where the model is unsure (docs/adaptive_snapshot_protocol.md).

  python experiments/adaptive_snapshot.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.family_set import RESULTS  # noqa: E402
from experiments.lineage_glue import Movie, image_shape_of  # noqa: E402
from experiments.lineage_raft import load, predict  # noqa: E402

SEEDS = ("raft_v3", "raft_v3_s1", "raft_v3_s2")
OUT = RESULTS / "adaptive_snapshot.parquet"


def ens(models, fa, fb, h, shape):
    return np.mean([predict(m, fa, fb, h, shape, a.get("no_shape", False), 1, init=a.get("init", "radial"))[0]
                    for m, a in models], 0)


def run():
    models = [load(t) for t in SEEDS]
    ref = pd.read_parquet(RESULTS / "raft_eval_raft_v3.parquet").query("method == 'raft_v3_argmax' and H in (66, 132)")
    rows = []
    for alias in ("M3", "M4"):
        mv, shape = Movie(alias), image_shape_of(alias)
        for wid, h in ref[ref.sequence == alias][["window_id", "H"]].drop_duplicates().itertuples(index=False):
            a, b = int(wid.split("_a")[1][:4]), int(wid.split("_b")[1][:4])
            m = (a + b) // 2
            if not all(t in mv.frames for t in (a, m, b)):
                continue
            fa, fb, label = mv.window(a, b)
            _, fm, _ = mv.window(a, m)
            ok = label >= 0
            if ok.sum() < 2 or len(fm["pos"]) < 2:
                continue
            pd_ = ens(models, fa, fb, b - a, shape)
            pmb = ens(models, fm, fb, b - m, shape)
            pam = ens(models, fa, fm, m - a, shape)
            pc = pmb[:, :-1] @ pam
            pc[:, -1] += pmb[:, -1]
            direct = pd_[:, :-1].argmax(1)
            chain = pc[:, :-1].argmax(1)
            rows.append({"sequence": alias, "H": h, "window_id": wid, "n_anchors": len(fa["pos"]),
                         "conf": float(pd_[ok, :-1].max(1).mean()),
                         "direct": float(np.mean(direct[ok] == label[ok])),
                         "chain": float(np.mean(chain[ok] == label[ok]))})
            print(alias, wid, rows[-1]["direct"], rows[-1]["chain"], flush=True)
    d = pd.DataFrame(rows)
    d.to_parquet(OUT, index=False)
    return d


def boot(x, n=4000, seed=0):
    b = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def summary(d=None):
    d = pd.read_parquet(OUT) if d is None else d
    for h, g in d.groupby("H"):
        for name, s in (("all", g), ("dense", g[g.n_anchors > 200])):
            m, lo, hi = boot((s.chain - s.direct).to_numpy())
            print(f"H={h} {name} n={len(s)}: direct {s.direct.mean():.3f} chain {s.chain.mean():.3f} "
                  f"gain {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        s = g.sort_values("conf")
        gain = (s.chain - s.direct).to_numpy()
        curve = []
        for f in (0, .1, .25, .5, .75, 1):
            k = int(round(f * len(s)))
            curve.append((f, s.direct.mean() + gain[:k].sum() / len(s), s.direct.mean() + f * gain.mean()))
        print("  f, accuracy (confidence policy), accuracy (random, expected):")
        for f, c, r in curve:
            print(f"  {f:.2f} {c:.3f} {r:.3f}")
        tot = gain.sum()
        half = gain[:int(round(.5 * len(s)))].sum()
        print(f"  share of full gain at f=0.5: {half / tot if tot else float('nan'):.2f}")


if __name__ == "__main__":
    summary() if sys.argv[1:] == ["summary"] else summary(run())
