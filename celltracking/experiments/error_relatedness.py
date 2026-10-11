"""Where do long-gap errors go? For each wrong target, how related are the predicted and the
true ancestor at the first snapshot (time since their lineages split)?

If most errors at long gaps confuse ancestors that are themselves recent sisters, the
remaining error is largely an identifiability limit: two sister cells born just before
snapshot a sit side by side and their descendant families intermix.
OIAM M3/M4, GT tracks for scoring; raft_v3 argmax and nearest.

  python experiments/error_relatedness.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.family_set import RESULTS  # noqa: E402
from experiments.lineage_glue import Movie  # noqa: E402
from experiments.lineage_raft import load, predict  # noqa: E402

HS = (33, 66, 132)


def chain(m, track):
    """[(track, birth frame)] from the track up to the root."""
    out = []
    while track and track in m.start:
        out.append((track, m.start[track]))
        track = m.parent.get(track, 0)
    return out


def split_age(m, a, t1, t2):
    """Minutes before frame a at which the lineages of tracks t1 and t2 diverged (inf if unrelated)."""
    c1, c2 = chain(m, t1), chain(m, t2)
    s2 = {t for t, _ in c2}
    for i, (t, _) in enumerate(c1):
        if t in s2:
            # the children below the common ancestor were born when the common ancestor divided
            j = [u for u, _ in c2].index(t)
            births = [c[k - 1][1] for c, k in ((c1, i), (c2, j)) if k > 0]
            return a - min(births) if births else 0
    return np.inf


def main():
    model, margs = load("raft_v3")
    rng = np.random.default_rng(2)
    rows = []
    for alias in ("M3", "M4"):
        m = Movie(alias)
        for H in HS:
            starts = [k for k in sorted(m.frames) if k + H in m.frames and len(m.frames[k]["pos"]) >= 2]
            for s in rng.choice(starts, 24, replace=False):
                s = int(s)
                fa, fb, label = m.window(s, s + H)
                ok = label >= 0
                sp = float(np.median(cKDTree(fa["pos"]).query(fa["pos"], k=2)[0][:, 1]))
                preds = {"nearest": cKDTree(fa["pos"]).query(fb["pos"])[1],
                         "raft_v3": predict(model, fa, fb, H, m.image_shape, margs.get("no_shape", False),
                                            init=margs.get("init", "radial"))[0][:, :-1].argmax(1)}
                for name, p in preds.items():
                    for j in np.flatnonzero(ok):
                        t, q = label[j], p[j]
                        age = 0.0 if t == q else split_age(m, s, int(fa["track"][t]), int(fa["track"][q]))
                        rows.append({"movie": alias, "H": H, "start": s, "method": name, "n_anchors": len(fa["pos"]),
                                     "correct": t == q, "split_age": age,
                                     "dist": float(np.linalg.norm(fa["pos"][t] - fa["pos"][q])) / sp})
            print(alias, H, flush=True)
    d = pd.DataFrame(rows)
    d.to_parquet(RESULTS / "error_relatedness.parquet", index=False)
    summary(d)


def summary(d=None):
    d = pd.read_parquet(RESULTS / "error_relatedness.parquet") if d is None else d
    for (H, name), g in d.groupby(["H", "method"]):
        e = g[~g.correct]
        acc = g.correct.mean()
        line = f"H={H:3d} {name:8s} acc {acc:.3f} | errors whose ancestors split <=:"
        for x in (15, 33, 66, 132):
            line += f" {x}m {np.mean(e.split_age <= x):.2f}"
        line += f" | acc if confusions split <=33m forgiven {np.mean(g.correct | (g.split_age <= 33)):.3f}"
        line += f" <=66m {np.mean(g.correct | (g.split_age <= 66)):.3f}"
        line += f" | err dist median {e.dist.median():.2f} sp, <=1sp {np.mean(e.dist <= 1.2):.2f}"
        print(line)


if __name__ == "__main__":
    if sys.argv[1:] == ["summary"]:
        summary()
    else:
        main()
