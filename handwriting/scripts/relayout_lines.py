"""Apply the post-hoc word re-layout fix (hwline/relayout.py) to a generator's val1 lines.

Real-data inputs (no val2/test):
  population stats and donor trajectories: real train lines (words_real32_train)
  style trajectories: the style reference line of each generated line (real val1, words_real32_val1)
Writes data/gen/fix/<src>_<kind>/<line_id>.png and a list CSV per kind.
"""
import argparse
import json
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from hwline.profile import add_features
from hwline.relayout import relayout, trajectory_stats

KINDS = ["ident", "jitter", "donor", "style"]


def line_tables(words_csv):
    f = add_features(pd.read_csv(words_csv, dtype={"writer_id": str}))
    out = {}
    for lid, g in f.groupby("line_id"):
        xc = g.x_c.to_numpy(float)
        out[lid] = (g.baseline.to_numpy(float), xc - np.nanmean(xc), g.gap.to_numpy(float)[:-1])
    return out


def population(train):
    sds, cvs = [], []
    for base, xc, gap in train.values():
        b = base[np.isfinite(base)]
        g = gap[np.isfinite(gap)]
        if len(b) >= 4:
            sds.append(b.std())
        if len(g) >= 3 and g.mean() > 0:
            cvs.append(g.std() / g.mean())
    return {"base_sd": float(np.median(sds)), "gap_cv": float(np.median(cvs))}


G = {}


def _work(rec):
    kind, out = G["kind"], G["out"]
    rng = np.random.default_rng(zlib.crc32(f"{rec['line_id']}/{kind}".encode()))
    kw = {"pop": G["pop"]}
    if kind == "style":
        kw["style"] = G["style"].get(rec["style_line"])
        if kw["style"] is None:
            return None
    if kind == "donor":
        def pick(n):
            pool = G["donors"].get(n)
            return pool[rng.integers(len(pool))] if pool else None
        kw["pick_donor"] = pick
    gray = np.asarray(Image.open(rec["image_path"]).convert("L"))
    im = relayout(gray, rec["transcript"], kind, rng, **kw)
    if im is None:
        return None
    p = out / f"{rec['line_id']}.png"
    Image.fromarray(im).save(p)
    return {**rec, "image_path": str(p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--kinds", nargs="+", default=KINDS)
    ap.add_argument("--root", type=Path, default=Path("data/gen/fix"))
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    train = line_tables("data/dyn/words_real32_train.csv.gz")
    val1 = line_tables("data/dyn/words_real32_val1.csv.gz")
    G["pop"] = population(train)
    donors = {}
    for t in train.values():
        if np.isfinite(t[0]).sum() >= 4:
            donors.setdefault(len(t[0]), []).append(t)
    G["donors"] = donors
    G["style"] = {lid: s for lid, t in val1.items() if (s := trajectory_stats(*t)) is not None}
    lst = pd.read_csv(a.list, dtype={"writer_id": str})
    lst = lst[[Path(p).exists() for p in lst.image_path]]
    for kind in a.kinds:
        G["kind"], G["out"] = kind, (a.root / f"{a.src}_{kind}").resolve()
        G["out"].mkdir(parents=True, exist_ok=True)
        with Pool(a.workers) as p:
            res = p.map(_work, lst.to_dict("records"), chunksize=8)
        rows = [r for r in res if r is not None]
        pd.DataFrame(rows).to_csv(a.root / f"list_{a.src}_{kind}.csv", index=False)
        print(json.dumps({"src": a.src, "kind": kind, "in": len(lst), "out": len(rows),
                          "pop": G["pop"]}), flush=True)


if __name__ == "__main__":
    main()
