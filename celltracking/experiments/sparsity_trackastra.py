"""Trackastra as the dense reference of the sparsity curve (docs/sparsity_trackastra_protocol.md).

Same windows, strides, composition (sparsity_curve.chain) and metric as sparsity_curve.py.
Trackastra general_2d (pretrained) is run once per window and stride on the n subsampled frames
(images + GT masks, I/O of trackastra_gap_baseline.py); its links between consecutive subsampled
frames (greedy linker / per-target argmax, nearest fallback) are composed to frame s.
Optionally the fine-tuned per-gap checkpoints (trackastra_finetune.py) where D equals their gap.

  CUDA_VISIBLE_DEVICES=1 TQDM_DISABLE=1 python experiments/sparsity_trackastra.py [--ft]
"""

import argparse
import logging
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.family_set import RESULTS, SEQUENCE  # noqa: E402
from experiments.lineage_glue import INDEX, Movie  # noqa: E402
from experiments.sparsity_curve import STRIDES, chain  # noqa: E402
from experiments.trackastra_finetune import CKPT  # noqa: E402

QUIET = lambda x, **k: x  # noqa: E731


def draw_starts(m, H, rng):
    """= sparsity_curve.main window draw."""
    starts = [k for k in sorted(m.frames) if k + H in m.frames and len(m.frames[k]["pos"]) >= 2]
    return [int(s) for s in rng.choice(starts, 16, replace=False)]


def run_links(model, imgs, masks):
    """{decoding: {(t1, target label): parent label at t1-1}} for one sequence."""
    pred = model._predict(imgs, masks, progbar_class=QUIET)
    node = {n["id"]: (int(n["time"]), int(n["label"])) for n in pred["nodes"]}
    best = {}
    for (u, v), wt in pred["weights"]:
        (t0, l0), (t1, l1) = sorted([node[int(u)], node[int(v)]])
        if t1 == t0 + 1 and wt > best.get((t1, l1), (-1, 0))[0]:
            best[(t1, l1)] = (float(wt), l0)
    graph = model._track_from_predictions(pred, mode="greedy")  # = model.track(..., mode="greedy")
    greedy = {}
    for u, v in graph.edges:
        du, dv = graph.nodes[u], graph.nodes[v]
        if int(dv["time"]) == int(du["time"]) + 1:
            greedy[(int(dv["time"]), int(dv["label"]))] = int(du["label"])
    return {"greedy": greedy, "argmax": {k: v[1] for k, v in best.items()}}


def make_linker(m, ts, link, stats):
    """linker(fa, fb, d) for sparsity_curve.chain: index in fa for every cell of fb."""
    pos_of = {id(m.frames[t]): k for k, t in enumerate(ts)}

    def linker(fa, fb, d):
        i = pos_of[id(fb)]
        assert pos_of[id(fa)] == i - 1
        col = {int(t): k for k, t in enumerate(fa["track"])}
        near = cKDTree(fa["pos"]).query(fb["pos"])[1]
        out = np.empty(len(fb["pos"]), dtype=np.int64)
        for k, lb in enumerate(fb["track"].astype(int)):
            p = link.get((i, int(lb)))
            out[k] = col[p] if p is not None and p in col else near[k]
            stats[0] += p is None
            stats[1] += 1
        return out
    return linker


def main():
    from trackastra.model import Trackastra
    logging.getLogger("trackastra").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--movies", nargs="+", default=["M2", "M3", "M4"])
    ap.add_argument("--ft", action="store_true", help="also fine-tuned per-gap checkpoints at D = their gap")
    ap.add_argument("--limit", type=int, default=None, help="windows per movie and H (smoke test)")
    ap.add_argument("--out", default="sparsity_curve_trackastra")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = {"trackastra": Trackastra.from_pretrained("general_2d", device=dev)}
    ft = {}
    if a.ft:
        for g in (33, 66, 132):
            mdl = Trackastra.from_pretrained("general_2d", device=dev)
            ck = torch.load(CKPT / f"h{g}" / "best.pt", map_location=dev)
            mdl.transformer.load_state_dict(ck["model"])
            mdl.transformer.eval()
            ft[g] = mdl
            print(f"ft h{g}: step {ck['step']} M2 val {ck['val_acc']:.4f}", flush=True)
    ref = pd.read_parquet(RESULTS / "sparsity_curve.parquet")
    rng, fig_starts = np.random.default_rng(0), {}
    for alias in ("M3", "M4"):  # re-draw the Fig. 3 windows and check them against the figure's parquet
        mv = Movie(alias)
        for H in (66, 132):
            fig_starts[alias, H] = draw_starts(mv, H, rng)
            fig = ref[(ref.movie == alias) & (ref.H == H)].start.unique().tolist()
            assert fig_starts[alias, H] == [int(x) for x in fig], (alias, H)
    del mv
    m2rng = np.random.default_rng(0)
    stats = {}
    for alias in a.movies:
        m = Movie(alias)
        fr = pd.read_parquet(INDEX / SEQUENCE[alias] / "frames.parquet").set_index("frame_index")
        img = lru_cache(maxsize=None)(lambda t: tifffile.imread(fr.image_path[t]))
        msk = lru_cache(maxsize=None)(lambda t: tifffile.imread(fr.mask_path[t]))
        rows = []
        for H in (66, 132):
            if alias == "M2":  # decoding selection: every frame only
                starts, strides = draw_starts(m, H, m2rng), (1,)
            else:
                starts, strides = fig_starts[alias, H], STRIDES[H]
            for s in starts[:a.limit]:
                t0 = time.time()
                _, _, label = m.window(s, s + H)
                ok = label >= 0
                for D in strides:
                    ts = list(range(s, s + H + 1, D))
                    imgs = np.stack([img(t) for t in ts])
                    masks = np.stack([msk(t) for t in ts])
                    runs = {"trackastra": models["trackastra"]}
                    if D in ft and alias != "M2":
                        runs["trackastra_ft"] = ft[D]
                    for mname, mdl in runs.items():
                        links = run_links(mdl, imgs, masks)
                        for dec, link in links.items():
                            name = f"{mname}_{dec}"
                            st = stats.setdefault((alias, H, D, name), [0, 0])
                            p = chain(m, s, H, D, make_linker(m, ts, link, st))
                            rows.append({"movie": alias, "H": H, "start": s, "D": D, "snapshots": H // D + 1,
                                         "method": name, "n": int(ok.sum()), "n_anchors": len(m.frames[s]["pos"]),
                                         "accuracy": float(np.mean(p[ok] == label[ok]))})
                print(alias, H, s, f"{time.time() - t0:.0f}s",
                      {r["method"]: round(r["accuracy"], 3) for r in rows if (r["H"], r["start"], r["D"]) == (H, s, 1)},
                      flush=True)
        d = pd.DataFrame(rows)
        out = RESULTS / (f"{a.out}_m2.parquet" if alias == "M2" else f"{a.out}_{alias}.parquet")
        d.to_parquet(out, index=False)
        print("wrote", out, flush=True)
        if alias == "M2":
            w = d.assign(c=d.accuracy * d.n)
            print("M2 every-frame, per H:\n", (w.groupby(["H", "method"]).c.sum() / w.groupby(["H", "method"]).n.sum())
                  .unstack("method").round(3).to_string())
            print("M2 pooled:\n", (w.groupby("method").c.sum() / w.groupby("method").n.sum()).round(4).to_string(),
                  flush=True)
    test = [RESULTS / f"{a.out}_{x}.parquet" for x in a.movies if x != "M2"]
    if test:
        d = pd.concat([pd.read_parquet(f) for f in test], ignore_index=True)
        d.to_parquet(RESULTS / f"{a.out}.parquet", index=False)
        for H, g in d.groupby("H"):
            w = g.assign(c=g.accuracy * g.n).groupby(["method", "snapshots"])[["c", "n"]].sum()
            print("H", H, "\n", (w.c / w.n).unstack("snapshots").round(3).to_string())
    fb = pd.DataFrame([{"movie": k[0], "H": k[1], "D": k[2], "method": k[3], "fallback": v[0] / max(v[1], 1)}
                       for k, v in stats.items()])
    fb.to_parquet(RESULTS / f"{a.out}_fallback.parquet", index=False)
    print("fallback rate (targets without a Trackastra link, per interval):\n",
          fb.groupby(["method", "H", "D"]).fallback.mean().unstack("D").round(3).to_string())


if __name__ == "__main__":
    main()
