"""How family centres move through the refinement rounds (inference only, no threshold).

For every OIAM M3/M4 window and gap, runs a trained model once and records, per round t = 0..T
(t = 0: initial centre, radial or Darcy), the distance of each ancestor's centre to the true centroid
of its scored descendants (in target spacings) and the recall of the true ancestor among each target's
k_out nearest centres. Dumps the centres of one dense window for the supplement figure.

  python experiments/centre_trajectory.py --tag raft_v3
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.loader import Windows  # noqa: E402
from experiments.family_set import BENCH, RESULTS, ROOT, examples  # noqa: E402
from experiments.lineage_glue import featurize, frame_dict, image_shape_of, spacing_of  # noqa: E402
from experiments.lineage_raft import initial_centres, load  # noqa: E402


@torch.no_grad()
def rounds(model, fa, fb, h, shape, no_shape, init, dev="cuda"):
    """Centres (pixels) of every round, t = 0..T, and the final argmax over candidates."""
    moved = initial_centres(fa, fb, h, shape, init)
    xa, xb, ca, cb = (t.to(dev) for t in featurize(fa, fb, h, shape, None, moved))
    if no_shape:
        xa[:, 4:8] = 0
        xb[:, 4:8] = 0
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, cand, centres, _ = model(xa, xb, ca, cb)
    s, mu_b = spacing_of(fb["pos"]), fb["pos"].mean(0)
    cs = [c.float().cpu().numpy() * s + mu_b for c in [ca, *centres]]
    p = torch.softmax(logits.float(), 1)[:, :-1]
    pred = cand.gather(1, p.argmax(1, keepdim=True)).squeeze(1).cpu().numpy()
    return cs, pred, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="raft_v3")
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--dump", default="M4_a0604_b0670_ends")
    a = ap.parse_args()
    model, margs = load(a.tag)
    windows = Windows(BENCH, ROOT / "data").windows
    rows, trows = [], []
    for alias in ("M3", "M4"):
        shape = image_shape_of(alias)
        for h in (33, 66, 132):
            for e in examples(alias, windows, compensate=False, horizon=h):
                anc, tgt = e["anchor"].reset_index(drop=True), e["target"].reset_index(drop=True)
                tg = e["targets"]
                ok = tg[tg.in_foi & (tg.target_status == "valid_anchor")]
                ro = {int(v): k for k, v in enumerate(tgt.local_detection_id)}
                co = {int(v): k for k, v in enumerate(anc.local_detection_id)}
                ok = ok[ok.target_detection_id.isin(ro) & ok.ancestor_detection_id.isin(co)]
                if not len(ok):
                    continue
                fa, fb = frame_dict(anc), frame_dict(tgt)
                cs, pred, s = rounds(model, fa, fb, h, shape, margs.get("no_shape", False), margs.get("init", "radial"))
                r, tr = ok.target_detection_id.map(ro).to_numpy(), ok.ancestor_detection_id.map(co).to_numpy()
                pb = fb["pos"]
                fam = pd.DataFrame({"a": tr, "y": pb[r, 0], "x": pb[r, 1]}).groupby("a")[["y", "x"]].mean()
                ids, cen = fam.index.to_numpy(), fam.to_numpy()
                for t, c in enumerate(cs):
                    err = np.linalg.norm(c[ids] - cen, axis=1) / s
                    d = np.linalg.norm(pb[r][:, None] - c[None], axis=-1)
                    kth = np.partition(d, min(a.k, d.shape[1]) - 1, axis=1)[:, min(a.k, d.shape[1]) - 1]
                    recall = (d[np.arange(len(r)), tr] <= kth).mean()
                    rows.append({"sequence": alias, "H": h, "window_id": e["window"]["window_id"],
                                 "n_anchors": len(anc), "round": t, "centre_err_mean": err.mean(),
                                 "centre_err_median": np.median(err), "recall_k": recall,
                                 "accuracy": (pred[r] == tr).mean()})
                size = pd.Series(tr).map(pd.Series(tr).value_counts()).to_numpy()
                pos = {v: k for k, v in enumerate(ids)}
                fe = [np.linalg.norm(c[ids] - cen, axis=1) / s for c in (cs[0], cs[-1])]
                k = np.array([pos[v] for v in tr])
                trows.append(pd.DataFrame({"H": h, "window_id": e["window"]["window_id"], "n_anchors": len(anc),
                                           "family_size": size, "correct": pred[r] == tr,
                                           "err_init": fe[0][k], "err_final": fe[1][k]}))
                if e["window"]["window_id"] == a.dump:
                    np.savez_compressed(RESULTS / f"centre_trajectory_{a.dump}.npz", centres=np.stack(cs),
                                        anchor_pos=fa["pos"], target_pos=pb, scored=r, truth=tr,
                                        family_ids=ids, family_centroid=cen, spacing=s)
            print(alias, h, flush=True)
    t = pd.concat(trows, ignore_index=True)
    t.to_parquet(RESULTS / f"centre_trajectory_targets_{a.tag}.parquet", index=False)
    t["size_bin"] = pd.cut(t.family_size, [0, 1, 2, 4, 8, 99], labels=["1", "2", "3-4", "5-8", ">8"])
    t["err_bin"] = pd.cut(t.err_final, [0, 0.5, 1, 2, 4, 99])
    print(t.pivot_table(index="size_bin", columns="H", values="correct", aggfunc=["mean", "size"], observed=True).round(3).to_string())
    print(t.pivot_table(index="err_bin", columns="H", values="correct", aggfunc=["mean", "size"], observed=True).round(3).to_string())
    df = pd.DataFrame(rows)
    df.to_parquet(RESULTS / f"centre_trajectory_{a.tag}.parquet", index=False)
    df["dense"] = df.n_anchors > 200
    for sub, d in (("all", df), ("dense", df[df.dense])):
        print(sub)
        print(d.pivot_table(index="round", columns="H", values=["centre_err_mean", "centre_err_median", "recall_k"]).round(3).to_string())


if __name__ == "__main__":
    main()
