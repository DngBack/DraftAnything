"""Robustness of ancestry to detection errors (oracle positions are perturbed).

Perturbations are applied independently to both snapshots: position jitter (in
units of the median nearest-neighbour spacing), missed detections, merges of a
cell with its nearest neighbour (under-segmentation) and spurious detections.
Only targets that survive unmerged and whose true ancestor survives unmerged
are scored, so every setting is scored on a subset of the clean targets.

  python experiments/detection_robustness.py --tags raft_v3 raft_sim_loc
"""

import argparse
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.baselines import expansion_compensate  # noqa: E402
from experiments.family_set import BENCH, RESULTS, ROOT  # noqa: E402
from experiments.lineage_glue import frame_dict, image_shape_of, mean_field, set_f1  # noqa: E402
from experiments.lineage_raft import load, predict  # noqa: E402

SETTINGS = [("clean", 0), ("jitter", 0.15), ("jitter", 0.3), ("jitter", 0.5), ("drop", 0.1), ("drop", 0.2),
            ("merge", 0.05), ("merge", 0.1), ("spurious", 0.1), ("spurious", 0.2)]


def perturb(df, kind, level, rng):
    """Return the perturbed frame and, per output row, the source row (-1 if merged or spurious)."""
    df = df.reset_index(drop=True)
    n = len(df)
    src = np.arange(n)
    if kind == "clean" or n < 3:
        return df, src
    pos = df[["center_y", "center_x"]].to_numpy(float)
    spacing = np.median(cKDTree(pos).query(pos, k=2)[0][:, 1])
    if kind == "jitter":
        df = df.copy()
        df[["center_y", "center_x"]] = pos + rng.normal(0, level * spacing, pos.shape)
        return df, src
    if kind == "drop":
        keep = rng.random(n) >= level
        return df[keep].reset_index(drop=True), src[keep]
    if kind == "merge":
        nn_ = cKDTree(pos).query(pos, k=2)[1][:, 1]
        used = np.zeros(n, bool)
        pairs = []
        for a in rng.permutation(n)[:int(level * n)]:
            b = nn_[a]
            if not (used[a] or used[b]):
                used[[a, b]] = True
                pairs.append((a, b))
        keep = ~used
        if not pairs:
            return df, src
        a, b = np.array(pairs).T
        merged = df.iloc[a].copy()
        merged[["center_y", "center_x"]] = (pos[a] + pos[b]) / 2
        merged["major_sd"] = (np.hypot(df.major_sd.to_numpy()[a], df.major_sd.to_numpy()[b]) +
                              np.linalg.norm(pos[a] - pos[b], axis=1) / 2)
        merged["local_detection_id"] = -1 - a
        out = pd.concat([df[keep], merged], ignore_index=True)
        return out, np.concatenate([src[keep], -np.ones(len(a), int)])
    if kind == "spurious":
        m = int(level * n)
        lo, hi = pos.min(0), pos.max(0)
        fake = df.iloc[rng.integers(0, n, m)].copy()
        fake[["center_y", "center_x"]] = lo + rng.random((m, 2)) * (hi - lo)
        th = rng.uniform(0, np.pi, m)
        fake["axis_y"], fake["axis_x"] = np.sin(th), np.cos(th)
        fake["local_detection_id"] = -1 - np.arange(m)
        return pd.concat([df, fake], ignore_index=True), np.concatenate([src, -np.ones(m, int)])
    raise ValueError(kind)


def main():
    from cellanc.loader import Windows
    from experiments.family_set import examples
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["raft_v3", "raft_sim_loc"])
    ap.add_argument("--eval", nargs="+", default=["M3", "M4"])
    ap.add_argument("--horizons", nargs="+", type=int, default=[33, 66, 132])
    ap.add_argument("--out", default="detection_robustness")
    a = ap.parse_args()
    models = {t: load(t) for t in a.tags}
    windows = Windows(BENCH, ROOT / "data").windows
    rows = []
    t0 = time.time()
    for alias in a.eval:
        shape = image_shape_of(alias)
        for h in a.horizons:
            for wi, e in enumerate(examples(alias, windows, compensate=False, horizon=h)):
                tg = e["targets"]
                ok = tg[tg.in_foi & (tg.target_status == "valid_anchor")]
                anc0, tgt0 = e["anchor"].reset_index(drop=True), e["target"].reset_index(drop=True)
                ro0 = {int(v): i for i, v in enumerate(tgt0.local_detection_id)}
                co0 = {int(v): i for i, v in enumerate(anc0.local_detection_id)}
                ok = ok[ok.target_detection_id.isin(ro0) & ok.ancestor_detection_id.isin(co0)]
                if not len(ok):
                    continue
                truth0 = dict(zip(ok.target_detection_id.map(ro0), ok.ancestor_detection_id.map(co0)))
                for kind, level in SETTINGS:
                    rng = np.random.default_rng(zlib.crc32(f"{alias}/{h}/{wi}/{kind}/{level}".encode()))
                    anc, sa = perturb(anc0, kind, level, rng)
                    tgt, sb = perturb(tgt0, kind, level, rng)
                    inv_a = {int(s): k for k, s in enumerate(sa) if s >= 0}
                    r, truth = [], []
                    for k, s in enumerate(sb):
                        if s >= 0 and s in truth0 and truth0[s] in inv_a:
                            r.append(k)
                            truth.append(inv_a[truth0[s]])
                    if len(r) == 0 or len(anc) < 2:
                        continue
                    r, truth = np.array(r), np.array(truth)
                    base = {"sequence": alias, "H": h, "window_id": e["window"]["window_id"], "kind": kind,
                            "level": level, "n_anchors": len(anc0), "n_scored": len(r)}
                    pa, pb = anc[["center_y", "center_x"]].to_numpy(float), tgt[["center_y", "center_x"]].to_numpy(float)
                    near = cKDTree(pa).query(pb)[1]
                    src = np.column_stack([np.arange(len(pa)), pa])
                    dst = np.column_stack([np.arange(len(pb)), pb])
                    comp = cKDTree(pa).query(expansion_compensate(src, dst)[:, 1:])[1]
                    preds = {"nearest": near, "nearest_comp": comp}
                    for tag, (model, margs) in models.items():
                        prob, fam, _ = predict(model, frame_dict(anc), frame_dict(tgt), h, shape,
                                               margs.get("no_shape", False), init=margs.get("init", "radial"))
                        q = prob[:, :-1] / np.maximum(prob[:, :-1].sum(1, keepdims=True), 1e-9)
                        preds[f"{tag}_argmax"] = q.argmax(1)
                        i, j, w = fam
                        preds[f"{tag}_mf_fam2"] = mean_field(q, i, j, w, 2).argmax(1) if len(i) else q.argmax(1)
                    for name, p in preds.items():
                        p = p[r]
                        rows.append({**base, "method": name, "accuracy": float(np.mean(p == truth)),
                                     "set_f1": set_f1(truth, p)})
                print(f"  {e['window']['window_id']} n={len(anc0)} {time.time() - t0:.0f}s", flush=True)
            print(alias, h, flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(RESULTS / f"{a.out}.parquet", index=False)
    print(df.pivot_table(index=["kind", "level"], columns=["method"], values="accuracy").round(3).to_string())
    for h in a.horizons:
        print(h, "\n", df[df.H == h].pivot_table(index=["kind", "level"], columns="method",
                                                 values="accuracy").round(3).to_string())


if __name__ == "__main__":
    main()
