"""Does the no-rotation-augmentation gain need prior knowledge of chamber orientation?

Protocol: docs/orientation_protocol.md (fixed before the M3/M4 run).

Every evaluation window is re-expressed in each of the 8 frames g of the image's
dihedral group D4 (exact transform of positions, cell axes and image shape, so
border/absolute features and the Darcy initialisation follow). Each model is run
once per (window, g); every arm is a composition of D4 elements applied to the
original frame and is read off these runs:
  A       no-rot model, original frame (g = e)
  B       no-rot model, frame g, no correction (mean over the 8 g)
  C_*     no-rot model after label-free canonicalisation of the observed frame g X.
          Chamber axis: sign of the nematic order <cos 2phi> of cell major axes
          (canonical < 0: cells along image x; else transpose). Residual ambiguity
          (the 4 axis-preserving elements K4) is either averaged over (``avg``,
          test-time averaging of probabilities) or resolved by the model's own
          label-free confidence (``conf``: candidate with the highest mean max
          ancestor probability over target cells). ``seq`` pools over every
          evaluated snapshot/window of the sequence, ``win`` uses one window only.
  M_conf  diagnostic, no geometry: all 8 candidates ranked by confidence.
  D       rotation-augmented model, frame g (original only, and mean over g)

  python experiments/orientation_canonicalisation.py --design   # M0-M2 (training movies) only
  python experiments/orientation_canonicalisation.py            # M3/M4, run once
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.family_set import BENCH, RESULTS, ROOT, examples  # noqa: E402
from experiments.lineage_glue import frame_dict, image_shape_of  # noqa: E402
from experiments.lineage_raft import load, predict  # noqa: E402

# D4 on (y, x) column vectors
I2 = np.eye(2, dtype=int)
R90 = np.array([[0, -1], [1, 0]])
FY = np.diag([-1, 1])
D4 = {"e": I2, "r90": R90, "r180": R90 @ R90, "r270": R90 @ R90 @ R90,
      "fy": FY, "fx": -FY, "tr": np.array([[0, 1], [1, 0]]), "atr": np.array([[0, -1], [-1, 0]])}
K4 = ("e", "fy", "fx", "r180")  # elements keeping the image axes (chamber axis) in place
SWAP = ("r90", "r270", "tr", "atr")


def name_of(m):
    return next(k for k, v in D4.items() if np.array_equal(v, m))


def mul(a, b):
    return name_of(D4[a] @ D4[b])


def apply(fd, g, shape):
    """Frame as a camera rotated/reflected by g would record it."""
    g = D4[g]
    c = np.array(shape, float) / 2
    shape2 = tuple(int(v) for v in np.abs(g) @ np.array(shape))
    c2 = np.array(shape2, float) / 2
    out = dict(fd)
    out["pos"] = (fd["pos"] - c) @ g.T + c2
    out["axis"] = fd["axis"] @ g.T
    return out, shape2


def cos2(fds):
    """Nematic order parameter <cos 2phi> of cell major axes, pooled over cells (phi from image y)."""
    ax = np.concatenate([f["axis"] for f in fds])
    ax = ax / np.maximum(np.linalg.norm(ax, axis=1, keepdims=True), 1e-6)
    return float(np.mean(ax[:, 0] ** 2 - ax[:, 1] ** 2))


def load_windows(aliases, horizons):
    from cellanc.loader import Windows
    windows = Windows(BENCH, ROOT / "data").windows
    out = []
    for alias in aliases:
        shape = image_shape_of(alias)
        for h in horizons:
            for e in examples(alias, windows, compensate=False, horizon=h):
                tg = e["targets"]
                ok = tg[tg.in_foi & (tg.target_status == "valid_anchor")]
                ro = {int(v): i for i, v in enumerate(e["target"].local_detection_id)}
                co = {int(v): i for i, v in enumerate(e["anchor"].local_detection_id)}
                ok = ok[ok.target_detection_id.isin(ro) & ok.ancestor_detection_id.isin(co)]
                if not len(ok):
                    continue
                out.append({"sequence": alias, "H": h, "window_id": e["window"]["window_id"], "shape": shape,
                            "fa": frame_dict(e["anchor"]), "fb": frame_dict(e["target"]),
                            "frames": (e["window"]["anchor_frame"], e["window"]["target_frame"]),
                            "n_anchors": len(co), "r": ok.target_detection_id.map(ro).to_numpy(),
                            "truth": ok.ancestor_detection_id.map(co).to_numpy()})
    return out


def acc(p, w):
    return float(np.mean(p[:, :-1].argmax(1)[w["r"]] == w["truth"]))


def conf(p):
    """Label-free confidence: mean over all target cells of the max (renormalised) ancestor probability."""
    a = p[:, :-1]
    return float(np.mean(a.max(1) / np.maximum(a.sum(1), 1e-9)))


def run(tags, aliases, horizons):
    """Raw rows: per (tag, window) accuracy/confidence in each of the 8 frames, K4-averaged accuracy per coset."""
    wins = load_windows(aliases, horizons)
    q_seq = {}
    for s in aliases:  # pooled over every distinct evaluated snapshot of the sequence (original frame)
        seen = {}
        for w in wins:
            if w["sequence"] == s:
                seen.setdefault(w["frames"][0], w["fa"])
                seen.setdefault(w["frames"][1], w["fb"])
        q_seq[s] = cos2(list(seen.values()))
    print("per-sequence <cos2phi>:", {k: round(v, 3) for k, v in q_seq.items()}, flush=True)
    rows = []
    for tag in tags:
        model, margs = load(tag)
        for w in wins:
            P = {}
            for g in D4:
                fa, shape2 = apply(w["fa"], g, w["shape"])
                fb, _ = apply(w["fb"], g, w["shape"])
                P[g] = predict(model, fa, fb, w["H"], shape2, margs.get("no_shape", False), 1,
                               init=margs.get("init", "radial"))[0]
            row = {"tag": tag, "sequence": w["sequence"], "H": w["H"], "window_id": w["window_id"],
                   "n_anchors": w["n_anchors"], "n_scored": len(w["r"]),
                   "q_win": cos2([w["fa"], w["fb"]]), "q_seq": q_seq[w["sequence"]],
                   "acc_K4avg": acc(sum(P[k] for k in K4) / 4, w),
                   "acc_SWAPavg": acc(sum(P[k] for k in SWAP) / 4, w)}
            for g in D4:
                row[f"acc_{g}"], row[f"conf_{g}"] = acc(P[g], w), conf(P[g])
            rows.append(row)
        print(tag, "done", flush=True)
    return pd.DataFrame(rows)


def arms(raw):
    """Long table: one row per (tag, window, g, arm). All arms are compositions on the original frame."""
    seqconf = raw.groupby(["tag", "sequence"])[[f"conf_{g}" for g in D4]].transform("mean")
    out = []
    for i, r in raw.iterrows():
        base = {k: r[k] for k in ("tag", "sequence", "H", "window_id", "n_anchors", "n_scored")}
        for g in D4:
            out.append({**base, "g": g, "arm": "frame_g", "accuracy": r[f"acc_{g}"], "recovered": np.nan})
            for pool, q, cf in (("seq", r.q_seq, seqconf.loc[i]), ("win", r.q_win, r)):
                q_obs = q if g in K4 else -q  # an axis swap flips <cos 2phi>; exact
                m = mul("e" if q_obs < 0 else "tr", g)  # original -> observed -> canonicalised
                coset = [mul(k, m) for k in K4]  # residual candidates, in original-frame terms
                rec = float(m in K4)
                out.append({**base, "g": g, "arm": f"C_{pool}_avg", "accuracy": r["acc_K4avg" if rec else "acc_SWAPavg"],
                            "recovered": rec})
                best = max(coset, key=lambda c: cf[f"conf_{c}"])
                out.append({**base, "g": g, "arm": f"C_{pool}_conf", "accuracy": r[f"acc_{best}"],
                            "recovered": rec, "exact": float(best == "e")})
                best8 = max(D4, key=lambda c: cf[f"conf_{c}"])  # equivariant: candidates k g^-1 cover D4
                out.append({**base, "g": g, "arm": f"M_{pool}_conf", "accuracy": r[f"acc_{best8}"],
                            "recovered": float(best8 in K4), "exact": float(best8 == "e")})
            out.append({**base, "g": g, "arm": "K4_avg_oracle", "accuracy": r["acc_K4avg"], "recovered": 1.0})
    return pd.DataFrame(out)


ROWS = [("A: no-rot, original frame", "norot", "frame_g", ("e",)),
        ("B: no-rot, D4 frame, no correction (mean 8 g)", "norot", "frame_g", tuple(D4)),
        ("B: no-rot, axis-swapping g only", "norot", "frame_g", SWAP),
        ("C_seq_avg: Q axis per sequence + K4 averaging", "norot", "C_seq_avg", tuple(D4)),
        ("C_seq_conf: Q axis per sequence + K4 by confidence", "norot", "C_seq_conf", tuple(D4)),
        ("C_win_avg: Q axis per window + K4 averaging", "norot", "C_win_avg", tuple(D4)),
        ("C_win_conf: Q axis per window + K4 by confidence", "norot", "C_win_conf", tuple(D4)),
        ("M_seq_conf: no geometry, 8 g by confidence", "norot", "M_seq_conf", tuple(D4)),
        ("K4 averaging on true frame (diagnostic)", "norot", "K4_avg_oracle", tuple(D4)),
        ("D: rot-aug, original frame", "rot", "frame_g", ("e",)),
        ("D: rot-aug, D4 frame (mean 8 g)", "rot", "frame_g", tuple(D4))]


def table(df, norot, rot):
    df = df.assign(dense=df.n_anchors > 200)
    out = {}
    for name, kind, arm, gs in ROWS:
        d = df[df.tag.isin(norot if kind == "norot" else rot) & (df.arm == arm) & df.g.isin(gs)]
        if not len(d):
            continue

        def cell(x):  # window mean per (tag, g), then mean over g, then over seeds
            return x.groupby(["tag", "g", "H"]).accuracy.mean().groupby(["tag", "H"]).mean().groupby("H").mean()
        out[name] = pd.concat([cell(d).rename(lambda h: f"all_{h}"), cell(d[d.dense]).rename(lambda h: f"dense_{h}")])
    t = pd.DataFrame(out).T
    return t[[f"{s}_{h}" for s in ("all", "dense") for h in (33, 66, 132)]].round(3)


def report(df, norot, rot):
    t = table(df, norot, rot)
    print(t.to_string())
    if rot:
        a, dg = t.loc["A: no-rot, original frame"], t.loc["D: rot-aug, D4 frame (mean 8 g)"]
        rec = t.drop(index=[n for n in t.index if n.startswith("D")]).apply(lambda r: (r - dg) / (a - dg), axis=1)
        print("fraction of the no-rot gain (A - D_g) recovered:\n", rec.round(2).to_string())
    c = df[df.tag.isin(norot) & df.arm.isin(["C_seq_conf", "C_win_conf", "M_seq_conf", "M_win_conf"])]
    print("axis recovered / exact frame chosen, by arm x sequence:")
    print(c.groupby(["arm", "sequence"])[["recovered", "exact"]].mean().round(3).to_string())
    print("per-g accuracy (no-rot, all windows):")
    print(df[df.tag.isin(norot) & (df.arm == "frame_g")].groupby(["g", "H"]).accuracy.mean().unstack().round(3).to_string())
    return t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--design", action="store_true", help="M0-M2 (training movies), first no-rot seed only")
    p.add_argument("--from-raw", action="store_true", help="recompute tables from saved raw rows")
    args = p.parse_args()
    if args.design:
        raw_path = RESULTS / "orientation_design_M012_raw.parquet"
        raw = pd.read_parquet(raw_path) if args.from_raw else run(["raft_v3"], ["M0", "M1", "M2"], [33, 66, 132])
        raw.to_parquet(raw_path, index=False)
        report(arms(raw), ["raft_v3"], [])
        return
    pairs = {"v3": (["raft_v3", "raft_v3_s1", "raft_v3_s2"], ["raft_v2_darcy_30k"]),
             "v1": (["raft_v1_norot"], ["raft_v1"])}
    raw_path = RESULTS / "orientation_canonicalisation_raw.parquet"
    if args.from_raw:
        raw = pd.read_parquet(raw_path)
    else:
        tags = [t for norot, rot in pairs.values() for t in norot + rot]
        raw = run(tags, ["M3", "M4"], [33, 66, 132])
        raw.to_parquet(raw_path, index=False)
    df = arms(raw)
    df["pair"] = df.tag.map({t: k for k, (a, b) in pairs.items() for t in a + b})
    df.to_parquet(RESULTS / "orientation_canonicalisation.parquet", index=False)
    print("per-sequence <cos2phi>:", raw.groupby("sequence").q_seq.first().round(3).to_dict(),
          "| per-window Q<0:", raw[raw.tag == "raft_v3"].assign(ok=lambda x: x.q_win < 0).groupby("sequence").ok.mean().round(3).to_dict())
    for k, (norot, rot) in pairs.items():
        print(f"\n== pair {k}: no-rot {norot} vs rot-aug {rot}")
        report(df[df.pair == k], norot, rot)


if __name__ == "__main__":
    main()
