"""Stronger baselines (OT, CPD, Darcy, FSA) on the windows of tab:external and blind test 4.

Protocol: docs/external_baselines_protocol.md. All hyperparameters frozen as selected on OIAM;
windows are built by the unchanged raft_external_eval.run / blind_test4 loops (rng 20261002).

  python experiments/external_baselines.py external --jobs 12
  python experiments/external_baselines.py blind4 --jobs 12
  python experiments/external_baselines.py summary
"""

import argparse
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.cpd_baseline import cpd  # noqa: E402
from experiments.darcy_flow_prior import darcy_assign  # noqa: E402
from experiments.family_set import RESULTS  # noqa: E402
from experiments.fast_microcolony_external_test import family_f1  # noqa: E402
from experiments.forward_sim_ancestry import assign as fsa_assign, rollout  # noqa: E402
from experiments.ot_baseline import align, couple  # noqa: E402
from experiments.radial_sector_probe import geometry_assignments  # noqa: E402

DATASETS = ["OIAM", "recA", "rpsM", "FAST", "LFCT", "T98G"]  # raft_external_eval default order (rng order)
OT = {"ot_uot_radial": ("radial", "uot", {"eps": 0.2, "rho": 1.0}),   # tab:oiam OT row (dev pick)
      "ot_uot_none": ("none", "uot", {"eps": 0.5, "rho": 10.0}),      # blind test 4 settings (dev picks)
      "ot_exact_none": ("none", "exact", {"c": 1.0})}
CPD = {"cpd_radial": ("radial", {"beta": 8.0, "lam": 4.0, "w": 0.0})}  # dev pick, as make_tables.oiam_rows
FSA_ROLLOUTS = 8                     # forward_sim_ancestry_final: align none + non-rigid warp
SPACING_PER_WIDTH = 1.13             # OIAM M0-M2 median NN spacing / cell width (masks, no labels)
LENGTH_WIDTHS = 2.48                 # OIAM M0-M2 median cell length / width
GEO = ("nearest_anchor", "radial_sector", "radial_expansion", "centroid_translation")


def pseudo_shape(pos, seed=0):
    """Positions-only stand-in for the mask shape FSA needs (protocol: fsa_pos)."""
    rng = np.random.default_rng(seed)
    sp = max(float(np.median(cKDTree(pos).query(pos, k=2)[0][:, 1])), 1.0)
    px = sp / SPACING_PER_WIDTH
    th = rng.uniform(0, np.pi, len(pos))
    return pd.DataFrame({"center_y": pos[:, 0], "center_x": pos[:, 1], "axis_y": np.sin(th), "axis_x": np.cos(th),
                         "minor_sd": 0.25 * px, "major_sd": LENGTH_WIDTHS * px / math.sqrt(12)})


def fsa(anchor_df, dst, minutes):
    sims = [rollout(anchor_df, len(dst), minutes, k) for k in range(FSA_ROLLOUTS)]
    tgt = pd.DataFrame({"center_y": dst[:, 0], "center_x": dst[:, 1]})
    return fsa_assign(anchor_df, tgt, sims, nonrigid=True, align="none")[0]


def predictions(e, minutes, shape, with_shape=False):
    import torch
    torch.set_num_threads(1)
    src = e["anchor"][["center_y", "center_x"]].to_numpy(float)
    dst = e["target"][["center_y", "center_x"]].to_numpy(float)
    names = [*GEO, *OT, *CPD, "darcy_68", "darcy_auto_68", "fsa_pos"] + (["fsa_shape"] if with_shape else [])
    if len(src) < 2:
        return {n: np.zeros(len(dst), int) for n in names}, {}
    out, sec = dict(geometry_assignments(e)), {}
    for name, (a, kind, p) in OT.items():
        out[name] = couple(align(src, dst, minutes, shape, a), dst, kind, p)
    for name, (a, p) in CPD.items():
        out[name] = cpd(align(src, dst, minutes, shape, a), dst, **p)
    ed = {**e, "window": {**e["window"], "horizon_frames": minutes}}
    out["darcy_68"] = darcy_assign(ed, shape, 68.0, auto=False)
    out["darcy_auto_68"] = darcy_assign(ed, shape, 68.0, auto=True)
    t = time.time()
    out["fsa_pos"] = fsa(pseudo_shape(src), dst, minutes)
    sec["fsa_pos"] = time.time() - t
    if with_shape:
        out["fsa_shape"] = fsa(e["anchor"].reset_index(drop=True), dst, minutes)
    return out, sec


def score_external(w):
    dataset, seq, wid, minutes, e, valid, truth, shape = w
    t0 = time.time()
    preds, sec = predictions(e, minutes, shape, with_shape=(dataset == "OIAM"))
    rows = []
    for m, p in preds.items():
        g = p[valid]
        rows.append({"dataset": dataset, "sequence": seq, "window_id": wid, "gap_minutes": minutes,
                     "n_anchors": len(e["anchor"]), "n_targets": len(e["target"]), "n_scored": len(truth),
                     "method": m, "accuracy": float(np.mean(g == truth)), "set_f1": family_f1(truth, g),
                     "fsa_seconds": sec.get("fsa_pos", 0.0), "seconds": time.time() - t0})
    return rows


def external_windows():
    import experiments.raft_external_eval as ree
    store = []
    orig = ree.Runner.window

    def window(self, dataset, seq, wid, minutes, e, valid, truth, shape):
        if len(truth):
            store.append((dataset, seq, wid, minutes, e, valid, truth, shape))
        orig(self, dataset, seq, wid, minutes, e, valid, truth, shape)  # geometry rows, for the identity check

    ree.Runner.window = window
    tmp = Path(tempfile.mkdtemp())
    ree.RESULTS = tmp  # run() writes its geometry-only parquet here, never into outputs/results
    ree.run(argparse.Namespace(tags=[], decoders=[], datasets=DATASETS, out="geo", ensemble=False))
    ree.Runner.window = orig
    return store, pd.read_parquet(tmp / "raft_external_geo.parquet")


def check_identity(new, ref, keys, methods):
    a = new[new.method.isin(methods)].set_index(keys + ["method"]).accuracy
    b = ref[ref.method.isin(methods)].set_index(keys + ["method"]).accuracy
    j = pd.concat([a.rename("new"), b.rename("ref")], axis=1)
    bad = j[(j.new - j.ref).abs().gt(1e-9) | j.isna().any(axis=1)]
    print(f"identity check: {len(j)} rows, {len(bad)} mismatches", flush=True)
    if len(bad):
        print(bad.head(20).to_string())
    return len(bad) == 0


def run_external(a):
    store, geo = external_windows()
    v1 = pd.read_parquet(RESULTS / "raft_external_v1.parquet")
    keys = ["dataset", "sequence", "window_id", "gap_minutes"]
    assert check_identity(geo, v1, keys, list(GEO)), "windows differ from raft_external_v1"
    if a.smoke:
        store = sorted(store, key=lambda w: -len(w[4]["target"]) * w[3])[:a.smoke]
    store.sort(key=lambda w: -len(w[4]["target"]) * w[3])  # largest first for load balance
    print("windows", len(store), flush=True)
    out = Parallel(n_jobs=a.jobs, verbose=5)(delayed(score_external)(w) for w in store)
    d = pd.DataFrame([r for rows in out for r in rows])
    if a.smoke:
        print(d.groupby("dataset")[["fsa_seconds", "seconds"]].max().to_string())
        return
    assert check_identity(d, v1, keys, list(GEO)), "geometry differs from raft_external_v1"
    d.to_parquet(RESULTS / "external_baselines.parquet", index=False)
    summary_external(d)


def blind4_windows():
    from experiments.blind_test4 import GAPS, MPF, SUBSETS, colonies
    from experiments.raft_external_eval import bbox_shape, reca_window
    rng = np.random.default_rng(20261002)
    store = []
    for subset in SUBSETS:
        for colony, (frames, mapping, tracks, invalid, _) in colonies(subset).items():
            shape = bbox_shape([f[:, ::-1] for f in frames.values() if len(f)])
            for gap in GAPS:
                for start in sorted(frames):
                    if start + gap not in frames or not len(frames[start]) or not len(frames[start + gap]):
                        continue
                    e, valid, truth = reca_window(start, gap, frames, mapping, tracks, invalid, rng)
                    if len(truth):
                        store.append((subset, colony, gap, f"{start}_{start + gap}", gap * MPF, e, valid, truth, shape))
    return store


def score_blind4(w):
    subset, colony, gap, wid, minutes, e, valid, truth, shape = w
    preds, _ = predictions(e, minutes, shape)
    return [{"dataset": subset, "sequence": colony, "gap": gap, "gap_minutes": minutes, "window_id": wid,
             "n_anchors": len(e["anchor"]), "n_scored": len(truth), "method": m,
             "accuracy": float(np.mean(p[valid] == truth))} for m, p in preds.items()]


def run_blind4(a):
    store = blind4_windows()
    store.sort(key=lambda w: -len(w[5]["target"]) * w[4])
    print("windows", len(store), flush=True)
    out = Parallel(n_jobs=a.jobs, verbose=5)(delayed(score_blind4)(w) for w in store)
    d = pd.DataFrame([r for rows in out for r in rows])
    ref = pd.read_parquet(RESULTS / "blind_test4.parquet")
    ref = ref.replace({"method": {"ot_uot": "ot_uot_none", "ot_exact": "ot_exact_none"}})
    assert check_identity(d, ref, ["dataset", "sequence", "gap", "window_id"],
                          list(GEO) + ["ot_uot_none", "ot_exact_none"]), "windows differ from blind_test4"
    d.to_parquet(RESULTS / "blind4_baselines_extra.parquet", index=False)
    summary_blind4(d)


NEW = ["ot_uot_radial", "ot_uot_none", "ot_exact_none", "cpd_radial", "darcy_68", "fsa_pos"]
BACTERIA = ["FAST", "recA", "rpsM", "OIAM"]


def table_mean(x):
    return x.groupby("gap_minutes").accuracy.mean().mean()


def summary_external(d=None):
    d = pd.read_parquet(RESULTS / "external_baselines.parquet") if d is None else d
    v1 = pd.read_parquet(RESULTS / "raft_external_v1.parquet")
    v3 = pd.read_parquet(RESULTS / "raft_external_v3.parquet")
    sim = pd.concat([v1[v1.method == "raft_sim_loc_argmax"].assign(method="rodsim_s0"),
                     v3[v3.method == "raft_sim_loc_s1_argmax"].assign(method="rodsim_s1"),
                     v3[v3.method == "raft_sim_loc_s2_argmax"].assign(method="rodsim_s2")])
    allm = pd.concat([d, sim[d.columns.intersection(sim.columns)]])
    base = list(GEO) + NEW
    extra = ["darcy_auto_68", "fsa_shape"]
    rows = []
    for ds in DATASETS:
        x = allm[allm.dataset == ds]
        acc = {m: table_mean(x[x.method == m]) for m in base + extra + ["rodsim_s0", "rodsim_s1", "rodsim_s2"]
               if (x.method == m).any()}
        acc["rodsim_3seed"] = np.mean([acc["rodsim_s0"], acc["rodsim_s1"], acc["rodsim_s2"]])
        bb = max(base, key=lambda m: acc[m])
        acc.update(best_baseline=bb, margin_s0=acc["rodsim_s0"] - acc[bb], margin_3seed=acc["rodsim_3seed"] - acc[bb])
        rows.append({"dataset": ds, **acc})
    t = pd.DataFrame(rows).set_index("dataset")
    pd.set_option("display.width", 250)
    print(t.T.to_string(float_format=lambda v: f"{v:.3f}"))
    # sequence wins, as in table_external (pooled window mean per sequence), and mean-over-gaps variant
    for how in ("pooled", "gapmean"):
        for tag in ("rodsim_s0", "rodsim_3seed"):
            won, tot, lines = 0, 0, []
            for ds in BACTERIA:
                x = allm[allm.dataset == ds]
                if how == "pooled":
                    s = x.pivot_table(index="sequence", columns="method", values="accuracy")
                else:
                    s = x.groupby(["sequence", "gap_minutes", "method"]).accuracy.mean().groupby(
                        ["sequence", "method"]).mean().unstack("method")
                s["rodsim_3seed"] = s[["rodsim_s0", "rodsim_s1", "rodsim_s2"]].mean(axis=1)
                w = s[tag] > s[base].max(axis=1)
                won += int(w.sum())
                tot += len(s)
                lines.append(f"{ds} {int(w.sum())}/{len(s)}")
                for q in s.index[~w]:
                    lines.append(f"   lost {q}: {tag} {s.loc[q, tag]:.3f} vs {s.loc[q, base].idxmax()} {s.loc[q, base].max():.3f}")
            print(f"sequence wins [{how}, {tag}] vs best baseline per sequence: {won}/{tot}; " + "; ".join(lines))
    return t


def summary_blind4(d=None):
    d = pd.read_parquet(RESULTS / "blind4_baselines_extra.parquet") if d is None else d
    ref = pd.read_parquet(RESULTS / "blind_test4.parquet")
    ref = ref.replace({"method": {"ot_uot": "ot_uot_none", "ot_exact": "ot_exact_none"}})
    keep = ref[ref.method.isin(["sim", "adapt", "adaptm"])]
    allm = pd.concat([d, keep[["dataset", "sequence", "gap", "window_id", "n_anchors", "method", "accuracy"]]])
    seq = allm.groupby(["dataset", "sequence", "gap", "method"]).accuracy.mean() \
        .groupby(["dataset", "sequence", "method"]).mean().unstack("method")
    base = list(GEO) + NEW
    src = seq.groupby("dataset").mean()
    cols = base + ["darcy_auto_68", "sim", "adapt", "adaptm"]
    t = src[cols].copy()
    t["best_baseline"] = src[base].idxmax(axis=1)
    t["margin_sim"] = src["sim"] - src[base].max(axis=1)
    t["colony_wins_sim"] = [f"{int((seq.loc[ds].sim > seq.loc[ds][base].max(axis=1)).sum())}/{len(seq.loc[ds])}"
                            for ds in t.index]
    old = list(GEO) + ["ot_uot_none", "ot_exact_none"]
    t["colony_wins_sim_old6"] = [f"{int((seq.loc[ds].sim > seq.loc[ds][old].max(axis=1)).sum())}/{len(seq.loc[ds])}"
                                 for ds in t.index]
    print(t.T.to_string(float_format=lambda v: f"{v:.3f}"))
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["external", "blind4", "summary"])
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--smoke", type=int, default=0, help="time the N largest external windows only; no output")
    a = ap.parse_args()
    if a.mode == "external":
        run_external(a)
    elif a.mode == "blind4":
        run_blind4(a)
    else:
        summary_external()
        summary_blind4()


if __name__ == "__main__":
    main()
