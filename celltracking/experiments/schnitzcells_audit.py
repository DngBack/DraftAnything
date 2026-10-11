"""Label-quality audit of Schnitzcells recA/rpsM lineages against curated FAST.

Protocol: docs/schnitzcells_audit_protocol.md. Descriptive statistics only; uses the
same loaders, mapped detections, parent links and windows as raft_external_eval.

  python experiments/schnitzcells_audit.py --jobs 8
"""

import argparse
import io
import re
import sys
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.io
from joblib import Parallel, delayed
from scipy.spatial import ConvexHull, QhullError

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.family_set import RESULTS, ROOT  # noqa: E402
from experiments import fast_microcolony_external_test as fastm  # noqa: E402
from experiments import reca_schnitzcells as schn  # noqa: E402
from experiments.fetch_rpsm import HELD_OUT as RPSM_HELD_OUT  # noqa: E402
from experiments.reca_microcolony_external_test import GAPS as RECA_GAPS, HELD_OUT as RECA_HELD_OUT  # noqa: E402

DATASETS = {"FAST": (fastm.HELD_OUT, fastm.GAPS, 5.0),
            "recA": (RECA_HELD_OUT, RECA_GAPS, 7.5),
            "rpsM": (RPSM_HELD_OUT, (10, 20, 40), 3.0)}
GEO = ["nearest_anchor", "radial_sector", "radial_expansion", "centroid_translation"]
DEV_TOL = 0.30
NOFLAG = np.zeros(0, int)


# ---------------------------------------------------------------- loading (common form)
def _seg_features(path):
    try:
        with h5py.File(path, "r") as f:
            g = f["cellSegData"]
            raw = np.asarray(g["cellCentroid"][()])
            L, A = np.asarray(g["length"][()]).ravel(), np.asarray(g["area"][()]).ravel()
    except OSError:
        g = scipy.io.loadmat(path, simplify_cells=True)["cellSegData"]
        raw, L, A = np.asarray(g["cellCentroid"]), np.atleast_1d(g["length"]).ravel(), np.atleast_1d(g["area"]).ravel()
    xy = raw.T if raw.shape[0] == 2 and raw.shape[1] != 2 else raw
    ok = np.isfinite(xy).all(1)
    return L[ok].astype(float), A[ok].astype(float)


def load_schnitz(root):
    frames, mapping, tracks, invalid, audit = schn.load_colony(root)
    off = audit["frame_offset"]
    feats = {}
    for p in sorted((root / "cellSegmentationData").glob("*_segData*.mat")):
        f = int(re.search(r"_segData(\d+)\.mat$", p.name).group(1)) + off
        feats[f] = _seg_features(p)
        assert len(feats[f][0]) == len(frames[f])
    parent, span = {}, {}
    for tid, tr in enumerate(tracks, start=1):
        fr = schn._as_int_array(tr["frames"])
        span[tid] = (int(fr.min()), int(fr.max())) if fr.size else None
        m = schn._as_int_array(tr["P"])
        ok = m.size and 1 <= m[0] <= len(tracks) and tid not in invalid
        parent[tid] = int(m[0]) if ok else 0
    ids = {f: m for f, m in mapping.items()}
    return dict(frames=frames, ids=ids, feats=feats, parent=parent, span=span,
                n_invalid=len(invalid), n_tracks=len(tracks),
                ancestor=lambda t, s: schn.ancestor_track(t, s, tracks, invalid), skip_empty=True)


def load_fast(colony):
    path = ROOT / "data/downloads" / f"fast_{colony}.zip"
    frames, mappings, tracks, _ = fastm.load_colony(path)
    with zipfile.ZipFile(path) as z, h5py.File(io.BytesIO(fastm._mat_bytes(z, "CellFeatures_ManualSeg.mat")), "r") as f:
        td = f["trackableData"]
        feats = {fr: (np.asarray(f[td["Length"][0, fr - 1]][()]).ravel().astype(float),
                      np.asarray(f[td["Area"][0, fr - 1]][()]).ravel().astype(float)) for fr in frames}
    for fr in frames:
        assert len(feats[fr][0]) == len(frames[fr])
    parent, span = {}, {}
    for tid, tr in enumerate(tracks, start=1):
        t = np.atleast_1d(tr["times"]).astype(int)
        span[tid] = (int(t.min()), int(t.max()))
        m = np.atleast_1d(tr["M"])
        parent[tid] = int(m[0]) if m.size and not tr.get("_invalid_mother", False) else 0
    ids = {fr: m[:, 0].astype(float) for fr, m in mappings.items()}
    return dict(frames=frames, ids=ids, feats=feats, parent=parent, span=span,
                n_invalid=sum(bool(tr.get("_invalid_mother", False)) for tr in tracks), n_tracks=len(tracks),
                ancestor=lambda t, s: fastm.ancestor_track(t, s, tracks), skip_empty=False)


# ---------------------------------------------------------------- statistics
def hull_depth(xy):
    if len(xy) < 3:
        return np.zeros(len(xy))
    try:
        eq = ConvexHull(xy).equations
    except QhullError:
        return np.zeros(len(xy))
    return np.maximum((-(xy @ eq[:, :2].T + eq[:, 2])).min(1), 0.0)


def audit_colony(dataset, colony):
    _, gaps, mpf = DATASETS[dataset]
    c = load_fast(colony) if dataset == "FAST" else load_schnitz(ROOT / "data/raw" / f"{dataset.lower()}_holdout" / colony)
    frames, ids, feats, parent, span = c["frames"], c["ids"], c["feats"], c["parent"], c["span"]
    f_first, f_last = min(frames), max(frames)
    med_len = float(np.nanmedian(np.concatenate([feats[f][0] for f in frames])))
    obs = {}  # tid -> {frame: (x, y, L, A, interior)}
    n_det = n_unmapped = 0
    for f, xy in frames.items():
        depth = hull_depth(xy)
        n_det += len(xy)
        for r, tid in enumerate(ids[f]):
            if not np.isfinite(tid):
                n_unmapped += 1
                continue
            obs.setdefault(int(tid), {})[f] = (xy[r, 0], xy[r, 1], feats[f][0][r], feats[f][1][r], depth[r] > med_len)
    children = {}
    for tid, m in parent.items():
        if m:
            children.setdefault(m, []).append(tid)
    nch = {t: len(children.get(t, [])) for t in parent}

    # (a) division events with != 2 daughters; (a2) interior dead ends
    div = [t for t in parent if nch[t] >= 1]
    div_not2 = np.mean([nch[t] != 2 for t in div]) if div else np.nan
    ended = [t for t in parent if span[t] and span[t][1] < f_last and t in obs]
    dead = [nch[t] == 0 and obs[t][max(obs[t])][4] for t in ended]
    # (b) orphans born inside the colony
    born = [t for t in parent if span[t] and span[t][0] > f_first and t in obs]
    orphan = [parent[t] == 0 and obs[t][min(obs[t])][4] for t in born]

    # (c) length / area conservation at clean two-daughter divisions; (d) link displacement
    len_ratio, area_ratio = {}, {}
    for m in div:
        if nch[m] != 2 or m not in obs:
            continue
        tm = max(obs[m])
        d = [obs.get(k, {}).get(tm + 1) for k in children[m]]
        if any(x is None or min(obs[k]) != tm + 1 for x, k in zip(d, children[m])):
            continue
        Lm, Am = obs[m][tm][2], obs[m][tm][3]
        if np.isfinite(Lm) and Lm > 0 and all(np.isfinite(x[2]) for x in d):
            len_ratio[m] = (d[0][2] + d[1][2]) / Lm
        if np.isfinite(Am) and Am > 0 and all(np.isfinite(x[3]) for x in d):
            area_ratio[m] = (d[0][3] + d[1][3]) / Am
    len_flag = {m: abs(r - 1) > DEV_TOL for m, r in len_ratio.items()}
    area_flag = {m: abs(r - 1) > DEV_TOL for m, r in area_ratio.items()}
    # post hoc (see Results): divide by the colony's median one-frame within-track length growth
    growth = float(np.nanmedian([o[f + 1][2] / o[f][2] for o in obs.values() for f in o
                                 if f + 1 in o and o[f][2] > 0]))
    len_flag_gc = {m: abs(r / growth - 1) > DEV_TOL for m, r in len_ratio.items()}
    # (d) raw displacement; post hoc: residual after the transition's least-squares affine motion
    # (removes stage drift and colony expansion) and within-track one-frame length change vs growth
    pairs = {}
    for t, o in obs.items():
        for f in o:
            if f + 1 in o:
                pairs.setdefault(f, []).append((o[f], o[f + 1]))
    motion = {}
    for f, pq in pairs.items():
        P = np.array([[x[0], x[1], 1.0] for x, _ in pq])
        Q = np.array([[y[0], y[1]] for _, y in pq])
        motion[f] = (np.linalg.lstsq(P, Q, rcond=None)[0] if len(pq) >= 4
                     else np.vstack([np.eye(2), np.median(Q - P[:, :2], 0)]))
    dist = lambda a, b: np.hypot(a[0] - b[0], a[1] - b[1]) / med_len  # noqa: E731
    resid = lambda f, a, b: (dist(np.array([a[0], a[1], 1.0]) @ motion[f], b)  # noqa: E731
                             if f in motion else dist(a, b))
    lenjump = lambda a, b: abs(b[2] / a[2] / growth - 1) > DEV_TOL if a[2] > 0 else False  # noqa: E731
    raw_flags, post_flags = {}, {}
    n_links = n_jumps = n_res = n_within = n_lenjump = 0
    for t, o in obs.items():
        fr, fp = [], []
        for f in sorted(o):
            if f + 1 in o:
                n_links += 1
                n_within += 1
                rj, sj, lj = dist(o[f], o[f + 1]) > 1, resid(f, o[f], o[f + 1]) > 1, lenjump(o[f], o[f + 1])
                n_jumps += rj
                n_res += sj
                n_lenjump += lj
                if rj:
                    fr.append(f)
                if sj or lj:
                    fp.append(f)
        raw_flags[t], post_flags[t] = np.array(fr, int), np.array(fp, int)
    edge_pre, edge_post = {}, {}
    for m in div:
        for k in children[m]:
            rj = sj = False
            if m in obs and k in obs and min(obs[k]) == max(obs[m]) + 1:
                f = max(obs[m])
                n_links += 1
                rj, sj = dist(obs[m][f], obs[k][f + 1]) > 1, resid(f, obs[m][f], obs[k][f + 1]) > 1
                n_jumps += rj
                n_res += sj
            edge_pre[k] = nch[m] != 2 or len_flag.get(m, False) or rj
            edge_post[k] = nch[m] != 2 or len_flag_gc.get(m, False) or sj

    # (e) cell-cycle durations
    cyc = np.array([(span[t][1] - span[t][0] + 1) * mpf for t in parent if parent[t] and nch[t] >= 1 and span[t]])
    cmed = float(np.median(cyc)) if len(cyc) else np.nan

    # (f) exposure on the evaluation's windows and scored targets
    win = []
    for gap in gaps:
        for s in sorted(frames):
            e = s + gap
            if e not in frames or (c["skip_empty"] and (not len(frames[s]) or not len(frames[e]))):
                continue
            src = {int(t) for t in ids[s] if np.isfinite(t)}
            n, hit = 0, np.zeros(2)
            for t in ids[e]:
                if not np.isfinite(t):
                    continue
                anc = c["ancestor"](int(t), s)
                if anc not in src:
                    continue
                n += 1
                k, flags = int(t), np.zeros(2, bool)
                while True:
                    for i, lf in enumerate((raw_flags.get(k, NOFLAG), post_flags.get(k, NOFLAG))):
                        flags[i] |= bool(np.any((lf >= s) & (lf < e)))
                    if k == anc:
                        break
                    flags |= (edge_pre[k], edge_post[k])
                    k = parent[k]
                hit += flags
            if n:
                win.append((gap * mpf, n, *(hit / n)))
    w = pd.DataFrame(win, columns=["gap_minutes", "n_scored", "exposed", "exposed_post"])
    m = lambda x: float(np.mean(x)) if len(x) else np.nan  # noqa: E731
    row = {"dataset": dataset, "colony": colony, "n_frames": len(frames), "minutes_per_frame": mpf,
           "n_tracks": c["n_tracks"], "n_detections": n_det, "median_cell_length": med_len,
           "n_divisions": len(div), "div_not2": div_not2, "dead_end": m(dead), "n_born": len(born),
           "orphan_interior": m(orphan), "n_div_len": len(len_flag), "div_len_dev": m(list(len_flag.values())),
           "div_area_dev": m(list(area_flag.values())), "n_links": n_links, "link_jump": n_jumps / max(n_links, 1),
           "n_cycles": len(cyc), "cycle_median_min": cmed, "cycle_lt10": m(cyc < 10),
           "cycle_gt3med": m(cyc > 3 * cmed), "invalid_parent_frac": c["n_invalid"] / c["n_tracks"],
           "unmapped": n_unmapped / n_det, "n_windows": len(w), "n_scored": int(w.n_scored.sum()),
           "exposure": w.groupby("gap_minutes").exposed.mean().mean(),
           "one_frame_growth": growth, "div_len_ratio_med": float(np.median(list(len_ratio.values()))),
           "div_len_ratio_iqr": float(np.subtract(*np.percentile(list(len_ratio.values()), [75, 25]))),
           "div_len_dev_gc": m(list(len_flag_gc.values())),
           "link_jump_resid": n_res / max(n_links, 1), "within_len_dev": n_lenjump / max(n_within, 1),
           "exposure_post": w.groupby("gap_minutes").exposed_post.mean().mean()}
    return row, w.assign(dataset=dataset, colony=colony)


STATS = ["div_not2", "dead_end", "orphan_interior", "div_len_dev", "div_area_dev", "link_jump",
         "cycle_median_min", "cycle_lt10", "cycle_gt3med", "unmapped", "exposure",
         "invalid_parent_frac", "one_frame_growth", "div_len_ratio_med", "div_len_ratio_iqr",
         "div_len_dev_gc", "link_jump_resid", "within_len_dev", "exposure_post"]  # supplementary / post hoc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=int, default=8)
    a = p.parse_args()
    todo = [(d, c) for d, (cols, _, _) in DATASETS.items() for c in cols]
    out = Parallel(n_jobs=a.jobs, verbose=0)(delayed(audit_colony)(d, c) for d, c in todo)
    rows = pd.DataFrame([r for r, _ in out])
    wins = pd.concat([w for _, w in out], ignore_index=True)
    ref = pd.read_parquet(RESULTS / "raft_external_v1.parquet")
    ref = ref[ref.method == "nearest_anchor"].groupby("dataset")[["n_scored"]].agg(["sum", "count"])
    for d in DATASETS:
        got = (int(rows[rows.dataset == d].n_scored.sum()), int(rows[rows.dataset == d].n_windows.sum()))
        exp = (int(ref.loc[d, ("n_scored", "sum")]), int(ref.loc[d, ("n_scored", "count")]))
        print(f"check {d}: scored/windows {got} vs raft_external_v1 {exp}", "OK" if got == exp else "MISMATCH")
    ex = ["exposed", "exposed_post"]
    pooled = (wins.groupby(["dataset", "gap_minutes"])[ex].mean().groupby("dataset").mean()
              .rename(columns=lambda x: x.replace("exposed", "exposure")).reset_index().assign(colony="_pooled"))
    # post hoc: per-colony tab:external margin, RodSim model minus the colony's best geometric baseline
    v1 = pd.read_parquet(RESULTS / "raft_external_v1.parquet")
    acc = (v1.groupby(["dataset", "sequence", "gap_minutes", "method"]).accuracy.mean()
           .groupby(["dataset", "sequence", "method"]).mean().unstack())
    margin = (acc["raft_sim_loc_argmax"] - acc[GEO].max(axis=1)).rename("margin_vs_best_geo")
    rows = rows.merge(margin, left_on=["dataset", "colony"], right_index=True, how="left")
    res = pd.concat([rows, pooled], ignore_index=True)
    res.to_parquet(RESULTS / "schnitzcells_audit.parquet", index=False)
    g = rows.groupby("dataset")[STATS]
    tab = pd.DataFrame({d: [f"{g.mean().loc[d, s]:.3f} [{g.min().loc[d, s]:.3f}-{g.max().loc[d, s]:.3f}]"
                            if s != "cycle_median_min" else
                            f"{g.mean().loc[d, s]:.1f} [{g.min().loc[d, s]:.1f}-{g.max().loc[d, s]:.1f}]"
                            for s in STATS] for d in DATASETS}, index=STATS)
    for x in ("exposure", "exposure_post"):
        tab.loc[x + "_pooled"] = [f"{pooled.set_index('dataset')[x][d]:.3f}" for d in DATASETS]
    pd.set_option("display.width", 200)
    print(tab.to_string())
    for d in ("recA", "rpsM"):
        r = rows[rows.dataset == d]
        lo = r.exposure_post <= r.exposure_post.median()
        print(f"{d}: margin min {r.margin_vs_best_geo.min():.3f}; low-exposure half {r[lo].margin_vs_best_geo.mean():.3f} "
              f"(n={lo.sum()}), high half {r[~lo].margin_vs_best_geo.mean():.3f}; "
              f"spearman {r.margin_vs_best_geo.corr(r.exposure_post, method='spearman'):.2f}")
    print(rows.groupby("dataset")[["n_divisions", "n_born", "n_div_len", "n_links", "n_cycles", "n_scored"]].sum().to_string())


if __name__ == "__main__":
    main()
