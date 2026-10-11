"""Can teacher reliability be judged without labels? (docs/teacher_reliability_protocol.md)

Per self-labelling unit (sequence/colony), label-free statistics U1..U6 of the one-frame teacher
(`raft_sim_loc` argmax, exactly as `self_label_adapt_ext.model_links`) against its ground-truth
one-frame link accuracy, and source means against the sign of (bootstrapped - Hungarian) adaptation.

  CUDA_VISIBLE_DEVICES=0 python experiments/teacher_reliability.py run
  python experiments/teacher_reliability.py summary
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments import blind_test4 as bt4  # noqa: E402
from experiments.blind_external_test import window as blind_window  # noqa: E402
from experiments.chained_pseudo_lineage import link_scaled  # noqa: E402
from experiments.family_set import RESULTS  # noqa: E402
from experiments.lineage_raft import featurize, load, predict  # noqa: E402
from experiments.raft_external_eval import bbox_shape, fast_window, reca_window  # noqa: E402
from experiments.self_label_adapt import all_movies  # noqa: E402
from experiments.self_label_adapt_ext import SETS, load_colony  # noqa: E402

OUT = RESULTS / "teacher_reliability.parquet"
GROUPS = {"ext": ("FAST", "recA", "rpsM"), "blind": ("TrackRefiner", "TrackMate", "LYN", "BSUB"),
          "bt4": ("trpL", "metA", "pheA")}
US = ["U1", "U2", "U3", "U4", "U5", "U6"]


class IdRng:
    """Identity permutation: GT windows built by the protocols' own builders, in original order."""

    def permutation(self, n):
        return np.arange(n)


def units():
    """(source, sequence, minutes per frame, yx frames, gt(a, b) -> (valid, truth))."""
    for source, name, mpf, frames, keys, anc in all_movies():
        def gt(a, b, frames=frames, keys=keys, anc=anc):
            _, v, t = blind_window(frames, keys, a, b, lambda o: anc(b, o, a), IdRng())
            return v, t
        yield source, name, mpf, frames, gt
    for name, held, base, _, mpf in SETS:
        for colony in held:
            frames, aux = load_colony(name, colony, base)
            yx = {t: f[:, ::-1] for t, f in frames.items()}

            def gt(a, b, frames=frames, aux=aux, name=name):
                fn = fast_window if name == "FAST" else reca_window
                _, v, t = fn(a, b - a, frames, *aux, IdRng())
                return v, t
            yield name, colony, mpf, yx, gt
    for subset in bt4.SUBSETS:
        for colony, (frames, mapping, tracks, invalid, _) in bt4.colonies(subset).items():
            yx = {t: f[:, ::-1] for t, f in frames.items()}

            def gt(a, b, frames=frames, aux=(mapping, tracks, invalid)):
                _, v, t = reca_window(a, b - a, frames, *aux, IdRng())
                return v, t
            yield subset, colony, bt4.MPF, yx, gt


def frame(p):
    n = len(p)
    return {"pos": p, "axis": np.zeros((n, 2)), "major": np.ones(n), "minor": np.ones(n)}


def unit_stats(model, mpf, yx, gt):
    shape = bbox_shape([f for f in yx.values() if len(f)])
    ks = sorted(yx)
    acc = {k: [] for k in ("top1", "ent", "u3", "u5", "u6", "u4", "tacc", "hacc")}
    n_pairs = 0
    for a, b in zip(ks[:-1], ks[1:]):
        pa, pb = yx[a], yx[b]
        if len(pa) < 2 or not len(pb):
            continue
        n_pairs += 1
        fa, fb = frame(pa), frame(pb)
        h = (b - a) * mpf
        prob, _, centre = predict(model, fa, fb, h, shape, no_shape=True)
        p = prob[:, :-1]
        p = p / np.maximum(p.sum(1, keepdims=True), 1e-9)
        t_link = prob[:, :-1].argmax(1)   # identical to model_links
        h_link = link_scaled(pa, pb)
        near = cKDTree(pa).query(pb)[1]
        cb = featurize(fa, fb, h, shape, None, None)[3].numpy()
        n_fam = np.bincount(t_link, minlength=len(pa))
        k2 = min(2, len(pb))
        nearest_t = cKDTree(cb).query(centre, k=k2)[1].reshape(len(pa), k2)
        cyc = np.array([j in nearest_t[t_link[j], :min(n_fam[t_link[j]], 2)] for j in range(len(pb))])
        acc["top1"].append(p.max(1))
        acc["ent"].append(-(p * np.log(np.where(p > 0, p, 1.0))).sum(1))
        acc["u3"].append(t_link == h_link)
        acc["u5"].append(t_link == near)
        acc["u6"].append(cyc)
        acc["u4"].append(n_fam > 0)
        valid, truth = gt(a, b)
        if len(truth):
            acc["tacc"].append(t_link[valid] == truth)
            acc["hacc"].append(h_link[valid] == truth)
    m = {k: (float(np.concatenate(v).mean()) if v else np.nan) for k, v in acc.items()}
    n_scored = int(sum(len(v) for v in acc["tacc"]))
    return {"n_pairs": n_pairs, "n_targets": int(sum(len(v) for v in acc["u3"])), "n_scored": n_scored,
            "U1": m["top1"], "U2": m["ent"], "U3": m["u3"], "U4": m["u4"], "U5": m["u5"], "U6": m["u6"],
            "teacher_acc": m["tacc"], "hungarian_acc": m["hacc"]}


def run():
    model = load("raft_sim_loc")[0]
    rows = []
    for source, seq, mpf, yx, gt in units():
        r = {"source": source, "sequence": seq, "mpf": mpf, **unit_stats(model, mpf, yx, gt)}
        rows.append(r)
        print(source, seq, r["n_pairs"], f"U3={r['U3']:.3f}", flush=True)
    d = pd.DataFrame(rows)
    d.to_parquet(OUT, index=False)
    summary(d)


def boot_rho(x, y, n=10000, seed=0):
    x, y = np.asarray(x), np.asarray(y)
    rho = spearmanr(x, y)[0]
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        r = spearmanr(x[i], y[i])[0]
        if np.isfinite(r):
            bs.append(r)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return rho, lo, hi


def outcome_b():
    def src_mean(d, gap):
        seq = d[d.method.isin(["adapt", "adaptm"])].groupby(["dataset", "sequence", gap, "method"]).accuracy.mean() \
            .groupby(["dataset", "sequence", "method"]).mean().unstack("method")
        return seq.groupby("dataset").mean()
    parts = [src_mean(pd.read_parquet(RESULTS / "self_label_adapt_ext2.parquet"), "gap"),
             src_mean(pd.read_parquet(RESULTS / "self_label_adapt_blind_boot.parquet"), "gap_minutes"),
             src_mean(pd.read_parquet(RESULTS / "blind_test4.parquet"), "gap")]
    s = pd.concat(parts)
    s["d"] = s.adaptm - s.adapt
    return s


def separation(v, win):
    """Min misclassified sources over thresholds/directions; perfect separation iff 0."""
    v, win = np.asarray(v, float), np.asarray(win, bool)
    cuts = np.concatenate([[-np.inf], np.sort(v) + 1e-12])
    best = (len(v), None, None)
    for c in cuts:
        for sign in (1, -1):
            pred = (v > c) if sign == 1 else (v <= c)
            err = int((pred != win).sum())
            if err < best[0]:
                best = (err, c, "high=win" if sign == 1 else "low=win")
    return best


def summary(d):
    pd.set_option("display.width", 200)
    print("\n== Sanity check (teacher / Hungarian one-frame accuracy) ==")
    for src in ("recA", "rpsM", "FAST"):
        g = d[d.source == src]
        w = g.n_scored
        print(f"{src}: teacher pooled {np.average(g.teacher_acc, weights=w):.3f} mean-over-colonies "
              f"{g.teacher_acc.mean():.3f} | Hungarian pooled {np.average(g.hungarian_acc, weights=w):.3f} "
              f"mean-over-colonies {g.hungarian_acc.mean():.3f} (n={len(g)})")
    a = d[d.n_scored > 0].copy()
    a["t_minus_h"] = a.teacher_acc - a.hungarian_acc
    print(f"\n== (a) Spearman rho with teacher_acc (units with scored targets: {len(a)} of {len(d)}) ==")
    for outcome in ("teacher_acc", "t_minus_h"):
        if outcome == "t_minus_h":
            print("\n-- secondary: rho with teacher_acc - hungarian_acc --")
        rows = []
        for gname, srcs in [("pooled", None), *GROUPS.items()]:
            g = a if srcs is None else a[a.source.isin(srcs)]
            for u in US:
                rho, lo, hi = boot_rho(g[u], g[outcome])
                rows.append({"group": gname, "n": len(g), "stat": u, "rho": rho, "ci_lo": lo, "ci_hi": hi})
        t = pd.DataFrame(rows).pivot_table(index="stat", columns="group", values=["rho", "ci_lo", "ci_hi"])
        for gname in ["pooled", *GROUPS]:
            n = int((a if gname == "pooled" else a[a.source.isin(GROUPS[gname])]).shape[0])
            print(f"[{gname}, n={n}] " + "  ".join(
                f"{u} {t.loc[u, ('rho', gname)]:+.2f} [{t.loc[u, ('ci_lo', gname)]:+.2f},{t.loc[u, ('ci_hi', gname)]:+.2f}]"
                for u in US))
    print("\n== Per-source means ==")
    b = outcome_b()
    a_all = d.copy()
    a_all["t_minus_h"] = a_all.teacher_acc - a_all.hungarian_acc
    s = a_all.groupby("source")[US + ["teacher_acc", "hungarian_acc", "t_minus_h"]].mean()
    s["n_units"] = d.groupby("source").size()
    s = s.join(b[["adapt", "adaptm", "d"]])
    s["boot_wins"] = s.d > 0
    print(s.sort_values("U3").round(3).to_string())
    print("\n== (b) single-threshold separation of winners/losers (10 sources) ==")
    for stat in ["U3", "U1", "U2", "U4", "U5", "U6", "teacher_acc", "t_minus_h"]:
        err, c, direction = separation(s[stat], s.boot_wins)
        tag = "PRIMARY" if stat == "U3" else ("label-using ref." if stat in ("teacher_acc", "t_minus_h") else "secondary")
        print(f"{stat:12s} ({tag}): separates={'yes' if err == 0 else 'no'}; min misclassified {err}/10 "
              f"({direction}; split: stat > {c:.4f} vs <= {c:.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "summary"])
    a = ap.parse_args()
    run() if a.mode == "run" else summary(pd.read_parquet(OUT))


if __name__ == "__main__":
    main()
