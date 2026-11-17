"""Do inter-word gaps carry linguistic structure? (docs/acl_prereg_gaps.md)

T1: between-writer agreement of residual gaps at the same boundary of the same text.
T2: partial effects of surprisal, parse-tree distance and clause onset on the gap.

Controls are removed from BOTH outcome and predictor (cross-fitted by prompt), then
both are centred within line, which is the partialling-out form of a line fixed-effects OLS.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

from analyze_line_dependence import GAP_SIDE, gap_frame, load

PRED = ["surprisal", "tree_dist", "clause_start"]
EDGE = list("abcdefghijklmnopqrstuvwxyz0123456789")
MAX_GROUP = 60


def build(paths, ling_path):
    words = load(paths)
    words["prompt"] = words.form_id.str.replace(r"[a-z]$", "", regex=True)
    g = gap_frame(words)
    w = words.set_index(["line_id", "pos"])
    g["text_l"] = w.text.reindex(pd.MultiIndex.from_arrays([g.line_id, g.pos])).to_numpy()
    g["text_r"] = w.text.reindex(pd.MultiIndex.from_arrays([g.line_id, g.pos + 1])).to_numpy()
    g["ink_x0_r"] = w.ink_x0.reindex(pd.MultiIndex.from_arrays([g.line_id, g.pos + 1])).to_numpy()
    g["line_w"] = w.line_w.reindex(pd.MultiIndex.from_arrays([g.line_id, g.pos])).to_numpy()
    g["prompt"] = g.form_id.str.replace(r"[a-z]$", "", regex=True)
    g = g[~g.text_r.astype(str).str.startswith("'")].copy()
    g["remain"] = (g.line_w - g.ink_x0_r) / g.med_core

    tok = pd.read_csv(ling_path, dtype={"writer_id": str}, keep_default_na=False, na_values=[""]).set_index(["line_id", "pos"])
    g["text_l"] = tok.text.reindex(pd.MultiIndex.from_arrays([g.line_id, g.pos])).to_numpy()
    g["text_r"] = tok.text.reindex(pd.MultiIndex.from_arrays([g.line_id, g.pos + 1])).to_numpy()
    g = g[~g.text_r.astype(str).str.startswith("'")].copy()
    alt = Path("data/ling/surprisal_pythia.csv.gz")
    if alt.exists():
        a = pd.read_csv(alt).set_index(["line_id", "pos"]).surprisal_alt
        g["surprisal_alt"] = a.reindex(pd.MultiIndex.from_arrays([g.line_id, g.pos + 1])).to_numpy()
    g["surprisal"] = tok.surprisal.reindex(pd.MultiIndex.from_arrays([g.line_id, g.pos + 1])).to_numpy()
    g["clause_start"] = tok.clause_start.reindex(pd.MultiIndex.from_arrays([g.line_id, g.pos + 1])).to_numpy()
    g["tree_dist"] = tok.tree_dist_next.reindex(pd.MultiIndex.from_arrays([g.line_id, g.pos])).to_numpy()
    g["sent_l"] = tok.sent.reindex(pd.MultiIndex.from_arrays([g.line_id, g.pos])).to_numpy()
    g["sent_r"] = tok.sent.reindex(pd.MultiIndex.from_arrays([g.line_id, g.pos + 1])).to_numpy()
    g["lob_l"] = tok.lob_tag.reindex(pd.MultiIndex.from_arrays([g.line_id, g.pos])).to_numpy()
    g["lob_r"] = tok.lob_tag.reindex(pd.MultiIndex.from_arrays([g.line_id, g.pos + 1])).to_numpy()

    # Frequencies come from Discovery text only, and are smoothed for unseen words.
    vc = pd.concat([words[words.split == "train"].text]).astype(str).str.lower().value_counts()
    total = vc.sum()
    lf = lambda s: np.log((s.astype(str).str.lower().map(vc).fillna(0) + 1) / total)
    g["l_logf"], g["r_logf"] = lf(g.text_l), lf(g.text_r)
    last = g.text_l.astype(str).str[-1].str.lower()
    first = g.text_r.astype(str).str[0].str.lower()
    g["last_c"], g["first_c"] = last, first
    for c in EDGE:
        g[f"e_l_{c}"] = (last == c).astype(float)
        g[f"e_r_{c}"] = (first == c).astype(float)
    return g.reset_index(drop=True)


def controls(g):
    cols = [f"{s}_{c}" for s in "lr" for c in GAP_SIDE] + ["l_logf", "r_logf", "rel_pos", "is_first", "is_last",
                                                           "n_tokens", "x_c", "remain"]
    cols += [f"e_{s}_{c}" for s in "lr" for c in EDGE]
    return g[cols].fillna(0).to_numpy(float)


def model(kind):
    if kind == "ridge":
        return RidgeCV(alphas=np.logspace(-2, 3, 12))
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, min_samples_leaf=40, random_state=0)


def crossfit(y, X, is_fit, groups, kind):
    """Train rows: out-of-fold by prompt. Other rows: model fit on all train rows."""
    ok = np.isfinite(y)
    res = np.full(y.shape, np.nan)
    fit, oth = np.nonzero(is_fit & ok)[0], np.nonzero(~is_fit & ok)[0]
    for tr, te in GroupKFold(5).split(fit, groups=groups[fit]):
        res[fit[te]] = y[fit[te]] - model(kind).fit(X[fit[tr]], y[fit[tr]]).predict(X[fit[te]])
    if len(oth):
        res[oth] = y[oth] - model(kind).fit(X[fit], y[fit]).predict(X[oth])
    return res


def center(v, lines):
    return v - pd.Series(v).groupby(lines).transform("mean").to_numpy()


def cluster_ols(y, X, cl):
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    uc, inv = np.unique(cl, return_inverse=True)
    S = np.zeros((X.shape[1], X.shape[1]))
    scores = np.zeros((len(uc), X.shape[1]))
    np.add.at(scores, inv, X * e[:, None])
    S = scores.T @ scores
    G = len(uc)
    V = XtX_inv @ S @ XtX_inv * G / (G - 1)
    return b, np.sqrt(np.diag(V))


def holm(ps):
    order = np.argsort(ps)
    adj = np.zeros(len(ps))
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def t2(g, split, kind, pred=None, extra=None, mask=None, outcome="gap", clusters=("writer", "prompt")):
    pred = PRED if pred is None else pred
    is_fit = (g.split == "train").to_numpy()
    groups = g.prompt.to_numpy()
    X = controls(g) if extra is None else np.hstack([controls(g), extra])
    ok = np.isfinite(g[pred + [outcome]].to_numpy(float)).all(1) & (g.sent_l == g.sent_r).to_numpy()
    if mask is not None:
        ok &= mask
    sd = g.loc[is_fit & ok, pred].std().to_numpy()
    y = np.where(ok, g[outcome].to_numpy(float), np.nan)
    ry = crossfit(y, X, is_fit, groups, kind)
    rx = np.column_stack([crossfit(np.where(ok, g[p].to_numpy(float), np.nan) / s, X, is_fit, groups, kind)
                          for p, s in zip(pred, sd)])
    sel = (g.split == split).to_numpy() & np.isfinite(ry)
    lines = g.line_id.to_numpy()[sel]
    cy = center(ry[sel], lines)
    cx = np.column_stack([center(rx[sel, j], lines) for j in range(len(pred))])
    rows = []
    for name, idx in [("joint", list(range(len(pred))))] + [("alone", [j]) for j in range(len(pred))]:
        for cname in clusters:
            cl = g[f"{cname}_id" if cname == "writer" else cname].to_numpy()[sel]
            b, se = cluster_ols(cy, cx[:, idx], cl)
            for k, j in enumerate(idx):
                z = b[k] / se[k]
                rows.append({"split": split, "kind": kind, "model": name, "cluster": cname, "predictor": pred[j],
                             "beta_per_sd": b[k], "se": se[k], "ci_lo": b[k] - 1.96 * se[k], "ci_hi": b[k] + 1.96 * se[k],
                             "p": 2 * norm.sf(abs(z)), "n": int(sel.sum()), "n_writers": len(np.unique(g.writer_id.to_numpy()[sel]))})
    out = pd.DataFrame(rows)
    for (m, c), h in out[out.model == "joint"].groupby(["model", "cluster"]):
        out.loc[h.index, "p_holm"] = holm(h.p.to_numpy())
    r2 = 1 - ((cy - cx @ np.linalg.lstsq(cx, cy, rcond=None)[0]) ** 2).sum() / (cy ** 2).sum()
    return out, r2, float(cy.std())


def tag_matrix(g):
    fit = g.split == "train"
    M = []
    for col in ["lob_l", "lob_r"]:
        top = g.loc[fit, col].value_counts().index[:30]
        M += [(g[col] == t).to_numpy(float) for t in top]
    return np.column_stack(M)


def robustness(g, split):
    lower = g.text_l.astype(str).str.fullmatch("[a-z]+") & g.text_r.astype(str).str.fullmatch("[a-z]+")
    lo, hi = g.loc[g.split == "train", "gap"].quantile([0.01, 0.99])
    g = g.assign(gap_clip=g.gap.clip(lo, hi))
    specs = {"primary": {}, "lower": {"mask": lower.to_numpy()}, "pos": {"extra": tag_matrix(g)},
             "clip": {"outcome": "gap_clip"}, "placebo_bg": {"outcome": "bg"}}
    rows = []
    for name, kw in specs.items():
        t, _, _ = t2(g, split, "ridge", **kw)
        rows.append(t[(t.model == "joint")].assign(spec=name))
    if "surprisal_alt" in g:
        t, _, _ = t2(g, split, "ridge", pred=["surprisal_alt", "tree_dist", "clause_start"])
        rows.append(t[t.model == "joint"].assign(spec="pythia"))
    t, _, _ = t2(g, split, "gbm")
    rows.append(t[t.model == "joint"].assign(spec="gbm"))
    return pd.concat(rows)


def heterogeneity(g, split, min_n=40):
    is_fit = (g.split == "train").to_numpy()
    X, groups = controls(g), g.prompt.to_numpy()
    ok = np.isfinite(g[PRED + ["gap"]].to_numpy(float)).all(1) & (g.sent_l == g.sent_r).to_numpy()
    ry = crossfit(np.where(ok, g.gap.to_numpy(float), np.nan), X, is_fit, groups, "ridge")
    out = {}
    sel = (g.split == split).to_numpy() & np.isfinite(ry)
    d = g.loc[sel, ["writer_id", "line_id"]].assign(y=center(ry[sel], g.line_id.to_numpy()[sel]))
    for p in PRED:
        rx = crossfit(np.where(ok, g[p].to_numpy(float), np.nan), X, is_fit, groups, "ridge")[sel]
        d["x"] = center(rx, g.line_id.to_numpy()[sel])
        sl = [np.polyfit(h.x, h.y, 1)[0] for _, h in d.groupby("writer_id") if len(h) >= min_n]
        out[p] = {"frac_writers_positive": float(np.mean(np.array(sl) > 0)), "n_writers": len(sl)}
    return out


def dose(g, split):
    is_fit = (g.split == "train").to_numpy()
    X, groups = controls(g), g.prompt.to_numpy()
    ok = np.isfinite(g[PRED + ["gap"]].to_numpy(float)).all(1) & (g.sent_l == g.sent_r).to_numpy()
    ry = crossfit(np.where(ok, g.gap.to_numpy(float), np.nan), X, is_fit, groups, "ridge")
    sel = (g.split == split).to_numpy() & np.isfinite(ry)
    d = g.loc[sel, PRED + ["writer_id", "line_id"]].assign(y=center(ry[sel], g.line_id.to_numpy()[sel]))
    rows = []
    d["sb"] = pd.qcut(d.surprisal, 10, labels=False, duplicates="drop")
    d["tb"] = d.tree_dist.clip(upper=10)
    for pname, col in [("surprisal", "sb"), ("tree_dist", "tb"), ("clause_start", "clause_start")]:
        for b, h in d.groupby(col):
            rows.append({"predictor": pname, "bin": float(b), "x_mean": float(h[pname].mean()), "y_mean": float(h.y.mean()),
                         "se": float(h.groupby("writer_id").y.mean().sem()), "n": len(h)})
    return pd.DataFrame(rows)


def pair_corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else np.nan


def t1(g, split, kind, n_boot, seed):
    rng = np.random.default_rng(seed)
    is_fit = (g.split == "train").to_numpy()
    ry = crossfit(g.gap.to_numpy(float), controls(g), is_fit, g.prompt.to_numpy(), kind)
    d = g.assign(r=ry)
    d = d[(d.split == split) & d.r.notna()].copy()
    d["r"] = center(d.r.to_numpy(), d.line_id.to_numpy())
    d["pair"] = d.text_l.astype(str).str.lower() + "|" + d.text_r.astype(str).str.lower()
    d["occ"] = d.groupby(["form_id", "pair"]).cumcount()
    d["bkey"] = d.prompt + "#" + d.pair + "#" + d.occ.astype(str)

    def pairs(key_col, other_prompt):
        A, B, C = [], [], []
        for _, h in d.groupby(key_col):
            if h.writer_id.nunique() < 2:
                continue
            if len(h) > MAX_GROUP:   # very common pairs ("of the") would otherwise dominate and exhaust memory
                h = h.sample(MAX_GROUP, random_state=seed)
            w, r, p = h.writer_id.to_numpy(), h.r.to_numpy(), h.prompt.to_numpy()
            for i in range(len(h)):
                for j in range(i + 1, len(h)):
                    if w[i] == w[j] or (other_prompt and p[i] == p[j]):
                        continue
                    A.append(r[i]); B.append(r[j]); C.append(p[i])
        return np.array(A), np.array(B), np.array(C)

    def boot(a, b, c):
        up, inv = np.unique(c, return_inverse=True)
        idx = [np.nonzero(inv == k)[0] for k in range(len(up))]
        vals = []
        for _ in range(n_boot):
            s = np.concatenate([idx[k] for k in rng.integers(0, len(up), len(up))])
            vals.append(pair_corr(a[s], b[s]))
        return [float(x) for x in np.percentile(vals, [2.5, 97.5])]

    out = {"split": split, "kind": kind, "gap_res_sd": float(d.r.std())}
    keep = d.groupby("bkey").writer_id.transform("nunique") >= 2
    a, b, c = pairs("bkey", False)
    out["same_boundary"] = {"r": pair_corr(a, b), "ci": boot(a, b, c), "n_pairs": len(a), "n_boundaries": int(d[keep].bkey.nunique())}

    # Same boundaries, but paired with the same word pair in a DIFFERENT prompt (other sentence).
    shared = set(d.loc[keep, "pair"])
    sub = d[d.pair.isin(shared)]
    d0 = d
    d = sub
    a, b, c = pairs("pair", True)
    d = d0
    if len(a) > 400000:
        s = rng.choice(len(a), 400000, replace=False); a, b, c = a[s], b[s], c[s]
    out["same_pair_other_prompt"] = {"r": pair_corr(a, b), "ci": boot(a, b, c), "n_pairs": len(a)}

    # Null: shuffle boundary identity within prompt, among writers of that prompt.
    nulls = []
    k = d[keep].copy()
    for _ in range(200):
        perm = k.groupby(["form_id"]).r.transform(lambda s: rng.permutation(s.to_numpy()))
        k2 = k.assign(r=perm.to_numpy())
        xa, xb = [], []
        for _, h in k2.groupby("bkey"):
            w, r = h.writer_id.to_numpy(), h.r.to_numpy()
            for i in range(len(h)):
                for j in range(i + 1, len(h)):
                    if w[i] != w[j]:
                        xa.append(r[i]); xb.append(r[j])
        nulls.append(pair_corr(np.array(xa), np.array(xb)))
    out["null_within_form_shuffle"] = {"mean": float(np.mean(nulls)), "p2.5_97.5": [float(x) for x in np.percentile(nulls, [2.5, 97.5])]}
    return out


def validate(g):
    """Pipeline checks that use no real gap-language relation.

    The outcome is a shuffled gap, so any linguistic signal is destroyed. A known effect is
    then planted: the estimator should return it. With none planted it should return 0."""
    rng = np.random.default_rng(1)
    base = g.copy()
    base["gap"] = base.groupby("line_id").gap.transform(lambda s: rng.permutation(s.to_numpy()))
    sd = base.loc[base.split == "train", PRED].std()
    for label, plant in [("no effect planted", {}),
                         ("planted surprisal +0.05", {"surprisal": 0.05}),
                         ("planted tree_dist +0.03", {"tree_dist": 0.03})]:
        d = base.copy()
        for p, b in plant.items():
            d["gap"] = d.gap + b * d[p].fillna(0) / sd[p]
        tab, _, _ = t2(d, "train", "ridge")
        t = tab[(tab.model == "joint") & (tab.cluster == "writer")]
        print(label, {r.predictor: (round(r.beta_per_sd, 4), round(r.ci_lo, 4), round(r.ci_hi, 4)) for r in t.itertuples()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train"])
    ap.add_argument("--out", type=Path, default=Path("reports/acl"))
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--kinds", nargs="+", default=["ridge", "gbm"])
    ap.add_argument("--skip-t1", action="store_true")
    ap.add_argument("--validate", action="store_true", help="planted-effect and shuffled-predictor checks only")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    paths = ["data/words_train.csv.gz", "data/words_val1.csv.gz"]
    if "confirm" in a.splits:   # sealed writers: only loaded for the single confirmation run
        paths.append("data/words_confirm.csv.gz")
    g = build(paths, "data/ling/tokens.csv.gz")
    print("boundaries", g.groupby("split").size().to_dict())
    print(g.gap.describe(percentiles=[.001, .01, .5, .99, .999]).round(3).to_dict())
    pd.set_option("display.width", 220, "display.max_columns", 30)
    if a.validate:
        validate(g)
        return
    tabs, res = [], []
    for kind in a.kinds:
        for split in a.splits:
            tab, r2, sd = t2(g, split, kind)
            tabs.append(tab)
            print(f"\n== T2 {split} {kind}: within-line R2 of the three predictors {r2:.5f}; residual gap SD {sd:.3f}")
            print(tab[(tab.cluster == "writer")].drop(columns=["cluster", "kind", "split"]).round(4).to_string(index=False))
            if kind == "ridge":
                rob = robustness(g, split)
                rob.to_csv(a.out / f"robust_{split}.csv", index=False)
                print(rob[(rob.cluster == "writer")][["spec", "predictor", "beta_per_sd", "ci_lo", "ci_hi", "p_holm", "n"]].round(4).to_string(index=False))
                print(rob[(rob.cluster == "prompt") & (rob.spec == "primary")][["predictor", "beta_per_sd", "ci_lo", "ci_hi"]].round(4).to_string(index=False))
                h = heterogeneity(g, split)
                print(json.dumps(h))
                (a.out / f"hetero_{split}.json").write_text(json.dumps(h))
                dose(g, split).to_csv(a.out / f"dose_{split}.csv", index=False)
            if not a.skip_t1:
                o = t1(g, split, kind, a.n_boot, 0)
                res.append(o)
                print(json.dumps(o, indent=1))
    pd.concat(tabs).to_csv(a.out / f"t2_{'_'.join(a.splits)}.csv", index=False)
    (a.out / f"t1_{'_'.join(a.splits)}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
