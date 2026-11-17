"""Exploratory: does the printed prompt's spacing explain the gap-language findings?"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gap_language import MAX_GROUP, build, center, controls, crossfit, pair_corr, t2

OUT = Path("reports/acl")


def attach(g):
    a = pd.read_csv("data/ling/printed_gaps.csv.gz", keep_default_na=False, na_values=[""])
    a = a[(a.print_gap_rel > 0.3) & (a.print_gap_rel < 2.5)]
    c = a.groupby(["prompt", "text_l", "text_r"]).size().rename("n").reset_index()
    u = a.merge(c[c.n == 1], on=["prompt", "text_l", "text_r"])
    u = u.drop_duplicates(["prompt", "text_l", "text_r"]).set_index(["prompt", "text_l", "text_r"])
    idx = pd.MultiIndex.from_arrays([g.prompt, g.text_l.astype(str), g.text_r.astype(str)])
    g["print_rel"] = u.print_gap_rel.reindex(idx).to_numpy()
    g["print_px"] = u.print_gap_px.reindex(idx).to_numpy()
    return g


def agreement(d, r):
    d = d.assign(r=r)
    d = d[d.r.notna()].copy()
    d["r"] = center(d.r.to_numpy(), d.line_id.to_numpy())
    d["pair"] = d.text_l.astype(str).str.lower() + "|" + d.text_r.astype(str).str.lower()
    d["occ"] = d.groupby(["form_id", "pair"]).cumcount()
    d["bkey"] = d.prompt + "#" + d.pair + "#" + d.occ.astype(str)
    A, B, P = [], [], []
    for _, h in d.groupby("bkey"):
        if h.writer_id.nunique() < 2:
            continue
        h = h.sample(min(len(h), MAX_GROUP), random_state=0)
        w, x, p = h.writer_id.to_numpy(), h.r.to_numpy(), h.prompt.to_numpy()
        for i in range(len(h)):
            for j in range(i + 1, len(h)):
                if w[i] != w[j]:
                    A.append(x[i]); B.append(x[j]); P.append(p[i])
    A, B, P = np.array(A), np.array(B), np.array(P)
    rng = np.random.default_rng(0)
    up, inv = np.unique(P, return_inverse=True)
    idx = [np.nonzero(inv == k)[0] for k in range(len(up))]
    v = []
    for _ in range(1000):
        s = np.concatenate([idx[k] for k in rng.integers(0, len(up), len(up))])
        v.append(pair_corr(A[s], B[s]))
    return {"r": pair_corr(A, B), "ci": [float(x) for x in np.percentile(v, [2.5, 97.5])], "n_pairs": len(A), "n_prompts": len(up)}


def main():
    g = attach(build(["data/words_train.csv.gz", "data/words_val1.csv.gz", "data/words_confirm.csv.gz"], "data/ling/tokens.csv.gz"))
    has = g.print_rel.notna().to_numpy()
    print("boundaries with printed gap by set:", g[has].groupby("split").size().to_dict(), "of", g.groupby("split").size().to_dict())
    extra = np.column_stack([g.print_rel.fillna(1).to_numpy(), g.print_px.fillna(g.print_px.median()).to_numpy() / 20])
    res = {}

    # Slope of the handwritten gap on the printed relative gap, in the same pipeline.
    g["print_rel_f"] = g.print_rel
    rows = []
    for split in [s for s in ["train", "val1", "confirm"] if g.loc[has, "split"].eq(s).sum() > 100]:
        t, _, _ = t2(g, split, "ridge", pred=["print_rel_f"], mask=has)
        rows.append(t[(t.model == "joint") & (t.cluster == "writer")].assign(spec="printed_alone"))
        for name, kw in [("without_printed", {"mask": has}), ("with_printed", {"mask": has, "extra": extra})]:
            t, _, _ = t2(g, split, "ridge", **kw)
            rows.append(t[(t.model == "joint") & (t.cluster == "writer")].assign(spec=name))
    tab = pd.concat(rows)
    tab.to_csv(OUT / "printed_control.csv", index=False)
    print(tab[["spec", "split", "predictor", "beta_per_sd", "ci_lo", "ci_hi", "n"]].round(4).to_string(index=False))
    pooled = {}
    for (sp, pr), h in tab.groupby(["spec", "predictor"], sort=False):
        w = 1 / h.se ** 2
        m, se = (w * h.beta_per_sd).sum() / w.sum(), w.sum() ** -0.5
        pooled[f"{sp}|{pr}"] = {"beta": float(m), "ci": [float(m - 1.96 * se), float(m + 1.96 * se)]}
    res["pooled"] = pooled
    print(json.dumps(pooled, indent=0))

    # Agreement between writers on Discovery + Replication boundaries that have a printed gap.
    is_fit = (g.split == "train").to_numpy()
    y = np.where(has, g.gap.to_numpy(float), np.nan)
    Xbase = controls(g)
    sel = g.split.isin(["train", "val1"]).to_numpy()
    r0 = crossfit(y, Xbase, is_fit, g.prompt.to_numpy(), "ridge")
    r1 = crossfit(y, np.hstack([Xbase, extra]), is_fit, g.prompt.to_numpy(), "ridge")
    res["agreement_no_printed"] = agreement(g[sel], r0[sel])
    res["agreement_with_printed"] = agreement(g[sel], r1[sel])
    print(json.dumps({k: v for k, v in res.items() if k.startswith("agreement")}, indent=1))
    (OUT / "printed_control.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
