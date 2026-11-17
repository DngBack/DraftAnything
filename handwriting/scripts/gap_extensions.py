"""Analyses added after the confirmation run. All are exploratory and labelled so in the paper.

- T1 across sets: sealed writers paired with other writers of the same prompt.
- Inverse-variance pooling of the three independent writer sets.
- Placebo effect expressed in outcome SD, so it is comparable with the gap effect.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gap_language import MAX_GROUP, build, center, controls, crossfit, pair_corr, t2

OUT = Path("reports/acl")


def residuals(g):
    is_fit = (g.split == "train").to_numpy()
    r = crossfit(g.gap.to_numpy(float), controls(g), is_fit, g.prompt.to_numpy(), "ridge")
    d = g.assign(r=r)
    d = d[d.r.notna()].copy()
    d["r"] = center(d.r.to_numpy(), d.line_id.to_numpy())
    d["pair"] = d.text_l.astype(str).str.lower() + "|" + d.text_r.astype(str).str.lower()
    d["occ"] = d.groupby(["form_id", "pair"]).cumcount()
    d["bkey"] = d.prompt + "#" + d.pair + "#" + d.occ.astype(str)
    d["is_c"] = d.split == "confirm"
    return d


def cross_pairs(d, key, other_prompt, seed):
    A, B, P = [], [], []
    for _, h in d.groupby(key):
        if h.is_c.sum() == 0 or (~h.is_c).sum() == 0:
            continue
        if len(h) > MAX_GROUP:
            h = h.sample(MAX_GROUP, random_state=seed)
        c, o = h[h.is_c], h[~h.is_c]
        for rc, wc, pc in zip(c.r, c.writer_id, c.prompt):
            for ro, wo, po in zip(o.r, o.writer_id, o.prompt):
                if wc != wo and (not other_prompt or pc != po) :
                    A.append(rc); B.append(ro); P.append(pc)
    return np.array(A), np.array(B), np.array(P)


def boot(a, b, p, rng, n=1000):
    up, inv = np.unique(p, return_inverse=True)
    idx = [np.nonzero(inv == k)[0] for k in range(len(up))]
    v = [pair_corr(a[s], b[s]) for s in (np.concatenate([idx[k] for k in rng.integers(0, len(up), len(up))]) for _ in range(n))]
    return [float(x) for x in np.percentile(v, [2.5, 97.5])], len(up)


def main():
    g = build(["data/words_train.csv.gz", "data/words_val1.csv.gz", "data/words_confirm.csv.gz"], "data/ling/tokens.csv.gz")
    d = residuals(g)
    rng = np.random.default_rng(0)
    res = {}
    pc = set(d[d.is_c].prompt) & set(d[~d.is_c].prompt)
    res["prompts_shared_between_confirm_and_other_writers"] = len(pc)
    print("prompts shared with sealed writers:", len(pc))

    def pool(f):
        rows = []
        for split, name in [("train", "robust_train.csv"), ("val1", "robust_val1.csv"), ("confirm", "robust_confirm.csv")]:
            t = pd.read_csv(OUT / name)
            rows.append(t[(t.cluster == "writer") & (t.spec == f)].assign(split=split))
        t = pd.concat(rows)
        out = {}
        for pred, h in t.groupby("predictor", sort=False):
            w = 1 / h.se ** 2
            m, se = (w * h.beta_per_sd).sum() / w.sum(), w.sum() ** -0.5
            out[pred] = {"beta": float(m), "ci": [float(m - 1.96 * se), float(m + 1.96 * se)]}
        return out

    res["pooled"] = {s: pool(s) for s in ["primary", "lower", "pos", "pythia", "gbm", "clip"]}
    print(json.dumps(res["pooled"], indent=1))
    # Placebo scale: effect per SD of predictor in within-line SD units of each outcome.
    for split in ["train", "val1", "confirm"]:
        sub = d[d.split == split]
        for o in ["gap", "bg"]:
            s = (sub[o] - sub.groupby("line_id")[o].transform("mean")).std()
            res[f"within_line_sd_{o}_{split}"] = float(s)
    print({k: round(v, 3) for k, v in res.items() if k.startswith("within")})
    (OUT / "extensions.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
