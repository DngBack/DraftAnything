"""Where does between-writer agreement (T1) come from? Exploratory diagnostic."""
import json

import numpy as np
import pandas as pd

from gap_language import MAX_GROUP, build, center, controls, crossfit, pair_corr

g = build(["data/words_train.csv.gz", "data/words_val1.csv.gz"], "data/ling/tokens.csv.gz")
is_fit = (g.split == "train").to_numpy()
d = g.assign(r=crossfit(g.gap.to_numpy(float), controls(g), is_fit, g.prompt.to_numpy(), "ridge"))
d = d[d.r.notna()].copy()
d["r"] = center(d.r.to_numpy(), d.line_id.to_numpy())
d["pair"] = d.text_l.astype(str).str.lower() + "|" + d.text_r.astype(str).str.lower()
d["occ"] = d.groupby(["form_id", "pair"]).cumcount()
d["bkey"] = d.prompt + "#" + d.pair + "#" + d.occ.astype(str)

rows = []
for (split, b), h in d.groupby(["split", "bkey"]):
    if h.writer_id.nunique() < 2:
        continue
    h = h.sample(min(len(h), MAX_GROUP), random_state=0)
    w, x, form, rp, last = h.writer_id.to_numpy(), h.r.to_numpy(), h.form_id.to_numpy(), h.rel_pos.to_numpy(), h.is_last.to_numpy()
    for i in range(len(h)):
        for j in range(i + 1, len(h)):
            if w[i] != w[j]:
                rows.append((split, h.prompt.iloc[0], h.text_l.iloc[0], h.text_r.iloc[0], x[i], x[j], abs(rp[i] - rp[j])))
p = pd.DataFrame(rows, columns=["split", "prompt", "l", "r", "a", "b", "dpos"])
p["group"] = p.prompt.str[0]
p["set3"] = p.prompt.str[:3]
print("pairs", len(p), "prompts", p.prompt.nunique())


def corr(x):
    return round(pair_corr(x.a.to_numpy(), x.b.to_numpy()), 3)


print("by first letter (train):")
print(p[p.split == "train"].groupby("group").apply(lambda x: pd.Series({"r": corr(x), "pairs": len(x), "prompts": x.prompt.nunique()})).to_string())
top = p[p.split == "train"].groupby("prompt").size().sort_values(ascending=False)
print("largest prompts:", top.head(8).to_dict())
for k in [0, 1, 3, 5, 10]:
    x = p[(p.split == "train") & ~p.prompt.isin(top.index[:k])]
    print("drop top", k, "r", corr(x), "pairs", len(x))
print("per-prompt r (prompts with >=200 pairs):")
q = p[p.split == "train"].groupby("prompt").filter(lambda x: len(x) >= 200).groupby("prompt").apply(lambda x: pd.Series({"r": corr(x), "pairs": len(x)}))
print(q.sort_values("r").round(3).to_string())
print("r by |rel position difference| (train):", p[p.split == "train"].groupby(pd.cut(p.dpos, [-.01, .1, .25, 1])).apply(corr).to_dict())
extreme = p[(p.split == "train")].assign(mag=lambda x: (x.a * x.b).abs()).sort_values("mag", ascending=False).head(12)
print(extreme[["prompt", "l", "r", "a", "b"]].round(2).to_string(index=False))
pv = p[p.split == "train"].assign(a=lambda x: x.a.clip(-2.5, 2.5), b=lambda x: x.b.clip(-2.5, 2.5))
print("winsorised at +/-2.5:", corr(pv))
print("spearman:", round(p[p.split == "train"][["a", "b"]].corr("spearman").iloc[0, 1], 3))
tr = p[p.split == "train"]
g06 = tr.prompt.str.startswith("g06")
print("g06 prompts", tr[g06].prompt.nunique(), "pairs", int(g06.sum()), "share", round(g06.mean(), 3), "r", corr(tr[g06]), "without g06: r", corr(tr[~g06]), "pairs", int((~g06).sum()), "prompts", tr[~g06].prompt.nunique())
big = tr[g06].groupby("prompt").size()
print("g06 prompts >=200 pairs", int((big >= 200).sum()), "share of all pairs", round(big[big >= 200].sum() / len(tr), 3))
v = p[p.split == "val1"]
print("VAL1 groups", v.groupby("group").apply(lambda x: (corr(x), len(x), x.prompt.nunique())).to_dict())
