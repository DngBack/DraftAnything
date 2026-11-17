"""Follow-up analyses A and B in docs/acl_prereg_gaps.md. Exploratory."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gap_language import PRED, build, t2

SPLITS = ["train", "val1", "confirm"]
SPECS = {
    "primary": PRED,
    "A_strict_clause": ["surprisal", "tree_dist", "clause_strict"],
    "B_context": ["surprisal", "surprisal_nc", "tree_dist", "clause_start"],
}


def pool(rows):
    w = 1 / rows.se.to_numpy() ** 2
    b = float((w * rows.beta_per_sd.to_numpy()).sum() / w.sum())
    se = float(np.sqrt(1 / w.sum()))
    return {"beta": b, "ci_lo": b - 1.96 * se, "ci_hi": b + 1.96 * se}


def main():
    g = build(["data/words_train.csv.gz", "data/words_val1.csv.gz", "data/words_confirm.csv.gz"], "data/ling/tokens.csv.gz")
    f = pd.read_csv("data/ling/followups.csv.gz").set_index(["line_id", "pos"])
    right = pd.MultiIndex.from_arrays([g.line_id, g.pos + 1])
    g["surprisal_nc"] = f.surprisal_nc.reindex(right).to_numpy()
    g["clause_strict"] = f.clause_strict.reindex(right).to_numpy()
    lower = (g.text_l.astype(str).str.fullmatch("[a-z]+") & g.text_r.astype(str).str.fullmatch("[a-z]+")).to_numpy()
    print("split labels", g.split.value_counts().to_dict())
    names = {"train": ["train"], "val1": ["val1"], "confirm": sorted(set(g.split) - {"train", "val1"})}
    if len(names["confirm"]) != 1:
        g.loc[g.split.isin(names["confirm"]), "split"] = "confirm"

    tabs = []
    for label, mask in [("all", None), ("lower", lower)]:
        for spec, pred in SPECS.items():
            for split in SPLITS:
                t, r2, _ = t2(g, split, "ridge", pred=pred, mask=mask, clusters=("writer",))
                tabs.append(t[t.model == "joint"].assign(spec=spec, subset=label, r2=r2))
    tab = pd.concat(tabs)
    out = Path("reports/acl")
    tab.to_csv(out / "followups.csv", index=False)

    summary = {}
    for (subset, spec, p), h in tab.groupby(["subset", "spec", "predictor"], sort=False):
        summary[f"{subset}|{spec}|{p}"] = {
            "all_three": pool(h),
            "val1_confirm": pool(h[h.split != "train"]),
            "by_split": {r.split: [round(r.beta_per_sd, 4), round(r.ci_lo, 4), round(r.ci_hi, 4)] for r in h.itertuples()},
        }
    (out / "followups.json").write_text(json.dumps(summary, indent=1))
    for k, v in summary.items():
        a, b = v["all_three"], v["val1_confirm"]
        print(f"{k:48s} pool {a['beta']:.4f} [{a['ci_lo']:.4f},{a['ci_hi']:.4f}]  v+c {b['beta']:.4f} [{b['ci_lo']:.4f},{b['ci_hi']:.4f}]  {v['by_split']}")
    d = g[(g.sent_l == g.sent_r)]
    print("corr surprisal, surprisal_nc:", d[["surprisal", "surprisal_nc"]].corr().iloc[0, 1])


if __name__ == "__main__":
    main()
