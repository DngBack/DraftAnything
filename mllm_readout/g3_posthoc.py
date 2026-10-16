"""G3 post-hoc checks (not pre-registered; docs/g3_report.md):
(a) loss errors the prompt alone could explain (text-only control right) -> image-attributable loss;
(b) best probe restricted to the last (answer) position vs image-token mean;
(c) [illusions] a linear probe on 32x32 grey pixels, same grouped folds."""
import glob, os, sys
import numpy as np, pandas as pd
from pilot import probe_cv, OUT
G3 = f"{OUT}/g3"
rows = []
for p in sorted(glob.glob(f"{G3}/*_states.npz")):
    name = os.path.basename(p)[:-11]
    if not any(name.endswith(s) for s in ("illusions", "pope", "countbench", "mmvp", "tally")):
        continue
    d = np.load(p)
    y, ans, scores, grp, key = d["y"], d["ans"], d["scores"], d["group"], d["key"]
    err = ans != y
    cal = np.empty_like(y)
    for g in np.unique(grp):
        k = grp == g
        cal[k] = probe_cv(scores[k], y[k], groups=key[k])[0] if len(np.unique(y[k])) > 1 else ans[k]
    u, inv = np.unique(key, return_inverse=True)
    fold = (np.random.default_rng(0).permutation(len(u)) % 5)[inv]
    text = np.empty_like(y)
    for f in range(5):
        tr = pd.DataFrame(dict(q=d["q"][fold != f], y=y[fold != f]))
        mp, maj = tr.groupby("q").y.agg(lambda v: v.mode()[0]), tr.y.mode()[0]
        text[fold == f] = [mp.get(q, maj) for q in d["q"][fold == f]]
    r = dict(cell=name, n_err=int(err.sum()))
    best = {}
    for pos in ("last", "imgm"):
        pr, _ = probe_cv(d[pos], y, groups=key)
        acc = (pr == y).mean(1); li = int(acc.argmax())
        best[pos] = (acc[li], li, pr[li])
        r[f"probe_{pos}"] = acc[li]; r[f"L_{pos}"] = li
    pos = max(best, key=lambda k: best[k][0])
    prb = best[pos][2]
    loss = err & (cal != y) & (prb == y)
    loss_last = err & (cal != y) & (best["last"][2] == y)
    ne = max(int(err.sum()), 1)
    r |= dict(loss=loss.sum() / ne, loss_img=(loss & (text != y)).sum() / ne,  # (a)
              loss_lastpos=loss_last.sum() / ne, loss_lastpos_img=(loss_last & (text != y)).sum() / ne)  # (b)
    rows.append(r)
    print(r, flush=True)
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/g3_posthoc.csv", index=False)
print(df.round(3).to_string(index=False))
