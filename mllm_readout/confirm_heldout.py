"""G3 addendum 4 (docs/g3_protocol.md): confirmatory steering (alpha=16 fixed) on held-out models.

usage: python confirm_heldout.py        (results/g3_heldout.csv)
"""
import glob, os
import numpy as np, pandas as pd
from pilot import OUT
from diag_gen import cell

D = f"{OUT}/g3_heldout"


def main():
    rows = []
    for p in sorted(glob.glob(f"{D}/*_states.npz")):
        r = cell(p)
        d, c = np.load(p), r["cell"]
        y, ans = d["y"], d["ans"]
        for tag, f in (("a1", f"{D}/{c}_steer.npz"), ("a16", f"{D}/{c}_steerA16.npz")):
            if os.path.exists(f):
                s = np.load(f)["steer"]
                r[f"steer_{tag}"] = np.mean(s == y)
                r[f"fixed_{tag}"] = np.sum((ans != y) & (s == y)) / max(np.sum(ans != y), 1)
                r[f"broken_{tag}"] = np.sum((ans == y) & (s != y)) / max(np.sum(ans == y), 1)
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/g3_heldout.csv", index=False)
    cols = [c for c in ("cell", "n", "nodec", "n_err", "model", "calib", "probe", "bias", "loss", "perc",
                        "steer_a1", "steer_a16", "fixed_a16", "broken_a16") if c in df]
    print(df[cols].round(3).to_string(index=False))
    ok = df[(df.n_err >= 30) & (df.nodec <= 0.10) & (df.text < 0.75)]  # validity as in G3
    ill, cb = ok[ok.cell.str.endswith("_illusions")], ok[ok.cell.str.endswith("_countbench")]

    def verdict(name, g, hit):
        if len(g) < 2:
            print(f"{name}: {len(g)} valid cells -> untestable")
        else:
            print(f"{name}: {int(hit.sum())}/{len(g)} -> {'holds' if hit.mean() >= 2 / 3 else 'fails'}")

    if "steer_a16" in ok:  # cells whose steering run has finished
        i, c = ill.dropna(subset=["steer_a16"]), cb.dropna(subset=["steer_a16"])
        verdict("H1 illusions steer16 >= max(raw, calib) + 10", i, i.steer_a16 >= np.maximum(i.model, i.calib) + 0.10)
        verdict("H2 CountBench steer16 < calib", c, c.steer_a16 < c.calib)
    verdict("H3a illusions loss > bias", ill, ill.loss > ill.bias)
    verdict("H3b CountBench bias > loss", cb, cb.bias > cb.loss)


if __name__ == "__main__":
    main()
