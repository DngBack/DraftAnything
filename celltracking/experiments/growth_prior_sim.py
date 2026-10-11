"""Binary-fission growth prior from RodSim only (no real labels): dispersion of descendant-family sizes.

For simulated colonies (full tree, no field-of-view crop) and gaps H, every ancestor at frame a has n_i descendants
at b = a + H. We record the per-window growth factor R = N_b / N_a and the Fano factor var(n) / mean(n). The fitted
kappa(R) sets the variance kappa * m of the family-size prior in genealogy_constrained_decode.py.

  python experiments/growth_prior_sim.py
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.family_set import RESULTS, ROOT  # noqa: E402

SIM = ROOT / "outputs/sim/v2"


def ancestor(start, parent, track, a):
    while start.get(track, -1) > a:
        track = parent.get(track, 0)
        if track == 0:
            return 0
    return track


def main():
    rows = []
    rng = np.random.default_rng(0)
    for path in sorted(SIM.glob("colony_*.pkl"))[:96]:
        d = pickle.loads(path.read_bytes())
        frames, start, parent = d["frames"], d["start"], d["parent"]
        keys = sorted(frames)
        for h in (33, 66, 132):
            ok = [t for t in keys if t + h in frames and len(frames[t]["track"]) >= 20]
            for a in rng.choice(ok, size=min(4, len(ok)), replace=False) if ok else []:
                b = a + h
                anc = np.array([ancestor(start, parent, int(t), a) for t in frames[b]["track"]])
                ids = frames[a]["track"]
                n = np.array([(anc == t).sum() for t in ids])
                rows.append({"H": h, "R": len(anc) / len(ids), "mean": n.mean(), "var": n.var(),
                             "frac0": float(np.mean(n == 0)), "doubling": d["p"]["doubling"]})
    r = np.array([[x["R"], x["var"] / x["mean"]] for x in rows])
    # Fano factor as a linear function of log2 R (least squares)
    A = np.column_stack([np.ones(len(r)), np.log2(r[:, 0])])
    coef = np.linalg.lstsq(A, r[:, 1], rcond=None)[0]
    out = {"n_windows": len(rows), "fano_intercept": float(coef[0]), "fano_slope_log2R": float(coef[1]),
           "fano_median": float(np.median(r[:, 1])), "frac0_max": float(max(x["frac0"] for x in rows))}
    for h in (33, 66, 132):
        s = [x for x in rows if x["H"] == h]
        out[f"H{h}"] = {"R_median": float(np.median([x["R"] for x in s])),
                        "fano_median": float(np.median([x["var"] / x["mean"] for x in s]))}
    print(json.dumps(out, indent=1))
    (RESULTS / "growth_prior_sim.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
