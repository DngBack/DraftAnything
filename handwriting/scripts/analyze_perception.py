"""Analyse perception-pilot responses (perception_*.json from study.html).

Participants below --min-catch accuracy on catch trials are excluded (reported). For each
manipulation: accuracy (share choosing the real line), 95% CI from a crossed bootstrap that
resamples participants and lines independently (both are random effects), and the share of
bootstrap samples <= 0.5. Line-level drivers: accuracy on shuffle trials vs how much the
manipulation moved the words (baseline SD and gap CV of the line), by median split.

  --simulate N  writes N fake participants (shuffle 0.6, detrend 0.55, flat 0.7, catch 0.95)
                to test the pipeline; never mix with real responses.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load(paths):
    rows = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        rows += d["responses"]
    df = pd.DataFrame(rows)
    return df[~df.practice]


def crossed_boot(df, n_boot, rng):
    ps, ls = df.pid.unique(), df.line.unique()
    pi = {p: i for i, p in enumerate(ps)}; li = {l: i for i, l in enumerate(ls)}
    P, L = df.pid.map(pi).to_numpy(), df.line.map(li).to_numpy()
    y = df.correct.to_numpy(float)
    out = np.empty(n_boot)
    for b in range(n_boot):
        wp = np.bincount(rng.integers(0, len(ps), len(ps)), minlength=len(ps))
        wl = np.bincount(rng.integers(0, len(ls), len(ls)), minlength=len(ls))
        w = wp[P] * wl[L]
        out[b] = (w * y).sum() / max(w.sum(), 1)
    return out


def simulate(n, stim, out_dir, seed):
    rng = np.random.default_rng(seed)
    p = {"shuffle": 0.6, "detrend": 0.55, "flat": 0.7, "catch": 0.95}
    plan = [("shuffle", 30), ("detrend", 14), ("flat", 14), ("catch", 6)]
    lines = stim.line_id.iloc[2:].to_numpy()
    for k in range(n):
        ls = rng.permutation(lines); i = 0; res = []
        for c, m in plan:
            for _ in range(m):
                res.append({"pid": f"sim{k}", "line": ls[i], "cond": c, "practice": False,
                            "correct": bool(rng.random() < p[c]), "rt_ms": 3000}); i += 1
        Path(out_dir, f"perception_sim{k}.json").write_text(json.dumps({"pid": f"sim{k}", "responses": res}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("responses", nargs="*")
    ap.add_argument("--stim", type=Path, default=Path("data/perception/stimuli.csv"))
    ap.add_argument("--min-catch", type=float, default=0.75)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--simulate", type=int, default=0)
    ap.add_argument("--sim-dir", type=Path, default=Path("reports/perception/sim"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    stim = pd.read_csv(a.stim)
    if a.simulate:
        a.sim_dir.mkdir(parents=True, exist_ok=True)
        simulate(a.simulate, stim, a.sim_dir, a.seed)
        a.responses = sorted(map(str, a.sim_dir.glob("perception_*.json")))
    df = load(a.responses)
    catch = df[df.cond == "catch"].groupby("pid").correct.mean()
    keep = catch[catch >= a.min_catch].index
    print(json.dumps({"participants": int(df.pid.nunique()), "kept": len(keep),
                      "excluded": sorted(set(df.pid) - set(keep)),
                      "median_rt_s": round(float(df.rt_ms.median()) / 1000, 1)}))
    df = df[df.pid.isin(keep)]
    rng = np.random.default_rng(a.seed)
    rows = []
    for c in ["shuffle", "detrend", "flat", "catch"]:
        d = df[df.cond == c]
        if not len(d):
            continue
        bs = crossed_boot(d, a.n_boot, rng)
        rows.append({"cond": c, "n_trials": len(d), "acc": d.correct.mean(),
                     "lo": np.quantile(bs, 0.025), "hi": np.quantile(bs, 0.975),
                     "p_le_half": float((bs <= 0.5).mean())})
    res = pd.DataFrame(rows)
    print(res.round(3).to_string(index=False))
    sh = df[df.cond == "shuffle"].merge(stim, left_on="line", right_on="line_id")
    for col in ["base_sd_core", "gap_cv", "slope_pct"]:
        v = sh[col].abs()
        hi = v > v.median()
        print(f"shuffle acc by |{col}|: low {sh.correct[~hi].mean():.3f}  high {sh.correct[hi].mean():.3f}")
    out = Path("reports/perception") / ("summary_sim.csv" if a.simulate else "summary.csv")
    res.to_csv(out, index=False)


if __name__ == "__main__":
    main()
