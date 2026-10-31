"""Evaluate the post-hoc re-layout fix on val1.

Common line set: CER <= 0.2 for real, raw Emuru and raw TS-GAN (the H2 set) and the line
was produced by every fix variant. Reports
  profile.csv     dynamics profile (lag-1, trend R2, within SD; writer-bootstrap CIs)
  distances.csv   Wasserstein-1 distance of per-line stats to real (slope, R2, baseline SD,
                  gap CV), with a writer-bootstrap CI, plus means
  cer.csv         TrOCR CER of each variant on the lines it produced vs the raw output
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from hwline.profile import dynamics_profile, per_line_stats

KINDS = ["ident", "jitter", "donor", "style"]
GENS = ["emuru", "tsgan"]
OUT = Path("reports/fix")


def sources():
    s = {"real": "data/dyn/words_real32_val1.csv.gz"}
    for g in GENS:
        s[g] = f"data/dyn/words_{g}32_val1.csv.gz"
        for k in KINDS:
            s[f"{g}_{k}"] = f"data/dyn/fix/words_{g}_{k}.csv.gz"
    return s


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ok = set(pd.read_csv("data/dyn/cer_real_val1.csv").query("cer <= 0.2").line_id)
    for g in GENS:
        ok &= set(pd.read_csv(f"data/dyn/cer_{g}_val1.csv").query("cer <= 0.2").line_id)
    words = {n: pd.read_csv(p, dtype={"writer_id": str}) for n, p in sources().items()}
    for n, w in words.items():
        if "_" in n:
            ok &= set(w.line_id)
    print("common lines:", len(ok))
    words = {n: w[w.line_id.isin(ok)].assign(source=n) for n, w in words.items()}
    writer = words["real"].drop_duplicates("line_id").set_index("line_id").writer_id

    # CER
    rows = []
    for g in GENS:
        raw = pd.read_csv(f"data/dyn/cer_{g}_val1.csv").set_index("line_id").cer
        for k in KINDS:
            c = pd.read_csv(f"data/dyn/fix/cer_{g}_{k}.csv").set_index("line_id").cer
            r = raw.reindex(c.index)
            rows.append({"gen": g, "kind": k, "n": len(c), "cer_raw_median": r.median(),
                         "cer_fix_median": c.median(), "share_le_0.2_raw": (r <= 0.2).mean(),
                         "share_le_0.2_fix": (c <= 0.2).mean(),
                         "mean_delta": (c - r).mean()})
    cer = pd.DataFrame(rows)
    cer.to_csv(OUT / "cer.csv", index=False)
    print(cer.round(3).to_string())

    # Per-line distribution distances to real
    stats = {n: per_line_stats(w).set_index("line_id") for n, w in words.items()}
    rng = np.random.default_rng(0)
    uw = writer.unique()
    rows = []
    for n, st in stats.items():
        row = {"source": n}
        for c in ["slope", "r2", "base_sd", "gap_cv"]:
            a, b = stats["real"][c].dropna(), st[c].dropna()
            row[f"{c}_mean"] = b.mean()
            row[f"{c}_w1"] = wasserstein_distance(a, b)
            bs = []
            for _ in range(300):
                ws = set(rng.choice(uw, len(uw)))
                ia = [l for l in a.index if writer.get(l) in ws]
                ib = [l for l in b.index if writer.get(l) in ws]
                bs.append(wasserstein_distance(a[ia], b[ib]))
            row[f"{c}_w1_lo"], row[f"{c}_w1_hi"] = np.percentile(bs, [2.5, 97.5])
        rows.append(row)
    dist = pd.DataFrame(rows)
    dist.to_csv(OUT / "distances.csv", index=False)
    print(dist[["source"] + [c for c in dist.columns if c.endswith("_w1") or c.endswith("_mean")]]
          .round(3).to_string())

    # Dynamics profile
    rows = []
    for n, w in words.items():
        for r in dynamics_profile(w, 200, 500, 0, features=["baseline", "gap"]):
            rows.append({"source": n, **r})
            print(n, r["feature"], {k: round(v, 3) for k, v in r.items() if isinstance(v, float)
                                    and k in ("lag1_excess", "trend_r2_excess", "within_sd")}, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "profile.csv", index=False)
    json.dump({"common_lines": len(ok)}, open(OUT / "meta.json", "w"))


if __name__ == "__main__":
    main()
