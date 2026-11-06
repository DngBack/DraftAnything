"""Protocol T analysis (docs/protocol_token.md). Written before any main-run output existed."""
import argparse
import json
import sys
from glob import glob
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_percept import auroc  # noqa: E402
from evc.data import DATA  # noqa: E402
from pilot_percept import dig, nums  # noqa: E402


def load(tag, name="token", setname="token_set.json"):
    S = {i["qid"]: i for i in json.load(open(DATA / setname))}
    rows = [json.loads(l) for p in sorted(glob(str(ROOT / "results" / f"{name}_{tag}_*.jsonl"))) for l in open(p)]
    qs = sorted({r["qid"] for r in rows})
    ix = {q: k for k, q in enumerate(qs)}
    T, LP, SZ = {}, {}, {}
    for r in rows:
        key = (r["level"], r["cond"])
        T.setdefault(key, np.full(len(qs), np.nan))[ix[r["qid"]]] = float(dig(S[r["qid"]]["word"]) in set(nums(r["text"])))
        LP.setdefault(key, np.full(len(qs), np.nan))[ix[r["qid"]]] = r["lp"]
        SZ.setdefault(key, np.full(len(qs), np.nan))[ix[r["qid"]]] = r["size"]
    return qs, T, LP, SZ


def paired(a, b, seed=0, B=2000):
    """Mean of a - b in points, with a document-level percentile bootstrap CI (one cell per document)."""
    d = a - b
    rng = np.random.default_rng(seed)
    means = [np.nanmean(d[rng.integers(0, len(d), len(d))]) for _ in range(B)]
    lo, hi = np.nanpercentile(means, [2.5, 97.5])
    return 100 * np.nanmean(d), 100 * lo, 100 * hi


def fmt(e):
    return f"{e[0]:+.1f} [{e[1]:+.1f}, {e[2]:+.1f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    qs, T, LP, SZ = load(a.tag)
    intern = any(c.startswith("tile") for _, c in T)
    L = [f"# Protocol T, {a.tag}, n = {len(qs)} ô", "", "| mức | điều kiện | kích thước TB (token hoặc tile) | T % | AUROC log-prob |", "|---|---|---|---|---|"]
    for (lv, c) in sorted(T, key=lambda k: (k[0], np.nanmean(SZ[k]), k[1])):
        ok = ~np.isnan(T[(lv, c)])
        au = auroc(LP[(lv, c)][ok], T[(lv, c)][ok] > 0) if 0 < np.nanmean(T[(lv, c)]) < 1 else float("nan")
        L.append(f"| {lv} | {c} | {np.nanmean(SZ[(lv, c)]):.0f} | {100 * np.nanmean(T[(lv, c)]):.1f} | {au:.3f} |")
    L += ["", "## Tiêu chí", ""]
    if intern:
        t3 = paired(T[("s100", "tile12")], T[("s100", "tile1")])
        L.append(f"- T3, s100, 12 tile − 1 tile: {fmt(t3)}; đạt: {t3[1] > 0 and t3[0] >= 10}")
        es = {lv: paired(T[(lv, "full")], T[(lv, "tile12")]) for lv in ("s40", "s30")}
        for lv, e in es.items():
            L.append(f"- T4, {lv}, full − 12 tile: {fmt(e)}")
        L.append(f"- T4 đạt: {all(e[1] > -5 and e[2] < 5 for e in es.values())}")
    else:
        es = {lv: paired(T[(lv, "full")], T[(lv, "k1")]) for lv in ("s40", "s30")}
        for lv, e in es.items():
            L.append(f"- T1, {lv}, full − k1: {fmt(e)}")
        L.append(f"- T1 đạt: {all(e[1] > 0 for e in es.values()) and es['s30'][0] >= 10}")
        d1, d2 = paired(T[("s30", "k2")], T[("s30", "k1")]), paired(T[("s30", "full")], T[("s30", "k2")])
        L.append(f"- T2, s30, k2 − k1: {fmt(d1)}; full − k2: {fmt(d2)}; đạt: {d1[1] > 0 and d2[1] > 0}")
        L += ["", "## Mô tả", ""]
        for lv in ("s40", "s30"):
            L.append(f"- {lv}, full_nearest − full: {fmt(paired(T[(lv, 'full_nearest')], T[(lv, 'full')]))}")
    out = ROOT / "results" / f"token_{a.tag}.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
