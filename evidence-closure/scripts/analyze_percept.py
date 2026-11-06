"""Pilot B analysis (docs/protocol_percept_cf.md). Written before any output existed."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from evc.data import DATA  # noqa: E402
from evc.degrade import SCALES  # noqa: E402
from evc.stimuli import swap_digit  # noqa: E402
from pilot_percept import dig, nums  # noqa: E402

LEVELS = ("s100", "s60", "s50", "s40", "s30")


def auroc(score, y):
    y = np.asarray(y, bool)
    if y.all() or not y.any():
        return float("nan")
    r = np.argsort(np.argsort(score)) + 1.0
    # average ranks over ties
    s = np.asarray(score)
    for v in np.unique(s):
        m = s == v
        r[m] = r[m].mean()
    n1, n0 = y.sum(), (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def threshold(h, y, iters=50):
    """Glyph height (px) at 50 % on a logistic fit against log height; nan if the slope is not positive."""
    x = np.c_[np.log(h), np.ones(len(h))]
    w = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-x @ w))
        g = x.T @ (y - p) - np.array([1e-3 * w[0], 0])
        H = (x * (p * (1 - p))[:, None]).T @ x + 1e-3 * np.eye(2)
        w += np.linalg.solve(H, g)
    return float(np.exp(-w[1] / w[0])) if w[0] > 0 else float("nan")


class Boot:
    def __init__(self, qids, n=2000):
        rng = np.random.default_rng(0)
        self.s = [rng.integers(0, len(qids), len(qids)) for _ in range(n)]

    def ci(self, fn):
        v = [fn(i) for i in self.s]
        return np.nanpercentile(v, [2.5, 97.5])


def fmt(e, ci, k=100):
    star = " *" if ci[0] > 0 or ci[1] < 0 else ""
    return f"{k * e:+.1f} [{k * ci[0]:+.1f}, {k * ci[1]:+.1f}]{star}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    S = {i["qid"]: i for i in json.load(open(DATA / "percept_set.json"))}
    rows = [json.loads(l) for l in open(ROOT / "results" / f"percept_{a.tag}.jsonl")]
    qs = sorted({r["qid"] for r in rows})
    ix = {q: k for k, q in enumerate(qs)}
    T, O, A, LP, U = {}, {}, {}, {}, {}
    for r in rows:
        it = S[r["qid"]]
        got = set(nums(r["text"]))
        orig, shown, plc = dig(it["word"]), dig(swap_digit(it["word"], it["pos"], it["d_s"])), dig(swap_digit(it["word"], it["pos"], it["d_a"]))
        key = (r["level"], r["cond"], r["prompt"])
        for D, v in ((T, (shown if r["cond"] == "cf" else orig) in got), (O, orig in got), (A, plc in got), (LP, r["lp"]), (U, "unreadable" in r["text"].lower())):
            D.setdefault(key, np.full(len(qs), np.nan))[ix[r["qid"]]] = float(v)
    B = Boot(qs)
    L = [f"# Pilot B, {a.tag}, n = {len(qs)} câu", "", "| mức | T orig copy | T null copy | T cf copy | T orig addr | T cf addr | T orig program | T cf program |", "|---|---|---|---|---|---|---|---|"]
    for lv in LEVELS:
        vals = [T.get((lv, c, p)) for c, p in (("orig", "copy"), ("null", "copy"), ("cf", "copy"), ("orig", "addr"), ("cf", "addr"), ("orig", "program"), ("cf", "program"))]
        L.append(f"| {lv} | " + " | ".join(f"{100 * np.nanmean(v):.1f}" if v is not None else "–" for v in vals) + " |")
    to, tn, tc = T[("s100", "orig", "copy")], T[("s100", "null", "copy")], T[("s100", "cf", "copy")]
    L += ["", f"**Hợp lệ (s100, copy):** T(null) − T(orig) = {fmt(np.nanmean(tn - to), B.ci(lambda i: np.nanmean((tn - to)[i])))}; T(cf) = {100 * np.nanmean(tc):.1f}", "",
          "## H-prior: O − A trên ảnh cf", "", "| mức | prompt | O | A | O − A |", "|---|---|---|---|---|"]
    for lv in LEVELS:
        for p in ("copy", "addr", "program"):
            o, x = O[(lv, "cf", p)], A[(lv, "cf", p)]
            L.append(f"| {lv} | {p} | {100 * np.nanmean(o):.1f} | {100 * np.nanmean(x):.1f} | {fmt(np.nanmean(o - x), B.ci(lambda i: np.nanmean((o - x)[i])))} |")
    L += ["", "## H-task: T(copy) − T(addr), ảnh orig", ""]
    for lv in LEVELS:
        d = T[(lv, "orig", "copy")] - T[(lv, "orig", "addr")]
        L.append(f"- {lv}: {fmt(np.nanmean(d), B.ci(lambda i: np.nanmean(d[i])))}")
    L += ["", "## H-meta: ảnh orig, prompt copy", "", "| mức | đúng % | AUROC log-prob [CI] | UNREADABLE % | P(UNREAD \\| copy sai) | P(UNREAD \\| copy đúng) |", "|---|---|---|---|---|---|"]
    for lv in LEVELS:
        y, lp, u = T[(lv, "orig", "copy")], LP[(lv, "orig", "copy")], U[(lv, "orig", "copyu")]
        m = ~np.isnan(y + lp + u)
        def au(i):
            j = i[m[i]]
            return auroc(lp[j], y[j] > 0)
        ci = B.ci(au)
        wr, ri = u[m & (y == 0)], u[m & (y == 1)]
        L.append(f"| {lv} | {100 * y[m].mean():.1f} | {auroc(lp[m], y[m] > 0):.2f} [{ci[0]:.2f}, {ci[1]:.2f}] | {100 * u[m].mean():.1f} | "
                 f"{100 * wr.mean() if len(wr) else float('nan'):.1f} (n={len(wr)}) | {100 * ri.mean() if len(ri) else float('nan'):.1f} (n={len(ri)}) |")
    L += ["", "## Mô tả: chiều cao chữ tại mức đọc đúng 50 % (px, log-logistic, ảnh orig)", ""]
    H0 = np.array([S[q]["bbox"][3] - S[q]["bbox"][1] for q in qs], float)
    for p in ("copy", "addr", "program"):
        h, y = [], []
        for lv in LEVELS:
            v = T[(lv, "orig", p)]
            m = ~np.isnan(v)
            h += list(H0[m] * SCALES[lv])
            y += list(v[m])
        L.append(f"- {p}: {threshold(np.array(h), np.array(y)):.1f} px")
    out = ROOT / "results" / f"percept_{a.tag}.md"
    out.write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
