"""H3 policies (docs/protocol_h3.md): fixed pipelines, confidence trigger, error-type routing, learned utility.

usage: h3_policy.py --tag qwen25vl7b            # 5-fold CV on train only (development)
       h3_policy.py --tag qwen25vl7b --test     # fit on all train, evaluate once on test
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_e8 import prog_ok  # noqa: E402
from evc.metrics import parse_nums, value_correct  # noqa: E402
from run_e8 import read_recall, safe_eval  # noqa: E402

VSRC = ("direct", "program", "zoom", "verify")
SSRC = ("direct", "ocr", "vlm")
EXTRA = ("zoom", "verify", "vlm")  # optional actions, 1 VLM call each (direct + program always run)
QRE = {k: re.compile(p, re.I) for k, p in dict(
    change=r"change|increase|decrease|difference|growth", avg=r"average|mean", pct=r"percent|percentage|%|proportion|ratio|rate|margin",
    total=r"total|sum|combined", resp=r"respective|respectively|both", what=r"^(what|which) (is|was|were|are) the (value|amount)").items()}


def has_expr(t):
    return t is not None and safe_eval(t) is not None and any(o in t for o in "+-*/")


def val_of(src, r):
    """Value correctness of a source, with the routing used everywhere (expression -> Python else text)."""
    if src == "direct":
        return bool(value_correct(r["direct"], r["answer"]))
    t = r[src]
    if t is None:
        t = r["program"]
    return bool(prog_ok(t, r, False)) if has_expr(t) else bool(value_correct(r["direct"], r["answer"]))


def num(t):
    if has_expr(t):
        return safe_eval(t)
    n = parse_nums(t or "")
    return n[0] if n else None


def agree(a, b):
    if a is None or b is None:
        return 0.0
    return float(abs(abs(a) - abs(b)) <= 1e-3 * max(1.0, abs(b)))


def scale_of(src, r):
    return {"direct": r["scale_direct"], "ocr": r["scale_ocr"], "vlm": r["scale_vlm"]}[src]


def base_feats(r):
    q, p, d = r["question"], r["program"], r["direct"]
    pv, dv = num(p), num(d)
    f = [r["direct_lp"], r["direct_lpmin"], r["program_lp"], r["program_lpmin"], float(has_expr(p)), r["program_ground"],
         agree(pv, dv), float(dv is not None), len(parse_nums(p)) / 5, p.count("/") / 2, p.count("*"), p.count("-") / 2,
         len(d) / 40, float("%" in d), float(r["scale_direct"] == r["scale_ocr"]), float(r["scale_direct"] == ""),
         float(r["scale_ocr"] == ""), float(r["scale_ocr"] == "percent")]
    f += [float(bool(rx.search(q))) for rx in QRE.values()]
    return f


def post_feats(r):
    pv = num(r["program"])
    out = []
    for k in ("zoom", "verify"):
        t = r[k]
        out += [float(has_expr(t)), r[f"{k}_ground"], agree(num(t), pv), float(t == r["program"])]
    out += [float(r["scale_vlm"] == r["scale_ocr"]), float(r["scale_vlm"] == r["scale_direct"]), float(r["scale_vlm"] == "")]
    return out


def pairs():
    return [(v, s) for v in VSRC for s in SSRC]


def pair_cost(v, s):
    return 2 + (v in ("zoom", "verify")) + (s == "vlm")


def label(r, v, s):
    return float(val_of(v, r) and scale_of(s, r) == r["scale"])


# ------------------------------------------------------------------ logistic regression (numpy)
class LR:
    def __init__(self, l2=1.0, iters=400, lr=0.3):
        self.l2, self.iters, self.lr = l2, iters, lr

    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = np.c_[(X - self.mu) / self.sd, np.ones(len(X))]
        self.w = np.zeros(Z.shape[1])
        for _ in range(self.iters):
            p = 1 / (1 + np.exp(-Z @ self.w))
            g = Z.T @ (p - y) / len(y) + self.l2 * np.r_[self.w[:-1], 0] / len(y)
            self.w -= self.lr * g
        return self

    def predict(self, X):
        Z = np.c_[(X - self.mu) / self.sd, np.ones(len(X))]
        return 1 / (1 + np.exp(-Z @ self.w))


def fit_models(R):
    Xb = np.array([base_feats(r) for r in R])
    Xp = np.c_[Xb, np.array([post_feats(r) for r in R])]
    M = {}
    for v, s in pairs():
        y = np.array([label(r, v, s) for r in R])
        M[("pre", v, s)] = LR().fit(Xb, y)   # predicted before running optional actions
        M[("post", v, s)] = LR().fit(Xp, y)  # predicted after running them (final choice)
    return M


def learned_policy(R, M, lam):
    """Run optional action a iff predicted gain of the best pair using a exceeds lam; then pick best executed pair."""
    Xb = np.array([base_feats(r) for r in R])
    Xp = np.c_[Xb, np.array([post_feats(r) for r in R])]
    pre = {k[1:]: m.predict(Xb) for k, m in M.items() if k[0] == "pre"}
    post = {k[1:]: m.predict(Xp) for k, m in M.items() if k[0] == "post"}
    correct, cost = [], []
    for i, r in enumerate(R):
        avail_v, avail_s = {"direct", "program"}, {"direct", "ocr"}
        base_best = max(pre[(v, s)][i] for v in avail_v for s in avail_s)
        gains = {}
        for a in EXTRA:
            vs = avail_v | ({a} if a in ("zoom", "verify") else set())
            ss = avail_s | ({"vlm"} if a == "vlm" else set())
            gains[a] = max(pre[(v, s)][i] for v in vs for s in ss) - base_best
        a = max(gains, key=gains.get)
        c = 2
        if gains[a] > lam:
            c += 1
            if a in ("zoom", "verify"):
                avail_v.add(a)
            else:
                avail_s.add("vlm")
        score = post if c > 2 else pre
        v, s = max(((v, s) for v in avail_v for s in avail_s), key=lambda t: score[t][i])
        correct.append(label(r, v, s))
        cost.append(c)
    return np.array(correct), np.array(cost)


def fixed(R, v, s):
    return np.array([label(r, v, s) for r in R]), np.full(len(R), pair_cost(v, s))


def route(R, extra, s):
    """Fixed pipeline: value = extra source if it gives an expression, else program route; scale from s."""
    c = np.array([label(r, extra if extra in ("zoom", "verify") and has_expr(r[extra]) else "program", s) for r in R])
    return c, np.full(len(R), 2 + (extra in ("zoom", "verify")) + (s == "vlm"))


def conf_policy(R, tau, extra, sb="ocr"):
    """Fixed-2 (scale from sb); if direct log-prob < tau also run `extra` and use it."""
    out, cost = [], []
    for r in R:
        trig = r["direct_lp"] < tau
        if trig and extra in ("zoom", "verify") and has_expr(r[extra]):
            out.append(label(r, extra, sb))
        elif trig and extra == "vlm":
            out.append(label(r, "program", "vlm"))
        else:
            out.append(label(r, "program", sb))
        cost.append(2 + trig)
    return np.array(out), np.array(cost)


def failure_type(r, sb="ocr"):
    """E8-style type of the Fixed-2 failure with scale source sb (ocr: error-type policy labels; direct: recovery table)."""
    if label(r, "program", sb):
        return "ok"
    if val_of("program", r):
        return "scale"
    if read_recall(r["program"], r["facts"]) < 1:
        return "read"
    return "reason" if r["answer_type"] == "arithmetic" else "span"


def errtype_policy(Rtr, R, thr, sb="ocr"):
    """Classifier: P(Fixed-2+OCR wrong for reason t); act with the fixed action for argmax type if P > thr."""
    amap = {"read": "zoom", "reason": "verify", "scale": "vlm", "span": None}
    Xtr = np.array([base_feats(r) for r in Rtr])
    X = np.array([base_feats(r) for r in R])
    clf = {t: LR().fit(Xtr, np.array([float(failure_type(r) == t) for r in Rtr])) for t in amap}
    P = {t: m.predict(X) for t, m in clf.items()}
    out, cost = [], []
    for i, r in enumerate(R):
        t = max(amap, key=lambda k: P[k][i])
        a = amap[t] if P[t][i] > thr else None
        if a in ("zoom", "verify") and has_expr(r[a]):
            out.append(label(r, a, sb))
        elif a == "vlm":
            out.append(label(r, "program", "vlm"))
        else:
            out.append(label(r, "program", sb))
        cost.append(2 + (a is not None))
    return np.array(out), np.array(cost)


def match_budget(fn, B, grid):
    """Pick the parameter whose mean cost is <= B and closest to it."""
    best = None
    for g in grid:
        c, k = fn(g)
        if k.mean() <= B + 1e-9 and (best is None or k.mean() > best[2].mean() or
                                      (abs(k.mean() - best[2].mean()) < 1e-9 and c.mean() > best[1].mean())):
            best = (g, c, k)
    return best


def boot(a, b, docs, n=2000, seed=0):
    ud = sorted(set(docs))
    idx = {d: [i for i, x in enumerate(docs) if x == d] for d in ud}
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        ii = [i for k in rng.choice(len(ud), len(ud)) for i in idx[ud[k]]]
        bs.append((a[ii] - b[ii]).mean())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return (a - b).mean(), lo, hi


def folds(R, k=5, seed=0):
    docs = sorted({r["uid"] for r in R})
    rng = np.random.default_rng(seed)
    rng.shuffle(docs)
    fold_of = {d: i % k for i, d in enumerate(docs)}
    return [r for r in R], [fold_of[r["uid"]] for r in R]


def evaluate(Rtr, Rte, B):
    """All policies at budget B; parameters chosen on train (CV for learned/conf/errtype)."""
    res = {}
    res["fixed2"] = fixed(Rte, "program", "direct")
    res["fixed2_ocr"] = fixed(Rte, "program", "ocr")
    sb = max(("direct", "ocr"), key=lambda s_: fixed(Rtr, "program", s_)[0].mean())  # best base scale on train
    tr_scores = {x: route(Rtr, x, "vlm" if x == "vlm" else sb)[0].mean() for x in EXTRA}
    bx = max(tr_scores, key=tr_scores.get)
    if B >= 3:
        res[f"fixed3_{bx}"] = route(Rte, bx, "vlm" if bx == "vlm" else sb)
    taus = np.quantile([r["direct_lp"] for r in Rtr], np.linspace(0, 1, 41))
    t = match_budget(lambda g: conf_policy(Rtr, g, bx, sb), B, taus)
    res["confidence"] = conf_policy(Rte, t[0], bx, sb)
    # error-type and learned: thresholds chosen by CV on train
    _, fo = folds(Rtr)
    thr_grid = np.linspace(0, 1, 21)
    lam_grid = np.r_[np.linspace(-0.2, 0.5, 36), 10]
    oof = {}
    for kf in range(5):
        tr = [r for r, f in zip(Rtr, fo) if f != kf]
        va = [r for r, f in zip(Rtr, fo) if f == kf]
        M = fit_models(tr)
        for lam in lam_grid:
            c, k = learned_policy(va, M, lam)
            oof.setdefault(("L", lam), []).append((c.sum(), k.sum(), len(va)))
        for thr in thr_grid:
            c, k = errtype_policy(tr, va, thr, sb)
            oof.setdefault(("E", thr), []).append((c.sum(), k.sum(), len(va)))
    def pick(tag, grid):
        cands = []
        for g in grid:
            s = oof[(tag, g)]
            acc = sum(x[0] for x in s) / sum(x[2] for x in s)
            cost = sum(x[1] for x in s) / sum(x[2] for x in s)
            if cost <= B + 1e-9:
                cands.append((acc, -abs(B - cost), g))
        return max(cands)[2]
    lam = pick("L", lam_grid)
    thr = pick("E", thr_grid)
    M = fit_models(Rtr)
    res["errtype"] = errtype_policy(Rtr, Rte, thr, sb)
    res["learned"] = learned_policy(Rte, M, lam)
    return {k: v for k, v in res.items() if v is not None}, dict(lam=float(lam), thr=float(thr), extra=bx, base_scale=sb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="qwen25vl7b")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    Rtr = [json.loads(l) for l in open(ROOT / "results" / f"h3_train_{a.tag}.jsonl")]
    if a.test:
        Rte = [json.loads(l) for l in open(ROOT / "results" / f"h3_test_{a.tag}.jsonl")]
        name = "test"
    else:  # development: hold out one train fold as pseudo-test
        _, fo = folds(Rtr, seed=1)
        Rte = [r for r, f in zip(Rtr, fo) if f == 0]
        Rtr = [r for r, f in zip(Rtr, fo) if f != 0]
        name = "train-holdout (dev only)"
    docs = [r["uid"] for r in Rte]
    L = [f"# H3 policies — {a.tag} — {name}", "", f"train n = {len(Rtr)}, eval n = {len(Rte)} câu, "
         f"{len(set(docs))} tài liệu. Metric chính thức: giá trị đúng (strict) **và** scale = scale gold.", ""]
    for B in (2, 3):
        res, par = evaluate(Rtr, Rte, B)
        L += [f"## Ngân sách B = {B} lượt gọi VLM / câu (trung bình)", "", f"Tham số chọn trên train: {par}", "",
              "| policy | accuracy | chi phí TB |", "|---|---|---|"]
        for k, (c, cost) in res.items():
            L.append(f"| {k} | {100 * c.mean():.1f} | {cost.mean():.2f} |")
        base = {k: v for k, v in res.items() if k != "learned"}
        bk = max(base, key=lambda k: base[k][0].mean())
        e, lo, hi = boot(res["learned"][0], base[bk][0], docs)
        L += ["", f"**learned − baseline tốt nhất trên tập đánh giá ({bk})** = {100 * e:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
              + (" *" if lo > 0 or hi < 0 else ""), ""]
        if B == 3:
            ft = [failure_type(r, "direct") for r in Rte]
            L += ["Recovery trên các câu Fixed-2 (scale từ direct) sai, theo loại lỗi E8 (số câu được policy trả lời đúng):", "", "| loại | n | " + " | ".join(res) + " |",
                  "|---|---|" + "---|" * len(res)]
            for t in ("scale", "read", "reason", "span"):
                ii = [i for i, x in enumerate(ft) if x == t]
                if ii:
                    L.append(f"| {t} | {len(ii)} | " + " | ".join(f"{int(res[k][0][ii].sum())}" for k in res) + " |")
            L.append("")
    out = ROOT / "results" / f"h3_policy_{a.tag}_{'test' if a.test else 'dev'}.md"
    out.write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
