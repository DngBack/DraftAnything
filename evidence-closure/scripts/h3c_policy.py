"""H3c (docs/protocol_h3c.md): do selectors trained on official-EM labels beat the same selectors trained on lenient labels?

usage: h3c_policy.py --tag qwen25vl7b          # document-level 5-fold CV on train only (development)
       h3c_policy.py --tag qwen25vl7b --dev2   # fit on all train, evaluate once on dev2
Everything here is scored with official EM; the training label is the only thing that differs.
"""
import argparse
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import h3_policy as H  # noqa: E402
from official_rescore import em, load, sel_h3, sel_rule, sel_v2  # noqa: E402

LENIENT_LABEL, LENIENT_VAL = H.label, H.val_of


def lab_em(r, v, s):
    return em(v, H.scale_of(s, r), r)


def val_em(v, r):
    return bool(em(v, r["scale"], r))


@contextmanager
def em_labels():
    H.label, H.val_of = lab_em, val_em
    try:
        yield
    finally:
        H.label, H.val_of = LENIENT_LABEL, LENIENT_VAL


def score(sel, R):
    return np.array([em(v, s, r) for (v, s), r in zip(sel, R)])


def lam_of(Rtr, R, B):
    return H.evaluate(Rtr, R, B)[1]["lam"]


def run(Rtr, R, B):
    """All conditions at budget B, every one scored with EM on R."""
    res = {"direct": score([("direct", r["scale_direct"]) for r in R], R),
           "fixed2": score([("program", r["scale_direct"]) for r in R], R)}
    # lenient-trained (as locked in H3/H3b)
    lam = lam_of(Rtr, R, B)
    res["H3-len"] = score(sel_h3(H.fit_models(Rtr), R, lam), R)
    res["rule-len"] = score(sel_rule(Rtr, R, B), R)
    res["v2-len"] = score(sel_v2(Rtr, R, B), R)
    # EM-trained
    with em_labels():
        lam = lam_of(Rtr, R, B)
        M = H.fit_models(Rtr)
    res["H3-em"] = score(sel_h3(M, R, lam), R)
    res["rule-em"] = score(sel_rule(Rtr, R, B, label=lab_em), R)
    res["v2-em"] = score(sel_v2(Rtr, R, B, valf=val_em), R)
    return res


COMPARE = (("v2-em", "v2-len"), ("H3-em", "H3-len"), ("v2-em", "H3-em"), ("v2-em", "rule-em"), ("H3-em", "rule-em"),
           ("v2-em", "fixed2"), ("H3-em", "fixed2"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="qwen25vl7b")
    ap.add_argument("--dev2", action="store_true")
    a = ap.parse_args()
    Rtr = load(a.tag, "train", "train")
    L = [f"# H3c — {a.tag} — {'dev2' if a.dev2 else 'train CV (dev)'}", "", "Mọi điều kiện chấm bằng EM chính thức.", ""]
    for B in (2, 3):
        if a.dev2:
            R = load(a.tag, "dev2", "dev")
            res, docs = run(Rtr, R, B), [r["uid"] for r in R]
        else:
            _, fo = H.folds(Rtr, seed=1)
            acc = {}
            docs = []
            for k in range(5):
                tr = [r for r, f in zip(Rtr, fo) if f != k]
                va = [r for r, f in zip(Rtr, fo) if f == k]
                for n, c in run(tr, va, B).items():
                    acc.setdefault(n, []).append(c)
                docs += [r["uid"] for r in va]
            res = {n: np.concatenate(v) for n, v in acc.items()}
        L += [f"## B = {B}, n = {len(docs)}, {len(set(docs))} tài liệu", "", "| policy | EM |", "|---|---|"]
        L += [f"| {n} | {100 * c.mean():.1f} |" for n, c in res.items()]
        L.append("")
        for x, y in COMPARE:
            e, lo, hi = H.boot(res[x], res[y], docs)
            L.append(f"- {x} − {y} = {100 * e:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]" + (" *" if lo > 0 or hi < 0 else ""))
        L.append("")
    out = ROOT / "results" / f"h3c_{a.tag}_{'dev2' if a.dev2 else 'dev'}.md"
    out.write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
