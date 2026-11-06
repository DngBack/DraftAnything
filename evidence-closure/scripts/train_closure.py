"""Train learned closure on TAT-DQA train (weak labels), evaluate vs heuristic on the E1 dev set.

Matched size: tau is chosen on a held-out slice of train so the learned closure has the same mean
number of words as the heuristic closure. Outputs results/closure_{geo,query}.pt and closure_eval.md.
"""
import json
import random
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evc.closure import closure_words  # noqa: E402
from evc.data import DATA, build_set, is_num, load_doc, page_words  # noqa: E402
from evc.learned import LearnedClosure, build_pairs, candidates, labels, train  # noqa: E402

torch.set_num_threads(8)


def words_of_split(split):
    @lru_cache(maxsize=4096)
    def f(uid):
        return page_words(load_doc(uid, split))

    return lambda it: f(it["uid"])


def eval_sets(items, wf, closers):
    """Weak-label recall/precision of each closure, plus mean size (words beyond seeds)."""
    res = {k: dict(tp=0, fp=0, fn=0, size=[]) for k in closers}
    for it in items:
        words, (W, H) = wf(it)
        pos = set()
        for s in it["fact_ids"]:
            if is_num(words[s]["text"]):
                c = candidates(words, s, W, H)
                y = labels(words, c, it["question"], it["scale"])
                pos |= {c[j] for j in range(len(c)) if y[j] > 0}
        pos -= set(it["fact_ids"])
        for k, fn in closers.items():
            got = set(fn(words, it["fact_ids"], W, H, it["question"])) - set(it["fact_ids"])
            r = res[k]
            r["tp"] += len(got & pos)
            r["fp"] += len(got - pos)
            r["fn"] += len(pos - got)
            r["size"].append(len(got))
    out = {}
    for k, r in res.items():
        p = r["tp"] / max(1, r["tp"] + r["fp"])
        rc = r["tp"] / max(1, r["tp"] + r["fn"])
        out[k] = dict(precision=p, recall=rc, f1=2 * p * rc / max(1e-9, p + rc), size=float(np.mean(r["size"])))
    return out


def pick_tau(model, items, wf, target, use_query):
    lo, hi = 0.0, 1.0
    for _ in range(18):
        mid = (lo + hi) / 2
        lc = LearnedClosure(model, mid, use_query)
        size = np.mean([len(lc(*wf(it)[:1], it["fact_ids"], *wf(it)[1], it["question"])) - len(set(it["fact_ids"]))
                        for it in items])
        lo, hi = (mid, hi) if size > target else (lo, mid)
    return (lo + hi) / 2


def main():
    tr_all, n_el = build_set(n=10**9, seed=0, split="train")
    print("train eligible", n_el, "items", len(tr_all))
    docs = sorted({it["uid"] for it in tr_all})
    random.Random(0).shuffle(docs)
    val_docs = set(docs[: len(docs) // 10])
    tr = [it for it in tr_all if it["uid"] not in val_docs]
    va = [it for it in tr_all if it["uid"] in val_docs]
    wtr = words_of_split("train")
    dev = json.load(open(DATA / "e1_set.json"))
    wdev = words_of_split("dev")

    def heur(words, seeds, W, H, q=None):
        return closure_words(words, seeds, W)

    target = np.mean([len(heur(*wtr(it)[:1], it["fact_ids"], *wtr(it)[1])) - len(set(it["fact_ids"])) for it in va])
    print(f"heuristic mean closure size (train-val) = {target:.2f}")
    closers = {"heuristic": heur}
    meta = {}
    for name, uq in (("geo", False), ("query", True)):
        X, Y, _ = build_pairs(tr, wtr, uq)
        print(f"[{name}] pairs {len(X)} positive rate {Y.mean():.3f}")
        m = train(X, Y)
        tau = pick_tau(m, va, wtr, target, uq)
        torch.save(dict(state=m.state_dict(), d=X.shape[1], tau=tau, use_query=uq), ROOT / "results" / f"closure_{name}.pt")
        closers[name] = LearnedClosure(m, tau, uq)
        meta[name] = dict(tau=tau, pairs=int(len(X)), pos_rate=float(Y.mean()))
    ev_va = eval_sets(va, wtr, closers)
    ev_dev = eval_sets(dev, wdev, closers)
    lines = ["# Learned closure vs heuristic (weak labels)", "",
             f"Train: {len(tr)} câu / {len({i['uid'] for i in tr})} tài liệu; held-out train: {len(va)} câu. "
             f"Dev = tập E1 (600 câu). τ chọn trên held-out train để kích thước closure bằng heuristic "
             f"({target:.2f} từ ngoài seed).", "",
             "Nhãn yếu: word mà câu hỏi nhắc tới, hoặc từ scale gold. `query` thấy đặc trưng trùng-từ-với-câu-hỏi "
             "nên recall của nó gần như là trần (nhãn được định nghĩa từ chính câu hỏi), chỉ để tham khảo.", "",
             "| Closure | split | size | precision | recall | F1 |", "|---|---|---|---|---|---|"]
    for split, ev in (("train-val", ev_va), ("dev E1", ev_dev)):
        for k, r in ev.items():
            lines.append(f"| {k} | {split} | {r['size']:.1f} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |")
    lines += ["", "```", json.dumps(meta, indent=1), "```"]
    (ROOT / "results" / "closure_eval.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
