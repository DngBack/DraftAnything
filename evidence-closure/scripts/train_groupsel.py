"""Train the learned group selector on TAT-DQA train; report recall@k on E1 dev and E3 test sets
against the lexical scorer. Saves results/groupsel.pt and results/groupsel_eval.md."""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evc.data import DATA, build_set, load_doc, page_words  # noqa: E402
from evc.groupsel import group_features, page_groups, train_listwise  # noqa: E402
from evc.learned import norm, q_tokens  # noqa: E402

torch.set_num_threads(8)
_cache = {}


def page(uid, split):
    k = (uid, split)
    if k not in _cache:
        words, (W, H) = page_words(load_doc(uid, split))
        _cache[k] = (words, W, H, page_groups(words, W))
    return _cache[k]


def sample(it, split):
    words, W, H, groups = page(it["uid"], split)
    if not groups:
        return None
    X = group_features(words, groups, it["question"], W, H)
    y = np.array([float(g["seed"] in set(it["fact_ids"])) for g in groups], dtype=np.float32)
    lex = np.array([len(({norm(words[j]["text"]) for j in g["words"][1:]} - {""}) & q_tokens(it["question"]))
                    for g in groups], dtype=np.float32)
    return X, y, lex


def recall_at(scores, y, ks=(1, 2, 4, 8)):
    o = np.argsort(-scores, kind="stable")
    pos = set(np.flatnonzero(y))
    return {k: len(set(o[:k]) & pos) / max(1, len(pos)) for k in ks}


def evaluate(model, items, split):
    R = {"lex": [], "learned": []}
    for it in items:
        s = sample(it, split)
        if s is None or s[1].sum() == 0:
            continue
        X, y, lex = s
        with torch.no_grad():
            p = model(torch.tensor(X)).numpy()
        R["lex"].append(recall_at(lex + 1e-6 * np.arange(len(lex))[::-1], y))
        R["learned"].append(recall_at(p, y))
    return {k: {m: float(np.mean([r[m] for r in v])) for m in (1, 2, 4, 8)} for k, v in R.items()}, len(R["lex"])


def main():
    tr, _ = build_set(n=10**9, seed=0, split="train")
    docs = sorted({i["uid"] for i in tr})
    random.Random(0).shuffle(docs)
    va_docs = set(docs[: len(docs) // 10])
    samples = []
    for it in tr:
        if it["uid"] in va_docs:
            continue
        s = sample(it, "train")
        if s is not None and s[1].sum() > 0:
            samples.append((s[0], s[1]))
    print("train questions", len(samples), flush=True)
    m = train_listwise(samples)
    torch.save(dict(state=m.state_dict(), d=samples[0][0].shape[1]), ROOT / "results" / "groupsel.pt")
    L = ["# Learned group selector — recall@k của nhóm gold", "",
         "Nhóm = số + closure heuristic. Học từ fact gold của TAT-DQA train (listwise). `lex` = số từ nhãn trùng câu hỏi.", "",
         "| tập | scorer | n | @1 | @2 | @4 | @8 |", "|---|---|---|---|---|---|---|"]
    for name, items, split in (("train-val", [i for i in tr if i["uid"] in va_docs], "train"),
                               ("dev E1", json.load(open(DATA / "e1_set.json")), "dev"),
                               ("test E3", json.load(open(DATA / "e3_test_set.json")), "test")):
        ev, n = evaluate(m, items, split)
        for k, r in ev.items():
            L.append(f"| {name} | {k} | {n} | " + " | ".join(f"{100 * r[x]:.1f}" for x in (1, 2, 4, 8)) + " |")
    (ROOT / "results" / "groupsel_eval.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
