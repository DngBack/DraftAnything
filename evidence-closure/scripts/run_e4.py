"""E4 (dev, exploratory): answerability-scored evidence groups.

Candidate groups = every numeric word on the page + its heuristic closure (row label, column
header, unit). Group scorers:
  lex : number of normalised label tokens shared with the question (+ tiny attention tie-break)
  vlm : logit(Yes) - logit(No) of "does the visible part contain the needed information?",
        with ONLY that group's tokens kept (one batched prefill for all groups of a page)
  mix : rank-average of lex and vlm
Selection: add whole groups best-first until K tokens, fill the rest by attention.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evc.closure import Grid, closure_words, fill  # noqa: E402
from evc.data import DATA, is_num, load_doc, page_words  # noqa: E402
from evc.learned import norm, q_tokens  # noqa: E402
from evc.metrics import scale_correct, value_correct  # noqa: E402
from evc.vlm import VLM  # noqa: E402

RHOS = (0.02, 0.05, 0.10)


def select(groups, order, sc, K):
    keep, s = [], set()
    for gi in order:
        if len(keep) >= K:
            break
        for c in groups[gi]["cells"]:
            if c not in s and len(keep) < K:
                keep.append(c)
                s.add(c)
    return fill(keep, sc, K)


def rankavg(*scores):
    r = [np.argsort(np.argsort(-np.asarray(s))) for s in scores]
    return -np.mean(r, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "hf" / "Qwen2.5-VL-7B-Instruct"))
    ap.add_argument("--tag", default="qwen25vl7b")
    ap.add_argument("--set", default="e1_set.json")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--prefix", default="e4")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bs", type=int, default=14)
    args = ap.parse_args()
    items = json.load(open(DATA / args.set))[: args.limit or None]
    out = ROOT / "results" / f"{args.prefix}_{args.tag}.jsonl"
    done = {json.loads(l)["qid"] for l in open(out)} if out.exists() else set()
    vlm = VLM(args.model, device="cuda:0")
    f = open(out, "a")
    t0 = time.time()
    for n, it in enumerate(items):
        if it["qid"] in done:
            continue
        words, (W, H) = page_words(load_doc(it["uid"], args.split))
        img = Image.open(DATA / "pages" / f"{it['uid']}.png").convert("RGB")
        inp = vlm.prepare(img, it["question"])
        emb, pos = vlm.encode(inp)
        sc = vlm.scores(inp, emb, pos)
        grid = Grid((W, H), inp.grid)
        fids = it["fact_ids"]
        fset_w = set(fids)
        qt = q_tokens(it["question"])
        wmax = np.array([max(sc[c] for c in grid.cells(w["bbox"])) for w in words])
        groups = []
        for i, w in enumerate(words):
            if not is_num(w["text"]):
                continue
            g = closure_words(words, [i], W)
            labels = {norm(words[j]["text"]) for j in g[1:]} - {""}
            groups.append(dict(seed=i, words=g, cells=grid.word_cells(words, g),
                               lex=len(labels & qt) + 1e-3 * float(wmax[g].mean())))
        t1 = time.time()
        rin = vlm.prepare_rel(img, it["question"])
        remb, rpos = vlm.encode(rin)
        vs = vlm.yes_scores(remb, rpos, rin, [g["cells"] for g in groups])
        t_vlm = time.time() - t1
        for g, v in zip(groups, vs):
            g["vlm"] = v
        scores = {"lex": [g["lex"] for g in groups], "vlm": [g["vlm"] for g in groups]}
        scores["mix"] = rankavg(scores["lex"], scores["vlm"])
        orders = {k: list(np.argsort(-np.asarray(v), kind="stable")) for k, v in scores.items()}
        gold_g = [k for k, g in enumerate(groups) if g["seed"] in fset_w]
        rec_at = {k: {str(m): len(set(o[:m]) & set(gold_g)) / max(1, len(gold_g)) for m in (1, 2, 4, 8)}
                  for k, o in orders.items()}
        fact = grid.word_cells(words, fids)
        clos = grid.word_cells(words, closure_words(words, fids, W))
        order_attn = [int(c) for c in np.argsort(-sc)]
        conds = {}
        for k in ("lex", "vlm", "mix"):  # top groups only, no fill: how good is the ranking itself
            m = max(1, it["n_facts"])
            conds[f"top_{k}"] = [c for gi in orders[k][:m] for c in groups[gi]["cells"]]
        for r in RHOS:
            K = max(1, round(r * grid.n))
            conds[f"B_attn@{r}"] = order_attn[:K]
            conds[f"C_gold@{r}"] = fill(clos, sc, K)
            for k in ("lex", "vlm", "mix"):
                conds[f"G_{k}@{r}"] = select(groups, orders[k], sc, K)
        preds = vlm.answer_batch(emb, pos, inp, list(conds.values()), bs=args.bs)
        rec = dict(qid=it["qid"], uid=it["uid"], answer_type=it["answer_type"], scale=it["scale"],
                   n_facts=it["n_facts"], answer=it["answer"], N=grid.n, n_groups=len(groups),
                   n_gold_groups=len(gold_g), recall_at=rec_at, t_vlm_score=t_vlm, res={})
        fset = set(fact)
        for (name, cells), pred in zip(conds.items(), preds):
            s = set(cells)
            rec["res"][name] = dict(pred=pred, v=value_correct(pred, it["answer"]),
                                    s=scale_correct(pred, it["scale"]), K=len(s),
                                    fact_rec=len(s & fset) / max(1, len(fset)))
        f.write(json.dumps(rec) + "\n")
        f.flush()
        print(f"[{n + 1}/{len(items)}] {(time.time() - t0) / 60:.1f} min groups={len(groups)} "
              f"vlm_score={t_vlm:.1f}s", flush=True)


if __name__ == "__main__":
    main()
