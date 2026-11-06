"""E2: learned closure vs heuristic closure inside the VLM at matched token budget.

Same set, same attention scores (deterministic recomputation) as E1. `values` and heuristic
`closure` are re-run here so every paired comparison is within one batch run.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evc.closure import Grid, attn_closure, closure_words, fill, group_closure  # noqa: E402
from evc.data import DATA, load_doc, page_words  # noqa: E402
from evc.learned import LearnedClosure, PairMLP  # noqa: E402
from evc.metrics import scale_correct, value_correct  # noqa: E402
from evc.vlm import VLM  # noqa: E402

RHOS = (0.02, 0.05, 0.10)


def load_closer(name):
    ck = torch.load(ROOT / "results" / f"closure_{name}.pt")
    m = PairMLP(ck["d"])
    m.load_state_dict(ck["state"])
    return LearnedClosure(m.eval(), ck["tau"], ck["use_query"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "hf" / "Qwen2.5-VL-7B-Instruct"))
    ap.add_argument("--tag", default="qwen25vl7b")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bs", type=int, default=14)
    args = ap.parse_args()
    geo, qry = load_closer("geo"), load_closer("query")
    items = json.load(open(DATA / "e1_set.json"))[: args.limit or None]
    out = ROOT / "results" / f"e2_{args.tag}.jsonl"
    done = {json.loads(l)["qid"] for l in open(out)} if out.exists() else set()
    vlm = VLM(args.model, device="cuda:0")
    f = open(out, "a")
    t0 = time.time()
    for n, it in enumerate(items):
        if it["qid"] in done:
            continue
        words, (W, H) = page_words(load_doc(it["uid"]))
        img = Image.open(DATA / "pages" / f"{it['uid']}.png").convert("RGB")
        inp = vlm.prepare(img, it["question"])
        emb, pos = vlm.encode(inp)
        sc = vlm.scores(inp, emb, pos)
        grid = Grid((W, H), inp.grid)
        fids = it["fact_ids"]
        fact = grid.word_cells(words, fids)
        wl = {"heur": closure_words(words, fids, W),
              "geo": geo(words, fids, W, H),
              "query": qry(words, fids, W, H, it["question"])}
        cl = {k: grid.word_cells(words, v) for k, v in wl.items()}
        conds = {"values": fact, "closure": cl["heur"], "closure_geo": cl["geo"], "closure_query": cl["query"]}
        for r in RHOS:
            K = max(1, round(r * grid.n))
            conds[f"C_gold_geo@{r}"] = fill(cl["geo"], sc, K)
            conds[f"C_gold_query@{r}"] = fill(cl["query"], sc, K)
            gfn = lambda ws, ids: geo(ws, ids, W, H)  # noqa: E731
            conds[f"C_attn_geo@{r}"] = attn_closure(words, grid, sc, K, W, clos_fn=gfn)
            conds[f"G_attn@{r}"] = group_closure(words, grid, sc, K, W)
            conds[f"G_attn_geo@{r}"] = group_closure(words, grid, sc, K, W, clos_fn=gfn)
        preds = vlm.answer_batch(emb, pos, inp, list(conds.values()), bs=args.bs)
        rec = dict(qid=it["qid"], uid=it["uid"], answer_type=it["answer_type"], scale=it["scale"],
                   n_facts=it["n_facts"], answer=it["answer"], N=grid.n,
                   closure_words={k: [words[i]["text"] for i in v] for k, v in wl.items()}, res={})
        fset = set(fact)
        for (name, cells), pred in zip(conds.items(), preds):
            s = set(cells)
            rec["res"][name] = dict(pred=pred, v=value_correct(pred, it["answer"]),
                                    s=scale_correct(pred, it["scale"]), K=len(s),
                                    fact_rec=len(s & fset) / max(1, len(fset)))
        f.write(json.dumps(rec) + "\n")
        f.flush()
        print(f"[{n + 1}/{len(items)}] {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
