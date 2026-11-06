"""E1: structural information gap. See docs/protocol.md."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evc.closure import Grid, attn_closure, closure_words, fill  # noqa: E402
from evc.data import DATA, load_doc, page_words  # noqa: E402
from evc.metrics import scale_correct, value_correct  # noqa: E402
from evc.vlm import VLM  # noqa: E402

RHOS = (0.02, 0.05, 0.10, 0.25, 0.50)


def conditions(words, grid, sc, fact_ids, page_w):
    N = grid.n
    fact_cells = grid.word_cells(words, fact_ids)
    clos_ids = closure_words(words, fact_ids, page_w)
    clos_cells = grid.word_cells(words, clos_ids)  # facts first
    n_rel = len(clos_cells) - len(fact_cells)
    local_cells = fact_cells + grid.ring(fact_cells, n_rel)
    order = [int(c) for c in np.argsort(-sc)]
    conds = {"values": fact_cells, "closure": clos_cells, "local": local_cells}
    for r in RHOS:
        K = max(1, round(r * N))
        conds[f"B_attn@{r}"] = order[:K]
        conds[f"C_attn@{r}"] = attn_closure(words, grid, sc, K, page_w)
        conds[f"B_gold@{r}"] = fill(fact_cells, sc, K)
        conds[f"C_gold@{r}"] = fill(clos_cells, sc, K)
        conds[f"L_gold@{r}"] = fill(local_cells, sc, K)
    meta = dict(N=N, n_fact_cells=len(fact_cells), n_clos_cells=len(clos_cells),
                closure_words=[words[i]["text"] for i in clos_ids])
    return conds, set(fact_cells), set(clos_cells), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path(__file__).resolve().parents[1] / "hf" / "Qwen2.5-VL-7B-Instruct"))
    ap.add_argument("--tag", default="qwen25vl7b")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bs", type=int, default=13)
    args = ap.parse_args()

    items = json.load(open(DATA / "e1_set.json"))
    if args.limit:
        items = items[: args.limit]
    out = Path(__file__).resolve().parents[1] / "results" / f"e1_{args.tag}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(l)["qid"] for l in open(out)}
    vlm = VLM(args.model, device=args.device)
    f = open(out, "a")
    t0 = time.time()
    for n, it in enumerate(items):
        if it["qid"] in done:
            continue
        words, size = page_words(load_doc(it["uid"]))
        img = Image.open(DATA / "pages" / f"{it['uid']}.png").convert("RGB")
        inp = vlm.prepare(img, it["question"])
        emb, pos = vlm.encode(inp)
        sc = vlm.scores(inp, emb, pos)
        grid = Grid(size, inp.grid)
        assert grid.n == len(sc), (grid.n, len(sc))
        conds, fset, cset, meta = conditions(words, grid, sc, it["fact_ids"], size[0])
        rec = dict(qid=it["qid"], uid=it["uid"], answer_type=it["answer_type"], scale=it["scale"],
                   n_facts=it["n_facts"], answer=it["answer"], **meta, res={})
        names = ["full"] + list(conds)
        preds = vlm.answer_batch(emb, pos, inp, [None] + [conds[k] for k in conds], bs=args.bs)
        rel = cset - fset
        for name, pred in zip(names, preds):
            s = set(range(grid.n)) if name == "full" else set(conds[name])
            rec["res"][name] = dict(pred=pred, v=value_correct(pred, it["answer"]),
                                    s=scale_correct(pred, it["scale"]), K=len(s),
                                    fact_rec=len(s & fset) / max(1, len(fset)),
                                    rel_rec=len(s & rel) / max(1, len(rel)))
        f.write(json.dumps(rec) + "\n")
        f.flush()
        el = time.time() - t0
        print(f"[{n + 1}/{len(items)}] {el / 60:.1f} min  full={rec['res']['full']['v']} "
              f"B10={rec['res']['B_attn@0.1']['v']} C10={rec['res']['C_attn@0.1']['v']} "
              f"vals={rec['res']['values']['v']} clos={rec['res']['closure']['v']}", flush=True)


if __name__ == "__main__":
    main()
