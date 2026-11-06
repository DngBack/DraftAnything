"""E3: confirmation on TAT-DQA test (unseen). Conditions locked after E1/E2 (docs/deviations.md #5-6).

Claims tested:
  (a) structural gap at tight budget: closure > values, closure > local, C_gold > L_gold @0.02
  (b) attention-seeded closure vs independent pruning: sign of C_attn - B_attn (model dependent in E1)
  (c) group scoring: G_attn vs C_attn / B_attn
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
from evc.closure import Grid, attn_closure, closure_words, fill, group_closure  # noqa: E402
from evc.data import DATA, load_doc, page_words  # noqa: E402
from evc.metrics import scale_correct, value_correct  # noqa: E402
from evc.vlm import load_vlm  # noqa: E402

RHOS = (0.02, 0.10, 0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "hf" / "Qwen2.5-VL-7B-Instruct"))
    ap.add_argument("--tag", default="qwen25vl7b")
    ap.add_argument("--bs", type=int, default=14)
    args = ap.parse_args()
    items = json.load(open(DATA / "e3_test_set.json"))
    out = ROOT / "results" / f"e3_{args.tag}.jsonl"
    done = {json.loads(l)["qid"] for l in open(out)} if out.exists() else set()
    vlm = load_vlm(args.model)
    f = open(out, "a")
    t0 = time.time()
    for n, it in enumerate(items):
        if it["qid"] in done:
            continue
        words, (W, H) = page_words(load_doc(it["uid"], "test"))
        img = Image.open(DATA / "pages" / f"{it['uid']}.png").convert("RGB")
        inp = vlm.prepare(img, it["question"])
        emb, pos = vlm.encode(inp)
        sc = vlm.scores(inp, emb, pos)
        grid = Grid((W, H), inp.grid)
        fids = it["fact_ids"]
        fact = grid.word_cells(words, fids)
        clos = grid.word_cells(words, closure_words(words, fids, W))
        local = fact + grid.ring(fact, len(clos) - len(fact))
        order = [int(c) for c in np.argsort(-sc)]
        conds = {"full": None, "values": fact, "closure": clos, "local": local}
        for r in RHOS:
            K = max(1, round(r * grid.n))
            conds[f"B_attn@{r}"] = order[:K]
            conds[f"C_attn@{r}"] = attn_closure(words, grid, sc, K, W)
            conds[f"G_attn@{r}"] = group_closure(words, grid, sc, K, W)
            conds[f"B_gold@{r}"] = fill(fact, sc, K)
            conds[f"C_gold@{r}"] = fill(clos, sc, K)
            conds[f"L_gold@{r}"] = fill(local, sc, K)
        preds = vlm.answer_batch(emb, pos, inp, list(conds.values()), bs=args.bs)
        rec = dict(qid=it["qid"], uid=it["uid"], answer_type=it["answer_type"], scale=it["scale"],
                   n_facts=it["n_facts"], answer=it["answer"], N=grid.n, n_fact_cells=len(fact),
                   n_clos_cells=len(clos), res={})
        fset = set(fact)
        for (name, cells), pred in zip(conds.items(), preds):
            s = set(range(grid.n)) if cells is None else set(cells)
            rec["res"][name] = dict(pred=pred, v=value_correct(pred, it["answer"]),
                                    s=scale_correct(pred, it["scale"]), K=len(s),
                                    fact_rec=len(s & fset) / max(1, len(fset)), rel_rec=float("nan"))
        f.write(json.dumps(rec) + "\n")
        f.flush()
        print(f"[{n + 1}/{len(items)}] {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
