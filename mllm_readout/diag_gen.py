"""G3 (docs/g3_protocol.md): the per-error decomposition at the decision step of free-form generation.

Same items, prompts, candidates, triggers and gates as gen_calib.py; generation is greedy and untouched and stops right
after the decision step. Saved per model x set: y, ans (arg-max of decision-step candidate scores), scores, hidden
states of the last prefix position (all layers) and the image-token mean (all layers, from the prompt pass).

usage: python diag_gen.py --model qwen3vl8b            (results/g3/<model>_<set>_states.npz)
       python diag_gen.py --analyze                    (results/g3_table.csv)
"""
import argparse, glob, os
import numpy as np, pandas as pd, torch
from transformers import LogitsProcessorList, StoppingCriteria, StoppingCriteriaList
from pilot import MODELS, OUT, probe_cv
from gen_calib import items, Decide, ILLUSIONS

G3 = os.environ.get("G3_DIR", f"{OUT}/g3")
SETS = {"illusions": ("vlmbias", sorted(ILLUSIONS)), "carlogos": ("vlmbias", ["Car Logos"]),
        "dice": ("vlmbias", ["Dice Pattern"]), "tally": ("vlmbias", ["Tally Pattern"]),
        "mmvp": ("mmvp", None), "pope": ("pope", None), "countbench": ("countbench", None)}


class Stop(StoppingCriteria):
    def __init__(self, d):
        self.d = d

    def __call__(self, input_ids, scores, **kw):
        return torch.tensor([self.d.done and not self.d.ban], device=input_ids.device)


def run(model):
    from pilot import Runner
    os.makedirs(G3, exist_ok=True)
    R = Runner(model, ["x"])
    tok = R.p.tokenizer
    ids_cache = {}

    def cand_ids(cands, paren=False):  # as gen_calib.run
        k = (tuple(cands), paren)
        if k not in ids_cache:
            pre = ("", " ", "(", " (") if paren else ("", " ")
            norm = (lambda t: t.strip().lstrip("(").lower()) if paren else (lambda t: t.strip().lower())
            ids_cache[k] = [sorted({i for p in pre for v in (c, c.lower(), c.capitalize())
                                    for i in tok.encode(p + v, add_special_tokens=False) if norm(tok.decode([i])) == c.lower()})
                            for c in cands]
        return ids_cache[k]

    digit_ids = sorted({i for d in range(10) for p in ("", " ") for i in tok.encode(p + str(d), add_special_tokens=False)
                        if tok.decode([i]).strip() == str(d)})

    @torch.no_grad()
    def measure(it):
        msgs = ([{"role": "system", "content": [{"type": "text", "text": it["sys"]}]}] if it["sys"] else []) + \
               [{"role": "user", "content": [{"type": "image", "image": it["image"]}, {"type": "text", "text": it["q"]}]}]
        x = R.p.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
        numeric = it["cands"][0].isdigit()
        d = Decide(cand_ids(it["cands"], it.get("gate") == "letter"), None, digit_ids if numeric else None, ())
        n0 = x["input_ids"].shape[1]
        if it.get("gate") == "{":
            d.ready = lambda ids, n0=n0: tok.decode(ids[0, n0:]).strip() == "" or tok.decode(ids[0, n0:][-4:]).rstrip().endswith("{")
        elif it.get("gate") == "letter":
            d.ready = lambda ids, n0=n0: tok.decode(ids[0, n0:]).strip().strip("*").strip() in ("", "(", "Answer:", "Answer: (")
        o = R.m.generate(**x, max_new_tokens=384 if it.get("gate") == "{" else 24, do_sample=False,
                         logits_processor=LogitsProcessorList([d]), stopping_criteria=StoppingCriteriaList([Stop(d)]),
                         output_hidden_states=True, return_dict_in_generate=True)
        if not d.done:
            return None
        t = o.sequences.shape[1] - n0 - 1  # the decided token is the last one generated
        hs = o.hidden_states  # per step: tuple over layers of (1, len, d)
        im = x["input_ids"][0] == R.img_id
        last = torch.stack([h[0, -1] for h in hs[t]]).float().cpu().half()
        imgm = torch.stack([h[0, im].mean(0) for h in hs[0]]).float().cpu().half()
        return d.scores, last, imgm

    cache = {}
    for name, (task, groups) in SETS.items():
        if name not in os.environ.get("G3_SETS", ",".join(SETS)).split(","):
            continue
        path = f"{G3}/{model}_{name}_states.npz"
        if os.path.exists(path):
            continue
        if task not in cache:
            cache = {task: items(task)}
        its = cache[task]
        if groups:
            its = [it for it in its if it["group"] in groups]
        if task == "pope":  # 1000 per setting, fixed draw
            rng = np.random.default_rng(0)
            idx = np.concatenate([rng.choice([i for i, it in enumerate(its) if it["group"] == g], 1000, replace=False)
                                  for g in ("adversarial", "popular", "random")])
            its = [its[i] for i in sorted(idx)]
        its = its[: int(os.environ.get("G3_LIMIT", 10 ** 9))]  # smoke tests only
        rows, last, imgm, nodec, offcand = [], [], [], 0, 0
        for j, it in enumerate(its):
            if it["y"] not in it["cands"]:
                offcand += 1
                continue
            r = measure(it)
            if r is None:
                nodec += 1
                continue
            sc, l, m = r
            rows.append((it["cands"].index(it["y"]), int(np.argmax(sc)), sc, it["group"], j,
                         str(it.get("key", it.get("pair", j))), it["q"]))
            last.append(l), imgm.append(m)
        y, ans, sc, grp, idx, key, q = map(np.array, zip(*rows))
        np.savez(path, y=y, ans=ans, scores=np.stack(sc), group=grp, idx=idx, key=key, q=q, last=torch.stack(last).numpy(),
                 imgm=torch.stack(imgm).numpy(), nodec=nodec, offcand=offcand, n_items=len(its))
        print(f"{model} {name}: n={len(y)} nodec={nodec} offcand={offcand} acc={np.mean(ans == y):.3f}", flush=True)


def cell(path):
    d = np.load(path)
    y, ans, scores, grp, key = d["y"], d["ans"], d["scores"], d["group"], d["key"]
    err = ans != y
    cal = np.empty_like(y)
    for g in np.unique(grp):  # calibrator per subset (as in generation); one subset except for the illusions
        k = grp == g
        cal[k] = probe_cv(scores[k], y[k], groups=key[k])[0] if len(np.unique(y[k])) > 1 else ans[k]
    # text-only control: majority train-fold label for the same prompt (folds grouped by image, as the probe's)
    u, inv = np.unique(key, return_inverse=True)
    fold = (np.random.default_rng(0).permutation(len(u)) % 5)[inv]
    text = np.empty_like(y)
    for f in range(5):
        tr = pd.DataFrame(dict(q=d["q"][fold != f], y=y[fold != f]))
        mp, maj = tr.groupby("q").y.agg(lambda v: v.mode()[0]), tr.y.mode()[0]
        text[fold == f] = [mp.get(q, maj) for q in d["q"][fold == f]]
    best = None
    for pos in ("last", "imgm"):
        pr, _ = probe_cv(d[pos], y, groups=key)
        acc = (pr == y).mean(1)
        li = int(acc.argmax())
        if best is None or acc[li] > best[0]:
            best = (acc[li], li, pos, pr[li])
    pacc, li, pos, prb = best
    ne = max(int(err.sum()), 1)
    bias = err & (cal == y)
    loss = err & (cal != y) & (prb == y)
    m, c = 1 - err.mean(), (cal == y).mean()
    nodec = int(d["nodec"]) / max(int(d["n_items"]) - int(d["offcand"]), 1)
    return dict(cell=os.path.basename(path).replace("_states.npz", ""), n=len(y), nodec=round(nodec, 3),
                text=(text == y).mean(), n_err=int(err.sum()),
                model=m, calib=c, probe=pacc, probe_at=f"L{li}/{'last' if pos == 'last' else 'img'}",
                bias=bias.sum() / ne, loss=loss.sum() / ne, perc=1 - (bias.sum() + loss.sum()) / ne,
                cal_recovers=(c - m) / (pacc - m) if pacc > m else np.nan)


def analyze():
    df = pd.DataFrame([cell(p) for p in sorted(glob.glob(f"{G3}/*_states.npz"))])
    df.to_csv(f"{OUT}/g3_table.csv", index=False)
    print(df.round(3).to_string(index=False))
    ok = df[(df.n_err >= 30) & (df.nodec <= 0.10) & (df.text < 0.75)]
    pcal = (ok.cal_recovers >= 0.5).mean()
    pread = ((ok.loss >= 0.3) & (ok.loss > ok.bias)).mean()
    print(f"\n[1] cells: {len(ok)} | Rec>=50%: {(ok.cal_recovers >= 0.5).sum()} ({pcal:.2f}) | loss-dominant: "
          f"{((ok.loss >= 0.3) & (ok.loss > ok.bias)).sum()} | verdict: {'P-cal' if pcal >= 2/3 else 'P-read' if pread >= 2/3 else 'mixed'}")
    for s, rule, txt in (("mmvp", lambda r: (r.perc > r.bias) & (r.perc > r.loss), "[2] MMVP perc > bias, loss"),
                         ("pope", lambda r: r.bias < 0.3, "[3] POPE bias < 30%")):
        c = ok[ok.cell.str.endswith("_" + s)]
        print(f"{txt}: {int(rule(c).sum())}/{len(c)} -> {'holds' if len(c) and rule(c).mean() >= 2/3 else 'fails'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    analyze() if a.analyze else run(a.model)
