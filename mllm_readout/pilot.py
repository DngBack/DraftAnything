"""Pilot G1 (docs/pilot_g1_protocol.md): probes, interchange patching, probe-guided repair.

usage: python pilot.py --model qwen3vl4b --family count --n 600
"""
import argparse, functools, json, os
import numpy as np, pandas as pd, torch
from transformers import AutoModelForImageTextToText, AutoProcessor
import stimuli as S

MODELS = {"qwen3vl4b": "Qwen/Qwen3-VL-4B-Instruct", "gemma3_4b": "google/gemma-3-4b-it",
          "qwen3vl8b": "Qwen/Qwen3-VL-8B-Instruct", "internvl35_8b": "OpenGVLab/InternVL3_5-8B-HF",
          "gemma3_27b": "google/gemma-3-27b-it",
          # competitor settings (2604.09364)
          "llava15_7b": "llava-hf/llava-1.5-7b-hf", "qwen2vl_7b": "Qwen/Qwen2-VL-7B-Instruct",
          "qwen3vl2b": "Qwen/Qwen3-VL-2B-Instruct",  # 2607.09544
          "qwen3vl32b": "Qwen/Qwen3-VL-32B-Instruct",
          # held-out models for the confirmatory steering test (docs/g3_protocol.md, addendum 4)
          "qwen35_2b": "Qwen/Qwen3.5-2B", "qwen35_4b": "Qwen/Qwen3.5-4B", "qwen35_9b": "Qwen/Qwen3.5-9B",
          "qwen2vl_2b": "Qwen/Qwen2-VL-2B-Instruct"}
WHATSUP_REL = {" on a ": "on", " under a ": "under", " to the left of a ": "left", " to the right of a ": "right",
               " in front of a ": "front", " behind a ": "behind"}
CANDS = {"count": [str(i) for i in range(1, 10)], "spatial": S.REL, "count_real": [str(i) for i in range(1, 10)],
         "attr": S.ATTR_COLORS, "spatial_real": list(WHATSUP_REL.values()),
         "color_cf": ["purple", "pink", "green", "blue", "yellow", "orange", "red", "brown"]}


def countbench_items():
    """CountBenchQA (real photos), counts 2-9 so every answer is one digit token."""
    from datasets import load_dataset
    out = []
    for r in load_dataset("vikhyatk/CountBenchQA", split="test"):
        if 2 <= int(r["number"]) <= 9:
            im = r["image"].convert("RGB")
            im.thumbnail((768, 768))
            out.append(dict(family="count_real", label=str(r["number"]), image=im,
                            q=r["question"] + " Answer with a single digit."))
    return out


def whatsup_items():
    """What'sUp Controlled_Images A+B (real photos); the first caption is the gold one."""
    from datasets import load_dataset
    out = []
    for sp in ("Controlled_Images_A", "Controlled_Images_B"):
        for r in load_dataset("ServiceNow/whatsup_all", split=sp):
            caps = r["caption_options"]
            rels = [next(v for k, v in WHATSUP_REL.items() if k in c) for c in caps]
            k = next(k for k in WHATSUP_REL if k in caps[0])
            a, b = caps[0][2:].split(k)
            opts = sorted(rels, key=CANDS["spatial_real"].index)
            im = r["image_options"].convert("RGB")
            im.thumbnail((768, 768))
            out.append(dict(family="spatial_real", label=rels[0], image=im,
                            q=f"Where is the {a} relative to the {b}? Answer with one word: {', '.join(opts[:-1])}, or {opts[-1]}."))
    return out


def counterfact_items():
    """Visual CounterFact colour split (2604.09364): object recoloured against its prior; gold = visible colour."""
    from datasets import load_dataset
    out = []
    for r in load_dataset("mgolov/Visual-Counterfact", split="color"):
        im = r["counterfact_image"].convert("RGB")
        im.thumbnail((768, 768))
        out.append(dict(family="color_cf", label=r["incorrect_answer"], image=im, prior=r["correct_answer"],
                        q=f"What color is the {r['object']} in the image? Answer with one word."))
    return out


OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


class Runner:
    def __init__(self, name, cands):
        self.m = AutoModelForImageTextToText.from_pretrained(MODELS[name], dtype=torch.bfloat16, device_map="cuda").eval()
        self.p = AutoProcessor.from_pretrained(MODELS[name])
        if name.startswith("qwen35"):  # non-thinking mode, the counterpart of the Instruct models
            self.p.apply_chat_template = functools.partial(self.p.apply_chat_template, enable_thinking=False)
        self.layers = self.m.model.language_model.layers
        cfg = self.m.config
        self.img_id = getattr(cfg, "image_token_id", None) or cfg.image_token_index
        tok = self.p.tokenizer
        # each class scored by its best surface variant's first token
        self.cand = [sorted({tok.encode(v, add_special_tokens=False)[0]
                             for v in (c, c.capitalize(), " " + c, " " + c.capitalize())}) for c in cands]

    def inputs(self, item):
        msgs = [{"role": "user", "content": [{"type": "image", "image": item["image"] if "image" in item else S.render(item["objs"])},
                                             {"type": "text", "text": item["q"]}]}]
        return self.p.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                          return_dict=True, return_tensors="pt").to("cuda")

    @torch.no_grad()
    def run(self, inp, edits=None, hidden=False):
        """edits: {layer: fn(h)->h} applied to that decoder layer's output hidden state (batch 1)."""
        hooks = []
        for L, fn in (edits or {}).items():
            def hook(mod, args, out, fn=fn):
                if isinstance(out, tuple):
                    return (fn(out[0]),) + out[1:]
                return fn(out)
            hooks.append(self.layers[L].register_forward_hook(hook))
        try:
            o = self.m(**inp, output_hidden_states=hidden)
        finally:
            for h in hooks:
                h.remove()
        lg = o.logits[0, -1].float()
        scores = torch.stack([lg[ids].max() for ids in self.cand])
        return scores, (o.hidden_states if hidden else None)


def probe_cv(X, y, k=5, steps=300, wd=1e-3, seed=0, groups=None):
    """Multinomial logistic regression on GPU, k-fold CV predictions.
    X: (n, d) -> preds (n,); or (n, B, d) -> one probe per slice, preds (B, n).
    groups: items sharing a group (e.g. one image at several resolutions) are kept in the same fold."""
    X = torch.as_tensor(X, dtype=torch.float32, device="cuda")
    squeeze = X.dim() == 2
    X = X[:, None] if squeeze else X
    y = torch.as_tensor(y, device="cuda")
    B, C = X.shape[1], int(y.max()) + 1
    g = torch.Generator().manual_seed(seed)
    if groups is None:
        fold = torch.randperm(len(y), generator=g).to("cuda") % k
    else:
        u, inv = np.unique(groups, return_inverse=True)
        fold = (torch.randperm(len(u), generator=g) % k)[torch.as_tensor(inv)].to("cuda")
    pred = torch.empty(B, len(y), dtype=torch.long, device="cuda")
    for f in range(k):
        tr, te = fold != f, fold == f
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-4
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        W = torch.zeros(B, X.shape[2], C, device="cuda", requires_grad=True)
        b = torch.zeros(B, C, device="cuda", requires_grad=True)
        opt = torch.optim.Adam([W, b], lr=1e-2, weight_decay=wd)
        ytr = y[tr][:, None].expand(-1, B).reshape(-1)
        for _ in range(steps):
            opt.zero_grad()
            lg = torch.einsum("nbd,bdc->nbc", Xtr, W) + b
            torch.nn.functional.cross_entropy(lg.reshape(-1, C), ytr).backward()
            opt.step()
        with torch.no_grad():
            pred[:, te] = (torch.einsum("nbd,bdc->nbc", Xte, W) + b).argmax(2).T
    pred = pred.cpu().numpy()
    return (pred[0] if squeeze else pred), fold.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--family", required=True, choices=CANDS)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--npairs", type=int, default=64)
    ap.add_argument("--states_only", action="store_true")  # R2 E: extraction only
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tag = f"{a.model}_{a.family}"
    cands = CANDS[a.family]
    if a.family in ("count_real", "spatial_real", "color_cf"):
        real = {"count_real": countbench_items, "spatial_real": whatsup_items, "color_cf": counterfact_items}[a.family]()
        a.n = min(a.n, len(real))
        gen = lambda s: (real[s], None)
    else:
        gen = {"count": S.count_item, "spatial": S.spatial_item, "attr": S.attr_item}[a.family]
    R = Runner(a.model, cands)
    nL = len(R.layers)

    # 1) answers + hidden states
    items, last, imgm, ans, scores = [], [], [], [], []
    for s in range(a.n):
        it, _ = gen(s)
        inp = R.inputs(it)
        sc, hs = R.run(inp, hidden=True)
        im = (inp["input_ids"][0] == R.img_id)
        last.append(torch.stack([h[0, -1] for h in hs]).float().cpu().half())
        imgm.append(torch.stack([h[0, im].mean(0) for h in hs]).float().cpu().half())
        ans.append(int(sc.argmax()))
        scores.append(sc.cpu().numpy())
        items.append(it)
    y = np.array([cands.index(it["label"]) for it in items])
    ans = np.array(ans)
    last, imgm = torch.stack(last).numpy(), torch.stack(imgm).numpy()  # (n, nL+1, d)
    correct = ans == y
    scores = np.stack(scores)
    np.savez(f"{OUT}/{tag}_states.npz", y=y, ans=ans, scores=scores, last=last, imgm=imgm)
    print(tag, "model acc", correct.mean(), flush=True)
    if a.states_only:
        return

    # 2) probes per layer (hidden_states[0] = embeddings, [i+1] = layer i output)
    rows = []
    yperm = np.random.default_rng(0).permutation(y)
    for pos, X in (("last", last), ("img", imgm)):
        pr, _ = probe_cv(X, y)
        ctl, _ = probe_cv(X, yperm)
        for li in range(nL + 1):
            rows.append(dict(layer=li, pos=pos, acc=(pr[li] == y).mean(), acc_on_err=(pr[li] == y)[~correct].mean(),
                             ctl_acc=(ctl[li] == yperm).mean()))
    probes = pd.DataFrame(rows)
    probes.to_parquet(f"{OUT}/{tag}_probes.parquet")

    # 3) interchange patching on counterfactual pairs where both runs are correct
    prow, used = [], 0
    for s in range(a.n):
        if used >= a.npairs:
            break
        it, pt = gen(s)
        if pt is None or not correct[s]:
            continue
        ti, si = R.inputs(it), R.inputs(pt)
        if ti["input_ids"].shape != si["input_ids"].shape:
            continue
        ssc, shs = R.run(si, hidden=True)
        if int(ssc.argmax()) != cands.index(pt["label"]):
            continue
        used += 1
        im = (ti["input_ids"][0] == R.img_id)
        for L in range(0, nL, 2):
            for pos in ("img", "last"):
                src = shs[L + 1][0]
                def fn(h, src=src, pos=pos):
                    h = h.clone()
                    if pos == "img":
                        h[0, im] = src[im]
                    else:
                        h[0, -1] = src[-1]
                    return h
                sc, _ = R.run(ti, {L: fn})
                prow.append(dict(pair=s, layer=L, pos=pos, to_source=int(sc.argmax()) == cands.index(pt["label"])))
    patch = pd.DataFrame(prow)
    patch.to_parquet(f"{OUT}/{tag}_patch.parquet")

    # 4) probe-guided repair (cross-fitted: probe + class means from train folds only)
    L0 = round(0.75 * nL)
    rrow = []
    k = 5
    fold = np.arange(a.n) % k
    for f in range(k):
        tr, te = fold != f, fold == f
        # choose probe layer by inner CV on train fold only
        cand_layers = list(range(1, nL + 1, 2))
        pr, _ = probe_cv(last[tr][:, cand_layers], y[tr], k=4)
        best = cand_layers[int((pr == y[tr]).mean(1).argmax())]
        Xtr = torch.tensor(last[tr, best], dtype=torch.float32, device="cuda")
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-4
        W = torch.zeros(Xtr.shape[1], len(cands), device="cuda", requires_grad=True)
        b = torch.zeros(len(cands), device="cuda", requires_grad=True)
        opt = torch.optim.Adam([W, b], lr=1e-2, weight_decay=1e-3)
        yt = torch.tensor(y[tr], device="cuda")
        for _ in range(300):
            opt.zero_grad()
            torch.nn.functional.cross_entropy(((Xtr - mu) / sd) @ W + b, yt).backward()
            opt.step()
        Xte = torch.tensor(last[te, best], dtype=torch.float32, device="cuda")
        ppred = (((Xte - mu) / sd) @ W + b).argmax(1).cpu().numpy()
        for L in sorted({L0, round(0.5 * nL), round(0.9 * nL)}):
            H = torch.tensor(last[tr, L + 1], dtype=torch.float32, device="cuda")
            means = {c: H[torch.tensor(y[tr] == c)].mean(0) for c in np.unique(y[tr])}
            for j, s in enumerate(np.where(te)[0]):
                p, m_ = int(ppred[j]), int(ans[s])
                for alpha in (0.5, 1.0, 2.0):
                    if p == m_ or p not in means or m_ not in means:
                        new = m_
                    else:
                        d = (alpha * (means[p] - means[m_])).to(torch.bfloat16)
                        def fn(h, d=d):
                            h = h.clone(); h[0, -1] += d; return h
                        sc, _ = R.run(R.inputs(items[s]), {L: fn})
                        new = int(sc.argmax())
                    rrow.append(dict(item=s, layer=L, alpha=alpha, primary=(L == L0 and alpha == 1.0),
                                     probe_layer=best, probe_pred=p, new=new, was_correct=bool(correct[s]), now_correct=new == y[s]))
    rep = pd.DataFrame(rrow)
    rep.to_parquet(f"{OUT}/{tag}_repair.parquet")

    # summary vs pre-registered criteria
    lp = probes[probes.pos == "last"]
    bestrow = probes.loc[probes.acc.idxmax()]
    prim = rep[rep.primary]
    cal_pred, _ = probe_cv(scores, y)
    src = last if bestrow.pos == "last" else imgm
    prb_pred, _ = probe_cv(src[:, int(bestrow.layer)], y)
    err = ~correct
    bias = err & (cal_pred == y)
    loss = err & (cal_pred != y) & (prb_pred == y)
    perc = err & (cal_pred != y) & (prb_pred != y)
    summ = dict(
        tag=tag, n=a.n, n_layers=nL, model_acc=float(correct.mean()), n_err=int((~correct).sum()),
        best_probe=dict(layer=int(bestrow.layer), pos=bestrow.pos, acc=float(bestrow.acc),
                        acc_on_err=float(bestrow.acc_on_err), ctl_acc=float(bestrow.ctl_acc)),
        last_probe_final_layer_acc=float(lp[lp.layer == nL].acc.iloc[0]),
        gap_pts=100 * float(bestrow.acc - correct.mean()),
        repair_fix_rate=float(prim[~prim.was_correct].now_correct.mean()),
        repair_break_rate=float(1 - prim[prim.was_correct].now_correct.mean()),
        repair_net_acc=float(prim.now_correct.mean()),
        patch_pairs=used,
        # control (added post-registration): affine recalibration of the model's own candidate logits
        calib_acc=float((cal_pred == y).mean()),
        calib_acc_on_err=float((cal_pred == y)[err].mean()),
        err_share=dict(bias=float(bias.sum() / max(err.sum(), 1)), loss=float(loss.sum() / max(err.sum(), 1)),
                       perc=float(perc.sum() / max(err.sum(), 1))),
        patch_img_by_layer=patch[patch.pos == "img"].groupby("layer").to_source.mean().round(3).to_dict() if used else {},
        patch_last_by_layer=patch[patch.pos == "last"].groupby("layer").to_source.mean().round(3).to_dict() if used else {},
    )
    summ["G1a"] = summ["gap_pts"] >= 10 and summ["best_probe"]["acc_on_err"] >= 0.5
    summ["G1b"] = summ["repair_fix_rate"] >= 0.30 and summ["repair_break_rate"] <= 0.05
    json.dump(summ, open(f"{OUT}/{tag}_summary.json", "w"), indent=1, default=float)
    print(json.dumps(summ, indent=1, default=float), flush=True)


if __name__ == "__main__":
    main()
