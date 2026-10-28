"""G3 addendum 3 (docs/g3_protocol.md): probe-guided steering at the generation decision step, cross-fitted by image.

usage: python steer_g3.py --model qwen3vl8b      (results/g3/<model>_<set>_steer.npz)
       python steer_g3.py --analyze              (results/g3_steer.csv)
"""
import argparse, glob, os
import numpy as np, pandas as pd, torch
from transformers import LogitsProcessorList, StoppingCriteriaList
from pilot import MODELS, OUT, probe_cv
from gen_calib import items, Decide
from diag_gen import G3, Stop
from diag_blank import item_list

USE = tuple(os.environ.get("STEER_SETS", "illusions,countbench").split(","))
ALPHA = float(os.environ.get("STEER_ALPHA", 1))  # pre-registered: 1; other values = post hoc dose-response
CTRL = os.environ.get("STEER_CTRL", "")  # R1 controls (docs/r1_protocol.md D): "random" direction, "shuffle"d labels
GRID = [float(a) for a in os.environ["STEER_GRID"].split(",")] if os.environ.get("STEER_GRID") else None  # R2 F
SRC = os.environ.get("STEER_SRC", "")  # G4 R3x: "ext" = target class from the SigLIP2 grid probe, not the MLLM probe;
# writeability controls (exploratory): "oracle" = the true class, "anti" = a wrong class (seeded) -- what steering writes when told
SUF = "steergrid" if GRID else ("steer" if ALPHA == 1 else f"steerA{ALPHA:g}") + (f"_{CTRL}" if CTRL else "") + (f"_{SRC}" if SRC else "")


def fit_probe(X, y, steps=300, wd=1e-3):
    """same optimiser as pilot.probe_cv; returns a predict function"""
    X = torch.as_tensor(X, dtype=torch.float32, device="cuda"); y = torch.as_tensor(y, device="cuda")
    mu, sd = X.mean(0), X.std(0) + 1e-4
    C = int(y.max()) + 1
    W = torch.zeros(X.shape[1], C, device="cuda", requires_grad=True); b = torch.zeros(C, device="cuda", requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=1e-2, weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad(); torch.nn.functional.cross_entropy(((X - mu) / sd) @ W + b, y).backward(); opt.step()
    return lambda Z: (((torch.as_tensor(Z, dtype=torch.float32, device="cuda") - mu) / sd) @ W + b).argmax(-1).cpu().numpy()


def run(model):
    from pilot import Runner
    R = Runner(model, ["x"])
    tok = R.p.tokenizer
    nL = len(R.layers)
    L = int(round(0.75 * nL))  # layer index; its output is hidden_states[L + 1]
    ids_cache = {}

    def cand_ids(cands, paren=False):
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
    def steered_scores(it, deltas):  # one generated prefix, one decision-step pass per delta
        msgs = ([{"role": "system", "content": [{"type": "text", "text": it["sys"]}]}] if it["sys"] else []) + \
               [{"role": "user", "content": [{"type": "image", "image": it["image"]}, {"type": "text", "text": it["q"]}]}]
        x = R.p.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
        numeric = it["cands"][0].isdigit()
        cids = cand_ids(it["cands"], it.get("gate") == "letter")
        d = Decide(cids, None, digit_ids if numeric else None, ())
        n0 = x["input_ids"].shape[1]
        if it.get("gate") == "{":
            d.ready = lambda ids, n0=n0: tok.decode(ids[0, n0:]).strip() == "" or tok.decode(ids[0, n0:][-4:]).rstrip().endswith("{")
        elif it.get("gate") == "letter":  # MMVP, as diag_gen.py
            d.ready = lambda ids, n0=n0: tok.decode(ids[0, n0:]).strip().strip("*").strip() in ("", "(", "Answer:", "Answer: (")
        o = R.m.generate(**x, max_new_tokens=384 if it.get("gate") == "{" else 24, do_sample=False,
                         logits_processor=LogitsProcessorList([d]), stopping_criteria=StoppingCriteriaList([Stop(d)]))
        if not d.done:
            return None, None
        pre = o[:, :-1]
        y = dict(x)
        for k, v in x.items():  # extend per-token inputs to the generated prefix
            if torch.is_tensor(v) and v.dim() == 2 and v.shape[1] == n0 and k != "input_ids":
                y[k] = torch.cat([v, (torch.ones if k == "attention_mask" else torch.zeros)(1, pre.shape[1] - n0, dtype=v.dtype, device=v.device)], 1)
        y["input_ids"] = pre
        res = []
        for delta in deltas:
            def hook(mod, args, out, delta=delta):
                h = out[0] if isinstance(out, tuple) else out
                h[:, -1] += delta.to(h.dtype)
                return out
            hd = R.layers[L].register_forward_hook(hook)
            try:
                lg = R.m(**y).logits[0, -1].float()
            finally:
                hd.remove()
            res.append(np.array([float(lg[i].max()) for i in cids]))
        return res, d.scores

    cache = {}
    for name in USE:
        st, out = f"{G3}/{model}_{name}_states.npz", f"{G3}/{model}_{name}_{SUF}.npz"
        if os.path.exists(out) or not os.path.exists(st):
            continue
        d = np.load(st)
        task = "vlmbias" if name == "illusions" else name
        if task not in cache:
            cache = {task: items(task)}
        its = item_list(name, cache[task])
        y, ans, key = d["y"], d["ans"], d["key"]
        H = d["last"][:, L + 1].astype(np.float32)
        if SRC == "ext":  # explore_input.py features in item_list order; 4x4 grid + mean, flattened
            E = np.load(f"{OUT}/explore_input/siglip2_{name}.npz")["feats"][d["idx"], 0].reshape(len(y), -1).astype(np.float32)
        u, inv = np.unique(key, return_inverse=True)
        fold = (np.random.default_rng(0).permutation(len(u)) % 5)[inv]
        steer, probe, edited, check = ans.copy(), ans.copy(), np.zeros(len(y), bool), []
        sgrid = np.repeat(ans[:, None], len(GRID or []), 1)
        for f in range(5):
            tr, te = fold != f, fold == f
            ytr = np.random.default_rng(f).permutation(y[tr]) if CTRL == "shuffle" else y[tr]
            pred = fit_probe(H[tr], ytr)
            means = {c: H[tr][ytr == c].mean(0) for c in np.unique(ytr)}
            if SRC == "ext":
                probe[te] = fit_probe(E[tr], y[tr])(E[te])
            elif SRC == "oracle":
                probe[te] = y[te]
            elif SRC == "anti":
                cls = np.array(sorted(means))
                probe[te] = [np.random.default_rng(int(j)).choice(cls[cls != y[j]]) for j in np.where(te)[0]]
            else:
                probe[te] = pred(H[te])
            for j in np.where(te)[0]:
                p, m = int(probe[j]), int(ans[j])
                if p == m or p not in means or m not in means:
                    continue
                delta = ALPHA * (means[p] - means[m])
                if CTRL == "random":  # same norm, random direction
                    r = np.random.default_rng(j).standard_normal(delta.shape).astype(np.float32)
                    delta = r * np.linalg.norm(delta) / np.linalg.norm(r)
                if GRID:
                    dl = [torch.as_tensor(a * (means[p] - means[m]), device="cuda") for a in GRID if a != 0]
                else:
                    dl = [torch.as_tensor(delta, device="cuda")]
                sc, unst = steered_scores(its[d["idx"][j]], dl)
                if sc is None:
                    continue
                edited[j] = True
                check.append(int(np.argmax(unst)) == m)
                if GRID:
                    it_ = iter(sc)
                    sgrid[j] = [m if a == 0 else int(np.argmax(next(it_))) for a in GRID]
                    steer[j] = sgrid[j, -1]
                else:
                    steer[j] = int(np.argmax(sc[0]))
        np.savez(out, steer=steer, probe=probe, edited=edited, layer=L, **({"grid": sgrid, "alphas": GRID} if GRID else {}))
        print(f"{model} {name} alpha={ALPHA:g}: L={L}/{nL} raw={np.mean(ans == y):.3f} probe={np.mean(probe == y):.3f} steer={np.mean(steer == y):.3f} "
              f"edited={edited.mean():.3f} prefix_reproduced={np.mean(check) if check else float('nan'):.3f}", flush=True)


def analyze():
    rows = []
    for p in sorted(glob.glob(f"{G3}/*_steer.npz")):
        cell = os.path.basename(p)[:-10]
        d = np.load(f"{G3}/{cell}_states.npz")
        y, ans, scores, grp, key = d["y"], d["ans"], d["scores"], d["group"], d["key"]
        cal = np.empty_like(y)
        for g in np.unique(grp):
            k = grp == g
            cal[k] = probe_cv(scores[k], y[k], groups=key[k])[0]
        r = dict(cell=cell, n=len(y), raw=np.mean(ans == y), calib=np.mean(cal == y))
        for tag, f in (("a1", p), ("a16", p.replace("_steer.npz", "_steerA16.npz"))):  # a16: post hoc dose
            if not os.path.exists(f):
                continue
            s_ = np.load(f)
            r[f"probe_L"] = np.mean(s_["probe"] == y)
            r[f"steer_{tag}"] = np.mean(s_["steer"] == y)
            r[f"broken_{tag}"] = np.sum((ans == y) & (s_["steer"] != y)) / max(np.sum(ans == y), 1)
            r[f"fixed_{tag}"] = np.sum((ans != y) & (s_["steer"] == y)) / max(np.sum(ans != y), 1)
        rows.append(r)
    df = pd.DataFrame(rows)
    df["a1_minus_cal"] = df.steer_a1 - df.calib
    df.to_csv(f"{OUT}/g3_steer.csv", index=False)
    print(df.round(3).to_string(index=False))
    ill = df[df.cell.str.endswith("illusions")]
    k = int((ill.a1_minus_cal >= 0.10).sum())
    print(f"[pre-registered] illusions: steered(a=1) >= calib + 10 pts in {k}/{len(ill)} -> {'holds' if len(ill) and k >= 2 / 3 * len(ill) else 'fails'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    analyze() if a.analyze else run(a.model)
