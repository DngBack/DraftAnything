"""G5 M (docs/g5_protocol.md, Addendum 2): image-position interchange at depth l, read at the decision step.

For each illusion item (base) a partner with the other label (same type, strength token, wording, image size and
number of image tokens) is run up to the input of layer l; the base item is then generated with its image-position
states at the input of layer l replaced by the partner's (prompt pass only), up to the G3 decision step. Read: the
model's answer and the decision-state probe (75% depth, trained on stored G3 states, folds grouped by type x strength).

usage: python g5_swap.py --model qwen3vl8b          (results/g5/swap_<model>.npz)
       python g5_swap.py --analyze
"""
import argparse, glob, os, re
import numpy as np, pandas as pd, torch
from transformers import LogitsProcessorList, StoppingCriteriaList
from pilot import OUT
from gen_calib import items, Decide
from diag_gen import Stop
from diag_blank import item_list
from steer_g3 import fit_probe

FR = (0.125, 0.25, 0.5, 0.75)
CTRL_FR = (0.25, 0.75)
D = f"{OUT}/g5"
LIMIT = int(os.environ.get("SWAP_LIMIT", 10 ** 9))  # smoke tests only (output suffix _smoke)
G6CTRL = os.environ.get("G6CTRL") == "1"  # G6 M': same-label swaps at 12.5 / 50% only (docs/g6_protocol.md)


class Halt(Exception):
    pass


def meta(key, q):
    n = os.path.basename(str(key)).replace("_notitle.png", "").split("_")  # Type_NNN_str.._diff..
    return n[0], n[2], n[3] != "diff0", q


def pairs(d, its, ntok):
    """base j -> (partner with other label, partner with same label) indices into the G3 arrays, or -1"""
    rng = np.random.default_rng(0)
    M = [meta(k, q) + (ntok[i],) for k, q, i in zip(d["key"], d["q"], d["idx"])]
    P, C = np.full(len(M), -1), np.full(len(M), -1)
    for j, (t, s, neq, q, nt) in enumerate(M):
        other = [i for i, m in enumerate(M) if m[0] == t and m[1] == s and m[2] != neq and m[3] == q and m[4] == nt]
        same = [i for i, m in enumerate(M) if i != j and m[0] == t and m[2] == neq and m[3] == q and m[4] == nt
                and str(d["key"][i]) != str(d["key"][j])]
        if other:
            P[j] = rng.choice(other)
        if same:
            C[j] = rng.choice(same)
    return P, C


def run(model):
    from pilot import Runner
    st = (glob.glob(f"{OUT}/g3/{model}_illusions_states.npz") + glob.glob(f"{OUT}/g3_heldout/{model}_illusions_states.npz"))[0]
    out = f"{D}/swap{'ctrl' if G6CTRL else ''}_{model}{'_smoke' if LIMIT < 10 ** 9 else ''}.npz"
    if os.path.exists(out):
        return
    R = Runner(model, ["x"])
    tok = R.p.tokenizer
    nL = len(R.layers)
    L = int(round(0.75 * nL))
    LS = sorted({int(round(f * nL)) for f in FR})
    d = np.load(st, allow_pickle=True)
    its = item_list("illusions", items("vlmbias"))

    def enc(it):
        msgs = ([{"role": "system", "content": [{"type": "text", "text": it["sys"]}]}] if it["sys"] else []) + \
               [{"role": "user", "content": [{"type": "image", "image": it["image"]}, {"type": "text", "text": it["q"]}]}]
        return R.p.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")

    xs = {int(i): enc(its[int(i)]) for i in np.unique(d["idx"])}
    ntok = {i: (int((x["input_ids"][0] == R.img_id).sum()), tuple(x["image_grid_thw"].flatten().tolist()) if "image_grid_thw" in x else ())
            for i, x in xs.items()}
    P, C = pairs(d, its, ntok)
    cids = None
    cache = {}

    @torch.no_grad()
    def capture(i):  # partner image-position states at the input of each layer in LS
        if i in cache:
            return cache[i]
        x, cap, hs = xs[i].to("cuda"), {}, []
        im = x["input_ids"][0] == R.img_id
        for l in LS:
            def h(mod, args, kwargs, l=l):
                cap[l] = (args[0] if args else kwargs["hidden_states"])[0, im].clone()
                if l == LS[-1]:
                    raise Halt
            hs.append(R.layers[l].register_forward_pre_hook(h, with_kwargs=True))
        try:
            R.m(**x)
        except Halt:
            pass
        finally:
            [k.remove() for k in hs]
        cache.clear()
        cache[i] = cap
        return cap

    @torch.no_grad()
    def generate(i, l=None, src=None):
        it, x = its[i], xs[i].to("cuda")
        im = x["input_ids"][0] == R.img_id
        n0 = x["input_ids"].shape[1]
        dd = Decide(cid(it), None, None, ())
        dd.ready = lambda ids, n0=n0: tok.decode(ids[0, n0:]).strip() == "" or tok.decode(ids[0, n0:][-4:]).rstrip().endswith("{")
        hd = None
        if l is not None:
            def h(mod, args, kwargs):
                hh = args[0] if args else kwargs["hidden_states"]
                if hh.shape[1] >= n0:
                    hh[0, : n0][im] = src.to(hh.dtype)
            hd = R.layers[l].register_forward_pre_hook(h, with_kwargs=True)
        try:
            o = R.m.generate(**x, max_new_tokens=384, do_sample=False, logits_processor=LogitsProcessorList([dd]),
                             stopping_criteria=StoppingCriteriaList([Stop(dd)]), output_hidden_states=True, return_dict_in_generate=True)
        finally:
            if hd is not None:
                hd.remove()
        if not dd.done:
            return -1, None
        t = o.sequences.shape[1] - n0 - 1
        return int(np.argmax(dd.scores)), o.hidden_states[t][L + 1][0, -1].float().cpu().numpy()

    ids_cache = {}

    def cid(it):
        k = tuple(it["cands"])
        if k not in ids_cache:
            ids_cache[k] = [sorted({t for p in ("", " ") for v in (c, c.lower(), c.capitalize())
                                    for t in tok.encode(p + v, add_special_tokens=False) if tok.decode([t]).strip().lower() == c.lower()})
                            for c in it["cands"]]
        return ids_cache[k]

    conds = [("swap", l) for l in LS] + [("ctrl", l) for l in sorted({int(round(f * nL)) for f in CTRL_FR})]
    if G6CTRL:
        conds = [("ctrl", l) for l in sorted({int(round(f * nL)) for f in (0.125, 0.5)})]
    A = np.full((len(P), len(conds)), -1)
    H = np.zeros((len(P), len(conds), d["last"].shape[-1]), np.float16)
    order = [j for j in np.argsort(d["idx"][np.maximum(C if G6CTRL else P, 0)], kind="stable") if P[j] >= 0]  # with a partner; grouped by partner (capture cache)
    for n, j in enumerate(order[:LIMIT]):
        for c, (kind, l) in enumerate(conds):
            p = P[j] if kind == "swap" else C[j]
            if p < 0:
                continue
            a, h = generate(int(d["idx"][j]), l, capture(int(d["idx"][p]))[l])
            A[j, c] = a
            if h is not None:
                H[j, c] = h
        if n % 50 == 0:
            print(model, n, len(P), flush=True)
    np.savez(out, ans=A, H=H, P=P, C=C, conds=np.array([f"{k}{l}" for k, l in conds]), layers=np.array(LS), L=L, nL=nL,
             ntok=np.array([ntok[int(i)][0] for i in d["idx"]]))
    print(model, "done", flush=True)


def analyze():
    rows = []
    for f in sorted(glob.glob(f"{D}/swap_*.npz")):
        model = os.path.basename(f)[5:-4]
        S = np.load(f)
        st = (glob.glob(f"{OUT}/g3/{model}_illusions_states.npz") + glob.glob(f"{OUT}/g3_heldout/{model}_illusions_states.npz"))[0]
        d = np.load(st, allow_pickle=True)
        L, y, ans = int(S["L"]), d["y"], d["ans"]
        X = d["last"][:, L + 1].astype(np.float32)
        g = np.array(["_".join(meta(k, "")[:2]) for k in d["key"]])
        ug, inv = np.unique(g, return_inverse=True)
        fold = (np.random.default_rng(0).permutation(len(ug)) % 5)[inv]
        Hs = S["H"].astype(np.float32)
        pr = np.full(S["ans"].shape, -1)
        base = np.full(len(y), -1)
        for k in range(5):
            f_ = fit_probe(X[fold != k], y[fold != k])
            te = fold == k
            base[te] = f_(X[te])
            for c in range(Hs.shape[1]):
                pr[te, c] = f_(Hs[te, c])
        P, C = S["P"], S["C"]
        r = dict(model=model, n_pairs=int((P >= 0).sum()), n_ctrl=int((C >= 0).sum()))
        ok = P >= 0
        # registered M1-M3 (layer-0 swap == running the partner item): from the stored states
        r["M1_probe_follows@0"] = np.mean(base[P[ok]] == y[P[ok]])
        r["M2_answer_follows@0"] = np.mean(ans[P[ok]] == y[P[ok]])
        okc = C >= 0
        r["M3_ctrl_probe_changes@0"] = np.mean(base[C[okc]] != base[okc])
        r["probe_base"] = np.mean(base[ok] == y[ok])
        r["answer_base"] = np.mean(ans[ok] == y[ok])
        for c, name in enumerate(S["conds"]):
            l = int(name[4:])
            fr = min(FR, key=lambda f: abs(f - l / int(S["nL"])))  # nominal depth (layer rounding differs per model)
            tag = f"{name[:4]}@{100 * fr:g}"
            m = (P >= 0) if name.startswith("swap") else (C >= 0)
            m &= S["ans"][:, c] >= 0
            q = P if name.startswith("swap") else C
            if name.startswith("swap"):
                r[f"probe->partner {tag}"] = np.mean(pr[m, c] == y[q[m]])
                r[f"answer->partner {tag}"] = np.mean(S["ans"][m, c] == y[q[m]])
            else:
                r[f"probe changes {tag}"] = np.mean(pr[m, c] != base[m])
                r[f"answer changes {tag}"] = np.mean(S["ans"][m, c] != ans[m])
        rows.append(r)
    df = pd.DataFrame(rows).set_index("model")
    df.to_csv(f"{D}/swap.csv")
    pd.set_option("display.width", 250)
    print((df * 1).round(3).T.to_string())
    m1, m3 = df["M1_probe_follows@0"].median(), df["M3_ctrl_probe_changes@0"].median()
    k2 = int(((df["M1_probe_follows@0"] - df["M2_answer_follows@0"]) >= 0.30).sum())
    print(f"M1 median {m1:.3f} (>= 0.70: {'PASS' if m1 >= 0.7 else 'FAIL'}); M2 {k2}/{len(df)} models; M3 median {m3:.3f} (<= 0.15: {'PASS' if m3 <= 0.15 else 'FAIL'})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    os.makedirs(D, exist_ok=True)
    analyze() if a.analyze else run(a.model)
