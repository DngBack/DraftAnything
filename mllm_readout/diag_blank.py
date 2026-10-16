"""G3 addendum 2 (docs/g3_protocol.md): text-only probe from the same prompts with a blank (grey) image.
Saves results/g3/<model>_<set>_blank.npz (last prompt position, all layers), aligned with the G3 states via idx.

usage: python diag_blank.py --model qwen3vl8b
       python diag_blank.py --analyze          (results/g3_blank.csv)
"""
import argparse, glob, os
import numpy as np, pandas as pd, torch
from PIL import Image
from pilot import MODELS, OUT, probe_cv
from gen_calib import items
from diag_gen import SETS, G3

USE = ("illusions", "tally", "mmvp", "pope", "countbench")


def item_list(name, its):  # exactly as diag_gen.run
    task, groups = SETS[name]
    if groups:
        its = [it for it in its if it["group"] in groups]
    if task == "pope":
        rng = np.random.default_rng(0)
        idx = np.concatenate([rng.choice([i for i, it in enumerate(its) if it["group"] == g], 1000, replace=False)
                              for g in ("adversarial", "popular", "random")])
        its = [its[i] for i in sorted(idx)]
    return its


@torch.no_grad()
def run(model):
    from pilot import Runner
    R = Runner(model, ["x"])
    cache = {}
    for name in USE:
        out = f"{G3}/{model}_{name}_blank.npz"
        st = f"{G3}/{model}_{name}_states.npz"
        if os.path.exists(out) or not os.path.exists(st):
            continue
        d = np.load(st)
        task = SETS[name][0]
        if task not in cache:
            cache = {task: items(task)}
        its = item_list(name, cache[task])
        H = []
        for j, q in zip(d["idx"], d["q"]):
            it = its[j]
            assert it["q"] == q
            blank = Image.new("RGB", it["image"].size, (128, 128, 128))
            msgs = ([{"role": "system", "content": [{"type": "text", "text": it["sys"]}]}] if it["sys"] else []) + \
                   [{"role": "user", "content": [{"type": "image", "image": blank}, {"type": "text", "text": it["q"]}]}]
            x = R.p.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
            hs = R.m(**x, output_hidden_states=True).hidden_states
            H.append(torch.stack([h[0, -1] for h in hs]).float().cpu().half())
        np.savez(out, last=torch.stack(H).numpy())
        print(f"{model} {name}: n={len(H)}", flush=True)


def analyze():
    t = pd.read_csv(f"{OUT}/g3_table.csv").set_index("cell")
    rows = []
    for p in sorted(glob.glob(f"{G3}/*_blank.npz")):
        cell = os.path.basename(p)[:-10]
        d, b = np.load(f"{G3}/{cell}_states.npz"), np.load(p)
        y, ans, scores, grp, key = d["y"], d["ans"], d["scores"], d["group"], d["key"]
        err = ans != y
        cal = np.empty_like(y)
        for g in np.unique(grp):
            k = grp == g
            cal[k] = probe_cv(scores[k], y[k], groups=key[k])[0] if len(np.unique(y[k])) > 1 else ans[k]
        best = None
        for pos in ("last", "imgm"):
            pr, _ = probe_cv(d[pos], y, groups=key)
            acc = (pr == y).mean(1); li = int(acc.argmax())
            if best is None or acc[li] > best[0]:
                best = (acc[li], pr[li])
        pt, _ = probe_cv(b["last"], y, groups=key)
        acc_t = (pt == y).mean(1); lt = int(acc_t.argmax())
        txt = pt[lt]
        ne = max(int(err.sum()), 1)
        loss = err & (cal != y) & (best[1] == y)
        rows.append(dict(cell=cell, n_err=int(err.sum()), model=1 - err.mean(), probe=best[0], text_probe=acc_t[lt], L_text=lt,
                         loss=loss.sum() / ne, loss_txt=(loss & (txt != y)).sum() / ne))
    df = pd.DataFrame(rows)
    df = df[df.cell.isin(t.index[(t.n_err >= 30) & (t.nodec <= 0.10) & (t.text < 0.75)])]
    df.to_csv(f"{OUT}/g3_blank.csv", index=False)
    print(df.round(3).to_string(index=False))
    df["set"] = df.cell.str.rsplit("_", n=1).str[1]
    print(df.groupby("set")[["model", "probe", "text_probe", "loss", "loss_txt"]].median().round(3))
    m = df.groupby("set").loss_txt.median()
    print(f"illusions loss_txt median {m.get('illusions', np.nan):.3f} -> {'stands' if m.get('illusions', 0) >= .3 else 'FAILS'}; "
          f"POPE {m.get('pope', np.nan):.3f} -> {'stands' if m.get('pope', 1) <= .1 else 'FAILS'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    analyze() if a.analyze else run(a.model)
