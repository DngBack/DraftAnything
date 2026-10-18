"""Exploratory (2026-10-16): external-encoder image control. A linear probe on an independent vision model's features
(SigLIP2-base, never part of the MLLM) uses the same items, labels and image-grouped folds as the G3 cells. If it matches
the MLLM's answer-position probe, a probe beating the model shows that the IMAGE determines the answer linearly, not
that the MLLM computed it. Features: mean patch embedding and a 4x4 average-pooled patch grid.

usage: python explore_extenc.py illusions countbench mmvp pope
"""
import sys, re, numpy as np, torch
from transformers import AutoModel, AutoProcessor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from gen_calib import items
from diag_gen import SETS
from pilot import OUT

M = "google/siglip2-base-patch16-512"
enc = AutoModel.from_pretrained(M, dtype=torch.float32).vision_model.cuda().eval()
proc = AutoProcessor.from_pretrained(M)


@torch.no_grad()
def feats(im):
    x = proc(images=im, return_tensors="pt")["pixel_values"].cuda()
    h = enc(pixel_values=x).last_hidden_state[0]  # (P, d)
    s = int(h.shape[0] ** 0.5)
    g = torch.nn.functional.adaptive_avg_pool2d(h.T.reshape(1, -1, s, s), 4).flatten()
    return h.mean(0).cpu().numpy(), g.cpu().numpy()


for name in sys.argv[1:]:
    task, groups = SETS[name]
    its = [it for it in items(task) if not groups or it["group"] in groups]
    if task == "pope":
        rng = np.random.default_rng(0)
        idx = np.concatenate([rng.choice([i for i, it in enumerate(its) if it["group"] == g], 1000, replace=False)
                              for g in ("adversarial", "popular", "random")])
        its = [its[i] for i in sorted(idx)]
    its = [it for it in its if it["y"] in it["cands"]]
    y = np.array([it["cands"].index(it["y"]) for it in its])
    grp = np.array([re.sub(r"_px\d+", "", str(it.get("key", it.get("pair", j)))) for j, it in enumerate(its)])
    cache = {}
    F = []
    for it in its:
        k = id(it["image"])
        if k not in cache:
            cache[k] = feats(it["image"])
        F.append(cache[k])
    mean, grid = (np.stack(f) for f in zip(*F))
    if task == "pope":  # the question names the object: image features alone cannot answer it
        print(name, "skipped (answer depends on the question)")
        continue
    ok = np.isin(y, np.flatnonzero(np.bincount(y) >= 5))
    res = {}
    for fn, X in (("mean", mean), ("grid4", grid)):
        X, yy, gg = X[ok], y[ok], grp[ok]
        X = (X - X.mean(0)) / (X.std(0) + 1e-4)
        p = np.zeros(len(yy), int)
        for tr, te in GroupKFold(5).split(X, yy, gg):
            p[te] = LogisticRegression(C=0.1, max_iter=3000).fit(X[tr], yy[tr]).predict(X[te])
        res[fn] = round(float((p == yy).mean()), 3)
    print(f"{name} n={ok.sum()} maj={np.bincount(y[ok]).max() / ok.sum():.3f} siglip2 probe: {res}", flush=True)
