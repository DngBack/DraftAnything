"""Exploratory (2026-10-16): input-side control for the per-error diagnostic.

For every cached cell (closed-form results/*_states.npz and generation results/g3/*_states.npz) the paper's three
predictors are recomputed (calibrator on candidate logits, probe on the answer-position state at the best layer) and
two input controls are added: a probe on the mean image-token state at layer 0 (the projector output, before any LLM
layer) and at the best layer. A LOSS error (calibrator wrong, answer probe right) whose answer the layer-0 input probe
also recovers is INHERITED (linearly available in what the vision side hands over); otherwise COMPUTED.
Also saves per-layer decodability (the evidence-flow profile). Output: results/explore_flow.csv, explore_flow_layers.csv
"""
import glob, os, re, numpy as np, pandas as pd, torch
from pilot import probe_cv, OUT

rows, prof = [], []
paths = sorted(glob.glob(f"{OUT}/*_states.npz")) + sorted(glob.glob(f"{OUT}/g3/*_states.npz"))
for p in paths:
    d = np.load(p, allow_pickle=True)
    if "imgm" not in d.files:
        continue
    name = os.path.basename(p).replace("_states.npz", "")
    setting = "gen" if "/g3/" in p else "closed"
    y, ans, sc = d["y"], d["ans"], np.asarray(d["scores"], np.float32)
    groups = np.array([re.sub(r"_px\d+", "", str(k)) for k in d["key"]]) if "key" in d.files else None
    # classes with too few items are dropped (as the paper's cells)
    ok = np.isin(y, np.flatnonzero(np.bincount(y) >= 5))
    y, ans, sc = y[ok], ans[ok], sc[ok]
    groups = groups[ok] if groups is not None else None
    yy = np.unique(y, return_inverse=True)[1]
    def clean(X):
        X = np.nan_to_num(X.astype(np.float32), posinf=0, neginf=0)
        return np.clip(X, -1e4, 1e4)
    cal = probe_cv(sc - sc.mean(1, keepdims=True), yy, groups=groups)[0]
    last = clean(d["last"][ok]); imgm = clean(d["imgm"][ok])
    pl = probe_cv(last, yy, groups=groups)[0]  # (L, n)
    pi = probe_cv(imgm, yy, groups=groups)[0]
    al, ai = (pl == yy).mean(1), (pi == yy).mean(1)
    bl = int(al.argmax())
    probe, inp0, inpb = pl[bl], pi[0], pi[int(ai.argmax())]
    err = ans != y
    bias = err & (cal == yy); loss = err & (cal != yy) & (probe == yy); perc = err & ~bias & ~loss
    inh = loss & (inp0 == yy)
    rows.append(dict(cell=name, setting=setting, n=len(y), errors=int(err.sum()), raw=np.mean(~err), calib=np.mean(cal == yy),
                     probe=al[bl], probe_layer=bl, input0=ai[0], input_best=ai.max(), input_best_layer=int(ai.argmax()),
                     bias=int(bias.sum()), loss=int(loss.sum()), perc=int(perc.sum()), loss_inherited=int(inh.sum()),
                     loss_inherited_best=int((loss & (inpb == yy)).sum()),
                     # errors the input probe fixes that the answer probe does not (evidence lost inside the LLM)
                     input_only=int((err & (inp0 == yy) & (probe != yy) & (cal != yy)).sum())))
    for l in range(len(al)):
        prof.append(dict(cell=name, setting=setting, layer=l, depth=l / (len(al) - 1), last=al[l], imgm=ai[l]))
    r = rows[-1]
    print(f"{name:32s} {setting:6s} raw {r['raw']:.2f} cal {r['calib']:.2f} probe {r['probe']:.2f}@L{bl} input0 {r['input0']:.2f} "
          f"| err {r['errors']} B/L/P {r['bias']}/{r['loss']}/{r['perc']} loss-inherited {r['loss_inherited']}", flush=True)
pd.DataFrame(rows).to_csv(f"{OUT}/explore_flow.csv", index=False)
pd.DataFrame(prof).to_csv(f"{OUT}/explore_flow_layers.csv", index=False)
