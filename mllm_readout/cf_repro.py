"""Reproduce 2604.09364's Visual CounterFact colour setting and compare calibration with their steering.

Their setting: prompt "What color is the {object}?", correct iff the visual colour's logit beats the prior colour's at the
final position (each colour = max over surface variants); steering learned on ~200 items, evaluated on 293 held-out.
Ours: same split sizes; (a) raw, (b) label-free batch calibration, (c) per-colour logit bias fitted on the 200 train items.

usage: python cf_repro.py --model llava15_7b [--fullres]  (scores cached to results/cfrepro_<model>[_full].npz)
       python cf_repro.py --report                    (table over all cached models)
"""
import argparse, ast, glob, os
import numpy as np, torch
from scipy.special import log_softmax
from pilot import MODELS, OUT


def items(fullres=False):
    from datasets import load_dataset
    out = []
    for r in load_dataset("mgolov/Visual-Counterfact", split="color"):
        im = r["counterfact_image"].convert("RGB")
        if not fullres:
            im.thumbnail((768, 768))
        prior = ast.literal_eval(r["correct_answer"])
        orig = r["original_image"].convert("RGB")
        if not fullres:
            orig.thumbnail((768, 768))
        out.append(dict(image=im, orig=orig, q=f"What color is the {r['object']}?", vis=r["incorrect_answer"],
                        prior=[c for c in prior if c != r["incorrect_answer"]]))
    return out


def score(model, fullres=False):
    from pilot import Runner
    its = items(fullres)
    vocab = sorted({it["vis"] for it in its} | {c for it in its for c in it["prior"]})
    R = Runner(model, vocab)
    S = np.stack([R.run(R.inputs(it))[0].cpu().numpy() for it in its])
    So = np.stack([R.run(R.inputs(dict(it, image=it["orig"])))[0].cpu().numpy() for it in its])  # original images (gold = prior)
    vis = np.array([vocab.index(it["vis"]) for it in its])
    prior = np.zeros((len(its), len(vocab)), bool)
    for i, it in enumerate(its):
        prior[i, [vocab.index(c) for c in it["prior"]]] = True
    np.savez(f"{OUT}/cfrepro_{model}{'_full' if fullres else ''}.npz", S=S, vis=vis, prior=prior, vocab=np.array(vocab))
    np.savez(f"{OUT}/cforig_{model}{'_full' if fullres else ''}.npz", S=So)


def steer(model, ntrain=200, nsel=100, alphas=(0.2, 0.5, 1.0, 1.5, 2.0, 3.0), every=4):
    """Their linear steering re-implemented in our pipeline: direction = mean(h_counterfact - h_original) over train items
    (sequence-mean hidden state of each layer), added at all positions of one layer. (layer, alpha) chosen on nsel train
    items; the test-oracle choice is also reported (upper bound for steering). Compared with fit_bias on the same split."""
    from pilot import Runner
    its = items()
    vocab = sorted({it["vis"] for it in its} | {c for it in its for c in it["prior"]})
    R = Runner(model, vocab)
    vis = np.array([vocab.index(it["vis"]) for it in its])
    prior = np.zeros((len(its), len(vocab)), bool)
    for i, it in enumerate(its):
        prior[i, [vocab.index(c) for c in it["prior"]]] = True
    perm = np.random.default_rng(0).permutation(len(its))
    tr, te = perm[:ntrain], perm[ntrain:]
    sel = tr[:nsel]
    diff, S0 = 0, []
    for i in range(len(its)):
        sc, hs = R.run(R.inputs(its[i]), hidden=i in set(tr))
        S0.append(sc.cpu().numpy())
        if i in set(tr):
            _, ho = R.run(R.inputs(dict(its[i], image=its[i]["orig"])), hidden=True)
            diff = diff + torch.stack([a[0].float().mean(0) - b[0].float().mean(0) for a, b in zip(hs[1:], ho[1:])])
    S0 = log_softmax(np.stack(S0).astype(np.float64), 1)
    d = diff / len(tr)  # (nL, dim)
    res = {}
    for L in range(0, len(R.layers), every):
        for al in alphas:
            v = (al * d[L]).to(torch.bfloat16)
            fn = lambda h, v=v: h + v
            S = np.stack([R.run(R.inputs(its[i]), {L: fn})[0].cpu().numpy() for i in np.concatenate([sel, te])])
            S = log_softmax(S.astype(np.float64), 1)
            res[(L, al)] = (wins(S[:nsel], vis[sel], prior[sel]).mean(), wins(S[nsel:], vis[te], prior[te]).mean())
    best = max(res, key=lambda k: res[k][0])
    oracle = max(res, key=lambda k: res[k][1])
    b = fit_bias(S0[tr], vis[tr], prior[tr])
    out = dict(model=model, raw=wins(S0[te], vis[te], prior[te]).mean(), batch=wins(S0[te] - S0[te].mean(0), vis[te], prior[te]).mean(),
               bias_fit=wins(S0[te] + b, vis[te], prior[te]).mean(), steer_sel=res[best][1], steer_sel_cfg=best,
               steer_oracle=res[oracle][1], steer_oracle_cfg=oracle)
    print(out, flush=True)
    import json
    json.dump({k: (list(v) if isinstance(v, tuple) else float(v) if not isinstance(v, str) else v) for k, v in out.items()},
              open(f"{OUT}/cfsteer_{model}.json", "w"), indent=1)


def wins(S, vis, prior):
    """visual colour beats every prior colour (items with no distinct prior count as visual-vs-nothing = win)"""
    v = S[np.arange(len(vis)), vis]
    p = np.where(prior, S, -np.inf).max(1)
    return v > p


def fit_bias(S, vis, prior, wd=1e-2, steps=500):
    """per-colour additive logit bias maximising P(visual beats prior) on train items (|vocab| parameters)."""
    S, pm = torch.tensor(S), torch.tensor(prior)
    b = torch.zeros(S.shape[1], dtype=S.dtype, requires_grad=True)
    opt = torch.optim.Adam([b], lr=0.05)
    idx = torch.arange(len(vis))
    for _ in range(steps):
        opt.zero_grad()
        T = S + b
        p = torch.where(pm, T, torch.tensor(-1e9, dtype=S.dtype)).logsumexp(1)
        loss = torch.nn.functional.softplus(p - T[idx, torch.tensor(vis)]).mean() + wd * (b ** 2).sum()
        loss.backward()
        opt.step()
    return b.detach().numpy()


def wins_set(S, tgt, oth):
    """best target-set logit beats best other-set logit (empty other set = win)"""
    return np.where(tgt, S, -np.inf).max(1) > np.where(oth, S, -np.inf).max(1)


def fit_bias_set(S, tgt, oth, wd=1e-2, steps=500):
    """per-colour bias maximising P(target set beats other set); generalises fit_bias to mixed original/counterfactual."""
    S, tm, om = torch.tensor(S), torch.tensor(tgt), torch.tensor(oth)
    b = torch.zeros(S.shape[1], dtype=S.dtype, requires_grad=True)
    opt = torch.optim.Adam([b], lr=0.05)
    neg = torch.tensor(-1e9, dtype=S.dtype)
    for _ in range(steps):
        opt.zero_grad()
        T = S + b
        loss = torch.nn.functional.softplus(torch.where(om, T, neg).logsumexp(1) - torch.where(tm, T, neg).logsumexp(1)).mean() + wd * (b ** 2).sum()
        loss.backward()
        opt.step()
    return b.detach().numpy()


def orig_control(ntrain=200, draws=20):
    """Label-shift control: does a calibrator fitted on counterfactual images only learn 'avoid the prior colour'?
    Original images have gold = prior colour. Compare a CF-only bias with one fitted on both images of the train objects."""
    print(f"{'model':14s} | raw CF  raw orig | CF-only bias: CF  orig | mixed bias: CF  orig   [test objects, {draws} splits]")
    for p in sorted(glob.glob(f"{OUT}/cforig_*.npz")):
        m, rng = os.path.basename(p)[7:-4], np.random.default_rng(0)  # per-model seed
        d = np.load(f"{OUT}/cfrepro_{m}.npz")
        S, So = log_softmax(d["S"].astype(np.float64), 1), log_softmax(np.load(p)["S"].astype(np.float64), 1)
        vis1 = np.eye(S.shape[1], dtype=bool)[d["vis"]]
        prior = d["prior"]
        has = prior.any(1)  # objects whose prior differs from the counterfactual colour
        r = {k: [] for k in ("rc", "ro", "cc", "co", "mc", "mo")}
        for _ in range(draws):
            perm = rng.permutation(len(S))
            tr, te = perm[:ntrain], perm[ntrain:]
            te = te[has[te]]
            r["rc"].append(wins_set(S[te], vis1[te], prior[te]).mean())
            r["ro"].append(wins_set(So[te], prior[te], vis1[te]).mean())
            b = fit_bias_set(S[tr], vis1[tr], prior[tr])
            r["cc"].append(wins_set(S[te] + b, vis1[te], prior[te]).mean())
            r["co"].append(wins_set(So[te] + b, prior[te], vis1[te]).mean())
            trh = tr[has[tr]]
            b = fit_bias_set(np.concatenate([S[tr], So[trh]]), np.concatenate([vis1[tr], prior[trh]]), np.concatenate([prior[tr], vis1[trh]]))
            r["mc"].append(wins_set(S[te] + b, vis1[te], prior[te]).mean())
            r["mo"].append(wins_set(So[te] + b, prior[te], vis1[te]).mean())
        f = lambda k: f"{100 * np.mean(r[k]):5.1f}"
        print(f"{m:14s} | {f('rc')}  {f('ro')}    |       {f('cc')}  {f('co')}  |     {f('mc')}  {f('mo')}")


def report(ntrain=200, draws=20):
    print(f"{'model':16s} raw(all)  raw(test)  batch-cal  bias-fit(200)   [mean over {draws} splits of 200/293]")
    for p in sorted(glob.glob(f"{OUT}/cfrepro_*.npz")):
        d, rng = np.load(p), np.random.default_rng(0)  # per-model seed: adding a model must not change the others' splits
        S, vis, prior = log_softmax(d["S"].astype(np.float64), 1), d["vis"], d["prior"]
        raw, bat, fit = [], [], []
        for _ in range(draws):
            perm = rng.permutation(len(vis))
            tr, te = perm[:ntrain], perm[ntrain:]
            raw.append(wins(S[te], vis[te], prior[te]).mean())
            bat.append(wins(S[te] - S[te].mean(0), vis[te], prior[te]).mean())
            fit.append(wins(S[te] + fit_bias(S[tr], vis[tr], prior[tr]), vis[te], prior[te]).mean())
        f = lambda x: f"{100 * np.mean(x):.1f}±{100 * np.std(x):.1f}"
        print(f"{os.path.basename(p)[8:-4]:16s} {100 * wins(S, vis, prior).mean():6.1f}    {f(raw):10s} {f(bat):10s} {f(fit)}")


def _check():
    S = np.array([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0]])
    assert wins(S, np.array([0, 2]), np.array([[0, 1, 0], [0, 1, 0]], bool)).tolist() == [True, False]
    # a constant bias against colour 1 is learnable
    S = np.random.default_rng(0).normal(size=(200, 3)); vis = np.full(200, 0); S[:, 1] += 2
    pr = np.zeros((200, 3), bool); pr[:, 1] = True
    assert wins(S + fit_bias(S, vis, pr), vis, pr).mean() > wins(S, vis, pr).mean() + 0.3


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--steer", action="store_true", help="re-implemented linear steering vs bias fit (one split)")
    ap.add_argument("--control", action="store_true", help="original-image label-shift control")
    ap.add_argument("--fullres", action="store_true", help="original image size (no 768px thumbnail)")
    a = ap.parse_args()
    _check()
    if a.model and a.steer:
        steer(a.model)
    elif a.model:
        score(a.model, a.fullres)
    if a.report:
        report()
    if a.control:
        orig_control()
