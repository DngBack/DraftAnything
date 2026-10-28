"""Train one variant on Controlled Long-KIE and evaluate on the frozen test grid.

python scripts/train.py --model A2_N4 --seed 0 [--steps 8000]
Writes results/stage0/<model>-s<seed>/{metrics.json,preds.npz,log.txt}. No checkpoint or
hyper-parameter selection on test: fixed steps, last weights are evaluated.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as Fn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sleepkie.data import Spec, Encoder, make_batch, TYPES
from sleepkie.model import build

TEST_T = (10, 25, 50, 100, 200)
TEST_SEED = 777_000   # test docs depend only on this, never on the training seed
DEV_SEED = 555_000    # pilot/tuning evaluations use this instead
AMP = False           # fp32 by default: bf16 diverged for N=4 variants in pilot


def losses(value, ev, b, T, lam_e):
    gold_ev = torch.cat([b['evid'], ~b['evid'].any(-1, keepdim=True)], -1)
    ev = ev.float()
    ev_nll = (torch.logsumexp(ev, -1) - torch.logsumexp(ev.masked_fill(~gold_ev, -1e4), -1)).mean()
    ce = Fn.cross_entropy(value.float().flatten(0, 1), b['gold'].flatten())
    return ce + lam_e * ev_nll, ce, ev_nll


@torch.no_grad()
def evaluate(model, spec, enc, F, n_docs, device, unseen=False, bs=64, base=TEST_SEED):
    model.eval()
    rows = {T: [] for T in TEST_T}
    for T in TEST_T:
        rng = np.random.default_rng(base + T + (10_000 if unseen else 0))
        for _ in range(n_docs // bs):
            b = make_batch(spec, enc, bs, T, F, rng, unseen, device)
            with torch.autocast('cuda', torch.bfloat16, enabled=AMP):
                value, ev = model(b['x'], b['svec'])
            ev = ev.float()
            gold_ev = torch.cat([b['evid'], ~b['evid'].any(-1, keepdim=True)], -1)
            hit = gold_ev.gather(-1, ev.argmax(-1, keepdim=True)).squeeze(-1)
            k = gold_ev.sum(-1)
            rank = ev.argsort(-1, descending=True).argsort(-1)
            recall = ((rank < k[..., None]) & gold_ev).sum(-1) / k
            rows[T].append(dict(pred=value.argmax(-1).cpu(), gold=b['gold'].cpu(), ftype=b['ftype'].cpu(),
                                dist=b['dist'].cpu(), ev_hit=hit.cpu(), ev_recall=recall.float().cpu()))
    model.train()
    return {T: {k: torch.cat([r[k] for r in rs]).numpy() for k in rs[0]} for T, rs in rows.items()}


def summarize(r, null):
    ok = r['pred'] == r['gold']
    out = dict(field_acc=ok.mean(), doc_em=ok.all(1).mean(), ev_hit=r['ev_hit'].mean(), ev_recall=r['ev_recall'].mean())
    cross = np.isin(r['ftype'], [1, 2])   # revision + link; aggregation is secondary (protocol)
    out['cross_page_acc'] = ok[cross].mean()
    for i, name in enumerate(TYPES):
        out[f'acc_{name}'] = ok[r['ftype'] == i].mean()
    pn, gn = r['pred'] == null, r['gold'] == null
    out['null_precision'] = (pn & gn).sum() / max(pn.sum(), 1)
    out['null_recall'] = (pn & gn).sum() / max(gn.sum(), 1)
    return {k: float(v) for k, v in out.items()}


@torch.no_grad()
def efficiency(model, spec, enc, F, device, T=200):
    """Memory-module cost only (frozen encoder excluded and identical across variants)."""
    model.eval()
    rng = np.random.default_rng(0)
    out = {}
    for bs in (1, 64):
        b = make_batch(spec, enc, bs, T, F, rng, device=device)
        with torch.autocast('cuda', torch.bfloat16, enabled=AMP):
            model(b['x'], b['svec'])
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated(); t0 = time.perf_counter()
            for _ in range(3):
                model(b['x'], b['svec'])
            torch.cuda.synchronize()
        out[f'bs{bs}_ms_per_doc'] = (time.perf_counter() - t0) / 3 / bs * 1e3
        out[f'bs{bs}_peak_mib'] = (torch.cuda.max_memory_allocated() - base) / 2**20
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=8000)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--fields', type=int, default=8)
    ap.add_argument('--tmin', type=int, default=10)
    ap.add_argument('--tmax', type=int, default=50)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--curriculum', type=float, default=.4, help='fraction of steps over which tmax grows from 4')
    ap.add_argument('--lam_e', type=float, default=.5)
    ap.add_argument('--test_docs', type=int, default=512)
    ap.add_argument('--tag', default='stage0')
    ap.add_argument('--amp', action='store_true', help='bf16 autocast')
    ap.add_argument('--dev', action='store_true', help='evaluate on dev documents (tuning only)')
    a = ap.parse_args()
    global AMP
    AMP = a.amp

    out = ROOT / 'results' / a.tag / f'{a.model}-s{a.seed}'
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / 'log.txt', 'w')
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = 'cuda'
    spec, enc = Spec(), Encoder(Spec(), seed=0)
    model = build(a.model, spec).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / 500) * .5 * (1 + np.cos(np.pi * min(s, a.steps) / a.steps)))
    t0 = time.time()
    for step in range(a.steps):
        hi = a.tmax if a.curriculum <= 0 else int(4 + (a.tmax - 4) * min(1, step / (a.curriculum * a.steps)))
        T = int(rng.integers(min(a.tmin, hi), hi + 1))
        b = make_batch(spec, enc, a.batch, T, a.fields, rng, device=device)
        with torch.autocast('cuda', torch.bfloat16, enabled=AMP):
            value, ev = model(b['x'], b['svec'])
        loss, ce, ev_nll = losses(value, ev, b, T, a.lam_e)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step(); sched.step()
        if step % 200 == 0 or step == a.steps - 1:
            acc = (value.argmax(-1) == b['gold']).float().mean().item()
            print(f'{step} T={T} loss={loss.item():.3f} ce={ce.item():.3f} ev={ev_nll.item():.3f} '
                  f'acc={acc:.3f} {time.time() - t0:.0f}s', file=log, flush=True)
    train_s = time.time() - t0

    metrics = dict(args=vars(a), params=n_params, train_seconds=train_s, efficiency=efficiency(model, spec, enc, a.fields, device))
    preds = {}
    for split, unseen in (('iid', False), ('unseen_entity', True)):
        res = evaluate(model, spec, enc, a.fields, a.test_docs, device, unseen, base=DEV_SEED if a.dev else TEST_SEED)
        metrics[split] = {T: summarize(r, spec.n_values) for T, r in res.items()}
        preds.update({f'{split}_T{T}_{k}': v for T, r in res.items() for k, v in r.items()})
    torch.save(model.state_dict(), out / 'model.pt')   # for warm-start variants
    np.savez_compressed(out / 'preds.npz', **preds)
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=1))
    print(json.dumps({T: round(m['cross_page_acc'], 3) for T, m in metrics['iid'].items()}), file=log, flush=True)


if __name__ == '__main__':
    main()
