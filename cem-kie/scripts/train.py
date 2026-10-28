"""Step 3: train one variant/seed on cached features; select epoch on val, report dev-0 (test).

results/{tag}/{variant}-s{seed}/metrics.json and preds.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cemkie.data import FIELDS, SHORT
from cemkie.metrics import counts, f1
from cemkie.model import VARIANTS, Model, nll

ROOT = Path(__file__).resolve().parents[1]
SPLIT_SEED, VAL_FRAC = 1234, 0.15
TOP_M = 10   # value-level candidates kept in preds for the joint solver


def flatten(doc, dev):
    """Per-doc candidate arrays over all fields; seg = field index, vid offset to be unique per doc."""
    parts, voff, values = [], 0, []
    for fi, f in enumerate(FIELDS):
        t = doc['fields'][f]
        n = len(t['m'])
        parts.append((t['m'], t['feat'], t['vid'] + voff, t['label'], torch.full((n,), fi)))
        values += t['values']; voff += len(t['values'])
    m, feat, vid, lab, seg = (torch.cat(x) for x in zip(*parts))
    return dict(m=m.to(dev), feat=feat.to(dev), vid=vid.to(dev), label=lab.to(dev), seg=seg.to(dev),
                n_values=voff, values=values)


def collate(flat, dev):
    F = len(FIELDS)
    out = {k: [] for k in ['m', 'feat', 'vid', 'label', 'seg']}
    voff = 0
    for b, x in enumerate(flat):
        for k in out:
            v = x[k]
            out[k].append(v + voff if k == 'vid' else v + b * F if k == 'seg' else v)
        voff += x['n_values']
    out = {k: torch.cat(v) for k, v in out.items()}
    out['G'] = len(flat) * F
    out['fid'] = out['seg'] % F
    out['seg_fid'] = torch.arange(F, device=dev).repeat(len(flat))
    return out


def targets(lp, idx, label):
    return (idx >= 0) & label[idx.clamp(min=0)]


@torch.no_grad()
def evaluate(model, emb, docs, flat, dev, bs=8):
    model.eval()
    rows, recall = [], []
    for i in range(0, len(docs), bs):
        fl = flat[i:i + bs]
        b = collate(fl, dev)
        out = model(emb, b)
        lp, idx = out['logp'].float().cpu().numpy(), out['idx'].cpu().numpy()
        vid = b['vid'].cpu().numpy()
        offs = np.cumsum([0] + [x['n_values'] for x in fl])
        values = [v for x in fl for v in x['values']]
        lab = b['label'].cpu().numpy()
        for g in range(b['G']):
            d, fi = divmod(g, len(FIELDS))
            doc, f = docs[i + d], FIELDS[fi]
            ok = idx[g] >= 0
            probs = {}
            for k in np.nonzero(ok)[0]:
                v = values[vid[idx[g, k]]]
                probs[v] = probs.get(v, 0.) + np.exp(lp[g, k])
            best = max(probs, key=probs.get) if probs else None
            pred = best if best is not None and probs[best] > np.exp(lp[g, -1]) else None
            gold = doc['gold'].get(f)
            top = sorted(probs.items(), key=lambda kv: -kv[1])[:TOP_M]
            rows.append(dict(id=doc['id'], field=SHORT[f], pred=pred, gold=gold, c=counts(pred, gold),
                             top=[(v, float(p)) for v, p in top], null=float(np.exp(lp[g, -1]))))
            if gold is not None and doc['fields'][f]['label'].any():
                recall.append(bool(lab[idx[g, ok]].any()))
    model.train()
    res = {fs: f1(r['c'] for r in rows if r['field'] == fs) for fs in SHORT.values()}
    res['micro'] = f1(r['c'] for r in rows)
    res['recall_at_K'] = float(np.mean(recall))
    return res, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=list(VARIANTS))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--K', type=int, default=64)
    ap.add_argument('--div', type=float, default=0.05)
    ap.add_argument('--tag', default='main')
    args = ap.parse_args()
    dev = 'cuda'
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out_dir = ROOT / 'results' / args.tag / f'{args.model}-s{args.seed}'
    out_dir.mkdir(parents=True, exist_ok=True)

    tr = torch.load(ROOT / 'cache' / 'train.pt', weights_only=False)
    te = torch.load(ROOT / 'cache' / 'dev-0.pt', weights_only=False)
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(tr['docs']))
    n_val = int(VAL_FRAC * len(perm))
    val_docs = [tr['docs'][i] for i in perm[:n_val]]
    tr_docs = [tr['docs'][i] for i in perm[n_val:]]
    test_docs = te['docs']
    emb_tr, emb_te = tr['emb'].to(dev), te['emb'].to(dev)
    fl_tr = [flatten(d, dev) for d in tr_docs]
    fl_val = [flatten(d, dev) for d in val_docs]
    fl_te = [flatten(d, dev) for d in test_docs]

    model = Model(n_fields=len(FIELDS), K=args.K, **VARIANTS[args.model]).to(dev)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(tr_docs) + args.bs - 1) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.05)
    log = open(out_dir / 'log.txt', 'w')
    print(f'{args.model} seed={args.seed} params={params:,} train={len(tr_docs)} val={len(val_docs)} '
          f'test={len(test_docs)}', file=log, flush=True)

    best, best_state, t0, step = -1, None, time.time(), 0
    rng = np.random.default_rng(args.seed)
    for ep in range(args.epochs):
        order = rng.permutation(len(tr_docs))
        tot = []
        for i in range(0, len(order), args.bs):
            b = collate([fl_tr[j] for j in order[i:i + args.bs]], dev)
            out = model(emb_tr, b)
            loss = nll(out['router'], targets(out['router'], out['router_idx'], b['label']))
            if model.kind != 'point':
                loss = loss + nll(out['logp'], targets(out['logp'], out['idx'], b['label']))
            if 'div' in out:
                loss = loss + args.div * out['div']
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            tot.append(loss.item())
        val, _ = evaluate(model, emb_tr, val_docs, fl_val, dev)
        print(f'ep {ep} step {step} loss {np.mean(tot):.3f} val micro {val["micro"]:.4f} '
              + ' '.join(f'{k}={v:.3f}' for k, v in val.items() if k != 'micro') + f' {time.time() - t0:.0f}s',
              file=log, flush=True)
        if val['micro'] > best:
            best, best_ep = val['micro'], ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    val, val_rows = evaluate(model, emb_tr, val_docs, fl_val, dev)
    test, rows = evaluate(model, emb_te, test_docs, fl_te, dev)
    json.dump(val_rows, open(out_dir / 'val_preds.json', 'w'))
    print(f'best ep {best_ep} val {val["micro"]:.4f} test {test["micro"]:.4f}', file=log, flush=True)
    json.dump(dict(args=vars(args), params=params, best_epoch=best_ep, val=val, test=test,
                   train_seconds=time.time() - t0), open(out_dir / 'metrics.json', 'w'), indent=1)
    json.dump(rows, open(out_dir / 'preds.json', 'w'))


if __name__ == '__main__':
    main()
