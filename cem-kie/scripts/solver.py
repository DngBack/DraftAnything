"""Step 6 (CoSolve oracle): exact joint inference over (income, spending) pairs with an arithmetic factor.

Unaries: LLM candidate log-probs (results/llm_score) and set4 top-M probs (results/topm, per seed).
Parameters are fit on val by NLL of the gold pair under softmax over all pairs; reported on dev-0.
See docs/protocol-cosolve.md. Writes results/cosolve/report.md and report.json.
"""
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cemkie.data import load_split, mentions
from cemkie.metrics import bootstrap_diff, counts, f1
from train import SPLIT_SEED, VAL_FRAC

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'cosolve'
FIELDS = ['income', 'spending']
EPS = 1e-6
KW = re.compile(r'net\s+(income|expenditure|movement|incoming|outgoing)|surplus|deficit', re.I)
N_PHI = 5


def money_values(text):
    return np.array(sorted({float(v) for t, _, _, vs, _ in mentions(text) if t == 'money' for v in vs}))


def build(split, seed):
    docs = load_split('train') if split == 'val' else load_split('dev-0')
    if split == 'val':
        perm = np.random.default_rng(SPLIT_SEED).permutation(len(docs))
        docs = [docs[i] for i in perm[:int(VAL_FRAC * len(perm))]]
    text = {d['id']: d['text'] for d in docs}
    llm = json.load(open(ROOT / 'results' / 'llm_score' / f'{split}.json'))
    sp = json.load(open(ROOT / 'results' / 'topm' / f'set4-s{seed}' / ('val_preds.json' if split == 'val' else 'preds.json')))
    st = {(r['id'], r['field']): (dict(r['top']), r['null']) for r in sp if r['field'] in FIELDS}
    items = []
    for x in llm:
        vals, ul, us = {}, {}, {}
        for f in FIELDS:
            V = sorted(v for v in x['cands'][f] if v != 'null') + ['null']
            top, pnull = st[(x['id'], f)]
            vals[f] = V
            ul[f] = np.array([x['cands'][f].get(v, -50.) for v in V])
            us[f] = np.log(np.array([pnull if v == 'null' else top.get(v, 0.) for v in V]) + EPS)
        mv = money_values(text[x['id']])
        I = np.array([float(v) if v != 'null' else np.nan for v in vals['income']])
        S = np.array([float(v) if v != 'null' else np.nan for v in vals['spending']])
        diff = np.abs(I[:, None] - S[None])
        if len(mv):
            pos = np.clip(np.searchsorted(mv, diff), 1, len(mv) - 1)
            near = np.minimum(np.abs(mv[pos - 1] - diff), np.abs(mv[np.clip(pos, 0, len(mv) - 1)] - diff))
            net = (near <= 1.5) & ~np.isnan(diff) & (diff >= 1)
        else:
            net = np.zeros_like(diff, bool)
        base = ul['income'][:, None] + ul['spending'][None] + us['income'][:, None] + us['spending'][None]
        ti, tj = np.unravel_index(base.argmax(), base.shape)
        nonnull = ~np.isnan(diff)
        marg = [np.sort(ul[f])[-1] - np.sort(ul[f])[-2] if len(ul[f]) > 1 else 10. for f in FIELDS]
        phi = np.array([bool(KW.search(text[x['id']])), net[nonnull].mean() if nonnull.any() else 0.,
                        float(net[ti, tj]), np.tanh(marg[0] / 5), np.tanh(marg[1] / 5)], np.float32)
        gidx = [V.index(x['gold'][f]) if x['gold'][f] in V else len(V) - 1 for f, V in
                ((f, vals[f]) for f in FIELDS)]
        items.append(dict(id=x['id'], gold=x['gold'], greedy=x['greedy'], vals=vals,
                          ul=[torch.tensor(ul[f], dtype=torch.float32) for f in FIELDS],
                          us=[torch.tensor(us[f], dtype=torch.float32) for f in FIELDS],
                          net=torch.tensor(net, dtype=torch.float32), phi=torch.tensor(phi), gidx=gidx))
    return items


class Solver(torch.nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.a, self.b = torch.nn.Parameter(torch.ones(1)), torch.nn.Parameter(torch.ones(1))
        self.nb = torch.nn.Parameter(torch.zeros(2))
        self.lam = torch.nn.Parameter(torch.zeros(1))
        self.w = torch.nn.Parameter(torch.zeros(N_PHI + 1))

    def score(self, it):
        u = [self.a * l + self.b * s for l, s in zip(it['ul'], it['us'])]
        for k in range(2):
            u[k] = u[k] + self.nb[k] * (torch.arange(len(u[k])) == len(u[k]) - 1).float()
        sc = u[0][:, None] + u[1][None]
        if self.mode in ('learned', 'evcond'):
            g = torch.sigmoid(self.w[:-1] @ it['phi'] + self.w[-1]) * 2 if self.mode == 'evcond' else 1.
            sc = sc + self.lam * g * it['net']
        return sc

    def decode(self, it):
        with torch.no_grad():
            sc = self.score(it)
            if self.mode == 'fixed' and it['net'].any():
                sc = sc.masked_fill(it['net'] == 0, -1e9)
        i, j = np.unravel_index(int(sc.argmax()), sc.shape)
        return [it['vals']['income'][i], it['vals']['spending'][j]]


def fit(mode, items, steps=400):
    m = Solver('indep' if mode == 'fixed' else mode)   # fixed = indep unaries + hard constraint at decode
    opt = torch.optim.Adam(m.parameters(), lr=0.05)
    for _ in range(steps):
        loss = 0.
        for it in items:
            sc = m.score(it)
            loss = loss - torch.log_softmax(sc.flatten(), 0)[it['gidx'][0] * sc.shape[1] + it['gidx'][1]]
        opt.zero_grad(); (loss / len(items)).backward(); opt.step()
    m.mode = mode
    return m


def rows_of(items, preds):
    out = []
    for it, p in zip(items, preds):
        for f, v in zip(FIELDS, p):
            v = None if v == 'null' else v
            out.append(dict(id=it['id'], field=f, pred=v, gold=it['gold'][f], c=counts(v, it['gold'][f]),
                            sat=None))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    modes = ['indep', 'fixed', 'learned', 'evcond']
    allrows = collections.defaultdict(list)
    params = {}
    for seed in range(3):
        val, test = build('val', seed), build('test', seed)
        for mode in modes:
            m = fit(mode, val)
            params[f'{mode}-s{seed}'] = {k: v.detach().numpy().round(3).tolist() for k, v in m.named_parameters()}
            preds = [m.decode(it) for it in test]
            rows = rows_of(test, preds)
            for k, (it, p) in enumerate(zip(test, preds)):
                i, j = it['vals']['income'].index(p[0]), it['vals']['spending'].index(p[1])
                rows[2 * k]['sat'] = rows[2 * k + 1]['sat'] = bool(it['net'][i, j])
            allrows[mode].append(rows)
        allrows['llm_greedy'].append(rows_of(test, [[it['greedy'][f] or 'null' for f in FIELDS] for it in test]))
        allrows['oracle'].append(rows_of(test, [[it['gold'][f] if it['gold'][f] in it['vals'][f] else 'null'
                                                 for f in FIELDS] for it in test]))

    def prec(rows):
        tp, fp, _ = np.array([r['c'] for r in rows]).sum(0)
        return tp / max(tp + fp, 1)

    table = {}
    for k, runs in allrows.items():
        table[k] = dict(money=[100 * f1(r['c'] for r in rows) for rows in runs],
                        income=[100 * f1(r['c'] for r in rows if r['field'] == 'income') for rows in runs],
                        spending=[100 * f1(r['c'] for r in rows if r['field'] == 'spending') for rows in runs],
                        precision=[100 * prec(rows) for rows in runs])

    def stacked(k):
        acc = collections.defaultdict(lambda: np.zeros(3))
        for rows in allrows[k]:
            for r in rows:
                acc[(r['id'], r['field'])] += r['c']
        return acc

    tests = []
    for a, b in [('learned', 'indep'), ('evcond', 'learned'), ('fixed', 'indep'), ('indep', 'llm_greedy'),
                 ('learned', 'llm_greedy')]:
        A, B = stacked(a), stacked(b)
        keys = sorted(A)
        d, lo, hi = bootstrap_diff([A[x] for x in keys], [B[x] for x in keys])
        tests.append(dict(a=a, b=b, diff=100 * d, lo=100 * lo, hi=100 * hi))

    # consistent-but-wrong: errors of `learned` where the chosen pair satisfies the net factor
    err = [r for rows in allrows['learned'] for r in rows if r['pred'] is not None and r['pred'] != r['gold']]
    cw = float(np.mean([r['sat'] for r in err])) if err else 0.

    pm = lambda x: f'{np.mean(x):.1f} ± {np.std(x):.1f}'
    L = ['# CoSolve oracle — income/spending trên Kleister-Charity dev-0', '',
         'F1 (%), mean ± std qua 3 seed unary set4 (llm_greedy, oracle lặp lại theo seed). Tham số solver học trên val.', '',
         '| hệ thống | money F1 | income | spending | precision |', '|---|---|---|---|---|']
    for k in ['llm_greedy', 'indep', 'fixed', 'learned', 'evcond', 'oracle']:
        t = table[k]
        L.append(f'| {k} | {pm(t["money"])} | {pm(t["income"])} | {pm(t["spending"])} | {pm(t["precision"])} |')
    L += ['', 'Paired bootstrap, Δ money F1 (pp), CI 95%:', '', '| a | b | Δ | CI |', '|---|---|---|---|']
    for t in tests:
        L.append(f'| {t["a"]} | {t["b"]} | {t["diff"]:+.2f} | [{t["lo"]:+.2f}, {t["hi"]:+.2f}] |')
    L += ['', f'Lỗi của `learned` mà cặp dự đoán vẫn thỏa ràng buộc net (nhất quán nhưng sai): {100 * cw:.0f}% '
          f'trên {len(err)} lỗi (3 seed).', '']
    (OUT / 'report.md').write_text('\n'.join(L))
    json.dump(dict(table=table, tests=tests, consistent_wrong=cw, params=params), open(OUT / 'report.json', 'w'),
              indent=1)
    json.dump(allrows, open(OUT / 'preds.json', 'w'))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
