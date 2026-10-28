"""Step 4: aggregate seeds, strata by field group / document length, paired bootstrap for the go criteria.

Writes results/{tag}/report.md and report.json.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cemkie.data import SHORT
from cemkie.metrics import bootstrap_diff, f1

ROOT = Path(__file__).resolve().parents[1]
MONEY, OTHER = ['income', 'spending'], ['date', 'number', 'postcode']
ORDER = ['point', 'pool', 'set2', 'set4', 'cem_r1', 'cem_r2', 'cem_r4', 'cem_r8', 'cem_r4_nores']
COMPARE = [('cem_r4', 'set2'), ('cem_r4', 'set4'), ('cem_r4', 'point'), ('cem_r4', 'cem_r1'),
           ('cem_r4', 'cem_r4_nores'), ('set2', 'point'), ('pool', 'point')]


def length_bins():
    te = torch.load(ROOT / 'cache' / 'dev-0.pt', weights_only=False)
    chars = {d['id']: d['chars'] for d in te['docs']}
    q = np.quantile(list(chars.values()), [1 / 3, 2 / 3])
    return {k: ['short', 'medium', 'long'][int(np.searchsorted(q, v))] for k, v in chars.items()}, q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='main')
    args = ap.parse_args()
    base = ROOT / 'results' / args.tag
    runs = collections.defaultdict(list)
    for d in sorted(base.glob('*-s*')):
        if (d / 'metrics.json').exists():
            runs[d.name.rsplit('-s', 1)[0]].append(
                (json.load(open(d / 'metrics.json')), json.load(open(d / 'preds.json'))))
    lbin, q = length_bins()
    groups = {'money': lambda r: r['field'] in MONEY, 'other': lambda r: r['field'] in OTHER,
              **{f: (lambda f: lambda r: r['field'] == f)(f) for f in SHORT.values()},
              **{f'money_{b}': (lambda b: lambda r: r['field'] in MONEY and lbin[r['id']] == b)(b)
                 for b in ['short', 'medium', 'long']}}

    table = {}
    for v in [x for x in ORDER if x in runs]:
        per = {g: [100 * f1(r['c'] for r in preds if fn(r)) for _, preds in runs[v]] for g, fn in groups.items()}
        per['micro'] = [100 * m['test']['micro'] for m, _ in runs[v]]
        table[v] = dict(seeds=len(runs[v]), params=runs[v][0][0]['params'],
                        recall=float(np.mean([m['test']['recall_at_K'] for m, _ in runs[v]])),
                        **{g: (float(np.mean(x)), float(np.std(x))) for g, x in per.items()})

    def stacked(v, fn):
        """Counts summed over seeds per decision, aligned across variants by (id, field)."""
        acc = collections.defaultdict(lambda: np.zeros(3))
        for _, preds in runs[v]:
            for r in preds:
                if fn(r):
                    acc[(r['id'], r['field'])] += r['c']
        return acc

    tests = []
    for a, b in COMPARE:
        if a not in runs or b not in runs:
            continue
        for g in ['money', 'other', 'money_long']:
            A, B = stacked(a, groups[g]), stacked(b, groups[g])
            keys = sorted(A)
            d, lo, hi = bootstrap_diff([A[k] for k in keys], [B[k] for k in keys])
            tests.append(dict(a=a, b=b, group=g, diff=100 * d, lo=100 * lo, hi=100 * hi))

    pm = lambda x: f'{x[0]:.1f} ± {x[1]:.1f}'
    L = [f'# Kết quả CEM-KIE trên Kleister-Charity dev-0 (tag `{args.tag}`)', '',
         'F1 (%) trên test (dev-0, 440 tài liệu), mean ± std qua seed. `money` = income + spending, '
         '`other` = date + number + postcode.', '',
         '| variant | seeds | params | recall@K | micro | money | other | income | spending | date | number | postcode |',
         '|' + '---|' * 12]
    for v, r in table.items():
        L.append(f'| {v} | {r["seeds"]} | {r["params"] / 1e6:.2f}M | {100 * r["recall"]:.1f} | {pm(r["micro"])} | '
                 f'{pm(r["money"])} | {pm(r["other"])} | ' + ' | '.join(pm(r[f]) for f in SHORT.values()) + ' |')
    L += ['', f'Money F1 theo độ dài tài liệu (tertile số ký tự: {q[0]:.0f} / {q[1]:.0f}):', '',
          '| variant | short | medium | long |', '|---|---|---|---|']
    for v, r in table.items():
        L.append(f'| {v} | ' + ' | '.join(pm(r[f'money_{b}']) for b in ['short', 'medium', 'long']) + ' |')
    L += ['', 'Paired bootstrap (2000 lần, theo quyết định, cộng dồn các seed), chênh lệch F1 a − b (pp), CI 95%:', '',
          '| a | b | nhóm | Δ | CI 95% |', '|---|---|---|---|---|']
    for t in tests:
        L.append(f'| {t["a"]} | {t["b"]} | {t["group"]} | {t["diff"]:+.2f} | [{t["lo"]:+.2f}, {t["hi"]:+.2f}] |')
    (base / 'report.md').write_text('\n'.join(L) + '\n')
    json.dump(dict(table=table, tests=tests), open(base / 'report.json', 'w'), indent=1)
    print('\n'.join(L))


if __name__ == '__main__':
    main()
