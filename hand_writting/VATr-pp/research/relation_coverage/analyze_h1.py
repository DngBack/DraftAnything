"""Analyse H1 records: paired within-(writer, word) contrasts with writer-cluster bootstrap CIs."""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])
recs = [json.loads(l) for l in open(out / 'records.jsonl')]
B = 4000
rng = random.Random(0)


def table(metric):
    # mean over draws per (writer, word, cond)
    acc = defaultdict(list)
    for r in recs:
        acc[(r['writer'], r['word_id'], r['cond'])].append(r[metric])
    m = {k: float(np.mean(v)) for k, v in acc.items()}
    units = sorted({(w, t) for w, t, _ in m})
    return m, units


def contrast(m, units, a, b):
    """mean over units of m[a]-m[b]; bootstrap writers."""
    byw = defaultdict(list)
    for w, t in units:
        if (w, t, a) in m and (w, t, b) in m:
            byw[w].append(m[(w, t, a)] - m[(w, t, b)])
    ws = sorted(byw)
    est = float(np.mean([x for w in ws for x in byw[w]]))
    boots = []
    for _ in range(B):
        s = [x for w in (rng.choice(ws) for _ in ws) for x in byw[w]]
        boots.append(np.mean(s))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    n = sum(len(v) for v in byw.values())
    return est, lo, hi, n, len(ws)


report = {}
for metric in ['d_pair', 'd_centroid']:
    m, units = table(metric)
    means = {c: float(np.mean([v for (w, t, cc), v in m.items() if cc == c])) for c in ['SELF', 'COV', 'RAND', 'UNC', 'OTHER']}
    res = {name: contrast(m, units, a, b) for name, (a, b) in {
        'join_effect UNC-COV': ('UNC', 'COV'),
        'writer_effect OTHER-COV': ('OTHER', 'COV'),
        'copy_effect COV-SELF': ('COV', 'SELF'),
        'RAND-COV': ('RAND', 'COV'),
    }.items()}
    je, we = res['join_effect UNC-COV'][0], res['writer_effect OTHER-COV'][0]
    print(f'\n== {metric} (lower = closer to the real writer) ==')
    print('  condition means:', {k: round(v, 4) for k, v in means.items()})
    for k, (e, lo, hi, n, nw) in res.items():
        print(f'  {k:26s} {e:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  units={n} writers={nw}')
    print(f'  join effect as % of wrong-writer effect: {100 * je / we:.1f}%')
    report[metric] = dict(means=means, contrasts={k: dict(est=v[0], lo=v[1], hi=v[2], units=v[3], writers=v[4])
                                                  for k, v in res.items()}, join_over_writer=je / we)

# observational dose-response in RAND: d_pair vs join coverage, controlling word length & letter instances
rand = [r for r in recs if r['cond'] == 'RAND']
X = np.array([[1, r['join_cov'], len(r['text']), r['letter_instances'], r['letter_cov']] for r in rand], float)
y = np.array([r['d_pair'] for r in rand])
# writer fixed effects via demeaning
wid = np.array([r['writer'] for r in rand])
for col in range(1, X.shape[1]):
    for w in set(wid):
        X[wid == w, col] -= X[wid == w, col].mean()
yd = y.copy()
for w in set(wid):
    yd[wid == w] -= y[wid == w].mean()
beta = np.linalg.lstsq(X[:, 1:], yd, rcond=None)[0]
print('\n== RAND observational (writer FE): d_pair ~ join_cov + len + letter_instances + letter_cov ==')
print('  coef join_cov (0->1 coverage):', round(beta[0], 4), ' len:', round(beta[1], 4),
      ' letter_inst:', round(beta[2], 5), ' letter_cov:', round(beta[3], 4))
print('  RAND join coverage: mean', round(float(np.mean([r['join_cov'] for r in rand])), 3))
# support balance check COV vs UNC
for c in ['COV', 'UNC']:
    rr = [r for r in recs if r['cond'] == c]
    print(f'  {c}: join_cov {np.mean([r["join_cov"] for r in rr]):.3f} letter_cov {np.mean([r["letter_cov"] for r in rr]):.3f} '
          f'letter_inst {np.mean([r["letter_instances"] for r in rr]):.1f} mean_len {np.mean([r["mean_len"] for r in rr]):.2f}')
report['rand_regression'] = dict(zip(['join_cov', 'len', 'letter_instances', 'letter_cov'], map(float, beta)))
json.dump(report, open(out / 'summary.json', 'w'), indent=1)
