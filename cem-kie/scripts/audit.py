"""Step 1: does competing evidence exist in Kleister-Charity, and how far do frequency/position heuristics go?

Writes results/audit.json and results/audit.md. CPU only.
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cemkie.data import FIELDS, FTYPE, SHORT, candidates, load_split
from cemkie.metrics import counts, f1

OUT = Path(__file__).resolve().parents[1] / 'results'


def subset(fc, gold):
    """single / repeated / competing / absent (gold not in text) / null (no gold)."""
    if gold is None:
        return 'null'
    if not fc['label'].any():
        return 'absent'
    mention_vals = collections.Counter()
    for m, v in set(zip(fc['cand_m'].tolist(), fc['cand_v'])):
        mention_vals[v] += 1
    g = mention_vals[gold]
    if any(c >= g for v, c in mention_vals.items() if v != gold):
        return 'competing'
    return 'single' if g == 1 else 'repeated'


def heuristics(fc, ftype):
    vals = [v for v, s in zip(fc['cand_v'], fc['cand_scale']) if s == 0]   # as written
    if not vals:
        return dict(first=None, last=None, frequent=None, max=None)
    freq = collections.Counter(vals).most_common(1)[0][0]
    h = dict(first=vals[0], last=vals[-1], frequent=freq, max=None)
    if ftype == 'money':
        h['max'] = max(vals, key=float)
    return h


def main():
    OUT.mkdir(exist_ok=True)
    report = {}
    for split in ['train', 'dev-0']:
        docs = load_split(split)
        stats = {f: collections.defaultdict(list) for f in FIELDS}
        for d in docs:
            c = candidates(d)
            pos = c['spans'][:, 0] / max(len(d['text']), 1)
            for f in FIELDS:
                fc, g = c['fields'][f], d['gold'].get(f)
                s = stats[f]
                sub = subset(fc, g)
                s['subset'].append(sub)
                s['n_values'].append(len(set(fc['cand_v'])))
                s['chars'].append(len(d['text']))
                if fc['label'].any():
                    gm = np.unique(fc['cand_m'][fc['label']])
                    s['gold_mentions'].append(len(gm))
                    s['gold_spread'].append(float(pos[gm].max() - pos[gm].min()))
                for k, v in heuristics(fc, FTYPE[f]).items():
                    if v is not None or FTYPE[f] == 'money' or k != 'max':
                        s['h_' + k].append(counts(v, g))
                s['h_oracle'].append(counts(g if fc['label'].any() else None, g))
        rep = {}
        for f in FIELDS:
            s = stats[f]
            sub = collections.Counter(s['subset'])
            rep[SHORT[f]] = dict(
                docs=len(docs), subsets=dict(sub),
                candidate_values_median=float(np.median(s['n_values'])),
                gold_mentions_median=float(np.median(s['gold_mentions'])) if s['gold_mentions'] else None,
                gold_spread_median=float(np.median(s['gold_spread'])) if s['gold_spread'] else None,
                f1={k[2:]: f1(v) for k, v in s.items() if k.startswith('h_') and v},
            )
        report[split] = rep
    (OUT / 'audit.json').write_text(json.dumps(report, indent=1))

    lines = ['# Audit Kleister-Charity (5 field có ứng viên đóng)', '']
    for split, rep in report.items():
        lines += [f'## {split} ({next(iter(rep.values()))["docs"]} tài liệu)', '',
                  '| field | null | absent | single | repeated | competing | #giá trị ứng viên (median) | '
                  'oracle F1 | first | last | frequent | max |', '|' + '---|' * 12]
        for f, r in rep.items():
            n = r['docs']; su = r['subsets']
            pc = lambda k: f'{100 * su.get(k, 0) / n:.0f}%'
            h = r['f1']
            fmt = lambda k: f'{100 * h[k]:.1f}' if k in h else '–'
            lines.append(f'| {f} | {pc("null")} | {pc("absent")} | {pc("single")} | {pc("repeated")} | '
                         f'{pc("competing")} | {r["candidate_values_median"]:.0f} | {fmt("oracle")} | {fmt("first")} | '
                         f'{fmt("last")} | {fmt("frequent")} | {fmt("max")} |')
        lines.append('')
    (OUT / 'audit.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
