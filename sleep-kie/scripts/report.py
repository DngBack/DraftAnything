"""Aggregate a results tag into docs tables and the Go-1 decision.

python scripts/report.py [--tag stage0]
Paired document bootstrap: test docs are identical across variants and seeds (fixed TEST_SEED),
so a difference is resampled over documents after averaging over seeds.
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sleepkie.data import TYPES
from train import TEST_T

CROSS = [1, 2]   # revision, link (aggregation secondary: unlearnable even for full-context in pilot)
ORDER = ['A0_N1', 'A1_N2', 'A2_N4', 'A3_N4_half', 'A4_N1_2x', 'A5_deep4', 'full']


def load(tag):
    runs = defaultdict(list)
    for d in sorted((ROOT / 'results' / tag).glob('*-s*')):
        if (d / 'metrics.json').exists():
            runs[d.name.rsplit('-s', 1)[0]].append((json.loads((d / 'metrics.json').read_text()), np.load(d / 'preds.npz')))
    return runs


def doc_score(p, split, T, types=CROSS):
    """Per-document cross-page accuracy (nan if the doc has no field of these types)."""
    ok = p[f'{split}_T{T}_pred'] == p[f'{split}_T{T}_gold']
    m = np.isin(p[f'{split}_T{T}_ftype'], types)
    with np.errstate(invalid='ignore'):
        return (ok & m).sum(1) / m.sum(1)


def seed_mean(runs, name, split, T):
    return np.mean([doc_score(p, split, T) for _, p in runs[name]], 0)


def boot(diff, n=5000, seed=0):
    diff = diff[~np.isnan(diff)]
    g = np.random.default_rng(seed)
    means = diff[g.integers(0, len(diff), (n, len(diff)))].mean(1)
    return diff.mean(), np.quantile(means, .025), np.quantile(means, .975)


def fmt(xs):
    xs = np.asarray(xs) * 100
    return f'{xs.mean():.1f}' + (f' ± {xs.std(ddof=1):.1f}' if len(xs) > 1 else '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='stage0')
    a = ap.parse_args()
    runs = load(a.tag)
    names = [n for n in ORDER if n in runs] + sorted(set(runs) - set(ORDER))
    lines = [f'# Kết quả {a.tag} (tự sinh bởi `scripts/report.py`)', '',
             'Giá trị: mean ± sd qua seed (%), 512 tài liệu test/độ dài, 8 field/tài liệu.', '']

    for split in ('iid', 'unseen_entity'):
        lines += [f'## Cross-page accuracy theo số trang — {split}', '',
                  '| variant | seeds | params | ' + ' | '.join(f'T={T}' for T in TEST_T) + ' |',
                  '|---|---|---|' + '---|' * len(TEST_T)]
        for n in names:
            ms = [m for m, _ in runs[n]]
            lines.append(f'| {n} | {len(ms)} | {ms[0]["params"] / 1e6:.2f}M | ' +
                         ' | '.join(fmt([m[split][str(T)]['cross_page_acc'] for m in ms]) for T in TEST_T) + ' |')
        lines.append('')

    for T in (50, 200):
        cols = ['field_acc', 'doc_em'] + [f'acc_{t}' for t in TYPES] + ['ev_hit', 'ev_recall', 'null_precision', 'null_recall']
        lines += [f'## Theo loại suy luận, iid, T={T}', '', '| variant | ' + ' | '.join(cols) + ' |', '|---|' + '---|' * len(cols)]
        for n in names:
            ms = [m['iid'][str(T)] for m, _ in runs[n]]
            lines.append(f'| {n} | ' + ' | '.join(fmt([m[c] for m in ms]) for c in cols) + ' |')
        lines.append('')

    lines += ['## Chi phí module memory (T=200, encoder loại trừ, GPU dùng chung)', '',
              '| variant | train s | ms/doc bs1 | ms/doc bs64 | peak MiB bs1 | peak MiB bs64 |', '|---|---|---|---|---|---|']
    for n in names:
        ms = [m for m, _ in runs[n]]
        e = lambda k: f'{np.mean([m["efficiency"][k] for m in ms]):.1f}'
        lines.append(f'| {n} | {np.mean([m["train_seconds"] for m in ms]):.0f} | {e("bs1_ms_per_doc")} | '
                     f'{e("bs64_ms_per_doc")} | {e("bs1_peak_mib")} | {e("bs64_peak_mib")} |')
    lines.append('')

    # distance curve (experiment B), T=200 iid, cross-page fields only
    bins = [(0, 25), (25, 50), (50, 100), (100, 150), (150, 200)]
    lines += ['## Accuracy theo khoảng cách bằng chứng đầu tiên → trang cuối (T=200, iid, cross-page)', '',
              '| variant | ' + ' | '.join(f'{lo}–{hi}' for lo, hi in bins) + ' |', '|---|' + '---|' * len(bins)]
    for n in names:
        cells = []
        for lo, hi in bins:
            accs = []
            for _, p in runs[n]:
                ok = p['iid_T200_pred'] == p['iid_T200_gold']
                d, ty = p['iid_T200_dist'], p['iid_T200_ftype']
                m = np.isin(ty, CROSS) & (d >= lo) & (d < hi)
                accs.append(ok[m].mean())
            cells.append(fmt(accs))
        lines.append(f'| {n} | ' + ' | '.join(cells) + ' |')
    lines.append('')

    # Go-1 contrasts (pre-registered in docs/stage0-protocol.md)
    contrasts = [('A2_N4', 'A0_N1'), ('A2_N4', 'A5_deep4'), ('A2_N4', 'A4_N1_2x'), ('A1_N2', 'A0_N1')]
    lines += ['## Go 1 — paired bootstrap trên tài liệu (cross-page acc, trung bình seed, pp)', '',
              '| contrast | split | T | Δ | 95% CI | seed nào cũng cùng dấu |', '|---|---|---|---|---|---|']
    verdict = {}
    for x, y in contrasts:
        if x not in runs or y not in runs:
            continue
        for split in ('iid', 'unseen_entity'):
            for T in (50, 100, 200):
                mean, lo, hi = boot(seed_mean(runs, x, split, T) - seed_mean(runs, y, split, T))
                per_seed = [np.nanmean(doc_score(px, split, T)) - np.nanmean(doc_score(py, split, T))
                            for (_, px), (_, py) in zip(runs[x], runs[y])]
                same = all(s > 0 for s in per_seed) or all(s < 0 for s in per_seed)
                verdict[(x, y, split, T)] = (mean, lo, hi, same)
                lines.append(f'| {x} − {y} | {split} | {T} | {mean * 100:+.1f} | [{lo * 100:+.1f}, {hi * 100:+.1f}] | {"có" if same else "không"} |')
    lines.append('')

    def passes(x, y, margin):
        cells = [verdict.get((x, y, 'iid', T)) for T in (50, 100)]
        return all(c and c[1] > 0 and c[0] >= margin and c[3] for c in cells)
    if ('A2_N4', 'A5_deep4', 'iid', 50) in verdict:
        g1a, g1b = passes('A2_N4', 'A0_N1', .05), passes('A2_N4', 'A5_deep4', .02)
        lines += ['## Quyết định Go 1 (tiêu chí khóa trước)', '',
                  f'- (a) A2_N4 hơn A0_N1 ≥ 5 pp, CI > 0, mọi seed cùng dấu, ở T=50 và T=100: **{"đạt" if g1a else "không đạt"}**',
                  f'- (b) A2_N4 hơn A5_deep4 ≥ 2 pp, CI > 0, mọi seed cùng dấu, ở T=50 và T=100: **{"đạt" if g1b else "không đạt"}**',
                  f'- **Go 1: {"ĐẠT" if g1a and g1b else "KHÔNG ĐẠT"}**', '']
    (ROOT / 'docs' / f'{a.tag}-results.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
