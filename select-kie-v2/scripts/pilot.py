"""Exploratory same-information baselines on existing frozen CORD predictions."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from functools import lru_cache

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('HF_HOME', str(ROOT / 'select-kie-v2' / '.cache' / 'huggingface'))
sys.path.insert(0, str(ROOT / 'selective-kie'))
import analyze as A
from common import load_cord, norm_value, aurc, rc_curve, threshold_at_risk, apply_threshold

OUT = ROOT / 'select-kie-v2' / 'results'
MODELS = ['Qwen3-VL-4B-Instruct', 'Qwen2-VL-2B-Instruct']
CONFIGS = {
    'field-only': ([], 'lr', True),
    'fusion': (A.FUSION, 'lr', True),
    'fusion+binding': (A.FUSION + A.BIND, 'lr', True),
    'same-info-HGB': (A.FUSION + A.BIND, 'hgb', True),
    'fusion-HGB': (A.FUSION, 'hgb', True),
    'no-field': (A.FUSION + A.BIND, 'lr', False),
    'no-key-competition': (A.FUSION + [x for x in A.BIND if x != 'key_lp'], 'lr', True),
    'no-value-rival': (A.FUSION + [x for x in A.BIND if x not in ['bind_margin', 'bind_argmax', 'bind_entropy']], 'lr', True),
}

@lru_cache(None)
def docs(split):
    return load_cord(split)

A.load_cord = docs

def matrix(rows, feats, field):
    return np.asarray([[r[f] for f in feats] + ([float(r['field'] == f) for f in A.PRIMARY] if field else []) for r in rows])

def metric(scores, err, threshold):
    c, r = apply_threshold(scores, err, threshold)
    accepted = scores >= threshold
    return {'coverage': float(c), 'risk': float(r) if np.isfinite(r) else None,
            'accepted': int(accepted.sum()), 'errors': int(err[accepted].sum()),
            'threshold': float(threshold) if np.isfinite(threshold) else None}

def bootstrap(v, t, sv, st, names, B, seed=20260928):
    rng = np.random.default_rng(seed)
    gv, gt = A.doc_index(v), A.doc_index(t)
    ev, et = np.array([r['err'] for r in v]), np.array([r['err'] for r in t])
    diffs, areas, risks, empty = [], [], {n: [] for n in names}, {n: 0 for n in names}
    for _ in range(B):
        iv = np.concatenate([gv[j] for j in rng.integers(len(gv), size=len(gv))])
        it = np.concatenate([gt[j] for j in rng.integers(len(gt), size=len(gt))])
        cov = {}
        for n in names:
            th = threshold_at_risk(sv[n][iv], ev[iv], .05)
            c, r = apply_threshold(st[n][it], et[it], th)
            cov[n] = c
            if np.isfinite(r):
                risks[n].append(r > .05)
            else:
                empty[n] += 1
        diffs.append(cov[names[0]] - cov[names[1]])
        areas.append(aurc(st[names[0]][it], et[it]) - aurc(st[names[1]][it], et[it]))
    return {'coverage_diff_ci95': np.percentile(diffs, [2.5, 97.5]).tolist(),
            'aurc_diff_ci95': np.percentile(areas, [2.5, 97.5]).tolist(),
            'risk_exceedance_given_nonempty': {n: float(np.mean(risks[n])) if risks[n] else None for n in names},
            'empty_acceptance_fraction': {n: empty[n] / B for n in names}}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    (OUT / 'heads').mkdir(exist_ok=True)
    manifest = {'outputs': {}, 'duplicates': {}, 'bootstrap': args.bootstrap, 'seed': 20260928,
                'dataset': 'naver-clova-ix/cord-v2',
                'dataset_revision': (Path(os.environ['HF_HOME']) / 'hub/datasets--naver-clova-ix--cord-v2/refs/main').read_text().strip(),
                'source_code_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in [Path(__file__), ROOT / 'selective-kie/common.py', ROOT / 'selective-kie/analyze.py']}}
    hashes = {}
    for split in ['train', 'validation', 'test']:
        for d in docs(split):
            im = d['image']
            h = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
            hashes.setdefault(h, []).append(d['id'])
        manifest.setdefault('annotation_sha256', {})[split] = hashlib.sha256(json.dumps(
            [{'id': d['id'], 'gold': d['gold'], 'lines': d['lines']} for d in docs(split)], sort_keys=True).encode()).hexdigest()
    manifest['duplicates'] = {h: ids for h, ids in hashes.items() if len({i.split('/')[0] for i in ids}) > 1}
    print('Exact cross-split image duplicates:', len(manifest['duplicates']), flush=True)
    results, audit = {}, []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, model in zip(axes, MODELS):
        print('Processing', model, flush=True)
        A.MD = str(ROOT / 'selective-kie' / 'outputs' / model)
        for fn in sorted(Path(A.MD).glob('*.jsonl')):
            manifest['outputs'][str(fn.relative_to(ROOT))] = hashlib.sha256(fn.read_bytes()).hexdigest()
        rows = [[r for r in A.rows(s) if r['field'] in A.PRIMARY] for s in ['train', 'validation', 'test']]
        tr, va, te = rows
        assert [len(A.doc_index(r)) for r in rows] == [800, 100, 100], 'Incomplete outputs'
        assert not (set(r['doc'] for r in tr) & set(r['doc'] for r in va + te))
        ev, et = np.array([r['err'] for r in va]), np.array([r['err'] for r in te])
        sv, st, measures = {}, {}, {}
        for name, (feats, kind, field) in CONFIGS.items():
            head = (HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7, min_samples_leaf=40,
                    l2_regularization=1., early_stopping=False, random_state=20260928) if kind == 'hgb'
                    else make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=5000)))
            head.fit(matrix(tr, feats, field), [1-r['err'] for r in tr])
            joblib.dump({'head': head, 'features': feats, 'field_identity': field,
                         'field_order': A.PRIMARY}, OUT / 'heads' / f'{model}.{name}.joblib')
            sv[name] = head.predict_proba(matrix(va, feats, field))[:, 1]
            st[name] = head.predict_proba(matrix(te, feats, field))[:, 1]
            measures[name] = {'aurc': aurc(st[name], et), 'brier': float(brier_score_loss(1-et, st[name])),
                              'targets': {str(alpha): metric(st[name], et, threshold_at_risk(sv[name], ev, alpha))
                                          for alpha in [.01, .02, .05, .10]}}
            print(name, measures[name]['aurc'], measures[name]['targets']['0.05'], flush=True)
        contrasts = {}
        for a, b in [('fusion+binding', 'fusion'), ('same-info-HGB', 'fusion+binding'),
                     ('same-info-HGB', 'fusion-HGB'), ('fusion+binding', 'no-key-competition')]:
            contrasts[a+' minus '+b] = bootstrap(va, te, sv, st, (a, b), args.bootstrap)
        tot = {r['doc']: norm_value(r['gold']) for r in va + te if r['field'] == 'total'}
        def keep(r):
            return not (r['field'] == 'subtotal' and norm_value(r['gold']) is None and norm_value(r['pred']) == tot.get(r['doc']))
        iv = np.array([keep(r) for r in va]); it = np.array([keep(r) for r in te])
        sensitivity = {}
        for n in CONFIGS:
            sensitivity[n] = metric(st[n][it], et[it], threshold_at_risk(sv[n][iv], ev[iv], .05))
        allr = tr+va+te
        results[model] = {'counts': [len(r) for r in rows], 'taxonomy_all': dict(Counter(r['etype'] for r in allr if r['err'])),
                          'taxonomy_test': dict(Counter(r['etype'] for r in te if r['err'])),
                          'test_error': float(et.mean()), 'measures': measures, 'contrasts': contrasts,
                          'sensitivity_dropped_val_test': [int((~iv).sum()), int((~it).sum())], 'sensitivity': sensitivity}
        rng = np.random.default_rng(20260928)
        # Random population sample and disjoint targeted error sample; do not mix for prevalence.
        random_ids = set(rng.choice(len(te), size=80, replace=False).tolist())
        targeted = [i for i, r in enumerate(te) if r['err'] and i not in random_ids]
        target_ids = set(rng.choice(targeted, size=min(60, len(targeted)), replace=False).tolist())
        for i in sorted(random_ids | target_ids):
            r = te[i]
            audit.append({'model': model, 'sample': 'random' if i in random_ids else 'targeted_error',
                          'doc': r['doc'], 'field': r['field'], 'pred': r['pred'], 'gold': r['gold'],
                          'auto_type': r['etype'] or 'correct', 'human_type': '', 'reviewer': '', 'notes': ''})
        with (OUT / f'predictions-{model}.csv').open('w') as f:
            writer = csv.writer(f); writer.writerow(['doc', 'field', 'error', 'auto_type'] + list(CONFIGS))
            for i, r in enumerate(te):
                writer.writerow([r['doc'], r['field'], r['err'], r['etype']] + [st[n][i] for n in CONFIGS])
        for n in ['fusion', 'fusion+binding', 'fusion-HGB', 'same-info-HGB']:
            cov, risk, _ = rc_curve(st[n], et)
            ax.plot(cov, risk, label=n)
        ax.axhline(.05, color='gray', linestyle='--'); ax.set(title=model, xlabel='Coverage', ylabel='Accepted error', ylim=(0,.15))
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / 'risk-coverage.png', dpi=170); plt.close(fig)
    with (OUT / 'audit.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(audit[0])); writer.writeheader(); writer.writerows(audit)
    (OUT / 'metrics.json').write_text(json.dumps(results, indent=2, allow_nan=False))
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Saved', OUT, flush=True)

if __name__ == '__main__':
    main()
