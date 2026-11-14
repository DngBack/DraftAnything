"""Score transcriptions joined by sample ID, with writer-cluster CIs.

Evaluator identity is recorded, not presumed independent. Transcription fidelity
is not handwriting aesthetics. Reference files come from train.py evaluation.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import unicodedata as ud
import numpy as np
from research.longtail.prepare import read_records, write_json
from research.vietcompose.orthography import component_errors


def alignment(a, b):
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+(a[i-1] != b[j-1]))
    pairs, i, j = [], n, m
    while i or j:
        if i and j and dp[i][j] == dp[i-1][j-1]+(a[i-1] != b[j-1]):
            pairs.append((a[i-1], b[j-1])); i -= 1; j -= 1
        elif i and dp[i][j] == dp[i-1][j]+1:
            pairs.append((a[i-1], None)); i -= 1
        else:
            pairs.append((None, b[j-1])); j -= 1
    return list(reversed(pairs))


def score(rows, targets):
    ref, pred, correct, roles = Counter(), Counter(), Counter(), Counter()
    errors = exact = 0
    for row in rows:
        a, b = row['reference'], row['prediction']
        exact += a == b
        ref.update(a); pred.update(b)
        for x, y in alignment(a, b):
            errors += x != y
            if x is not None and x == y: correct[x] += 1
        roles.update(component_errors(a, b))
    supported = [c for c in targets if ref[c] > 0]
    per = {c: {'reference_occurrences': ref[c], 'predicted_occurrences': pred[c],
               'correct_aligned': correct[c], 'f1': 2*correct[c]/max(ref[c]+pred[c], 1)} for c in targets}
    return {'cer': errors/max(sum(ref.values()), 1), 'word_exact': exact/max(len(rows), 1),
            'tail_macro_f1': sum(per[c]['f1'] for c in supported)/len(supported) if supported else None,
            'tail_reference_coverage': len(supported), 'tail_requested': len(targets), 'per_target': per,
            'component_error_rates': {k: v/max(sum(ref.values()), 1) for k, v in roles.items()
                if k in ('base_errors', 'shape_errors', 'tone_errors', 'case_errors')}}


def evaluate(references, predictions, targets, repeats=1000, seed=742):
    if repeats < 1: raise ValueError('Positive bootstrap count required')
    targets = ''.join(sorted(set(ud.normalize('NFC', targets))))
    if not references or not targets: raise ValueError('Nonempty references and target inventory required')
    ref_by_id = {r['sample_id']: r for r in references}
    pred_by_id = {r['sample_id']: r for r in predictions}
    if len(ref_by_id) != len(references) or len(pred_by_id) != len(predictions):
        raise ValueError('Duplicate sample ID')
    if ref_by_id.keys() != pred_by_id.keys():
        raise ValueError('Missing or extra predictions: evaluation coverage must match exactly')
    rows = []
    for key, r in ref_by_id.items():
        if not r['reference'] or not isinstance(r['writer'], str) or not r['writer']:
            raise ValueError('Nonempty reference and genuine writer required')
        prediction = pred_by_id[key]['prediction']
        if not isinstance(prediction, str): raise ValueError('Prediction must be a string; empty allowed')
        rows.append({'sample_id': key, 'writer': r['writer'], 'reference': ud.normalize('NFC', r['reference']),
                     'prediction': ud.normalize('NFC', prediction)})
    result = score(rows, targets)
    clusters = defaultdict(list)
    for r in rows: clusters[r['writer']].append(r)
    rng, draws = random.Random(seed), defaultdict(list)
    names = sorted(clusters)
    supported = {c for c in targets if result['per_target'][c]['reference_occurrences']}
    for _ in range(repeats):
        sample = [r for name in rng.choices(names, k=len(names)) for r in clusters[name]]
        metric = score(sample, targets)
        for key in ('cer', 'word_exact'): draws[key].append(metric[key])
        # Keep the glyph inventory fixed instead of changing macro denominators.
        if supported and all(metric['per_target'][c]['reference_occurrences'] for c in supported):
            draws['tail_macro_f1'].append(metric['tail_macro_f1'])
    result['writer_cluster_percentile_95_ci'] = {
        key: {'low': float(np.quantile(values, .025)), 'high': float(np.quantile(values, .975)),
              'valid_replicates': len(values)} for key, values in draws.items() if values}
    result.update(writers=len(clusters), samples=len(rows), bootstrap_requested=repeats)
    result['ci_note'] = 'Descriptive percentile CI; one writer provides no between-writer evidence. Tail CI conditions on all observed target glyphs: inspect valid_replicates.'
    result['warning'] = 'Transcription only. Unsupported target glyphs excluded from macro and explicitly counted; evaluator independence requires external provenance.'
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--references', type=Path, required=True)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--targets', default='ỄẮỠ')
    p.add_argument('--evaluator-id', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--bootstrap', type=int, default=1000)
    args = p.parse_args()
    result = evaluate(read_records(args.references), read_records(args.predictions), args.targets, args.bootstrap)
    result['evaluator_id'] = args.evaluator_id
    write_json(args.output, result)


if __name__ == '__main__': main()
