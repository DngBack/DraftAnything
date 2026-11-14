"""Paired writer-cluster differences for two transcription systems."""
import argparse
import json
from pathlib import Path
import numpy as np
from research.longtail.evaluate import alignment, evaluate
from research.longtail.prepare import read_records, write_json


def paired_compare(references, first, second, targets, repeats=2000, seed=742):
    # Reuse strict ID/NFC/type validation; the one-draw CIs are not retained.
    a = evaluate(references, first, targets, repeats=1)
    b = evaluate(references, second, targets, repeats=1)
    chars = sorted(a['per_target']); k = len(chars)
    names = sorted({r['writer'] for r in references}); name_index = {w: i for i, w in enumerate(names)}
    values = []
    import unicodedata
    for predictions in (first, second):
        mapping = {r['sample_id']: unicodedata.normalize('NFC', r['prediction']) for r in predictions}
        matrix = np.zeros((len(names), 4+3*k), dtype=np.float64)
        for r in references:
            ref = unicodedata.normalize('NFC', r['reference']); pred = mapping[r['sample_id']]
            row = matrix[name_index[r['writer']]]; pairs = alignment(ref, pred)
            row[:4] += [sum(x != y for x, y in pairs), len(ref), ref == pred, 1]
            for i, char in enumerate(chars):
                row[4+i] += ref.count(char); row[4+k+i] += pred.count(char)
                row[4+2*k+i] += sum(x == char and y == char for x, y in pairs)
        values.append(matrix)
    if repeats < 1: raise ValueError('Positive bootstrap count required')
    weights = np.random.default_rng(seed).multinomial(len(names), np.ones(len(names))/len(names), size=repeats)
    samples = [weights @ matrix for matrix in values]
    supported = values[0][:, 4:4+k].sum(0) > 0
    def metrics(sample):
        ref, pred, correct = sample[:, 4:4+k], sample[:, 4+k:4+2*k], sample[:, 4+2*k:]
        good = (ref[:, supported] > 0).all(1) if supported.any() else np.zeros(repeats, dtype=bool)
        f1 = (2*correct[:, supported]/np.maximum(ref[:, supported]+pred[:, supported], 1)).mean(1) if supported.any() else np.full(repeats, np.nan)
        return {'cer': sample[:, 0]/np.maximum(sample[:, 1], 1),
                'word_exact': sample[:, 2]/np.maximum(sample[:, 3], 1), 'tail_macro_f1': f1}, good
    am, good_a = metrics(samples[0]); bm, good_b = metrics(samples[1])
    intervals = {}
    for key in am:
        valid = good_a & good_b if key == 'tail_macro_f1' else np.ones(repeats, dtype=bool)
        delta = (bm[key]-am[key])[valid]
        if len(delta): intervals[key] = {'difference_second_minus_first': b[key]-a[key],
                                         'low_95': float(np.quantile(delta, .025)), 'high_95': float(np.quantile(delta, .975)),
                                         'valid_replicates': len(delta)}
    return {'first': {key: a[key] for key in ('cer', 'word_exact', 'tail_macro_f1')},
            'second': {key: b[key] for key in ('cer', 'word_exact', 'tail_macro_f1')},
            'paired_writer_cluster_difference': intervals, 'writers': len(names), 'samples': len(references),
            'note': 'Second minus first. Negative CER is better; positive F1/exact is better. Single-seed exploratory percentile intervals; tail CI conditional on reference glyph coverage.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--references', type=Path, required=True); p.add_argument('--first', type=Path, required=True)
    p.add_argument('--second', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--targets', default='ễỡẵ'); args = p.parse_args()
    write_json(args.output, paired_compare(read_records(args.references), read_records(args.first), read_records(args.second), args.targets))


if __name__ == '__main__': main()
