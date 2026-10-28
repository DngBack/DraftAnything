"""Kleister-style F1 on key=value pairs (one predicted value per key, abstaining allowed) and paired bootstrap."""
import numpy as np


def counts(pred, gold):
    """pred, gold: value str or None. Returns (tp, fp, fn)."""
    tp = int(pred is not None and pred == gold)
    return tp, int(pred is not None and not tp), int(gold is not None and not tp)


def f1(rows):
    """rows: iterable of (tp, fp, fn)."""
    tp, fp, fn = np.asarray(list(rows), float).reshape(-1, 3).sum(0)
    return 2 * tp / max(2 * tp + fp + fn, 1)


def bootstrap_diff(a, b, n=2000, seed=0):
    """a, b: [N, 3] per-decision counts of two systems on the same items. CI of F1(a) - F1(b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    fa = lambda x: 2 * x[..., 0] / np.maximum(2 * x[..., 0] + x[..., 1] + x[..., 2], 1)
    d = fa(a[idx].sum(1)) - fa(b[idx].sum(1))
    return float(fa(a.sum(0)) - fa(b.sum(0))), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
