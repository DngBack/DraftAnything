import numpy as np
import pytest
from scipy import ndimage

from hwline.dependence import (center_lines, lag_stat, perm_test, permute_within_lines,
                               writer_bootstrap)
from hwline.descriptors import slant


def _lines(rng, L=400, n=10, phi=0.0, p_miss=0.2, line_sd=2.0):
    R = np.zeros((L, n))
    e = rng.normal(size=(L, n))
    R[:, 0] = e[:, 0]
    for t in range(1, n):
        R[:, t] = phi * R[:, t - 1] + e[:, t]
    R += rng.normal(scale=line_sd, size=(L, 1))       # line constant must not matter
    R[rng.random(R.shape) < p_miss] = np.nan
    return center_lines(R)


def test_permutation_keeps_line_multiset_and_mask():
    rng = np.random.default_rng(0)
    R = _lines(rng, L=20)
    P = permute_within_lines(R, rng)
    assert np.array_equal(np.isfinite(R), np.isfinite(P))
    for a, b in zip(R, P):
        assert np.allclose(np.sort(a[np.isfinite(a)]), np.sort(b[np.isfinite(b)]))


def test_permutation_respects_bins():
    rng = np.random.default_rng(1)
    R = _lines(rng, L=30)
    bins = rng.integers(0, 3, R.shape).astype(float)
    P = permute_within_lines(R, rng, bins)
    for k in range(3):
        for a, b, bb in zip(R, P, bins):
            m = np.isfinite(a) & (bb == k)
            assert np.allclose(np.sort(a[m]), np.sort(b[m]))


def test_exact_null_matches_monte_carlo_and_iid_not_rejected():
    rng = np.random.default_rng(2)
    R = _lines(rng, phi=0.0)
    r = perm_test(R, [1], 300, 0)[1]
    assert abs(r["exact_null"] - r["null_mean"]) < 3 * r["null_sd"] / np.sqrt(300) + 1e-3
    assert r["exact_null"] < 0          # centring bias is negative, handled by the null
    assert r["p"] > 0.01


def test_ar_detected_and_ci_excludes_zero():
    rng = np.random.default_rng(3)
    R = _lines(rng, phi=0.5)
    r = perm_test(R, [1], 200, 0)[1]
    assert r["excess"] > 0.2 and r["p"] < 0.01
    lo, hi = writer_bootstrap(R, np.arange(R.shape[0]) // 4, 1, 300, 0)
    assert lo > 0.1 and hi > lo


@pytest.mark.parametrize("angle", [-20, 0, 15, 30])
def test_slant_recovers_synthetic_strokes(angle):
    img = np.zeros((60, 160), bool)
    t = np.tan(np.deg2rad(angle))
    for x0 in range(30, 130, 18):
        for y in range(10, 50):
            x = int(round(x0 + (30 - y) * t))
            img[y, x:x + 3] = True
    est = slant(img)
    assert est is not None and abs(est - angle) <= 3


@pytest.mark.parametrize("angle", [-20, 0, 15, 30])
def test_slant_gradient_recovers_synthetic_strokes(angle):
    from hwline.descriptors import slant_gradient
    img = np.zeros((60, 160), bool)
    t = np.tan(np.deg2rad(angle))
    for x0 in range(30, 130, 18):
        for y in range(10, 50):
            x = int(round(x0 + (30 - y) * t))
            img[y, x:x + 3] = True
    gray = np.where(ndimage.gaussian_filter(img.astype(float), 0.7) > 0.3, 30, 240).astype(np.uint8)
    est = slant_gradient(gray, gray < 128)
    assert est is not None and abs(est - angle) <= 4


def test_detrend_vectorised_matches_lstsq():
    from hwline.dependence import detrend_lines
    rng = np.random.default_rng(5)
    R = _lines(rng, L=50)
    X = np.cumsum(rng.uniform(1, 3, R.shape), 1)
    D = detrend_lines(R, X)
    for i in range(R.shape[0]):
        v = np.isfinite(R[i])
        if v.sum() < 3:
            continue
        A = np.c_[np.ones(v.sum()), X[i, v]]
        c, *_ = np.linalg.lstsq(A, R[i, v], rcond=None)
        assert np.allclose(D[i, v], R[i, v] - A @ c)


def test_detrend_null_calibrated_for_trend_plus_iid():
    """Per-line linear trend + iid noise: after detrending there is no local dependence,
    so the detrend-aware null must not reject while the naive exact null is biased."""
    from hwline.dependence import detrend_lines
    rng = np.random.default_rng(6)
    L, n = 600, 10
    X = np.cumsum(rng.uniform(1, 3, (L, n)), 1)
    R = rng.normal(size=(L, 1)) * 0.3 * X + rng.normal(size=(L, n))
    R[rng.random(R.shape) < 0.2] = np.nan
    post = lambda M: center_lines(detrend_lines(M, X), min_valid=4)
    Rd = post(R)
    r = perm_test(Rd, [1], 200, 0, post=post)[1]
    assert r["p"] > 0.01 and abs(r["excess"]) < 0.03
    assert r["exact_null"] - r["obs"] > 0.05          # naive null would report a negative effect
