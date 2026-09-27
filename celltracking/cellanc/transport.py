"""Mass-conserving transport between observed frames (experiments E1, E2).

Each cell carries mass = its mask area. Between two kept frames the colony's area grows by
g = sum(area_dst) / sum(area_src), measured from the frames themselves (no labels), so source i
is expected to supply area_i * g of descendants. Semi-relaxed entropic OT: every target's mass
must come from some source (hard marginal); a source may deviate from its expected supply at
KL price tau (soft, per-cell capacity, no child-count ceiling). tau=None is balanced OT.

Chaining over a schedule: `soft` multiplies the per-link P(parent | child) matrices and decides
only at b; hard commits to the argmax parent at every link (like the chained baselines).
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import logsumexp

from .baselines import expansion_compensate


def log_plan(src, dst, m_src, m_dst, eps, tau, axis=None, aniso=1.0, iters=1000, tol=1e-4):
    """Log transport plan (n_src x n_dst) from rows [id, y, x] and masses. Cost is squared
    distance after expansion compensation, in units of the squared median NN spacing at src.
    With `axis` (n_src x 2 unit major axes), the along-axis part is divided by `aniso`: rod
    cells push their descendants out along their long axis."""
    dst = expansion_compensate(src, dst)
    spacing2 = np.median(cKDTree(src[:, 1:]).query(src[:, 1:], k=2)[0][:, 1]) ** 2
    d = dst[None, :, 1:] - src[:, None, 1:]
    c = (d ** 2).sum(-1)
    if axis is not None and aniso != 1.0:
        par = ((d * axis[:, None, :]).sum(-1)) ** 2
        c = c - par + par / aniso
    lK = -c / (spacing2 * eps)
    la = np.log(m_src * (m_dst.sum() / m_src.sum()))
    lb = np.log(m_dst)
    f = 1.0 if tau is None else tau / (tau + eps)
    lu = np.zeros(len(src))
    for _ in range(iters):
        lv = lb - logsumexp(lu[:, None] + lK, axis=0)
        lu_new = f * (la - logsumexp(lK + lv[None, :], axis=1))
        if np.abs(lu_new - lu).max() < tol:
            lu = lu_new
            break
        lu = lu_new
    lv = lb - logsumexp(lu[:, None] + lK, axis=0)  # target marginal exact
    return lu[:, None] + lK + lv[None, :]


def ot_ancestry(win: dict, mass: dict, eps: float, tau, soft: bool, axis: dict | None = None,
                aniso: float = 1.0) -> dict[int, int]:
    """{target id at b: anchor id at a}. mass[t], axis[t] align with win['detections'][t] rows."""
    det, fr = win["detections"], win["window"]["observed_frames"]
    A = np.eye(len(det[fr[0]]))  # P(anchor | cell at current frame), anchors x cells
    for prev, cur in zip(fr, fr[1:]):
        if len(det[prev]) < 2:
            cond = np.ones((len(det[prev]), len(det[cur])))
        else:
            lp = log_plan(det[prev], det[cur], mass[prev], mass[cur], eps, tau,
                          None if axis is None else axis[prev], aniso)
            cond = np.exp(lp - logsumexp(lp, axis=0, keepdims=True))  # P(parent | child)
        if not soft:
            cond = (cond == cond.max(0, keepdims=True)).astype(float)
            cond /= cond.sum(0, keepdims=True)
        A = A @ cond
    anchors = det[fr[0]][:, 0].astype(int)
    return dict(zip(det[fr[-1]][:, 0].astype(int), anchors[A.argmax(0)]))
