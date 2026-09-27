"""Parameter-free baselines on oracle locations (doc §15 day 3).

  nearest_anchor      b -> nearest cell at a (endpoints only)
  chained_nearest     frame-to-frame nearest neighbour, composed back to a
  chained_lap         frame-to-frame assignment, each parent takes <= 2 children
  chained_lap_c4/_c8  same with <= 4 / <= 8 children (several generations per link)
  chained_lap_soft    no hard cap: capacity grows with n(dst)/n(src), only the per-child
                      penalty limits family size
  *_rc                same, after compensating colony expansion between the two frames

A LAP link that is infeasible (more children than capacity allows) falls back to nearest;
_chain counts those links in win["link_stats"] so the caller can report them.

Each takes a loaded window (cellanc.loader) and returns
  ancestors: {target_local_id at b: anchor_local_id at a}
  siblings:  set of (id, id) pairs at b predicted to be direct sisters
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


def _nearest(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """For each row of dst, the id (col 0) of the nearest row of src by (y, x)."""
    _, idx = cKDTree(src[:, 1:]).query(dst[:, 1:])
    return src[idx, 0].astype(int)


def expansion_compensate(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Map dst points into src's frame: match centroid and RMS radius of the point clouds.

    A colony that grows in place expands roughly radially; this undoes the first-order
    part of that motion without using identities.
    """
    cs, cd = src[:, 1:].mean(0), dst[:, 1:].mean(0)
    rs = np.sqrt(((src[:, 1:] - cs) ** 2).sum(1).mean())
    rd = np.sqrt(((dst[:, 1:] - cd) ** 2).sum(1).mean())
    out = dst.astype(float).copy()
    out[:, 1:] = (dst[:, 1:] - cd) * (rs / rd if rd > 0 else 1.0) + cs
    return out


def _lap(src: np.ndarray, dst: np.ndarray, cap: int | None = 2) -> tuple[np.ndarray, bool]:
    """(parent id for each row of dst, fell_back); each src row takes at most `cap` children.

    Squared-distance cost; every child after a parent's first pays one division cost equal
    to the squared median nearest-neighbour spacing at src (each extra descendant needs one
    more division), so a parent only grows its family when that is cheaper than sending a
    child elsewhere. cap=2 is the v0 baseline. cap=None
    (soft) sizes the capacity so the problem is always feasible. When dst > cap * src the
    problem has no solution and the link falls back to nearest (fell_back=True).
    """
    if len(src) < 2:
        return _nearest(src, dst), False
    if cap is None:
        cap = max(2, 2 * -(-len(dst) // len(src)))  # ponytail: 2x mean family size, raise if slots run out
    if len(dst) > cap * len(src):
        return _nearest(src, dst), True
    d2 = ((dst[:, None, 1:] - src[None, :, 1:]) ** 2).sum(-1)
    spacing = np.median(cKDTree(src[:, 1:]).query(src[:, 1:], k=2)[0][:, 1]) ** 2
    rows, cols = linear_sum_assignment(np.concatenate([d2] + [d2 + spacing] * (cap - 1), axis=1))
    parent = np.empty(len(dst), dtype=int)
    parent[rows] = src[cols % len(src), 0].astype(int)
    return parent, False


def _link(src, dst, method: str, rc: bool, cap) -> tuple[np.ndarray, bool]:
    if rc:
        dst = expansion_compensate(src, dst)
    return _lap(src, dst, cap) if method == "lap" else (_nearest(src, dst), False)


_CACHE: dict = {}  # (sequence, prev, cur, method, rc, cap) -> (parent ids, fell_back)


def _chain(win: dict, method: str, rc: bool, endpoints: bool = False, cap=2) -> dict[int, int]:
    det, w = win["detections"], win["window"]
    frames = [w["anchor_frame"], w["target_frame"]] if endpoints else w["observed_frames"]
    lineage = {int(i): int(i) for i in det[frames[0]][:, 0]}  # id at current frame -> anchor id
    n_fb = 0
    for prev, cur in zip(frames, frames[1:]):
        key = (w["sequence_id"], prev, cur, method, rc, cap)
        if key not in _CACHE:
            if len(_CACHE) > 200_000:
                _CACHE.clear()
            _CACHE[key] = _link(det[prev], det[cur], method, rc, cap)
        parent, fb = _CACHE[key]
        n_fb += fb
        lineage = {int(c): lineage[int(p)] for c, p in zip(det[cur][:, 0], parent)}
    win["link_stats"] = (n_fb, len(frames) - 1)  # read by the caller right after this call
    return lineage


def nearest_anchor(win: dict) -> dict[int, int]:
    """Endpoints only: every cell at b takes the nearest cell at a. Ignores intermediate frames."""
    return _chain(win, "nearest", rc=False, endpoints=True)


def chained_nearest(win: dict) -> dict[int, int]:
    """Link each observed frame to the previous one by nearest neighbour, compose back to a.

    Many-to-one links allow divisions. Uses every observed frame, so it should improve with
    denser schedules if the task needs intermediate evidence.
    """
    return _chain(win, "nearest", rc=False)


def nearest_anchor_rc(win: dict) -> dict[int, int]:
    return _chain(win, "nearest", rc=True, endpoints=True)


def chained_nearest_rc(win: dict) -> dict[int, int]:
    return _chain(win, "nearest", rc=True)


def chained_lap(win: dict) -> dict[int, int]:
    return _chain(win, "lap", rc=False)


def chained_lap_rc(win: dict) -> dict[int, int]:
    return _chain(win, "lap", rc=True)


def _lap_variant(cap, rc):
    return lambda win: _chain(win, "lap", rc=rc, cap=cap)


def mutual_nearest_siblings(win: dict) -> set[tuple[int, int]]:
    """Mutual nearest neighbours at b as predicted direct sisters."""
    d = win["detections"][win["window"]["target_frame"]]
    if len(d) < 2:
        return set()
    _, idx = cKDTree(d[:, 1:]).query(d[:, 1:], k=2)
    nn = idx[:, 1]
    ids = d[:, 0].astype(int)
    return {tuple(sorted((ids[i], ids[j]))) for i, j in enumerate(nn) if nn[j] == i}


BASELINES = {"nearest_anchor": nearest_anchor, "chained_nearest": chained_nearest,
             "nearest_anchor_rc": nearest_anchor_rc, "chained_nearest_rc": chained_nearest_rc,
             "chained_lap": chained_lap, "chained_lap_rc": chained_lap_rc,
             **{f"chained_lap_{tag}{'_rc' if rc else ''}": _lap_variant(cap, rc)
                for tag, cap in [("c4", 4), ("c8", 8), ("soft", None)] for rc in (False, True)}}
