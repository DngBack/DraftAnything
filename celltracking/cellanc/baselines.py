"""Parameter-free baselines on oracle locations (doc §15 day 3).

  nearest_anchor      b -> nearest cell at a (endpoints only)
  chained_nearest     frame-to-frame nearest neighbour, composed back to a
  chained_lap         frame-to-frame assignment, each parent takes <= 2 children
  *_rc                same, after compensating colony expansion between the two frames

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


def _lap(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Parent id for each row of dst; each src row can take at most 2 children.

    Squared-distance cost; the second slot of a parent pays an extra division cost equal to
    the squared median nearest-neighbour spacing at src, so a parent only splits when that
    is cheaper than sending a child elsewhere. Falls back to nearest when dst > 2 * src.
    """
    if len(dst) > 2 * len(src) or len(src) < 2:
        return _nearest(src, dst)
    d2 = ((dst[:, None, 1:] - src[None, :, 1:]) ** 2).sum(-1)
    spacing = np.median(cKDTree(src[:, 1:]).query(src[:, 1:], k=2)[0][:, 1]) ** 2
    rows, cols = linear_sum_assignment(np.concatenate([d2, d2 + spacing], axis=1))
    parent = np.empty(len(dst), dtype=int)
    parent[rows] = src[cols % len(src), 0].astype(int)
    return parent


def _link(src, dst, method: str, rc: bool) -> np.ndarray:
    if rc:
        dst = expansion_compensate(src, dst)
    return _lap(src, dst) if method == "lap" else _nearest(src, dst)


_CACHE: dict = {}  # (sequence, prev, cur, method, rc) -> parent ids; schedules share frame pairs


def _chain(win: dict, method: str, rc: bool, endpoints: bool = False) -> dict[int, int]:
    det, w = win["detections"], win["window"]
    frames = [w["anchor_frame"], w["target_frame"]] if endpoints else w["observed_frames"]
    lineage = {int(i): int(i) for i in det[frames[0]][:, 0]}  # id at current frame -> anchor id
    for prev, cur in zip(frames, frames[1:]):
        key = (w["sequence_id"], prev, cur, method, rc)
        if key not in _CACHE:
            if len(_CACHE) > 200_000:
                _CACHE.clear()
            _CACHE[key] = _link(det[prev], det[cur], method, rc)
        lineage = {int(c): lineage[int(p)] for c, p in zip(det[cur][:, 0], _CACHE[key])}
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
             "chained_lap": chained_lap, "chained_lap_rc": chained_lap_rc}
