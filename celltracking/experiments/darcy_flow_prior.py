"""Growth-pressure (Darcy) flow prior for sparse ancestry.

A dense growing monolayer behaves like an active porous medium: local volume
growth g produces pressure p with  -lap(p) = g  inside the colony, p = 0 on
free boundaries (colony edge and the field-of-view edge, through which cells
leave), and velocity v = -grad p. For a disc this is exactly the radial
expansion v = g r / 2; for a colony cropped by a rectangular field of view it
is a mostly uniaxial flow toward the nearest open side. One scalar (g, from a
fixed doubling time) is the only parameter; nothing is fitted to lineage.

Each anchor is advected for H minutes (RK2 on fields re-solved on the anchor
and target domains, linearly blended in time); targets go to the nearest
advected anchor.

  python experiments/darcy_flow_prior.py --datasets OIAM recA rpsM FAST LFCT T98G
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parents[1]))

DOUBLING_MIN = 68.0  # fixed prior; OIAM-typical, not fitted per window


def colony_mask(pos, shape, cell, spacing):
    """Rasterise the colony domain on a grid of `cell` pixels."""
    ny, nx = int(np.ceil(shape[0] / cell)) + 1, int(np.ceil(shape[1] / cell)) + 1
    occ = np.zeros((ny, nx), bool)
    iy = np.clip((pos[:, 0] / cell).astype(int), 0, ny - 1)
    ix = np.clip((pos[:, 1] / cell).astype(int), 0, nx - 1)
    occ[iy, ix] = True
    r = max(1, int(round(1.5 * spacing / cell)))
    occ = ndimage.binary_dilation(occ, iterations=r)
    occ = ndimage.binary_fill_holes(occ)
    if r > 1:
        occ = ndimage.binary_erosion(occ, iterations=r - 1)
    return occ


def solve_pressure(mask, g, cell):
    """-lap p = g on mask, p = 0 outside mask and outside the grid (open)."""
    ny, nx = mask.shape
    idx = -np.ones(mask.shape, int)
    idx[mask] = np.arange(mask.sum())
    n = int(mask.sum())
    if n == 0:
        return np.zeros(mask.shape)
    rows, cols, vals = [], [], []
    yy, xx = np.nonzero(mask)
    me = idx[yy, xx]
    rows.append(me); cols.append(me); vals.append(np.full(n, 4.0))
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        y2, x2 = yy + dy, xx + dx
        inside = (y2 >= 0) & (y2 < ny) & (x2 >= 0) & (x2 < nx)
        nb = np.full(n, -1)
        nb[inside] = idx[y2[inside], x2[inside]]
        ok = nb >= 0
        rows.append(me[ok]); cols.append(nb[ok]); vals.append(np.full(ok.sum(), -1.0))
    a = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(n, n)).tocsr()
    p = np.zeros(mask.shape)
    p[mask] = spsolve(a, np.full(n, g * cell * cell))
    return p


def velocity_field(pos, shape, g, cell, spacing):
    mask = colony_mask(pos, shape, cell, spacing)
    p = solve_pressure(mask, g, cell)
    vy, vx = np.gradient(-p, cell)
    ys = np.arange(mask.shape[0]) * cell
    xs = np.arange(mask.shape[1]) * cell
    fy = RegularGridInterpolator((ys, xs), vy, bounds_error=False, fill_value=None)
    fx = RegularGridInterpolator((ys, xs), vx, bounds_error=False, fill_value=None)
    return lambda q: np.column_stack([fy(q), fx(q)])


def fully_visible(pos, shape, spacing, tol=0.02):
    margin = 2 * spacing
    near = ((pos[:, 0] < margin) | (pos[:, 1] < margin) |
            (pos[:, 0] > shape[0] - margin) | (pos[:, 1] > shape[1] - margin))
    return near.mean() <= tol


def advect(src, dst, horizon, shape, doubling=DOUBLING_MIN, steps=12, auto=True):
    """Move anchors along the blended Darcy field for `horizon` minutes.

    auto: if both colonies are fully inside the field of view, the growth rate
    is the observed count ratio and the result is re-centred on the target
    centroid (drift); otherwise use the doubling-time prior and no shift."""
    spacing = max(float(np.median(cKDTree(src).query(src, k=2)[0][:, 1])), 1.0) if len(src) > 1 else 10.0
    visible = auto and fully_visible(src, shape, spacing) and fully_visible(dst, shape, spacing)
    if visible:
        g = np.log(max(len(dst) / len(src), 1.0)) / horizon
    else:
        g = np.log(2) / doubling
    cell = max(spacing / 2, 2.0)
    va = velocity_field(src, shape, g, cell, spacing)
    vb = velocity_field(dst, shape, g, cell, spacing)
    x = src.copy()
    dt = horizon / steps
    for k in range(steps):
        s0, s1 = k / steps, (k + 0.5) / steps
        v0 = (1 - s0) * va(x) + s0 * vb(x)
        xm = x + 0.5 * dt * v0
        v1 = (1 - s1) * va(xm) + s1 * vb(xm)
        x = x + dt * v1
    if visible:
        x = x - x.mean(0) + dst.mean(0)
    return x


def darcy_assign(e, shape, doubling=DOUBLING_MIN, auto=True):
    src = e["anchor"][["center_y", "center_x"]].to_numpy(float)
    dst = e["target"][["center_y", "center_x"]].to_numpy(float)
    if len(src) < 2:
        return np.zeros(len(dst), int)
    moved = advect(src, dst, e["window"]["horizon_frames"], shape, doubling, auto=auto)
    return cKDTree(moved).query(dst)[1]


def main():
    from experiments.raft_external_eval import Runner, run  # noqa: F401
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["OIAM"])
    ap.add_argument("--doubling", type=float, nargs="+", default=[DOUBLING_MIN])
    ap.add_argument("--out", default="darcy")
    a = ap.parse_args()
    import experiments.raft_external_eval as ree
    orig = ree.Runner.window

    def window(self, dataset, seq, wid, minutes, e, valid, truth, shape):
        if not len(truth):
            return
        e["window"]["horizon_frames"] = minutes
        n0 = len(self.rows)
        orig(self, dataset, seq, wid, minutes, e, valid, truth, shape)
        base = self.rows[n0]
        from experiments.fast_microcolony_external_test import family_f1
        for d in a.doubling:
            for auto in (False, True):
                g = darcy_assign(e, shape, d, auto)[valid]
                self.rows.append({**base, "method": f"darcy{'_auto' if auto else ''}_{int(d)}",
                                  "accuracy": float(np.mean(g == truth)),
                                  "set_f1": family_f1(truth, g)})
    ree.Runner.window = window
    ree.run(argparse.Namespace(tags=[], decoders=[], datasets=a.datasets, out=a.out))


if __name__ == "__main__":
    main()
