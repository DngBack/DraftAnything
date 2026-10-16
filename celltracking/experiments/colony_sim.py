"""Agent-based 2D simulator of growing rod-shaped bacterial monolayers.

Cells are spherocylinders (sphere chains for contacts) with exponential growth,
adder-type division, overdamped soft-contact mechanics and small noise. Every
simulated minute is recorded with track IDs and parents, so a simulated colony
plugs into the same window sampler as a GT-tracked OIAM movie. Parameters are
domain-randomised per colony; no real lineage label is used anywhere.

  python experiments/colony_sim.py --n 64 --out outputs/sim/v1   # generate
  python experiments/colony_sim.py --preview                      # one quick colony
"""

import argparse
import math
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from experiments.lineage_glue import Movie  # noqa: E402


MIDPOINT = {"birth_len": 2.0, "doubling": 75.0, "growth_cv": 0.15, "adder_cv": 0.15, "angle_noise": 0.07,
            "diffusion": 0.01, "rot_diffusion": 0.015, "friction_aniso": 1.0}


def random_params(rng, fixed_physics=False, division_modes=False):
    """fixed_physics: ablation without physics randomisation (FOV/scale still random).
    division_modes (sim v2): also randomise asymmetric division, V-snapping of the
    daughters, inherited cell width and per-frame measurement noise."""
    p = {
        "radius": 0.5,
        "birth_len": rng.uniform(1.0, 3.0),          # segment length at birth (um)
        "doubling": rng.uniform(40, 110),            # minutes
        "growth_cv": rng.uniform(0.05, 0.3),
        "adder_cv": rng.uniform(0.05, 0.25),
        "angle_noise": rng.uniform(0.0, 0.15),       # rad at division
        "diffusion": rng.uniform(0.0, 0.02),         # um per sqrt(min)
        "rot_diffusion": rng.uniform(0.0, 0.03),
        "friction_aniso": rng.uniform(0.5, 2.0),
        "substeps": 8,
        "n_init": int(rng.integers(1, 5)),
        "minutes": 1500,
        "max_cells": int(rng.integers(900, 2200)),
        "px_per_um": rng.uniform(6.0, 14.0),
        "fov_frac": rng.uniform(0.35, 1.2),          # FOV half-size relative to final colony radius
        "fov_offset": rng.uniform(-0.4, 0.4, size=2),
    }
    if division_modes:
        p.update({
            "div_asym": rng.uniform(0.0, 0.08),      # sd of the division-plane position (fraction of length)
            "snap": rng.uniform(0.0, 1.2),           # mean V-snapping angle between daughters (rad)
            "width_cv": rng.uniform(0.0, 0.06),      # inherited width noise per generation
            "obs_noise": rng.uniform(0.0, 0.08),     # per-frame noise on measured lengths and widths
            "obs_angle": rng.uniform(0.0, 0.15),     # per-frame noise on measured orientation (rad)
        })
    if fixed_physics:
        p.update(MIDPOINT)
    return p


class Colony:
    def __init__(self, p, rng):
        self.p, self.rng = p, rng
        n = p["n_init"]
        self.pos = rng.normal(0, 2.0, size=(n, 2))
        self.theta = rng.uniform(0, math.pi, n)
        self.length = p["birth_len"] * rng.uniform(1.0, 1.9, n)
        self.born_len = self.length / rng.uniform(1.0, 1.9, n)
        self.rate = self._rates(n)
        self.target_add = self._adds(n)
        self.track = np.arange(1, n + 1)
        self.next_id = n + 1
        self.start = {int(t): 0 for t in self.track}
        self.parent = {int(t): 0 for t in self.track}
        self.width = np.exp(rng.normal(0, p.get("width_cv", 0.0), n))

    def _rates(self, n):
        base = math.log(2) / self.p["doubling"]
        return base * np.exp(self.rng.normal(0, self.p["growth_cv"], n))

    def _adds(self, n):
        return self.p["birth_len"] * np.exp(self.rng.normal(0, self.p["adder_cv"], n))

    def divide(self, minute):
        idx = np.flatnonzero(self.length - self.born_len >= self.target_add)
        if not len(idx):
            return
        r = self.p["radius"]
        k = len(idx)
        th = self.theta[idx]
        u = np.stack([np.cos(th), np.sin(th)], 1)
        L = self.length[idx]                      # total length incl. caps
        frac = np.clip(0.5 + self.rng.normal(0, self.p.get("div_asym", 0.0), k), 0.3, 0.7)
        len1, len2 = L * frac, L * (1 - frac)
        septum = self.pos[idx] + ((len1 - L / 2))[:, None] * u
        # V-snapping: daughters hinge at the septum and open by alpha (0 = symmetric split)
        alpha = np.abs(self.rng.normal(0, self.p.get("snap", 0.0) * 1.2533, k))
        side = self.rng.choice([-1.0, 1.0], k)
        an = self.p["angle_noise"]
        th1 = th - side * alpha / 2 + self.rng.normal(0, an, k)
        th2 = th + side * alpha / 2 + self.rng.normal(0, an, k)
        u1 = np.stack([np.cos(th1), np.sin(th1)], 1)
        u2 = np.stack([np.cos(th2), np.sin(th2)], 1)
        keep = np.setdiff1d(np.arange(len(self.track)), idx)
        pos = np.concatenate([self.pos[keep], septum - (len1 / 2)[:, None] * u1,
                              septum + (len2 / 2)[:, None] * u2])
        theta = np.concatenate([self.theta[keep], th1, th2])
        length = np.concatenate([self.length[keep], len1, len2])
        wcv = self.p.get("width_cv", 0.0)
        self.width = np.concatenate([self.width[keep], self.width[idx] * np.exp(self.rng.normal(0, wcv, k)),
                                     self.width[idx] * np.exp(self.rng.normal(0, wcv, k))])
        ids = np.arange(self.next_id, self.next_id + 2 * k)
        self.next_id += 2 * k
        for c, par in zip(ids, np.concatenate([self.track[idx], self.track[idx]])):
            self.start[int(c)] = minute
            self.parent[int(c)] = int(par)
        self.born_len = np.concatenate([self.born_len[keep], len1, len2])
        self.pos, self.theta, self.length = pos, theta, length
        self.rate = np.concatenate([self.rate[keep], self._rates(2 * k)])
        self.target_add = np.concatenate([self.target_add[keep], self._adds(2 * k)])
        self.track = np.concatenate([self.track[keep], ids])

    def relax(self, dt):
        r = self.p["radius"]
        n = len(self.track)
        u = np.stack([np.cos(self.theta), np.sin(self.theta)], 1)
        seg = np.maximum(self.length - 2 * r, 0.0)
        nseg = np.maximum(2, np.ceil(seg / r).astype(int) + 1)
        owner = np.repeat(np.arange(n), nseg)
        frac = np.concatenate([np.linspace(-0.5, 0.5, m) for m in nseg])
        s = frac * seg[owner]
        pts = self.pos[owner] + s[:, None] * u[owner]
        pairs = cKDTree(pts).query_pairs(2 * r, output_type="ndarray")
        if len(pairs):
            i, j = pairs.T
            m = owner[i] != owner[j]
            i, j = i[m], j[m]
            d = pts[j] - pts[i]
            dist = np.maximum(np.linalg.norm(d, axis=1), 1e-6)
            f = ((2 * r - dist) / dist)[:, None] * d       # force on j (away from i)
            fx = np.zeros((n, 2))
            np.add.at(fx, owner[j], f)
            np.add.at(fx, owner[i], -f)
            cross = lambda a, b: a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
            tq = np.zeros(n)
            np.add.at(tq, owner[j], s[j] * cross(u[owner[j]], f))
            np.add.at(tq, owner[i], s[i] * cross(u[owner[i]], -f))
            ltot = self.length
            # anisotropic drag: easier along the rod axis
            along = (fx * u).sum(1, keepdims=True) * u
            perp = fx - along
            a = self.p["friction_aniso"]
            v = (a * along + perp) / ltot[:, None]
            self.pos += dt * 0.5 * v
            self.theta += dt * 0.5 * tq / (ltot ** 3 / 12 + 1e-3) * 2

    def step(self, minute):
        p = self.p
        self.length = self.length * np.exp(self.rate)
        self.divide(minute)
        for _ in range(p["substeps"]):
            self.relax(1.0)
        n = len(self.track)
        self.pos += self.rng.normal(0, p["diffusion"], (n, 2))
        self.theta += self.rng.normal(0, p["rot_diffusion"], n)

    def snapshot(self):
        r = self.p["radius"]
        u = np.stack([np.sin(self.theta), np.cos(self.theta)], 1)  # (y, x) like OIAM axis
        return {"track": self.track.copy(), "pos_um": self.pos[:, ::-1].copy(), "axis": u,
                "major_um": self.length / math.sqrt(12), "minor_um": self.width * r / 2}


def simulate(seed, fixed_physics=False, division_modes=False, max_cells=None):
    rng = np.random.default_rng(seed)
    p = random_params(rng, fixed_physics, division_modes)
    if max_cells:  # sim v3: larger colonies, hence crowded windows with large expansion displacements
        p["max_cells"] = int(rng.integers(*max_cells))
    col = Colony(p, rng)
    frames = {}
    for minute in range(p["minutes"]):
        col.step(minute + 1)
        frames[minute + 1] = col.snapshot()
        if len(col.track) >= p["max_cells"]:
            break
    return p, frames, col.start, col.parent


def observe(p, frames, start, parent):
    """Crop a field of view in pixels and express it like an OIAM Movie."""
    last = frames[max(frames)]["pos_um"]
    centre = last.mean(0)
    rad = np.sqrt(np.mean(np.sum((last - centre) ** 2, 1))) * 1.4 + 1
    half = p["fov_frac"] * rad
    c = centre + p["fov_offset"] * rad
    lo = c - half
    px = p["px_per_um"]
    shape = (int(2 * half * px), int(2 * half * px))
    out = {}
    rng = np.random.default_rng(int(p["max_cells"]) * 7919 + int(px * 1000))
    sn, sa = p.get("obs_noise", 0.0), p.get("obs_angle", 0.0)
    for t, f in frames.items():
        pos = (f["pos_um"] - lo) * px
        m = (pos[:, 0] >= 0) & (pos[:, 1] >= 0) & (pos[:, 0] < shape[0]) & (pos[:, 1] < shape[1])
        n = int(m.sum())
        axis, major, minor = f["axis"][m], f["major_um"][m] * px, f["minor_um"][m] * px
        if sn or sa:  # segmentation-like measurement noise, independent per frame
            major = major * np.exp(rng.normal(0, sn, n))
            minor = minor * np.exp(rng.normal(0, sn, n))
            da = rng.normal(0, sa, n)
            c, s = np.cos(da), np.sin(da)
            axis = np.stack([c * axis[:, 0] + s * axis[:, 1], c * axis[:, 1] - s * axis[:, 0]], 1)
        out[t] = {"track": f["track"][m], "pos": pos[m], "axis": axis, "major": major, "minor": minor}
    return out, shape


class SimMovie:
    """Duck-types experiments.lineage_glue.Movie for the training sampler."""

    def __init__(self, path, full_fov=False):
        d = pickle.loads(Path(path).read_bytes())
        if full_fov:  # ablation: whole colony always visible, no FOV crop
            d["p"] = {**d["p"], "fov_frac": 1.3, "fov_offset": np.zeros(2)}
        self.frames, self.image_shape = observe(d["p"], d["frames"], d["start"], d["parent"])
        # tracks re-entering the FOV keep their IDs; ancestry uses the full simulated tree
        self.start, self.parent = d["start"], d["parent"]
        self.valid_a = [t for t, f in self.frames.items() if len(f["track"]) >= 2]

    ancestor = Movie.ancestor
    window = Movie.window


def _job(args):
    seed, out, fixed, modes, mc = args
    p, frames, start, parent = simulate(seed, fixed, modes, mc)
    Path(out, f"colony_{seed:05d}.pkl").write_bytes(pickle.dumps(
        {"p": p, "frames": frames, "start": start, "parent": parent}))
    return seed, len(frames), len(frames[max(frames)]["track"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "outputs/sim/v1"))
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--fixed-physics", action="store_true")
    ap.add_argument("--division-modes", action="store_true", help="sim v2: asymmetric/snapping division, widths")
    ap.add_argument("--max-cells", type=int, nargs=2, default=None, help="sim v3: colony size range")
    a = ap.parse_args()
    if a.preview:
        import time
        t0 = time.time()
        p, frames, start, parent = simulate(a.seed0, a.fixed_physics, a.division_modes, a.max_cells)
        print(p, len(frames), len(frames[max(frames)]["track"]), f"{time.time()-t0:.1f}s")
        Path(a.out).mkdir(parents=True, exist_ok=True)
        Path(a.out, "preview.pkl").write_bytes(pickle.dumps(
            {"p": p, "frames": frames, "start": start, "parent": parent}))
        return
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with Pool(a.workers) as pool:
        for seed, nf, nc in pool.imap_unordered(_job, [(a.seed0 + i, a.out, a.fixed_physics, a.division_modes, a.max_cells) for i in range(a.n)]):
            print(seed, nf, nc, flush=True)


if __name__ == "__main__":
    main()
