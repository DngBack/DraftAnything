"""Annotation-free lineage from dense unlabeled time-lapses by chaining one-frame links.

Between consecutive frames cells barely move, so ancestry over one frame is a
near-trivial capacity-2 assignment (binary fission: a cell has at most two
children). Composing these links gives pseudo-ancestry for any horizon H, with no
track IDs or lineage annotation. GT track IDs are used here only to score it.

  python experiments/chained_pseudo_lineage.py --movies M0 M1 M2
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.lineage_glue import Movie  # noqa: E402

BIG = 1e6


def link(fa, fb, div_cost=2.0, max_px=60.0, k=8):
    """Parent index in frame a for every cell of frame b (-1 = no parent).
    Each parent offers three slots: stay (at its centre) or divide, the two daughters
    sitting at centre +- L/4 along its axis (L = sqrt(12) * major s.d.), each daughter
    slot costing `div_cost` px extra. Solved exactly by linear assignment."""
    pa, pb = fa["pos"], fb["pos"]
    if len(pa) == 0 or len(pb) == 0:
        return -np.ones(len(pb), int)
    if "axis" in fa:
        off = fa["axis"] * (fa["major"] * np.sqrt(12) / 4)[:, None]
    else:  # positions only: both daughter slots at the parent centre
        off = np.zeros_like(pa)
    n_a = len(pa)
    k = min(k, n_a)
    cost = np.full((len(pb), 3 * n_a + len(pb)), BIG)
    rows = np.repeat(np.arange(len(pb)), k)
    for slot, centre, extra in ((0, pa, 0.0), (1, pa + off, div_cost), (2, pa - off, div_cost)):
        d, j = cKDTree(centre).query(pb, k=k)
        cost[rows, slot * n_a + j.reshape(len(pb), k).ravel()] = d.reshape(len(pb), k).ravel() + extra
    cost[np.arange(len(pb)), 3 * n_a + np.arange(len(pb))] = max_px      # enters / no parent
    r, c = linear_sum_assignment(cost)
    par = -np.ones(len(pb), int)
    par[r] = np.where(c < 3 * n_a, c % n_a, -1)
    return par


def link_scaled(pa, pb, div_rel=0.55, max_rel=3.5):
    """Positions-only link with costs in units of the median nearest-neighbour spacing
    (the OIAM defaults, 10 px and 60 px, are 0.55 and 3.5 spacings)."""
    q = pa if len(pa) >= 2 else pb
    s = np.median(cKDTree(q).query(q, k=2)[0][:, 1]) if len(q) >= 2 else 20.0
    return link({"pos": pa}, {"pos": pb}, div_cost=div_rel * s, max_px=max_rel * s)


def chain_positions(frames):
    """frames: {t: (N, 2) positions}. Returns {t: parent index in the previous frame}."""
    ks = sorted(frames)
    links = {ks[0]: -np.ones(len(frames[ks[0]]), int)}
    for a, b in zip(ks[:-1], ks[1:]):
        links[b] = link_scaled(frames[a], frames[b])
    return links


def compose(links, a, b, n_b):
    """Pseudo-ancestor index in frame a for each of the n_b cells of frame b."""
    idx = np.arange(n_b)
    for t in sorted(k for k in links if a < k <= b)[::-1]:
        idx = np.where(idx >= 0, links[t][np.maximum(idx, 0)], -1)
    return idx


class PseudoLineage:
    """Wraps a Movie: same frames, but ancestry comes from chained links."""

    def __init__(self, movie, **kw):
        self.m = movie
        ks = sorted(movie.frames)
        self.links = {ks[0]: -np.ones(len(movie.frames[ks[0]]["pos"]), int)}
        for a, b in zip(ks[:-1], ks[1:]):
            self.links[b] = link(movie.frames[a], movie.frames[b], **kw)

    def ancestors(self, a, b):
        idx = np.arange(len(self.m.frames[b]["pos"]))
        for t in range(b, a, -1):
            ok = idx >= 0
            idx = np.where(ok, self.links[t][np.maximum(idx, 0)], -1)
        return idx


class PseudoMovie:
    """Duck-types lineage_glue.Movie for the training sampler with chained pseudo-lineage
    (links cached under outputs/results/pseudo_links)."""

    def __init__(self, alias, div_cost=10.0):
        import pickle
        from experiments.family_set import RESULTS
        m = Movie(alias)
        cache = RESULTS / "pseudo_links" / f"{alias}_dc{div_cost:g}.pkl"
        if cache.exists():
            links = pickle.loads(cache.read_bytes())
        else:
            links = PseudoLineage(m, div_cost=div_cost).links
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(pickle.dumps(links))
        self.pl = PseudoLineage.__new__(PseudoLineage)
        self.pl.m, self.pl.links = m, links
        self.frames, self.image_shape, self.valid_a = m.frames, m.image_shape, m.valid_a

    def window(self, a, b):
        return self.frames[a], self.frames[b], self.pl.ancestors(a, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--movies", nargs="+", default=["M0", "M1", "M2"])
    ap.add_argument("--div-cost", type=float, default=2.0)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    for alias in a.movies:
        m = Movie(alias)
        pl = PseudoLineage(m, div_cost=a.div_cost)
        ks = sorted(m.frames)
        for h in (1, 8, 33, 66, 132, 240):
            accs, cov, w = [], [], []
            for s in rng.choice([k for k in ks if k + h in m.frames and len(m.frames[k]["pos"]) >= 2], 40):
                _, _, label = m.window(s, s + h)
                p = pl.ancestors(s, s + h)
                ok = label >= 0
                accs.append(np.mean(p[ok] == label[ok]))
                cov.append(np.mean(p[ok] >= 0))
                w.append(ok.sum())
            w = np.array(w)
            print(f"{alias} H={h:3d} acc={np.average(accs, weights=w):.3f} (unweighted {np.mean(accs):.3f}) "
                  f"has-parent={np.average(cov, weights=w):.3f}", flush=True)


if __name__ == "__main__":
    main()
