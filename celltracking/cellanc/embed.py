"""Learned metric-embedding cost for ancestry chaining (method v0 -- go/no-go for a learned method).

Every baseline so far uses a hand-set cost: squared pixel distance (baselines.py), optionally
after expansion compensation, or a hand-set entropic-OT cost (transport.py). PyUAT's ILP tracker
(experiments/pyuat_run.py) shows shape + motion/division likelihoods beat plain position by a
wide margin at moderate sparsity (s16/s32: ~0.6-0.8 vs ~0.45-0.55 for the best OT/LAP heuristic,
see outputs/tables/go_no_go.csv) -- but it needs several observed frames to fit those likelihoods
and collapses at 2 frames (endpoints), where it drops below nearest-neighbour.

This module replaces the hand-set cost with a *learned* one, meant to keep PyUAT's advantage at
moderate sparsity while not collapsing at 2 frames:

  phi(features) -> R^d       shared per-cell embedding: position (normalised per-frame like
                              baselines.expansion_compensate), local density, shape axis +
                              eccentricity from the oracle mask (cellanc/../experiments/shape.py)
  g(phi(cell), Delta_t)      residual motion predictor: where that cell's embedding should be
                              Delta_t minutes later
  cost(i@t, j@t+Delta)       = || g(phi(i), Delta_t) - phi(j) ||^2

Feeding Delta_t explicitly -- and training on a curriculum of gaps from one real frame up to the
largest test horizon (experiments/train_embed.py) -- is the fix for PyUAT's endpoints collapse:
the model has to learn what a family looks like at every gap, not just adjacent frames.

Trained with a listwise softmax cross-entropy per target cell (true ancestor vs every other
candidate in the same source frame): the natural loss for something decoded by nearest neighbour,
LAP or entropic OT, all of which just read this pairwise cost matrix off the shelf.

Oracle features only: shape comes from the oracle mask, looked up via the evaluator-side
gt_mapping, exactly like experiments/decide.py's OT experiments. This is method development on
top of oracle locations (v1 report limits), not a deployable detector-based pipeline.
"""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

N_FEATURES = 11  # y_norm, x_norm, axis_y, axis_x, log_major_sd, log_minor_sd, log_local_spacing,
# vy_norm, vx_norm, has_velocity, log_nearest_dist -- velocity is (this cell's displacement since
# the previous *observed* frame) / that gap, using the same track id; 0 + flag=0 when the track
# wasn't present then (window's first frame, or a division happened since). This is the one signal
# PyUAT's motion likelihood has that a single-frame embedding does not (diag: PyUAT's 8-hop chain
# at s16 beats this model's single-hop match, see experiments/train_embed.py's build_example).
# log_nearest_dist (distance to the single closest neighbour, separate from log_local_spacing's
# k=5 average) flags a just-divided pair: two daughters sit much closer to each other than the
# general local density would predict -- PyUAT's division likelihood uses the same kind of cue.


class Embed(nn.Module):
    def __init__(self, n_in: int = N_FEATURES, d: int = 16, hidden: int = 64):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, d))
        self.pred = nn.Sequential(nn.Linear(d + 1, hidden), nn.ReLU(), nn.Linear(hidden, d))
        self.log_tau = nn.Parameter(torch.zeros(()))

    def embed(self, feats: torch.Tensor) -> torch.Tensor:
        return self.phi(feats)

    def predict(self, e_src: torch.Tensor, dt_min: float) -> torch.Tensor:
        """Predicted embedding of each src cell dt_min minutes later (residual motion model)."""
        dt = torch.full((e_src.shape[0], 1), float(dt_min), device=e_src.device)
        return e_src + self.pred(torch.cat([e_src, torch.log1p(dt)], -1))

    def cost(self, e_src: torch.Tensor, e_tgt: torch.Tensor, dt_min: float) -> torch.Tensor:
        """(n_src, n_tgt) squared distance in embedding space after motion prediction."""
        pred = self.predict(e_src, dt_min)
        d2 = ((pred[:, None, :] - e_tgt[None, :, :]) ** 2).sum(-1)
        return d2 * torch.exp(-self.log_tau)


def soft_transition(cost: np.ndarray, temp: float = 1.0) -> np.ndarray:
    """(n_src, n_tgt) cost -> (n_src, n_tgt) column-stochastic transition, softmax over src for
    each tgt -- the same normalisation axis as the training cross-entropy (cost.T rows = targets,
    columns = candidate ancestors), so belief propagated with this matches what the model was
    actually optimised for, no retraining needed.
    """
    z = -cost / temp
    z = z - z.max(axis=0, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=0, keepdims=True)


def lap_decode(cost: np.ndarray, cap: int | None = 2) -> np.ndarray:
    """cost is (n_src, n_tgt); returns the src row index chosen for each target column, each
    src row taking <= cap columns.

    Same capacity trick as baselines._lap (replicate src rows `cap` times, extra copies pay one
    penalty), but on a learned cost matrix instead of raw squared pixel distance -- the penalty
    is set from the cost scale itself (median best-match cost) since there is no pixel spacing
    to anchor it to. Falls back to nearest (argmin) if infeasible (more targets than cap * src).

    cap=None (soft, like baselines.chained_lap_soft): size the capacity to the family ratio of
    THIS link (2x mean children per src row), so one run adapts to a short step (~2 children,
    e.g. s16) and a long jump (~4+ children, e.g. `ends` at 2 cycles) without a hand-picked cap.
    """
    n_src, n_tgt = cost.shape
    if cap is None:
        cap = max(2, 2 * -(-n_tgt // n_src))
    if n_src < 2 or n_tgt > cap * n_src:
        return cost.argmin(0)
    penalty = np.median(cost.min(0))
    rep = np.concatenate([cost] + [cost + penalty] * (cap - 1), axis=0)  # (n_src * cap, n_tgt)
    rows, cols = linear_sum_assignment(rep)
    parent = np.empty(n_tgt, dtype=int)
    parent[cols] = rows % n_src
    return parent
