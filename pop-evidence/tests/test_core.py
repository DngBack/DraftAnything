import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from popev import metrics, pooling  # noqa: E402
from popev.bags import BagSampler, eval_bags  # noqa: E402
from popev.models import DeepSets, PopEvidence, context_consistency  # noqa: E402
from popev.phoebi import Manifest, View  # noqa: E402

SYNTH = ROOT / "data/synth"


@pytest.mark.parametrize("model", [
    DeepSets(32, 6), *[PopEvidence(32, 6, pool=p, class_specific=c) for p in ("mean", "meanstd", "quantile", "max") for c in (True, False)]])
def test_permutation_invariance_and_set_size(model):
    model.eval(); x = torch.randn(3, 20, 32)
    e1, _ = model(x); e2, _ = model(x[:, torch.randperm(20)])
    assert torch.allclose(e1, e2, atol=1e-5)
    assert model(torch.randn(2, 7, 32))[0].shape == (2, 6)


@pytest.mark.parametrize("name", ["mean", "meanstd", "meanquant", "kme"])
def test_representation_invariance(name):
    rep = pooling.make(name).fit(torch.randn(500, 32), 0)
    x = torch.randn(4, 16, 32)
    assert torch.allclose(rep(x), rep(x[:, torch.randperm(16)]), atol=1e-5)


def test_class_specific_score_reads_only_its_species():
    m = PopEvidence(8, 3, m=2, pool="meanstd", class_specific=True).eval()
    x = torch.randn(1, 5, 8, requires_grad=False)
    e, a = m(x)
    s = torch.zeros(1, 3, 4); s[0, 1] = 1.0
    assert ((s * m.w).sum(-1) + m.b)[0, 0] == m.b[0]   # pooled evidence of species 1 does not enter e_0


def test_consistency_has_no_constant_minimum():
    y = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1.0]]); c = torch.tensor([0, 1, 2, 3])
    const = context_consistency(torch.zeros(4, 2), y, c)
    good = context_consistency(torch.tensor([[5, -5], [5, -5], [-5, 5], [-5, 5.0]]), y, c)
    assert good < const


def test_metrics():
    y = np.array([1, 1, 0, 0]); s = np.array([.9, .8, .1, .2])
    assert metrics.auroc(s, y) == 1.0 and metrics.average_precision(s, y) == 1.0
    assert metrics.auroc(np.ones(4), y) == 0.5
    Y = np.array([[1, 0], [1, 1], [0, 1]]); r = metrics._point(Y * 1.0, Y, np.ones_like(Y))
    assert abs(r["f1_margin"]) < 1e-9 and r["n_pred"] == 2


def test_bags():
    v = View(np.arange(10), np.repeat([[1, 0], [0, 1]], 5, 0), np.array(["a"] * 5 + ["b"] * 5))
    pos, y, c = eval_bags(v, 2, 0)
    assert pos.shape == (4, 2) and all(v.combos[p].tolist().count(cc) == 2 for p, cc in zip(pos, c))
    pos, y, _ = BagSampler(v, 3, 0).sample(8)
    assert all(len(set(v.combos[p])) == 1 and len(set(p)) == 3 for p in pos)


@pytest.mark.skipif(not (SYNTH / "splits_lco.json").exists(), reason="run scripts/make_synthetic.py")
def test_folds_never_touch_test():
    man = Manifest(SYNTH / "splits_lco.json", SYNTH / "lco_dev_folds.json")
    for f in range(len(man.folds)):
        sp = man.fold(f)
        for v in (sp.train, sp.calib, sp.eval):
            assert not set(v.combos) & set(man.heldout)
        assert not set(sp.eval.combos) & set(sp.train.combos)


def test_abmil_invariance_and_evidence_shape():
    from popev.models import ClassAttentionMIL
    m = ClassAttentionMIL(32, 5).eval(); x = torch.randn(3, 20, 32)
    e1, a = m(x); e2, _ = m(x[:, torch.randperm(20)])
    assert torch.allclose(e1, e2, atol=1e-5) and a.shape == (3, 20, 5, 1)


def test_context_prior_learns_cooccurrence_and_removes_it():
    from popev.cla import ContextPrior
    rng = np.random.default_rng(0)
    a = rng.random(400) < 0.5
    b = np.where(a, rng.random(400) < 0.9, rng.random(400) < 0.1)          # b co-occurs with a
    Y = np.c_[a, b, rng.random(400) < 0.3].astype(int)
    pr = ContextPrior(Y, "conditional", C=10.0)
    assert pr.W[1, 0] > 1.5 and abs(pr.W[2, 0]) < 0.5 and pr.W[0, 0] == 0
    m = ContextPrior(Y, "marginal")
    assert np.all(m.W == 0)
    out = pr.logits(torch.tensor(Y[:5], dtype=torch.float32))
    assert out.shape == (5, 3)


def test_tile_loss_label_valid_variant():
    """With use_pure=False and q=1.0 (max), a positive bag is satisfied by a single strong tile: the loss
    asks nothing of the other tiles of a positive bag (only that one tile carries the class)."""
    from popev.models import tile_evidence_loss
    T = 20
    a_one = torch.full((1, T, 1, 1), -10.0); a_one[0, 3, 0, 0] = 10.0       # one evidence tile
    a_all = torch.full((1, T, 1, 1), 10.0)                                 # every tile carries evidence
    y = torch.ones(1, 1)
    l_one = tile_evidence_loss(a_one, y, q=1.0, use_pure=False)
    l_all = tile_evidence_loss(a_all, y, q=1.0, use_pure=False)
    assert l_one < 1e-3 and l_all < 1e-3                                   # both satisfy "at least one tile"
    assert tile_evidence_loss(a_one, y, q=0.9, use_pure=False) > 1.0       # q=0.9 demands ~10% of tiles
    assert tile_evidence_loss(a_one, y, q=1.0, use_pure=True) > 1.0        # 'pure' demands every tile
