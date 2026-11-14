import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from train_vietnamese import override_resume_lrs
from util.vietnamese_benchmark import FixedBenchmark


class ExperimentTests(unittest.TestCase):
    def test_resume_override_preserves_adam_moments(self):
        p = torch.nn.Parameter(torch.tensor([1.]))
        q = torch.nn.Parameter(torch.tensor([2.]))
        opt = torch.optim.Adam([{'params': [p], 'lr': 1e-3}, {'params': [q], 'lr': 1e-3}])
        (p.square()+q.square()).sum().backward()
        opt.step()
        restored = torch.optim.Adam([{'params': [p]}, {'params': [q]}])
        restored.load_state_dict(opt.state_dict())
        moment = restored.state[p]['exp_avg'].clone()
        override_resume_lrs({'G': restored}, SimpleNamespace(encoder_lr=2.5e-6, lr=1e-5, d_lr=1e-5, writer_lr=2e-5))
        self.assertEqual([g['lr'] for g in restored.param_groups], [2.5e-6, 1e-5])
        torch.testing.assert_close(moment, restored.state[p]['exp_avg'])
        self.assertEqual(restored.state[p]['step'].item(), 1)

    def test_benchmark_restores_rng_and_mode_on_failure(self):
        bench = object.__new__(FixedBenchmark)
        generator = torch.nn.Linear(1, 1).train()
        py, npstate, ts = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        def fail(*args):
            generator.eval()
            random.random(); np.random.rand(); torch.rand(3)
            raise RuntimeError('evaluation failed')
        with patch.object(bench, '_render', side_effect=fail), patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'evaluation failed'):
                bench.render(generator, None, None, 0)
        self.assertTrue(generator.training)
        self.assertEqual(random.getstate(), py)
        np.testing.assert_array_equal(np.random.get_state()[1], npstate[1])
        self.assertEqual(np.random.get_state()[2:], npstate[2:])
        torch.testing.assert_close(torch.get_rng_state(), ts)
