import tempfile
import unittest
from unittest.mock import patch

import torch

from util.vietnamese_visual import feature_matching_loss, paired_pixel_loss, VisualBenchmark


class Features(torch.nn.Module):
    def return_features(self,x):
        return [x,x*2,x*3]


class VisualTests(unittest.TestCase):
    def test_features_detach_target_and_mask_padding(self):
        real=torch.ones(2,1,32,48,requires_grad=True)
        fake=torch.zeros_like(real,requires_grad=True)
        widths=torch.tensor([16,48])
        loss=feature_matching_loss(Features(),fake,real,widths)
        self.assertAlmostEqual(float(loss.detach()),1.)
        loss.backward()
        self.assertIsNone(real.grad)
        self.assertEqual(float(fake.grad[0,:,:,16:].abs().sum()),0.)
        self.assertGreater(float(fake.grad[1].abs().sum()),0.)
        self.assertAlmostEqual(float(feature_matching_loss(Features(),real.detach(),real.detach(),widths)),0.)
        self.assertAlmostEqual(float(paired_pixel_loss(real.detach(),real.detach(),widths)),0.)

    def test_eval_support_excludes_target_and_target_transcript(self):
        class Data:
            pools={35:[0,1,2,3],106:[4,5,6,7]}
            labels=['ab','ab','cd','ef','gh','gh','ij','kl']
        with tempfile.TemporaryDirectory() as directory,patch('util.vietnamese_visual.PackedH5',return_value=Data()):
            b=VisualBenchmark({},15,directory,'cpu',Features())
            self.assertEqual(len(b.records),4)
            for r in b.records:
                self.assertNotIn(r['target'],r['support'])
                self.assertTrue(all(Data.labels[i]!=r['text'] for i in r['support']))
                self.assertTrue(set(r['support']).issubset(Data.pools[r['writer']]))
