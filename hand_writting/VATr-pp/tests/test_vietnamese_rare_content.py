import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from data.vietnamese_rare_content import RareHatContent, rare_reconstruction_loss


class RareContentTests(unittest.TestCase):
    def test_split_by_transcript_no_pseudo_writer_targets(self):
        class FakeData:
            labels = [f'Ễ{i//2}' for i in range(80)]
            def image(self, index):
                a = np.full((32, 40), 180, dtype='uint8')
                a[8:24, 10:25] = 30
                return Image.fromarray(a)
        with patch('data.vietnamese_rare_content.PackedH5', return_value=FakeData()):
            content = RareHatContent({'indices': {'hat_ocr': list(range(80))}})
        self.assertTrue(content.train_indices and content.eval_indices)
        self.assertFalse(set(content.data.labels[i] for i in content.train_indices) & set(content.data.labels[i] for i in content.eval_indices))
        batch = content.batch(2, 15, 'cpu')
        self.assertNotIn('writer', batch)
        self.assertEqual(tuple(batch['style'].shape), (2, 15, 32, 192))
        self.assertTrue(torch.isfinite(batch['target']).all())
        self.assertLessEqual(float(batch['target'].max()), 1)
        self.assertGreaterEqual(float(batch['target'].min()), -1)

    def test_reconstruction_masks_padding_and_has_gradient(self):
        fake = torch.zeros(2, 1, 32, 48, requires_grad=True)
        target = torch.ones_like(fake)
        widths = torch.tensor([16, 48])
        loss = rare_reconstruction_loss(fake, target, widths)
        self.assertAlmostEqual(float(loss.detach()), 1.)
        loss.backward()
        self.assertEqual(float(fake.grad[0, :, :, 16:].abs().sum()), 0.)
        self.assertGreater(float(fake.grad[1].abs().sum()), 0.)
        with self.assertRaisesRegex(ValueError, 'shape mismatch'):
            rare_reconstruction_loss(fake[..., :16], target, widths)
