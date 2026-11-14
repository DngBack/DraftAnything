import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from data.vietnamese import PackedH5, collate, CorpusSampler
from models.OCR_network import CRNN, strLabelConverter
from train_vietnamese import decoded, edit_distance, ocr_loss


class VietnameseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_nfc_last_character_and_unknown(self):
        converter = strLabelConverter('aệ')
        tokens, lengths, _ = converter.encode('e\u0323\u0302')
        self.assertEqual(tokens.tolist(), [[2]])
        self.assertEqual(lengths.tolist(), [1])
        with self.assertRaisesRegex(ValueError, 'outside'):
            converter.encode(['á'])

    def test_ctc_blank_logprob_true_width_and_gradient(self):
        args = SimpleNamespace(alphabet='aệ', special_alphabet='', resolution=16)
        network = CRNN(args)
        converter = strLabelConverter(args.alphabet)
        images = torch.randn(2, 1, 32, 32, requires_grad=True)
        predictions = network(images)
        self.assertEqual(predictions.shape, (9, 2, 3))
        torch.testing.assert_close(predictions.logsumexp(-1), torch.zeros(9, 2), atol=1e-5, rtol=0)
        loss = ocr_loss(network, converter, images, ['ệ', 'aa'], torch.tensor([12, 32]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(images.grad).all())
        self.assertGreater(images.grad.abs().sum().item(), 0)
        with self.assertRaisesRegex(ValueError, 'geometry'):
            ocr_loss(network, converter, images, ['aaaa', 'a'], torch.tensor([8, 32]))

    def test_decoder_ctc_repeat_and_true_length(self):
        logits = torch.full((6, 1, 3), -10.)
        for t, token in enumerate([2, 2, 0, 2, 1, 1]):
            logits[t, 0, token] = 10
        self.assertEqual(decoded(logits, [12], 'aệ'), ['ệệ'])
        self.assertEqual(edit_distance('Việt', 'Viet'), 1)

    def test_packed_support_and_writer_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'train.h5'
            widths = np.array([32, 240, 32, 32, 32])
            labels = ['a', 'ệ', 'a', 'ệ', 'a']
            with h5py.File(path, 'w') as f:
                pixels = np.full((32, sum(widths)), 255, dtype='uint8')
                pixels[16, :] = 0
                f['imgs'] = pixels
                f['img_lens'] = widths
                f['img_seek_idxs'] = np.r_[0, np.cumsum(widths)[:-1]]
                f['wids'] = [0, 0, 106, 106, 295]
                f['lbs'] = [ord(t) for t in labels]
                f['lb_lens'] = np.ones(5, dtype=int)
                f['lb_seek_idxs'] = np.arange(5)
            manifest = {'source_train': str(path), 'indices': {'train': [0, 1, 2, 3], 'hat_ocr': [4]}}
            data = PackedH5(manifest, style=True, num_examples=3)
            item = data[0]
            self.assertEqual(item['support_indices'], [1, 1, 1])
            self.assertEqual(tuple(item['style'].shape), (3, 32, 192))
            self.assertEqual(data[2]['writer'], 1)
            self.assertEqual(len(data.writer_map), 2)
            self.assertEqual(collate([data[0], data[1]])['widths'].tolist(), [16, 16])
            with self.assertRaises(ValueError):
                PackedH5(manifest, style=True, include_hat=True)
            self.assertEqual(len(PackedH5(manifest, include_hat=True)), 5)
            self.assertLessEqual(set(CorpusSampler(labels).sample(20)), set(labels))


if __name__ == '__main__':
    unittest.main()
