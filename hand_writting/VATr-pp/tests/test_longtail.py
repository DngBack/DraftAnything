"""Leakage, sampling and train/eval integration tests without real handwriting."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from PIL import Image
import torch
from research.longtail.protocol import audit, make_protocol, ExposurePolicy, sampling_weights, digest, validate_records
from research.longtail.data import PairedEpisodes, read_image
from research.longtail.evaluate import evaluate
from research.longtail.train import paired_loss, run
from research.longtail.models import PairedGenerator


def record(key, text, writer='train:w', split='train', source='real'):
    return {'sample_id': key, 'text': text, 'writer': writer, 'split': split,
            'source': source, 'pixel_sha256': hashlib.sha256(key.encode()).hexdigest()}


def records():
    return [record('a', 'E'), record('b', 'ê'), record('c', 'Ễ'),
            record('hat', 'Ễ', None, source='HAT'),
            record('v1', 'Ễ', 'val:w', 'val'), record('v2', 'E', 'val:w', 'val')]


class PolicyTests(unittest.TestCase):
    def test_zero_holdout_removed_from_every_training_consumer(self):
        rows = records(); p = ExposurePolicy(rows, make_protocol(rows, 'Ễ'))
        for purpose in ('image', 'ocr', 'corpus', 'support', 'memory'):
            self.assertTrue(all('Ễ' not in r['text'] for r in p.training(purpose)))
        with self.assertRaises(ValueError): p.assert_training_texts(['NGUYỄN'])
        p.assert_training_texts(['E'])
        self.assertEqual([r['sample_id'] for r in p.supports_for(rows[4], True)], ['v2'])
        with self.assertRaises(ValueError): p.supports_for(rows[2])

    def test_fewshot_real_only_and_nested_budgets(self):
        rows = [record(str(i), t) for i, t in enumerate(['Ễ', 'ỄỠ', 'Ỡ', 'ỄỠ', 'Ỡ', 'Ễ'])]
        rows.append(record('hat', 'Ễ', None, source='HAT')); prior = set()
        for k in range(4):
            p = make_protocol(rows, 'ỄỠ', k)
            self.assertTrue(prior <= set(p['acquired']))
            self.assertNotIn('hat', p['train_allowed'])
            self.assertTrue(all(v <= k for v in p['acquired_images_per_glyph'].values()))
            prior = set(p['acquired'])
        self.assertEqual(ExposurePolicy(rows, make_protocol(rows, 'ỄỠ', 1)).training('support'), [])

    def test_reject_writer_pixel_id_leakage_and_hat_writer(self):
        for mutation in ('writer', 'pixel', 'id', 'hat'):
            rows = records()
            if mutation == 'writer': rows[4]['writer'] = 'train:w'
            if mutation == 'pixel': rows[4]['pixel_sha256'] = rows[0]['pixel_sha256']
            if mutation == 'id': rows[4]['sample_id'] = rows[0]['sample_id']
            if mutation == 'hat': rows[3]['writer'] = 'fake:295'
            with self.assertRaises(ValueError, msg=mutation): validate_records(rows)

    def test_protocol_and_record_tampering_rejected(self):
        rows = records(); p = make_protocol(rows, 'Ễ'); p['train_allowed'].append('hat')
        with self.assertRaises(ValueError): ExposurePolicy(rows, p)
        p = make_protocol(rows, 'Ễ'); rows[0]['text'] = 'A'
        with self.assertRaises(ValueError): ExposurePolicy(rows, p)

    def test_audit_counts_images_and_real_writers(self):
        rows = records(); rows[0]['text'] = 'EEE'; a = audit(rows)['train/real']
        self.assertEqual(a['character_occurrences']['E'], 3)
        self.assertEqual(a['group_images']['glyph:E'], 1)
        self.assertEqual(a['group_writers']['glyph:E'], 1)
        self.assertEqual(audit(rows)['train/HAT']['genuine_writers'], 0)

    def test_sampling_no_nan_and_rare_has_more_mass(self):
        rows = [record(str(i), 'E' if i < 9 else 'Ễ') for i in range(10)]
        for mode in ('natural', 'glyph', 'interaction'):
            weights = sampling_weights(rows, mode)
            self.assertAlmostEqual(sum(weights), 1)
            self.assertTrue(all(w > 0 and np.isfinite(w) for w in weights))
            if mode != 'natural': self.assertGreater(weights[-1], weights[0])
        with self.assertRaises(ValueError): sampling_weights(rows, beta=1)


class EvaluationTests(unittest.TestCase):
    def test_insertions_penalized_and_missing_glyphs_reported(self):
        refs = [{'sample_id': 'a', 'reference': 'Ễ', 'writer': 'w1'}, {'sample_id': 'b', 'reference': 'Ễ', 'writer': 'w2'}]
        preds = [{'sample_id': 'a', 'prediction': 'ỄỄ'}, {'sample_id': 'b', 'prediction': ''}]
        r = evaluate(refs, preds, 'ỄẮ', 30)
        self.assertEqual(r['cer'], 1); self.assertAlmostEqual(r['tail_macro_f1'], .5)
        self.assertEqual(r['per_target']['Ắ']['reference_occurrences'], 0)
        self.assertEqual(r['tail_reference_coverage'], 1)
        with self.assertRaises(ValueError): evaluate(refs, preds[:1], 'Ễ')

    def test_nfc_and_repeatable_writer_cluster_bootstrap(self):
        refs = [{'sample_id': 'a', 'reference': 'Ễ', 'writer': 'w1'}]
        pred = [{'sample_id': 'a', 'prediction': 'E\u0302\u0303'}]
        self.assertEqual(evaluate(refs, pred, 'Ễ', 10), evaluate(refs, pred, 'Ễ', 10))
        self.assertEqual(evaluate(refs, pred, 'Ễ', 10)['cer'], 0)


class TrainingTests(unittest.TestCase):
    def test_padding_does_not_affect_loss(self):
        fake = torch.zeros(2, 1, 32, 32, requires_grad=True)
        real = torch.zeros_like(fake); real[0, :, :, 16:] = 1
        loss = paired_loss(fake, real, [16, 32]); self.assertEqual(float(loss.detach()), 0)
        loss.backward(); self.assertEqual(float(fake.grad[0, :, :, 16:].abs().sum()), 0)

    def test_both_models_have_style_gradient_and_white_padding(self):
        torch.set_num_threads(2)
        glyphs = {c: np.eye(16, dtype=np.float32) for c in 'EỄ'}
        for mode in ('glyph', 'factorized'):
            net = PairedGenerator(mode, hidden=16, glyphs=glyphs)
            out = net(['E', 'ỄE'], torch.randn(2, 2, 1, 32, 192))['image']
            torch.testing.assert_close(out[0, :, :, 16:], torch.ones(1, 32, 16))
            out.square().mean().backward(); grad = next(net.support.parameters()).grad
            self.assertTrue(torch.isfinite(grad).all()); self.assertGreater(float(grad.abs().sum()), 0)

    def test_cpu_end_to_end_both_arms_and_pixel_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); rows = []
            for i, (split, text) in enumerate([('train', 'E'), ('train', 'ê'), ('train', 'Ễ'), ('val', 'E'), ('val', 'ê'), ('val', 'Ễ')]):
                array = np.full((32, 20+i), 255, np.uint8); array[5:25, 4:7+i] = 10+i
                path = root/f'{i}.png'; Image.fromarray(array).save(path)
                row = record(str(i), text, split+':writer', split); row['image_path'] = str(path)
                row['pixel_sha256'] = hashlib.sha256(str(array.shape).encode()+array.tobytes()).hexdigest(); rows.append(row)
            records_path = root/'records.jsonl'; records_path.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows))
            protocol = make_protocol(rows, 'Ễ'); protocol_path = root/'protocol.json'; protocol_path.write_text(json.dumps(protocol))
            episodes = PairedEpisodes(ExposurePolicy(rows, protocol), split='val', support_count=2)
            self.assertNotIn('5', episodes[2]['support_ids']); self.assertNotIn('2', protocol['train_allowed'])
            import pickle
            glyph_path = root/'glyphs.pickle'
            with glyph_path.open('wb') as f: pickle.dump([{'idx': [ord(c)], 'mat': np.eye(16)} for c in 'EêỄ'], f)
            for mode in ('glyph', 'factorized'):
                output = root/mode
                args = SimpleNamespace(records=records_path, protocol=protocol_path, output=output, mode=mode,
                    sampler='interaction', glyphs=glyph_path, steps=2, eval_every=2, batch_size=2,
                    support_count=2, hidden=16, max_length=20, seed=742, lr=.001, device='cpu')
                run(args)
                self.assertTrue((output/'last.pth').exists())
                self.assertEqual(json.loads((output/'completion.json').read_text())['completed_updates'], 2)
                checkpoint = torch.load(output/'last.pth', weights_only=False)
                self.assertEqual(checkpoint['protocol']['records_sha256'], digest(rows))
            from research.longtail.recognizer import FactorizedCTC
            teacher = root/'teacher.pth'
            torch.save({'architecture': 'factorized-joint-ctc-v1',
                'config': {'alphabet': 'EêỄ', 'hidden': 16, 'protocol_sha256': digest(protocol), 'records_sha256': digest(rows)},
                'model': FactorizedCTC('EêỄ', 16).state_dict()}, teacher)
            args.output = root/'guided'; args.reconstruction = 'balanced-bce'; args.foreground_weight = 4.
            args.ctc_weight = .2; args.ocr_checkpoint = teacher; args.adversarial_weight = .1; args.feature_weight = 1.
            run(args)
            saved = torch.load(args.output/'last.pth', weights_only=False)
            self.assertIn('discriminator', saved)
            self.assertTrue((args.output/'code/research/longtail/train.py').exists())
            args.resume = args.output/'last.pth'; args.output = root/'continued'; args.steps = 4
            run(args)
            self.assertEqual(json.loads((args.output/'completion.json').read_text())['completed_updates'], 4)
            Image.new('L', (20, 32), 0).save(rows[0]['image_path'])
            with self.assertRaises(ValueError): read_image(rows[0])



class ExportAndSuiteTests(unittest.TestCase):
    def test_hdf5_export_preserves_sources_and_pixel_integrity(self):
        import h5py
        from research.longtail.prepare import export_records
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, texts, writers in [('train', ['E', 'ê', 'Ễ', 'A'], [0, 0, 295, 1]),
                                          ('test', ['E', 'Ễ'], [0, 0])]:
                widths = [20+i for i in range(len(texts))]
                arrays = []
                for i, width in enumerate(widths):
                    array = np.full((32, width), 255, dtype=np.uint8)
                    array[3:20, 2:5+i] = i+(50 if name == 'test' else 0)
                    arrays.append(array)
                with h5py.File(root/(name+'.h5'), 'w') as f:
                    f['imgs'] = np.concatenate(arrays, axis=1)
                    f['img_seek_idxs'] = np.cumsum([0]+widths[:-1]); f['img_lens'] = widths
                    f['lbs'] = [ord(c) for text in texts for c in text]
                    lens = list(map(len, texts)); f['lb_lens'] = lens
                    f['lb_seek_idxs'] = np.cumsum([0]+lens[:-1]); f['wids'] = writers
            manifest = {'source_train': str(root/'train.h5'), 'source_test': str(root/'test.h5'),
                        'indices': {'train': [0, 1], 'hat_ocr': [2], 'val': [3], 'test': [0, 1]}}
            rows = export_records(manifest); validate_records(rows)
            hat = next(r for r in rows if r['source'] == 'HAT')
            self.assertIsNone(hat['writer']); self.assertEqual(hat['text'], 'Ễ')
            for row in rows: self.assertEqual(read_image(row).height, 32)

    def test_suite_does_not_launch_with_missing_primary_ground_truth(self):
        import sys
        from unittest.mock import patch
        from research.longtail.suite import main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); rows = records(); rows[4]['text'] = 'ê'
            path = root/'records.jsonl'
            path.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows))
            argv = ['suite', '--records', str(path), '--output', str(root/'suite'),
                    '--holdout', 'Ễ', '--primary', 'Ễ', '--budgets', '0', '--seeds', '742', '--execute']
            with patch.object(sys, 'argv', argv), patch('research.longtail.suite.subprocess.run') as launch:
                with self.assertRaises(SystemExit): main()
                launch.assert_not_called()
            plan = json.loads((root/'suite/plan.json').read_text())
            self.assertFalse(plan['scientific_ready'])
            self.assertEqual(plan['status'], 'not-executed-incomplete-data')

class StructuredRecognizerTests(unittest.TestCase):
    def test_joint_ctc_logprobs_shared_roles_and_live_gradient(self):
        from research.longtail.recognizer import FactorizedCTC, ctc_loss
        net = FactorizedCTC('eEêÊẽẼễỄ', hidden=16)
        # Same tilde class across lower/upper and with/without circumflex.
        self.assertEqual(int(net.map_tone[4]), int(net.map_tone[7]))
        image = torch.randn(2, 1, 32, 32, requires_grad=True)
        output = net(image)
        torch.testing.assert_close(output.exp().sum(-1), torch.ones(8, 2))
        loss = ctc_loss(net, image, ['ê', 'Ẽ'], [32, 28]); loss.backward()
        self.assertTrue(torch.isfinite(image.grad).all())
        self.assertGreater(float(net.heads['tone'].weight.grad.abs().sum()), 0)
        # Semantic none is a factor class, while blank is its own prediction head.
        self.assertEqual(output.shape[-1], 9)
        with self.assertRaises(ValueError): ctc_loss(net, image, ['eeeeee', 'ê'], [8, 32])

    def test_decode_repeats_and_true_width(self):
        from research.longtail.recognizer import decode
        probabilities = torch.full((5, 1, 3), -10.)
        for t, index in enumerate([1, 1, 0, 1, 2]): probabilities[t, 0, index] = 0.
        self.assertEqual(decode(probabilities, [16], 'eE'), ['ee'])


class PairedComparisonTests(unittest.TestCase):
    def test_identity_has_zero_paired_difference(self):
        from research.longtail.compare import paired_compare
        refs = [{'sample_id': 'a', 'reference': 'Ễ', 'writer': 'w1'}, {'sample_id': 'b', 'reference': 'Ễ', 'writer': 'w2'}]
        preds = [{'sample_id': 'a', 'prediction': 'Ễ'}, {'sample_id': 'b', 'prediction': 'E'}]
        result = paired_compare(refs, preds, preds, 'Ễ', repeats=30)
        for metric in result['paired_writer_cluster_difference'].values():
            self.assertEqual(metric['low_95'], 0); self.assertEqual(metric['high_95'], 0)

    def test_inventory_is_unique_and_roundtrips(self):
        from research.longtail.collection import inventory
        from research.vietcompose.orthography import compose_character, decompose_character
        rows = inventory(); self.assertEqual(len(rows), 145)
        self.assertEqual(len({r['prompt_id'] for r in rows}), 145)
        for row in rows[:-1]: self.assertEqual(compose_character(decompose_character(row['text'])), row['text'])


class GlyphSemanticControlTests(unittest.TestCase):
    def test_canonical_glyph_control_has_finite_unseen_label_scores(self):
        from research.longtail.recognizer import GlyphPrototypeCTC, ctc_loss
        glyphs = {c: np.random.default_rng(ord(c)).random((16, 16)).astype('float32') for c in 'EêỄ'}
        net = GlyphPrototypeCTC('EêỄ', glyphs, hidden=16)
        image = torch.randn(2, 1, 32, 32, requires_grad=True)
        logits = net(image)
        self.assertEqual(tuple(logits.shape), (8, 2, 4))
        self.assertTrue(torch.isfinite(logits[:, :, 3]).all())
        loss = ctc_loss(net, image, ['E', 'ê'], [32, 32]); loss.backward()
        self.assertTrue(torch.isfinite(net.glyph_projection[0].weight.grad).all())
        self.assertGreater(float(net.glyph_projection[0].weight.grad.abs().sum()), 0)


if __name__ == '__main__': unittest.main()
