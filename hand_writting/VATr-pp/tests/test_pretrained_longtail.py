import random
import unittest
from data.vietnamese import CorpusSampler
from util.vietnamese_longtail import ComponentCorpusSampler, interaction_groups


class PretrainedSamplingTests(unittest.TestCase):
    def test_zero_mix_preserves_legacy_rng_and_transcripts(self):
        labels = ['E', 'ê', 'Ễ', 'Ễ', 'NGUYỄN', 'mã', 'mả']
        old = CorpusSampler(labels); new = ComponentCorpusSampler(labels, component_mix=0)
        self.assertEqual(old.weights, new.weights)
        random.seed(742); before = old.sample(100)
        random.seed(742); self.assertEqual(before, new.sample(100))

    def test_case_and_joint_marks_are_distinct(self):
        upper = interaction_groups('Ễ'); lower = interaction_groups('ễ')
        self.assertIn('pair:circumflex|tilde|upper', upper)
        self.assertNotIn('pair:circumflex|tilde|upper', lower)
        self.assertIn('pair:circumflex|tilde|upper', interaction_groups('Ỗ'))

    def test_no_invented_text_and_analytic_exposure(self):
        labels = ['E', 'Ê', 'Ễ', 'NGUYỄN', 'mã']
        sampler = ComponentCorpusSampler(labels, rare_fraction=0)
        self.assertEqual(sampler.exposure('Ễ')['Ễ'], 2/5)
        self.assertEqual(sampler.exposure('Ỡ')['Ỡ'], 0)
        self.assertTrue(set(sampler.sample(100)) <= set(labels))
        self.assertEqual(sampler.words, CorpusSampler(labels).words)
        self.assertTrue(all(weight > 0 for weight in sampler.weights))


if __name__ == '__main__': unittest.main()
