"""Opt-in component/case-aware transcript sampling for pretrained VATr++.

Keeps empirical text draws, source proportions and the rare-draw probability.
Only the rare-draw weighting changes; this is not new visual supervision.
"""
from collections import Counter
import itertools
from functools import lru_cache
from data.vietnamese import CorpusSampler
from research.longtail.protocol import groups


@lru_cache(maxsize=65536)
def interaction_groups(text):
    return frozenset(groups(text, 'interaction'))


class ComponentCorpusSampler(CorpusSampler):
    def __init__(self, labels, max_length=20, rare_fraction=.3, component_mix=.5):
        if not 0 <= component_mix <= 1:
            raise ValueError('component_mix must be in [0,1]')
        super().__init__(labels, max_length, rare_fraction)
        counts = Counter(key for text in self.words for key in interaction_groups(text))
        self.group_counts = counts
        self.component_mix = component_mix
        self.legacy_weights = list(self.weights)
        component_weights = [min(20., sum((len(self.words)/counts[key])**.5 for key in interaction_groups(text))/len(interaction_groups(text)))
                             for text in self.words]
        self.weights = [(1-component_mix)*old+component_mix*new for old, new in zip(self.legacy_weights, component_weights)]
        self.cumulative_weights = list(itertools.accumulate(self.weights))

    def exposure(self, characters='ỄỠẴẮÊễỡẵ'):
        total = sum(self.weights)
        return {char: (1-self.rare_fraction)*sum(char in text for text in self.words)/len(self.words)
                +self.rare_fraction*sum(weight for text, weight in zip(self.words, self.weights) if char in text)/total
                for char in characters}
