"""Cross-font glyph sampling for stage-1 training and evaluation."""
from collections import defaultdict

import numpy as np
import torch

from .holdout import held_chars


def allowed_chars(chars, variant):
    """Characters the tokenizer may train on. 'strict' drops the VN held-out set."""
    if variant == 'full':
        return list(chars)
    if variant == 'strict':
        held = held_chars('vn')
        out = [c for c in chars if c not in held]
        assert len(out) == len(chars) - len(held)
        return out
    raise ValueError(variant)


class GlyphSampler:
    """Samples (input in font A, target in font B, style ref = another char in font B).

    Fonts are drawn family-uniformly so that families with many weights do not
    dominate. With probability `same_font`, B = A.
    """

    def __init__(self, glyphs, chars, fonts, split, variant, device='cpu', same_font=0.25, seed=0):
        self.font_ids = [f['index'] for f in fonts if f['split'] == split]
        fam = defaultdict(list)
        for f in fonts:
            if f['split'] == split:
                fam[f['family']].append(f['index'])
        self.families = [np.array(v) for _, v in sorted(fam.items())]
        self.chars = allowed_chars(chars, variant)
        self.char_ids = np.array([chars.index(c) for c in self.chars])
        self.glyphs = torch.as_tensor(glyphs, device=device)  # uint8, all fonts/chars
        self.same_font = same_font
        self.rng = np.random.default_rng(seed)
        self.device = device

    def _fonts(self, n):
        fam = self.rng.integers(len(self.families), size=n)
        return np.array([self.families[k][self.rng.integers(len(self.families[k]))] for k in fam])

    def sample(self, n):
        a = self._fonts(n)
        b = np.where(self.rng.random(n) < self.same_font, a, self._fonts(n))
        ci = self.rng.integers(len(self.char_ids), size=n)
        # style char: any other allowed char
        di = (ci + self.rng.integers(1, len(self.char_ids), size=n)) % len(self.char_ids)
        c, d = self.char_ids[ci], self.char_ids[di]
        assert (c != d).all()
        g = self.glyphs
        to = lambda t: torch.as_tensor(t, device=self.device)
        x = g[to(a), to(c)].float().div(255)[:, None]
        y = g[to(b), to(c)].float().div(255)
        s = g[to(b), to(d)].float().div(255)[:, None]
        return x, y, s, {'font_a': a, 'font_b': b, 'char': c, 'style_char': d}
