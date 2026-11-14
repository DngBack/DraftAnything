"""Scratch paired controls. These are NOT reproductions of VATr or SOTA models."""
import pickle
import unicodedata

import numpy as np
import torch
from torch import nn

from research.vietcompose.renderer import LayeredInkComposer


class SupportEncoder(nn.Module):
    def __init__(self, dimension=128):
        super().__init__()
        self.network = nn.Sequential(nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.GELU(),
                                     nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
                                     nn.Conv2d(64, dimension, 3, stride=2, padding=1), nn.GELU(),
                                     nn.AdaptiveAvgPool2d(1))

    def forward(self, support):
        b, n, c, h, w = support.shape
        return self.network(support.reshape(b*n, c, h, w)).reshape(b, n, -1).mean(1)


def load_glyphs(path):
    # Only load the trusted local project Unifont artifact.
    with open(path, 'rb') as handle:
        rows = pickle.load(handle)
    result = {chr(row['idx'][0]): np.asarray(row['mat'], dtype=np.float32) for row in rows}
    if any(array.shape != (16, 16) or not np.isfinite(array).all() for array in result.values()):
        raise ValueError('Expected finite 16x16 canonical glyphs')
    return result


class GlyphRenderer(nn.Module):
    def __init__(self, glyphs, hidden=128, max_length=64):
        super().__init__()
        self.characters = ''.join(sorted(glyphs))
        self.lookup = {c: i+1 for i, c in enumerate(self.characters)}
        matrices = np.stack([np.zeros((16, 16), dtype=np.float32)]+[glyphs[c] for c in self.characters])
        self.register_buffer('glyphs', torch.from_numpy(matrices).flatten(1))
        self.max_length = max_length
        self.project = nn.Linear(256, hidden)
        self.position = nn.Embedding(max_length, hidden)
        layer = nn.TransformerEncoderLayer(hidden, 4, hidden*2, dropout=0., batch_first=True, activation='gelu')
        self.context = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(hidden, hidden*2), nn.GELU(), nn.Linear(hidden*2, 32*16), nn.Tanh())
        self.refine = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 1, 3, padding=1))

    def forward(self, texts, style):
        texts = [unicodedata.normalize('NFC', t) for t in texts]
        if not texts or any(not t or len(t) > self.max_length for t in texts):
            raise ValueError('Empty or overlong text')
        length = max(map(len, texts))
        ids = torch.zeros(len(texts), length, dtype=torch.long, device=style.device)
        for i, text in enumerate(texts):
            try:
                ids[i, :len(text)] = torch.tensor([self.lookup[c] for c in text], device=style.device)
            except KeyError as exc:
                raise ValueError('Canonical glyph absent') from exc
        valid = ids.ne(0)
        value = self.project(self.glyphs[ids])+self.position(torch.arange(length, device=style.device))[None]+style[:, None]
        value = self.context(value, src_key_padding_mask=~valid)
        image = self.head(value).reshape(len(texts), length, 1, 32, 16).permute(0, 2, 3, 1, 4).reshape(len(texts), 1, 32, length*16)
        image = (image+.05*self.refine(image).tanh()).clamp(-1, 1)
        mask = valid[:, :, None].expand(-1, -1, 16).reshape(len(texts), 1, 1, -1)
        return {'image': torch.where(mask, image, torch.ones_like(image))}


class PairedGenerator(nn.Module):
    def __init__(self, mode, hidden=128, max_length=20, glyphs=None):
        super().__init__()
        if hidden % 4 or mode not in ('glyph', 'factorized'):
            raise ValueError('Unknown model or hidden dimension not divisible by four')
        self.support = SupportEncoder(hidden)
        self.renderer = (LayeredInkComposer(hidden, hidden, max_length) if mode == 'factorized'
                         else GlyphRenderer(glyphs, hidden, max_length))

    def forward(self, texts, support):
        return self.renderer(texts, self.support(support))
