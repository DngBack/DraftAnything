import math
import os

import torch
import pickle
import numpy as np


def gauss(x, sigma=1.0):
    return (1.0 / math.sqrt(2.0 * math.pi) * sigma) * math.exp(-x**2 / (2.0 * sigma**2))


class UnifontModule(torch.nn.Module):
    def __init__(self, out_dim, alphabet, device='cuda', input_type='unifont', projection='linear', glyph_path=None, glyph_canvas=None):
        super(UnifontModule, self).__init__()
        self.projection_type = projection
        self.device = device
        self.alphabet = alphabet
        self.glyph_path = glyph_path
        self.glyph_canvas = glyph_canvas
        self.register_buffer('symbols', self.get_symbols('unifont'), persistent=False)
        self.register_buffer('symbols_repr', self.get_symbols(input_type), persistent=False)

        self._table_cache = None
        if projection == 'linear':
            self.linear = torch.nn.Linear(self.symbols_repr.shape[1], out_dim)
        elif projection == 'cnn':
            from models.glyph_encoders import GlyphCNN
            side = math.isqrt(self.symbols_repr.shape[1])
            if side*side != self.symbols_repr.shape[1]:
                raise ValueError('CNN glyph encoder needs square glyphs')
            self.glyph_side = side
            self.encoder = GlyphCNN(out_dim, side)
        else:
            self.linear = torch.nn.Identity()

    def get_symbols(self, input_type):
        with open(self.glyph_path or f"files/{input_type}.pickle", "rb") as f:
            symbols = pickle.load(f)

        all_symbols = {sym['idx'][0]: sym['mat'].astype(np.float32) for sym in symbols}
        symbols = []
        for char in self.alphabet:
            im = all_symbols[ord(char)]
            if self.glyph_canvas is not None:
                size = self.glyph_canvas
                if im.ndim != 2 or im.shape[0] != im.shape[1] or size < im.shape[0] or size % im.shape[0]:
                    raise ValueError('Glyph canvas must be an integer multiple of square native glyph size')
                if not np.isfinite(im).all() or im.min() < 0 or im.max() > 1:
                    raise ValueError('Glyph ablation expects finite ink=1/background=0 in [0,1]')
                factor = size // im.shape[0]
                im = np.repeat(np.repeat(im, factor, axis=0), factor, axis=1)
            im = im.flatten()
            symbols.append(im)

        symbols.insert(0, np.zeros_like(symbols[0]))
        symbols = np.stack(symbols)
        return torch.from_numpy(symbols).float().to(self.device)

    def encode_indices(self, indices):
        """Embeddings for a 1-D tensor of symbol indices (0 is padding)."""
        if self.projection_type == 'cnn':
            glyphs = self.symbols_repr[indices].view(-1, 1, self.glyph_side, self.glyph_side)
            return self.encoder(glyphs)
        return self.linear(self.symbols_repr[indices])

    def embedding_table(self):
        return self.encode_indices(torch.arange(self.symbols_repr.shape[0], device=self.symbols_repr.device))

    def _parameter_versions(self):
        return tuple((p.data_ptr(), p._version) for p in self.parameters())

    def forward(self, QR):
        if self.projection_type != 'cnn':
            return self.linear(self.symbols_repr[QR])
        if not torch.is_grad_enabled():
            # Inference/benchmark: one table per weight version instead of a CNN pass per call.
            versions = self._parameter_versions()
            if self._table_cache is None or self._table_cache[0] != versions:
                self._table_cache = (versions, self.embedding_table())
            return self._table_cache[1][QR]
        # GroupNorm is per-sample, so encoding only unique symbols is exact.
        unique, inverse = QR.unique(return_inverse=True)
        return self.encode_indices(unique)[inverse]


class LearnableModule(torch.nn.Module):
    def __init__(self, out_dim, device='cuda'):
        super(LearnableModule, self).__init__()
        self.device = device
        self.param = torch.nn.Parameter(torch.zeros(1, 1, 256, device=device))
        self.linear = torch.nn.Linear(256, out_dim)

    def forward(self, QR):
        return self.linear(self.param).repeat((QR.shape[0], 1, 1))


