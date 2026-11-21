"""[A]+[B] Content pathway: frozen part tokens -> context encoder -> per-char widths.

`ContentTable` stores, for every character and several reference fonts, the
stage-1 tokenizer output (FSQ code, where-vector, validity of each slot). The
generator never sees glyph pixels or Unicode parts, only these tokens.

`ContentEncoder` mode
  * 'parts': one token per valid slot = code embedding + where + char position.
    Codes are shared across characters, so what is learned for the circumflex
    of ê is reused for Ê or ễ.
  * 'nfd': factorised tokens from the Unicode decomposition: [letter (lower-
    cased base) + case], [modifier], [tone], each with a role embedding. Added
    after the stage-1 NO-GO of the learned slots (runs/probes/verdict.txt,
    18/11/2026): ễ = e + circumflex + tilde, all seen in training, and an
    uppercase letter reuses the marks learned on lowercase.
  * 'char': one learned embedding per character (internal baseline, stage 3).
    A held-out character keeps its random initial embedding, which is the
    honest behaviour of an atomic-character model.
"""
import hashlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .charset import ALPHABET, MARK_NAMES, MARKS, MODIFIERS, parts
from .layers import Attention, FeedForward, sinusoid

INK_SHARE_MIN = 0.05   # same "slot used" criterion as scripts/train_tokenizer.py


@torch.no_grad()
def build_content_table(tokenizer, glyphs, chars, font_indices, style_ref='n', device='cuda'):
    """Run the frozen tokenizer on every char of every reference font.

    Returns a dict saved with torch.save; validity = the slot owns >= 5% of ink.
    """
    tokenizer.eval().to(device)
    sref = chars.index(style_ref)
    codes, where, valid = [], [], []
    for f in font_indices:
        x = torch.as_tensor(glyphs[f], device=device).float().div(255)[:, None]
        st = torch.as_tensor(glyphs[f, [sref] * len(chars)], device=device).float().div(255)[:, None]
        tok = tokenizer.tokenize(x)
        _, alpha, _ = tokenizer.decode(tok, st)
        ink = x
        share = (alpha.float() * ink).sum((-1, -2)) / ink.sum((-1, -2)).clamp(min=1)
        v = share >= INK_SHARE_MIN
        v[v.sum(1) == 0] = True          # never leave a glyph without tokens
        codes.append(tok['codes'].cpu())
        where.append(tok['where'].float().cpu())
        valid.append(v.cpu())
    return {'chars': ''.join(chars), 'fonts': list(map(int, font_indices)),
            'codes': torch.stack(codes), 'where': torch.stack(where), 'valid': torch.stack(valid),
            'n_codes': int(tokenizer.fsq.n_codes)}


def table_digest(table):
    h = hashlib.sha256()
    for k in ('codes', 'where', 'valid'):
        h.update(table[k].numpy().tobytes())
    h.update(table['chars'].encode())
    return h.hexdigest()


class ContextBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn, self.ff = Attention(dim, heads), FeedForward(dim)

    def forward(self, x, mask):
        x = x + self.attn(self.n1(x), key_mask=mask)
        return x + self.ff(self.n2(x))


class ContentEncoder(nn.Module):
    """texts -> (tokens, token_char, token_mask, char_feats, char_mask)."""

    def __init__(self, table=None, mode='parts', dim=384, depth=4, heads=6, max_chars=48):
        super().__init__()
        assert mode in ('parts', 'char', 'nfd')
        self.mode, self.dim, self.max_chars = mode, dim, max_chars
        if mode == 'parts':
            assert table is not None
            self.register_buffer('codes', table['codes'], persistent=False)    # (F, C, S)
            self.register_buffer('where', table['where'], persistent=False)    # (F, C, S, 5)
            self.register_buffer('valid', table['valid'], persistent=False)    # (F, C, S)
            self.table_chars = table['chars']
            self.code_emb = nn.Embedding(table['n_codes'], dim)
            self.where_mlp = nn.Sequential(nn.Linear(5, dim), nn.SiLU(), nn.Linear(dim, dim))
            self.n_slots = table['codes'].shape[-1]
        elif mode == 'nfd':
            self.table_chars = ''.join(c for c in ALPHABET if not c.isspace())
            letters, ids, upper, marks = [], [], [], []
            for c in self.table_chars:
                base, names = parts(c)
                low = base.lower()
                if low not in letters:
                    letters.append(low)
                ids.append(letters.index(low))
                upper.append(int(base != low))
                marks.append([MARK_NAMES.index(n) for n in names] + [-1] * (2 - len(names)))
            self.register_buffer('nfd_letter', torch.tensor(ids), persistent=False)
            self.register_buffer('nfd_upper', torch.tensor(upper), persistent=False)
            self.register_buffer('nfd_marks', torch.tensor(marks), persistent=False)          # (C, 2), -1 = none
            modifier_ids = [MARK_NAMES.index(n) for n in MODIFIERS.values()]
            self.register_buffer('mark_role', torch.tensor(                                   # 1 modifier, 2 tone
                [1 if i in modifier_ids else 2 for i in range(len(MARK_NAMES))]), persistent=False)
            self.letter_emb = nn.Embedding(len(letters), dim)
            self.case_emb = nn.Embedding(2, dim)
            self.mark_emb = nn.Embedding(len(MARK_NAMES), dim)
            self.role_emb = nn.Embedding(3, dim)                                               # letter / modifier / tone
            self.n_slots = 3
        else:
            self.table_chars = ''.join(c for c in ALPHABET if not c.isspace())
            self.char_emb = nn.Embedding(len(self.table_chars), dim)
            self.n_slots = 1
        self.index = {c: i for i, c in enumerate(self.table_chars)}
        self.space = nn.Parameter(torch.randn(dim) * 0.02)
        self.blocks = nn.ModuleList(ContextBlock(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)

    def supports(self, text):
        return all(c == ' ' or c in self.index for c in text)

    def forward(self, texts, font_choice=None):
        """font_choice: (B,) reference-font index into the table, or None -> font 0."""
        b, s = len(texts), self.n_slots
        lc = max(len(t) for t in texts)
        assert lc <= self.max_chars, f'text longer than {self.max_chars}'
        dev = self.space.device
        cid = torch.full((b, lc), -1, dtype=torch.long)
        for i, t in enumerate(texts):
            for j, c in enumerate(t):
                cid[i, j] = -2 if c == ' ' else self.index[c]
        cid = cid.to(dev)
        char_mask = cid != -1
        is_space = cid == -2
        safe = cid.clamp(min=0)
        if self.mode == 'parts':
            fi = torch.zeros(b, dtype=torch.long, device=dev) if font_choice is None else font_choice.to(dev)
            codes = self.codes[fi[:, None], safe]                     # (B, Lc, S)
            where = self.where[fi[:, None], safe]                     # (B, Lc, S, 5)
            slot_ok = self.valid[fi[:, None], safe]                   # (B, Lc, S)
            tok = self.code_emb(codes) + self.where_mlp(where)
        elif self.mode == 'nfd':
            letter = self.letter_emb(self.nfd_letter[safe]) + self.case_emb(self.nfd_upper[safe]) + self.role_emb.weight[0]
            mk = self.nfd_marks[safe]                                                         # (B, Lc, 2)
            mtok = self.mark_emb(mk.clamp(min=0)) + self.role_emb(self.mark_role[mk.clamp(min=0)])
            tok = torch.cat([letter[:, :, None], mtok], 2)                                    # (B, Lc, 3, d)
            slot_ok = torch.cat([torch.ones_like(mk[..., :1], dtype=torch.bool), mk >= 0], 2)
        else:
            tok = self.char_emb(safe)[:, :, None]
            slot_ok = torch.ones(b, lc, 1, dtype=torch.bool, device=dev)
        # Space: a single learned token in slot 0.
        sp = torch.zeros_like(slot_ok)
        sp[..., 0] = True
        slot_ok = torch.where(is_space[..., None], sp, slot_ok) & char_mask[..., None]
        tok = torch.where(is_space[..., None, None], self.space.expand_as(tok), tok)
        pos = sinusoid(torch.arange(lc, device=dev), self.dim)       # char position in the word
        tok = tok + pos[None, :, None]
        x = tok.reshape(b, lc * s, self.dim)
        mask = slot_ok.reshape(b, lc * s)
        token_char = torch.arange(lc, device=dev).repeat_interleave(s)[None].expand(b, -1)
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.norm(x)
        m = mask.reshape(b, lc, s, 1).float()
        char_feats = (x.reshape(b, lc, s, -1) * m).sum(2) / m.sum(2).clamp(min=1)
        return {'tokens': x, 'token_char': token_char, 'token_mask': mask,
                'char_feats': char_feats, 'char_mask': char_mask}


class WidthHead(nn.Module):
    """Per-character width (latent columns), conditioned on the writer vector."""

    def __init__(self, dim, style_dim, init_width=4.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim + style_dim, dim), nn.SiLU(), nn.Linear(dim, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, float(np.log(init_width)))

    def forward(self, char_feats, char_mask, g):
        z = torch.cat([char_feats, g[:, None].expand(-1, char_feats.shape[1], -1)], -1)
        w = self.net(z).squeeze(-1).clamp(-4, 6).exp()
        return w * char_mask


def width_loss(widths, true_total):
    """Squared log error of the predicted total word width."""
    return (torch.log(widths.sum(1).clamp(min=1e-3)) - torch.log(true_total.float())).pow(2).mean()


def fit_widths(widths, total):
    """Rescale per-char widths so they sum to `total` (teacher forcing during training)."""
    return widths * (total.float() / widths.sum(1).clamp(min=1e-3))[:, None]


def char_layout(widths):
    """Centres and widths of each character, in latent columns."""
    right = widths.cumsum(1)
    return right - widths / 2, widths
