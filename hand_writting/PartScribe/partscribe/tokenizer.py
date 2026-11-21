"""Stage-1 part tokenizer: glyph image -> set of (what, where) part slots.

* Encoder: CoordConv CNN, 96x96 -> 24x24x128 feature map.
* Slot attention (Locatello et al., 2020) with learned initial slots, so the
  output is deterministic given the input.
* what: FSQ (Mentzer et al., 2023) on each slot, levels (8, 6, 5) = 240 codes.
* where: centroid and spread of the slot's attention mass (analytic, 4 numbers)
  plus log mass.
* Decoder: spatial-broadcast per slot, conditioned on (what, where, style).
  Style comes from a *different* character of the target font, so character
  identity can only reach the decoder through the slots.

`GlobalAE` is the matched non-slot baseline: same encoder and decoder budget,
one continuous content vector, no quantisation.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

FSQ_LEVELS = (8, 6, 5)
GRID = 24


def coord_grid(h, w, device):
    ys = torch.linspace(-1, 1, h, device=device)
    xs = torch.linspace(-1, 1, w, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    return torch.stack([xx, yy])  # (2, h, w): x first


class FSQ(nn.Module):
    """Finite scalar quantisation (Mentzer et al., 2023), straight-through.

    Level L gives integer values {-(L//2) .. L-1-(L//2)} (e.g. L=8: -4..3,
    L=5: -2..2); outputs are divided by L//2.
    """

    def __init__(self, levels=FSQ_LEVELS, eps=1e-3):
        super().__init__()
        L = torch.tensor(levels, dtype=torch.float32)
        self.register_buffer('L', L, persistent=False)
        self.register_buffer('half_l', (L - 1) * (1 + eps) / 2, persistent=False)
        self.register_buffer('offset', torch.where(L % 2 == 0, 0.5, 0.0), persistent=False)
        self.register_buffer('half_width', torch.floor(L / 2), persistent=False)
        self.register_buffer('basis', torch.cumprod(torch.tensor([1.0] + list(levels[:-1])), 0), persistent=False)
        self.n_codes = int(math.prod(levels))

    def forward(self, z):
        shift = torch.atanh(self.offset / self.half_l)
        bounded = torch.tanh(z + shift) * self.half_l - self.offset
        q = bounded + (torch.round(bounded) - bounded).detach()
        return q / self.half_width

    def codes(self, qn):
        idx = torch.round(qn * self.half_width + self.half_width)
        return (idx * self.basis).sum(-1).long()


class Encoder(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        g = lambda c: nn.GroupNorm(8, c)
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), g(64), nn.SiLU(),
            nn.Conv2d(64, 64, 3, 2, 1), g(64), nn.SiLU(),      # 48
            nn.Conv2d(64, dim, 3, 2, 1), g(dim), nn.SiLU(),    # 24
            nn.Conv2d(dim, dim, 3, 1, 1), g(dim), nn.SiLU(),
            nn.Conv2d(dim, dim, 3, 1, 1))
        self.pos = nn.Conv2d(2, dim, 1)

    def forward(self, x):
        b, _, h, w = x.shape
        grid = coord_grid(h, w, x.device).expand(b, 2, h, w)
        f = self.net(torch.cat([x, grid], 1))
        f = f + self.pos(coord_grid(f.shape[2], f.shape[3], x.device))[None]
        return f  # (b, dim, 24, 24)


class SlotAttention(nn.Module):
    def __init__(self, n_slots=6, dim=128, iters=3, eps=1e-8):
        super().__init__()
        self.n, self.iters, self.eps, self.scale = n_slots, iters, eps, dim ** -0.5
        self.init = nn.Parameter(torch.randn(1, n_slots, dim) * dim ** -0.5)
        self.norm_in, self.norm_slot, self.norm_mlp = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.q, self.k, self.v = nn.Linear(dim, dim, bias=False), nn.Linear(dim, dim, bias=False), nn.Linear(dim, dim, bias=False)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim))

    def forward(self, feats):
        # feats: (b, n_in, dim)
        b, _, d = feats.shape
        x = self.norm_in(feats)
        k, v = self.k(x), self.v(x)
        slots = self.init.expand(b, -1, -1)
        for _ in range(self.iters):
            prev = slots
            q = self.q(self.norm_slot(slots))
            logits = torch.einsum('bsd,bnd->bsn', q, k) * self.scale
            attn = logits.softmax(dim=1) + self.eps            # compete over slots
            weights = attn / attn.sum(-1, keepdim=True)        # weighted mean over inputs
            upd = torch.einsum('bsn,bnd->bsd', weights, v)
            slots = self.gru(upd.reshape(-1, d), prev.reshape(-1, d)).reshape(b, self.n, d)
            slots = slots + self.mlp(self.norm_mlp(slots))
        return slots, attn  # attn: (b, s, n_in), sums to 1 over slots


def where_from_attn(attn, h=GRID, w=GRID):
    """(b, s, h*w) attention -> (b, s, 5): cx, cy, sx, sy, log-mass."""
    grid = coord_grid(h, w, attn.device).reshape(2, -1)  # (2, n)
    mass = attn.sum(-1, keepdim=True)                     # (b, s, 1)
    p = attn / mass
    mu = torch.einsum('bsn,cn->bsc', p, grid)
    var = torch.einsum('bsn,bscn->bsc', p, (grid[None, None] - mu[..., None]) ** 2)
    sd = (var + 1e-6).sqrt()
    return torch.cat([mu, sd, torch.log(mass / (h * w))], -1)


class StyleEncoder(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        g = lambda c: nn.GroupNorm(8, c)
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), g(32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1), g(64), nn.SiLU(),
            nn.Conv2d(64, 64, 3, 2, 1), g(64), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, dim))

    def forward(self, x):
        return self.net(x)


class BroadcastDecoder(nn.Module):
    """Decode each conditioning vector independently on a 24x24 grid, upsample to 96."""

    def __init__(self, cond_dim, hidden=96, relative=True):
        super().__init__()
        self.relative = relative
        g = lambda c: nn.GroupNorm(8, c)
        in_ch = cond_dim + (4 if relative else 2)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, 1, 1), g(hidden), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, 1, 1), g(hidden), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(hidden, hidden // 2, 3, 1, 1), g(hidden // 2), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(hidden // 2, 32, 3, 1, 1), g(32), nn.SiLU(),
            nn.Conv2d(32, 2, 3, 1, 1))  # ink logit, alpha logit

    def forward(self, cond, centre=None):
        # cond: (m, c); centre: (m, 4) = cx, cy, sx, sy
        m = cond.shape[0]
        grid = coord_grid(GRID, GRID, cond.device).expand(m, 2, GRID, GRID)
        planes = [cond[:, :, None, None].expand(-1, -1, GRID, GRID), grid]
        if self.relative:
            c = centre[:, :2, None, None]
            s = centre[:, 2:4, None, None].clamp(min=0.05)
            planes.append((grid - c) / s)
        return self.net(torch.cat(planes, 1))  # (m, 2, 96, 96)


class PartTokenizer(nn.Module):
    def __init__(self, n_slots=6, dim=128, style_dim=64, iters=3):
        super().__init__()
        self.enc = Encoder(dim)
        self.slot_attn = SlotAttention(n_slots, dim, iters)
        self.to_what = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, len(FSQ_LEVELS)))
        self.fsq = FSQ()
        self.code_proj = nn.Linear(len(FSQ_LEVELS), 64)
        self.style = StyleEncoder(style_dim)
        self.dec = BroadcastDecoder(64 + 5 + style_dim)
        self.n_slots = n_slots

    def tokenize(self, x):
        """x: (b, 1, 96, 96) in [0, 1]. Returns dict with what (normalised FSQ), codes, where, attn."""
        f = self.enc(x)
        b, d, h, w = f.shape
        slots, attn = self.slot_attn(f.flatten(2).transpose(1, 2))
        what = self.fsq(self.to_what(slots))
        return {'what': what, 'codes': self.fsq.codes(what.detach()), 'where': where_from_attn(attn, h, w),
                'attn': attn.reshape(b, -1, h, w)}

    def decode(self, tok, style_img):
        b, s, _ = tok['what'].shape
        st = self.style(style_img)                                       # (b, style)
        cond = torch.cat([self.code_proj(tok['what']), tok['where'], st[:, None].expand(-1, s, -1)], -1)
        out = self.dec(cond.reshape(b * s, -1), tok['where'].reshape(b * s, -1)[:, :4])
        out = out.reshape(b, s, 2, *out.shape[-2:])
        ink, alpha = out[:, :, 0], out[:, :, 1].softmax(1)
        recon_logit = torch.logsumexp(torch.log(alpha + 1e-8) + F.logsigmoid(ink), 1)  # log p(ink)
        return recon_logit, alpha, ink

    def forward(self, x, style_img):
        tok = self.tokenize(x)
        recon_logp, alpha, _ = self.decode(tok, style_img)
        return recon_logp, tok, alpha


class GlobalAE(nn.Module):
    """Non-slot baseline: one continuous content vector, same encoder/decoder family."""

    def __init__(self, dim=128, content_dim=64, style_dim=64):
        super().__init__()
        self.enc = Encoder(dim)
        self.head = nn.Sequential(nn.Flatten(), nn.LayerNorm(dim * 36), nn.Linear(dim * 36, 256), nn.SiLU(),
                                  nn.Linear(256, content_dim))
        self.pool = nn.AdaptiveAvgPool2d(6)
        self.style = StyleEncoder(style_dim)
        self.dec = BroadcastDecoder(content_dim + style_dim, hidden=128, relative=False)

    def content(self, x):
        return self.head(self.pool(self.enc(x)))

    def forward(self, x, style_img):
        c = self.content(x)
        out = self.dec(torch.cat([c, self.style(style_img)], -1))
        return F.logsigmoid(out[:, 0]), {'content': c}, None


def recon_loss(logp, target, ink_weight=4.0):
    """Ink-weighted BCE. logp = log p(ink); target in [0, 1]."""
    log1mp = torch.log1p(-logp.exp().clamp(max=1 - 1e-6))
    bce = -(target * logp + (1 - target) * log1mp)
    w = 1 + ink_weight * target
    return (bce * w).sum((-1, -2)).div(w.sum((-1, -2))).mean()
