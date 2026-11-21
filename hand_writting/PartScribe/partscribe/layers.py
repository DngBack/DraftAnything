"""Shared building blocks: positional encodings, masked attention, gradient reversal."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoid(pos, dim, max_period=10000.0):
    """pos: (...,) float -> (..., dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=pos.device, dtype=torch.float32) / half)
    ang = pos.float()[..., None] * freqs
    return torch.cat([ang.sin(), ang.cos()], -1)


def sinusoid_2d(rows, cols, dim, device):
    """(rows*cols, dim): first half encodes the row, second half the column."""
    r = sinusoid(torch.arange(rows, device=device), dim // 2)
    c = sinusoid(torch.arange(cols, device=device), dim // 2)
    return torch.cat([r[:, None].expand(-1, cols, -1), c[None].expand(rows, -1, -1)], -1).reshape(rows * cols, dim)


class Attention(nn.Module):
    """Multi-head attention with an optional additive bias and a key padding mask.

    bias: (B, Lq, Lk) or (B, 1|H, Lq, Lk); key_mask: (B, Lk) bool, True = valid.
    Every query must have at least one valid key.
    """

    def __init__(self, dim, heads, ctx_dim=None):
        super().__init__()
        ctx_dim = ctx_dim or dim
        self.h = heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(ctx_dim, 2 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, ctx=None, bias=None, key_mask=None):
        ctx = x if ctx is None else ctx
        b, lq, d = x.shape
        q = self.q(x).view(b, lq, self.h, -1).transpose(1, 2)
        k, v = self.kv(ctx).view(b, ctx.shape[1], 2, self.h, -1).permute(2, 0, 3, 1, 4)
        mask = None
        if bias is not None or key_mask is not None:
            # Memory-efficient SDPA needs the mask's last-dim stride aligned to 16; otherwise it falls back
            # to the math kernel and materialises (B, H, Lq, Lk). Without a bias the mask is (B, 1, 1, Lk).
            lk = ctx.shape[1]
            lk_pad = (lk + 15) // 16 * 16
            rows = lq if bias is not None else 1
            mask = torch.zeros(b, 1, rows, lk_pad, device=x.device, dtype=q.dtype)[..., :lk]
            if bias is not None:
                mask = mask + (bias[:, None] if bias.dim() == 3 else bias).to(q.dtype)
                mask = torch.zeros(b, 1, lq, lk_pad, device=x.device, dtype=q.dtype)[..., :lk].copy_(mask)
            if key_mask is not None:
                mask.masked_fill_(~key_mask[:, None, None, :], float('-inf'))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(y.transpose(1, 2).reshape(b, lq, d))


class FeedForward(nn.Sequential):
    def __init__(self, dim, mult=4):
        super().__init__(nn.Linear(dim, dim * mult), nn.GELU(approximate='tanh'), nn.Linear(dim * mult, dim))


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def grad_reverse(x, lam=1.0):
    return _GradReverse.apply(x, lam)


class ResBlock(nn.Module):
    def __init__(self, cin, cout=None, groups=32):
        super().__init__()
        cout = cout or cin
        self.net = nn.Sequential(nn.GroupNorm(min(groups, cin), cin), nn.SiLU(), nn.Conv2d(cin, cout, 3, 1, 1),
                                 nn.GroupNorm(min(groups, cout), cout), nn.SiLU(), nn.Conv2d(cout, cout, 3, 1, 1))
        self.skip = nn.Identity() if cin == cout else nn.Conv2d(cin, cout, 1)

    def forward(self, x):
        return self.skip(x) + self.net(x)
