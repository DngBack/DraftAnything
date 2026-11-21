"""[C] Two-level style.

* M: patch memory of all support images (local "how this writer draws a part").
* g: n_g global tokens pooled from M by learned queries (writer identity).

Training signals on g (both optional, weights in the trainer):
  * supervised contrastive loss over writers in the batch;
  * gradient reversal against a bag-of-characters head, so g cannot carry the
    content of the support words (anti-copy). M keeps local content on purpose.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import Attention, ResBlock, grad_reverse, sinusoid_2d

DOWN = 8   # style feature stride (image px per memory cell)


class StyleEncoder(nn.Module):
    def __init__(self, dim=384, n_global=4, heads=6, n_chars=None):
        super().__init__()
        self.dim = dim
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 2, 1), ResBlock(64),           # no full-resolution block: 15 supports x B
            nn.Conv2d(64, 128, 3, 2, 1), ResBlock(128),
            nn.Conv2d(128, 256, 3, 1, 1), ResBlock(256),
            nn.Conv2d(256, dim, 3, 2, 1), ResBlock(dim), ResBlock(dim),
            nn.GroupNorm(32, dim))
        self.queries = nn.Parameter(torch.randn(n_global, dim) * 0.02)
        self.pool = Attention(dim, heads)
        self.pool_norm = nn.LayerNorm(dim)
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 128))   # SupCon head
        self.bag = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, n_chars)) if n_chars else None

    def forward(self, support, support_w):
        """support: (B, K, H, Ws) ink in [0,1], zero-padded; support_w: (B, K) valid widths in px."""
        b, k, h, w = support.shape
        f = self.cnn(support.reshape(b * k, 1, h, w))                       # (BK, d, h', w')
        hh, ww = f.shape[-2:]
        f = f + sinusoid_2d(hh, ww, self.dim, f.device).T.reshape(1, self.dim, hh, ww).to(f.dtype)
        cols = torch.arange(ww, device=f.device)
        col_ok = cols[None, None] < torch.ceil(support_w.float() / DOWN)[..., None]   # (B, K, w')
        mem = f.reshape(b, k, self.dim, hh, ww).permute(0, 1, 3, 4, 2).reshape(b, k * hh * ww, self.dim)
        mask = col_ok[:, :, None, :].expand(b, k, hh, ww).reshape(b, -1)
        q = self.queries[None].expand(b, -1, -1)
        g_tokens = self.pool_norm(q + self.pool(q, mem, key_mask=mask))
        return {'memory': mem, 'memory_mask': mask, 'g_tokens': g_tokens, 'g': g_tokens.mean(1)}


def supcon_loss(g, writers, tau=0.1):
    """Supervised contrastive loss (Khosla et al., 2020); anchors without positives are skipped."""
    z = F.normalize(g.float(), dim=-1)
    sim = z @ z.T / tau
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, float('-inf'))
    pos = (writers[:, None] == writers[None]) & ~eye
    logp = sim.log_softmax(1)
    has = pos.any(1)
    if not has.any():
        return sim.new_zeros(())
    return -(logp.masked_fill(~pos, 0).sum(1)[has] / pos.sum(1)[has]).mean()


def anti_content_loss(style_enc, g, bag_targets, lam=1.0):
    """BCE of the bag-of-characters head on reversed-gradient g (encoder is pushed to remove content)."""
    logits = style_enc.bag(grad_reverse(g, lam))
    return F.binary_cross_entropy_with_logits(logits.float(), bag_targets.float())
