"""Streaming schema-conditioned memory with recurrent consolidation, plus a full-context reference.

Single-read: each page's tokens enter `write` once; consolidation iterates on slots and that
page's selected tokens only. `tied=True` is Sleep (shared G_phi, N steps); `tied=False` is the
compute-matched deeper updater (N distinct blocks, same FLOPs, N x params).
"""
import math
import torch
from torch import nn
import torch.nn.functional as Fn


def page_codes(T, d, device):
    t = torch.arange(T, device=device, dtype=torch.float32)[:, None]
    w = torch.exp(torch.arange(0, d, 2, device=device) * (-math.log(1e4) / d))
    return torch.cat([torch.sin(t * w), torch.cos(t * w)], -1)


PAGE_DIM = 32   # page code is concatenated to token features; adding it swamped content in debug runs


def with_page(x, T):
    """x [B, T, n, d_in] -> [B, T, n, d_in + PAGE_DIM]."""
    c = page_codes(T, PAGE_DIM, x.device)[None, :, None].expand(*x.shape[:3], -1)
    return torch.cat([x, c.to(x.dtype)], -1)


def cap(u, max_rms=1.):
    """Shrink a slot only when its RMS exceeds max_rms; keeps tied recurrence finite without
    LayerNorm's full renormalisation (which stalled learning in the debug runs)."""
    rms = u.float().pow(2).mean(-1, keepdim=True).sqrt()
    return u * (max_rms / rms.clamp(min=max_rms)).to(u.dtype)


class Block(nn.Module):
    """Slots self-attend, read page tokens, MLP. Pre-LN residual."""

    def __init__(self, d, heads):
        super().__init__()
        self.n1, self.n2, self.n3, self.nz = (nn.LayerNorm(d) for _ in range(4))
        self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.cross = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, u, z):
        h = self.n1(u); u = u + self.self_attn(h, h, h, need_weights=False)[0]
        zz = self.nz(z); u = u + self.cross(self.n2(u), zz, zz, need_weights=False)[0]
        return u + self.mlp(self.n3(u))


class Heads(nn.Module):
    """Schema queries read the final state; value classifier (+null) and page-evidence pointer (+none)."""

    def __init__(self, d, heads, n_values, layers=2):
        super().__init__()
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.value = nn.Linear(d, n_values + 1)
        self.ev_q, self.ev_k, self.ev_none = nn.Linear(d, d), nn.Linear(PAGE_DIM, d), nn.Linear(d, 1)

    def forward(self, q, state, T):
        # start from each field's own slot (it already contains q). No final LayerNorm: in the
        # debug runs it stalled learning of even single-page lookup (docs/stage0-protocol.md).
        q = state[:, :q.shape[1]]
        for b in self.blocks:
            q = b(q, state)
        pages = self.ev_k(page_codes(T, PAGE_DIM, q.device))
        ev = torch.einsum('bfd,td->bft', self.ev_q(q), pages) / math.sqrt(q.shape[-1])
        return self.value(q), torch.cat([ev, self.ev_none(q)], -1)


class SleepKIE(nn.Module):
    def __init__(self, d_in, n_fields, n_values, d=256, heads=4, free_slots=24, n_iter=1, tied=True,
                 select_k=64, gate=True):
        super().__init__()
        self.d, self.n_iter, self.tied, self.k, self.use_gate = d, n_iter, tied, select_k, gate
        self.inp = nn.Linear(d_in + PAGE_DIM, d)
        self.field_q = nn.Linear(d_in, d)   # schema names via the frozen encoder
        self.free = nn.Parameter(torch.randn(free_slots, d) * .02)
        self.sel_q, self.sel_k = nn.Linear(d, d), nn.Linear(d, d)
        self.write_norm = nn.LayerNorm(d)
        self.write = nn.MultiheadAttention(d, heads, batch_first=True)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(1 if tied else n_iter))
        self.gate = nn.Linear(2 * d, d)
        self.heads = Heads(d, heads, n_values)

    def select(self, m, z):
        """Memory-conditioned top-K (memory is initialised from the schema). Scores gate kept tokens
        so selection receives gradient; hard top-K at train and test."""
        s = torch.einsum('bmd,bnd->bmn', self.sel_q(m), self.sel_k(z)).amax(1) / math.sqrt(self.d)
        if self.k >= z.shape[1]:
            return z * torch.sigmoid(s)[..., None]
        top = s.topk(self.k, -1).indices
        zs = z.gather(1, top[..., None].expand(-1, -1, self.d))
        return zs * torch.sigmoid(s.gather(1, top))[..., None]

    def step(self, m, z):
        z = self.select(m, z)
        zz = self.write_norm(z)
        u = cap(m + self.write(m, zz, zz, need_weights=False)[0])
        for k in range(self.n_iter):
            # each of the N steps contributes 1/N of its residual update (N=1 unchanged)
            u = cap(u + (self.blocks[0 if self.tied else k](u, z) - u) / self.n_iter)
        if not self.use_gate:
            return u
        g = torch.sigmoid(self.gate(torch.cat([m, u], -1)))
        return g * m + (1 - g) * u

    def forward(self, x, schema):
        B, T = x.shape[:2]
        q = self.field_q(schema)
        m = torch.cat([q, self.free.expand(B, -1, -1)], 1)
        x = with_page(x, T)
        for t in range(T):
            m = self.step(m, self.inp(x[:, t]))
        return self.heads(q, m, T)


class FullContext(nn.Module):
    """Non-streaming reference: schema queries attend over every token of every page.
    State grows linearly with T (no memory bound)."""

    def __init__(self, d_in, n_fields, n_values, d=256, heads=4, layers=4):
        super().__init__()
        self.d = d
        self.inp = nn.Linear(d_in + PAGE_DIM, d)
        self.field_q = nn.Linear(d_in, d)   # schema names via the frozen encoder
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.heads = Heads(d, heads, n_values)

    def forward(self, x, schema):
        B, T, n = x.shape[:3]
        z = self.inp(with_page(x, T)).reshape(B, T * n, -1)
        q = self.field_q(schema)
        h = q
        for b in self.blocks:
            h = b(h, z)
        return self.heads(q, h, T)


def build(name, spec, d=256):
    """A-matrix variants from docs/idea.md section 6, at d=256 (half the proposal's d)."""
    common = dict(d_in=spec.d_in, n_fields=spec.n_fields, n_values=spec.n_values, d=d)
    table = {
        'A0_N1': dict(n_iter=1),
        'A1_N2': dict(n_iter=2),
        'A2_N4': dict(n_iter=4),
        'A3_N4_half': dict(n_iter=4, free_slots=8),
        'A4_N1_2x': dict(n_iter=1, free_slots=56),
        'A5_deep4': dict(n_iter=4, tied=False),
    }
    if name == 'full':
        return FullContext(**common)
    return SleepKIE(**common, **table[name])
