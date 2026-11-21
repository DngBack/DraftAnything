"""[D] Denoiser: a DiT on the VAE latent with role-specific conditioning.

Per block:
  1. AdaLN-zero self-attention over latent tokens; the modulation comes from
     timestep + global writer vector g ("global how": slant, weight, size).
  2. Cross-attention to content tokens ("what, where"), with a soft monotonic
     prior: a Gaussian bias around each character's centre from the width head.
  3. Cross-attention to the style patch memory M ("local how").
  4. AdaLN-zero MLP.
Both cross-attentions have zero-initialised gates. A learned null token is
always prepended to each context; classifier-free dropout masks the rest of the
context, so the same weights give unconditional / content-only / full outputs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import Attention, FeedForward, sinusoid, sinusoid_2d


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.n2, self.n3 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.n4 = nn.LayerNorm(dim, elementwise_affine=False)
        self.self_attn = Attention(dim, heads)
        self.content_attn = Attention(dim, heads)
        self.style_attn = Attention(dim, heads)
        self.ff = FeedForward(dim)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)
        self.gate_c = nn.Parameter(torch.zeros(dim))
        self.gate_s = nn.Parameter(torch.zeros(dim))
        self.align_temp = nn.Parameter(torch.zeros(()))   # softplus -> sigma multiplier

    def forward(self, x, cond, x_mask, content, content_mask, align, memory, memory_mask):
        s1, c1, g1, s2, c2, g2 = self.ada(cond).chunk(6, -1)
        x = x + g1[:, None] * self.self_attn(modulate(self.n1(x), s1, c1), key_mask=x_mask)
        bias = None
        if align is not None:
            dx, sigma = align                      # (B, Lq, Lk), (B, 1, Lk)
            sig = sigma * F.softplus(self.align_temp + 0.5413) + 0.5   # softplus(0.5413) = 1
            bias = -0.5 * (dx / sig) ** 2
        x = x + self.gate_c * self.content_attn(self.n2(x), content, bias=bias, key_mask=content_mask)
        x = x + self.gate_s * self.style_attn(self.n3(x), memory, key_mask=memory_mask)
        return x + g2[:, None] * self.ff(modulate(self.n4(x), s2, c2))


class DiT(nn.Module):
    def __init__(self, z_ch=4, dim=384, depth=8, heads=6, patch=(2, 2), style_dim=384):
        super().__init__()
        self.dim, self.patch, self.z_ch = dim, patch, z_ch
        ph, pw = patch
        self.embed = nn.Linear(z_ch * ph * pw, dim)
        self.t_mlp = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.g_mlp = nn.Sequential(nn.Linear(style_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.null_g = nn.Parameter(torch.zeros(style_dim))
        self.null_content = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.null_memory = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.content_in = nn.Linear(dim, dim)
        self.memory_in = nn.Linear(style_dim, dim)
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.final = nn.Linear(dim, z_ch * ph * pw)
        for m in (self.final_ada[-1], self.final):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def patchify(self, z):
        b, c, h, w = z.shape
        ph, pw = self.patch
        assert h % ph == 0 and w % pw == 0, (h, w)
        return z.reshape(b, c, h // ph, ph, w // pw, pw).permute(0, 2, 4, 1, 3, 5).reshape(b, (h // ph) * (w // pw), -1)

    def unpatchify(self, x, h, w):
        b = x.shape[0]
        ph, pw = self.patch
        return x.reshape(b, h // ph, w // pw, self.z_ch, ph, pw).permute(0, 3, 1, 4, 2, 5).reshape(b, self.z_ch, h, w)

    def forward(self, z, t, lat_width, g, content, layout, memory, memory_mask, drop_content, drop_style):
        """
        z: (B, C, h, w) noisy latent; t: (B,) in [0, 1]; lat_width: (B,) valid latent columns.
        g: (B, style_dim); content: dict from ContentEncoder; layout: (centres, widths) (B, Lc) latent cols.
        memory: (B, Lm, style_dim) + mask; drop_*: (B,) bool classifier-free dropout.
        """
        b, _, h, w = z.shape
        ph, pw = self.patch
        rows, cols = h // ph, w // pw
        x = self.embed(self.patchify(z)) + sinusoid_2d(rows, cols, self.dim, z.device)[None].to(z.dtype)
        col = torch.arange(cols, device=z.device).repeat(rows)                     # token -> column
        x_mask = col[None] < torch.ceil(lat_width.float() / pw)[:, None]
        g = torch.where(drop_style[:, None], self.null_g.expand_as(g), g)
        cond = self.t_mlp(sinusoid(t * 1000, 256)) + self.g_mlp(g)

        tok = self.content_in(content['tokens'])
        cmask = content['token_mask'] & ~drop_content[:, None]
        ctx = torch.cat([self.null_content.expand(b, -1, -1), tok], 1)
        cmask = torch.cat([torch.ones(b, 1, dtype=torch.bool, device=z.device), cmask], 1)
        centres, widths = layout
        tc = content['token_char']
        k_centre = torch.gather(centres, 1, tc)                                    # (B, Lk)
        k_width = torch.gather(widths, 1, tc)
        q_x = (col.float() + 0.5) * pw                                             # latent columns
        dx = q_x[None, :, None] - k_centre[:, None, :]
        dx = torch.cat([torch.zeros_like(dx[..., :1]), dx], -1)                   # null token: no bias
        sigma = torch.cat([torch.full_like(k_width[:, :1], 1e4), k_width / 2], -1)[:, None]

        mem = torch.cat([self.null_memory.expand(b, -1, -1), self.memory_in(memory)], 1)
        mmask = torch.cat([torch.ones(b, 1, dtype=torch.bool, device=z.device),
                           memory_mask & ~drop_style[:, None]], 1)
        for blk in self.blocks:
            x = blk(x, cond, x_mask, ctx, cmask, (dx, sigma), mem, mmask)
        shift, scale = self.final_ada(cond).chunk(2, -1)
        out = self.final(modulate(self.final_norm(x), shift, scale))
        return self.unpatchify(out, h, w)
