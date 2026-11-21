"""PartScribe generator: content [A+B] + style [C] + DiT [D] on a frozen VAE latent [E]."""
import unicodedata as ud
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from . import flow
from .content import ContentEncoder, WidthHead, char_layout, fit_widths, width_loss
from .dit import DiT
from .style import StyleEncoder, anti_content_loss, supcon_loss
from .vae import FACTOR


@dataclass
class Config:
    content: str = 'parts'        # 'parts' (learned slots) | 'nfd' (factorised Unicode parts) | 'char' (baseline)
    dim: int = 384
    depth: int = 8
    heads: int = 6
    context_depth: int = 4
    n_global: int = 4
    z_ch: int = 4
    patch_h: int = 2
    patch_w: int = 2
    p_drop_content: float = 0.1
    p_drop_style: float = 0.1
    w_width: float = 0.1
    w_supcon: float = 0.05
    w_anti: float = 0.05
    anti_lambda: float = 1.0
    content_font_aug: bool = True  # random reference font per word during training


def strip_marks(text):
    """Remove Vietnamese combining marks, keeping length (ẹ -> e, Ễ -> E; đ stays đ)."""
    return ''.join(ud.normalize('NFD', c)[0] if c not in 'đĐ' else c for c in text)


class PartScribe(nn.Module):
    def __init__(self, cfg: Config, table=None, alphabet=None):
        super().__init__()
        self.cfg = cfg
        self.content = ContentEncoder(table, cfg.content, cfg.dim, cfg.context_depth, cfg.heads)
        self.alphabet = alphabet or self.content.table_chars
        self.char_index = {c: i for i, c in enumerate(self.alphabet)}
        self.style = StyleEncoder(cfg.dim, cfg.n_global, cfg.heads, n_chars=len(self.alphabet))
        self.width = WidthHead(cfg.dim, cfg.dim)
        self.dit = DiT(cfg.z_ch, cfg.dim, cfg.depth, cfg.heads, (cfg.patch_h, cfg.patch_w), cfg.dim)
        self.n_fonts = table['codes'].shape[0] if table is not None else 1

    # -- helpers -------------------------------------------------------------
    def bag(self, support_texts, device):
        """Multi-hot of characters appearing in each sample's support transcripts."""
        out = torch.zeros(len(support_texts), len(self.alphabet))   # built on CPU: one copy, no per-item syncs
        for i, texts in enumerate(support_texts):
            idx = [self.char_index[c] for c in set(''.join(texts)) if c in self.char_index]
            out[i, idx] = 1
        return out.to(device, non_blocking=True)

    def condition(self, texts, support, support_w, font_choice=None):
        c = self.content(texts, font_choice)
        s = self.style(support, support_w)
        widths = self.width(c['char_feats'], c['char_mask'], s['g'])
        return c, s, widths

    # -- training ------------------------------------------------------------
    def loss(self, z0, lat_width, texts, support, support_w, writers, support_texts, pix_weight=None):
        """z0: (B, C, h, w) scaled VAE latents, zero beyond lat_width (latent columns)."""
        b, dev = z0.shape[0], z0.device
        font = None
        if self.cfg.content == 'parts' and self.cfg.content_font_aug and self.training:
            font = torch.randint(self.n_fonts, (b,), device=dev)
        # writer < 0: content-only target (HAT2023, no writer identity): style is always dropped,
        # the width head sees the null style, and SupCon / anti-content skip it.
        co = writers < 0
        c = self.content(texts, font)
        s = self.style(support, support_w)
        g_w = torch.where(co[:, None], self.dit.null_g.to(s['g'].dtype).expand_as(s['g']), s['g'])
        widths = self.width(c['char_feats'], c['char_mask'], g_w)
        l_width = width_loss(widths, lat_width)
        layout = char_layout(fit_widths(widths, lat_width))
        drop_c = torch.rand(b, device=dev) < self.cfg.p_drop_content
        drop_s = (torch.rand(b, device=dev) < self.cfg.p_drop_style) | co
        t = flow.sample_t(b, dev)
        noise = torch.randn_like(z0)
        zt, v = flow.interpolate(z0, noise, t)
        pred = self.dit(zt, t, lat_width, s['g'], c, layout, s['memory'], s['memory_mask'], drop_c, drop_s)
        lw = None
        if pix_weight is not None:   # pixel weights -> latent cells (max over each FACTOR x FACTOR block)
            lw = torch.nn.functional.max_pool2d(pix_weight, FACTOR)[..., :z0.shape[-1]]
        l_flow = flow.masked_mse(pred, v, lat_width, lw)
        real = ~co
        l_sup = supcon_loss(self.style.proj(s['g'][real]), writers[real])
        l_anti = anti_content_loss(self.style, s['g'][real],
                                   self.bag([t for t, r in zip(support_texts, real.tolist()) if r], dev),
                                   self.cfg.anti_lambda)
        total = l_flow + self.cfg.w_width * l_width + self.cfg.w_supcon * l_sup + self.cfg.w_anti * l_anti
        return total, {'flow': l_flow.item(), 'width': l_width.item(), 'supcon': l_sup.item(),
                       'anti': l_anti.item()}

    # -- sampling ------------------------------------------------------------
    @torch.no_grad()
    def sample(self, texts, support, support_w, height_px, steps=50, w_content=3.0, w_style=1.5,
               widths_override=None, generator=None, w_mark=0.0):
        """Returns scaled latents (B, C, height/4, W) and the latent width of each word.

        w_mark > 0 adds mark guidance: w_mark * (v_full(text) - v_full(strip_marks(text))), with the
        layout of `text` for both, so the difference is the velocity of the marks alone.
        """
        b, dev = len(texts), support.device
        c, s, widths = self.condition(texts, support, support_w)
        if widths_override is not None:
            widths = fit_widths(widths, widths_override)
        mult = self.cfg.patch_w
        lat_width = torch.ceil(widths.sum(1) / mult) * mult
        w = int(lat_width.max().item())
        h = height_px // FACTOR
        layout = char_layout(widths)
        noise = torch.randn(b, self.cfg.z_ch, h, w, device=dev, generator=generator)
        cols = torch.arange(w, device=dev)[None] < lat_width[:, None]
        n = 4 if w_mark else 3
        rep = lambda x: torch.cat([x] * n)
        if w_mark:
            plain = self.content([strip_marks(t) for t in texts])
            cc = {k: torch.cat([c[k]] * 3 + [plain[k]]) for k in c}
        else:
            cc = {k: rep(v) for k, v in c.items()}
        g_, mem_, mm_ = rep(s['g']), rep(s['memory']), rep(s['memory_mask'])
        lay_ = tuple(rep(x) for x in layout)
        lw_ = rep(lat_width)
        ones, zeros = torch.ones(b, dtype=torch.bool, device=dev), torch.zeros(b, dtype=torch.bool, device=dev)
        drop_c = torch.cat([ones, zeros, zeros] + [zeros] * (n - 3))
        drop_s = torch.cat([ones, ones, zeros] + [zeros] * (n - 3))

        def velocity(x, t):
            v = self.dit(rep(x), rep(t), lw_, g_, cc, lay_, mem_, mm_, drop_c, drop_s).float()
            parts = v.chunk(n)
            out = flow.guided(parts[0], parts[1], parts[2], w_content, w_style)
            if w_mark:
                out = out + w_mark * (parts[2] - parts[3])
            return out

        z = flow.euler(velocity, noise, steps)
        return z * cols[:, None, None], lat_width

    def config_dict(self):
        return asdict(self.cfg)
