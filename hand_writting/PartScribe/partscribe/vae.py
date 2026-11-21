"""[E] Handwriting VAE, f=4, single channel, variable width.

Images are ink maps in [0, 1] (ink = 1, background and padding = 0), so zero
padding is background. Losses: masked L1 + Sobel-edge L1 (sharpness) + tiny KL,
and a PatchGAN hinge loss switched on after a warm-up.
The latent std is measured after training and stored as `latent_scale`.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ResBlock

FACTOR = 4


EDGE_COLS = 4


def clean_right_edge(ink, width_px):
    """Remove the decoder's boundary line from generated ink maps, keeping strokes.

    The generator zeroes latent columns beyond the predicted width, but 0 is not the latent
    of blank paper, so the decoder draws a faint, uniform, full-height line in the last
    pixel columns. Per-column median ink (0-255, strokes ignored) over val outputs, last 6
    columns: 3.9 3.1 3.9 4.8 11.2 22.6 (whole image ~1.6); real images and encoder latents
    do not have it. Cropping is wrong: strokes reach the edge in about half the words.
    A stroke covers few rows of a column, so the column median is the line itself:
    subtract it in the last EDGE_COLS columns.

    ink: (B, 1, H, W) in [0, 1], ink = 1. width_px: (B,) valid pixel width per item.
    """
    out = ink.clone()
    for k in range(ink.shape[0]):
        w = int(width_px[k])
        a = max(0, w - EDGE_COLS)
        if w <= a:
            continue
        cols = out[k, :, :, a:w]
        out[k, :, :, a:w] = (cols - cols.median(dim=-2, keepdim=True).values).clamp(min=0)
    return out


def decoded_width(lat_w):
    """Pixel width of a decoded generator latent of latent width `lat_w`."""
    return (lat_w * FACTOR).clamp(min=1)


class VAE(nn.Module):
    def __init__(self, z_ch=4, ch=(64, 128, 256)):
        super().__init__()
        c0, c1, c2 = ch
        self.enc = nn.Sequential(
            nn.Conv2d(1, c0, 3, 1, 1), ResBlock(c0),
            nn.Conv2d(c0, c1, 3, 2, 1), ResBlock(c1),
            nn.Conv2d(c1, c2, 3, 2, 1), ResBlock(c2), ResBlock(c2),
            nn.GroupNorm(32, c2), nn.SiLU(), nn.Conv2d(c2, 2 * z_ch, 3, 1, 1))
        self.dec = nn.Sequential(
            nn.Conv2d(z_ch, c2, 3, 1, 1), ResBlock(c2), ResBlock(c2),
            nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(c2, c1, 3, 1, 1), ResBlock(c1), ResBlock(c1),
            nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(c1, c0, 3, 1, 1), ResBlock(c0), ResBlock(c0),
            nn.GroupNorm(32, c0), nn.SiLU(), nn.Conv2d(c0, 1, 3, 1, 1))
        self.z_ch = z_ch
        self.register_buffer('latent_scale', torch.ones(()))

    def encode(self, x):
        mu, logvar = self.enc(x).chunk(2, 1)
        return mu, logvar.clamp(-20, 10)

    def decode(self, z):
        return torch.sigmoid(self.dec(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def to_latent(self, x):
        return self.encode(x)[0] * self.latent_scale

    @torch.no_grad()
    def from_latent(self, z):
        return self.decode(z / self.latent_scale)


class PatchDisc(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, ch, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(ch, ch * 2, 4, 2, 1), nn.GroupNorm(32, ch * 2), nn.LeakyReLU(0.2),
            nn.Conv2d(ch * 2, ch * 4, 3, 1, 1), nn.GroupNorm(32, ch * 4), nn.LeakyReLU(0.2),
            nn.Conv2d(ch * 4, 1, 3, 1, 1))

    def forward(self, x):
        return self.net(x)


_SOBEL = torch.tensor([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]])


def sobel(x):
    k = torch.stack([_SOBEL, _SOBEL.T])[:, None].to(x)
    return F.conv2d(x, k, padding=1)


def col_mask(width_px, w, device):
    """(B, 1, 1, w) float mask of valid columns."""
    return (torch.arange(w, device=device)[None] < width_px[:, None].to(device)).float()[:, None, None]


def recon_losses(rec, x, mu, logvar, width_px, kl_weight=1e-6):
    m = col_mask(width_px, x.shape[-1], x.device)
    denom = m.sum() * x.shape[-2]
    l1 = ((rec - x).abs() * m).sum() / denom
    edge = ((sobel(rec) - sobel(x)).abs() * m).sum() / (2 * denom)
    ml = m[..., ::FACTOR][..., :mu.shape[-1]]
    kl = (0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar) * ml).sum() / (ml.sum() * mu.shape[1] * mu.shape[2])
    return {'l1': l1, 'edge': edge, 'kl': kl * kl_weight}


def decode_generated(vae, z, lat_w):
    """Generator latents -> clean ink maps (B, 1, H, W) and the valid pixel width of each.

    Masks everything beyond the predicted width (the VAE is only trained on valid columns)
    and removes the boundary line (see clean_right_edge)."""
    img = vae.from_latent(z).float().clamp(0, 1)
    wpx = decoded_width(lat_w)
    img = img * (torch.arange(img.shape[-1], device=img.device)[None] < wpx[:, None])[:, None, None]
    return clean_right_edge(img, wpx), wpx
