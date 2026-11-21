"""Rectified flow (flow matching with straight paths).

x_t = (1 - t) * x_data + t * noise, target velocity v = noise - x_data.
Sampling integrates dx/dt = v from t = 1 (noise) to t = 0 (data).
"""
import torch


def sample_t(n, device, mean=0.0, std=1.0):
    """Logit-normal timesteps (Esser et al., 2024)."""
    return torch.sigmoid(torch.randn(n, device=device) * std + mean)


def interpolate(x0, noise, t):
    tt = t.view(-1, 1, 1, 1)
    return (1 - tt) * x0 + tt * noise, noise - x0


def masked_mse(pred, target, lat_width, weight=None):
    """MSE over valid latent columns; `weight` (B, 1, h, w) re-weights positions (normalised by its sum)."""
    m = (torch.arange(pred.shape[-1], device=pred.device)[None] < lat_width[:, None]).float()[:, None, None]
    if weight is not None:
        m = m * weight.float()
    return ((pred.float() - target.float()) ** 2 * m).sum() / (m.expand_as(pred).sum()).clamp(min=1)


def guided(v_none, v_content, v_full, w_content, w_style):
    """Compositional CFG: content guidance, then style guidance on top of content."""
    return v_none + w_content * (v_content - v_none) + w_style * (v_full - v_content)


@torch.no_grad()
def euler(velocity_fn, noise, steps=50):
    """velocity_fn(x, t) -> v; t is a (B,) tensor."""
    x = noise
    ts = torch.linspace(1, 0, steps + 1, device=noise.device)
    for k in range(steps):
        t = ts[k].expand(x.shape[0])
        x = x + (ts[k + 1] - ts[k]) * velocity_fn(x, t)
    return x
