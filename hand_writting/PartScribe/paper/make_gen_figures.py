"""Figures that need the frozen model (GPU): counterfactual pairs, denoising trajectory, style transfer.

    python paper/make_gen_figures.py      (from hand_writting/PartScribe)
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from partscribe import flow  # noqa: E402
from partscribe import generator as gen_mod  # noqa: E402
from partscribe.datasets import Batcher, load_vn  # noqa: E402
from partscribe.generator import Config, PartScribe, strip_marks  # noqa: E402
from partscribe.runtime import VATR, load_vae  # noqa: E402
from partscribe.vae import decode_generated  # noqa: E402

OUT = ROOT / 'paper' / 'figures'
FONT = ImageFont.truetype('/opt/conda/fonts/DejaVuSans.ttf', 22)
FONT_S = ImageFont.truetype('/opt/conda/fonts/DejaVuSans.ttf', 18)
DEV = 'cuda'


def load(arm):
    st = torch.load(ROOT / f'runs/gen_vn_{arm}_v3/snap_350000.pt', map_location='cpu', weights_only=False)
    m = PartScribe(Config(**st['meta']['config']), None).to(DEV).eval()
    m.load_state_dict(st['ema'])
    return m, st


def to_img(ink, scale=3):
    a = ((1 - ink.clamp(0, 1).cpu().numpy()) * 255).astype(np.uint8)
    im = Image.fromarray(a)
    return im.resize((im.width * scale, im.height * scale), Image.LANCZOS)


def grid(rows, header, row_labels, path, label_w=200, pad=12):
    """rows: list of list of PIL images."""
    ncol = len(rows[0])
    cw = [max(r[c].width for r in rows) for c in range(ncol)]
    rh = [max(im.height for im in r) for r in rows]
    W = label_w + sum(cw) + pad * (ncol + 1)
    H = 40 + sum(rh) + pad * (len(rows) + 1)
    out = Image.new('L', (W, H), 255)
    d = ImageDraw.Draw(out)
    x = label_w + pad
    for c in range(ncol):
        d.text((x, 8), header[c], fill=0, font=FONT_S); x += cw[c] + pad
    y = 40 + pad
    for r, lab in zip(rows, row_labels):
        d.text((6, y + rh[rows.index(r)] // 2 - 12), lab, fill=0, font=FONT)
        x = label_w + pad
        for c, im in enumerate(r):
            out.paste(im, (x, y)); x += cw[c] + pad
        y += max(im.height for im in r) + pad
    out.save(path)


def batch_for(bt, writer, n):
    i = bt.targets[writer][0]
    sup = bt.support_for(i)
    return bt.collate([i] * n, [sup] * n), sup


@torch.no_grad()
def fig_pairs(models, bt, vae, h):
    words = ['mẹ', 'vẻ', 'lễ', 'ngỡ', 'sử', 'thẳng']
    w = bt.writers[3]
    rows = []
    for t in words:
        rows.append([])
    for arm in ('nfd', 'char'):
        m = models[arm]
        b, _ = batch_for(bt, w, len(words))
        g = torch.Generator(device=DEV).manual_seed(1)
        z, lw = m.sample(words, b['support'].to(DEV), b['support_width'].to(DEV), h, 50, 3.0, 1.5, generator=g)
        img, wpx = decode_generated(vae, z, lw)
        for k in range(len(words)):
            rows[k].append(to_img(img[k, 0, :, :int(wpx[k])]))
        if arm == 'nfd':
            g = torch.Generator(device=DEV).manual_seed(1)
            zp, lwp = m.sample([strip_marks(t) for t in words], b['support'].to(DEV), b['support_width'].to(DEV), h, 50,
                               3.0, 1.5, widths_override=lw, generator=g)
            ip, wpp = decode_generated(vae, zp, lwp)
            for k in range(len(words)):
                rows[k].append(to_img(ip[k, 0, :, :int(wpp[k])]))
    rows = [[r[0], r[1], r[2]] for r in rows]
    grid(rows, ['nfd, có dấu', 'nfd, bỏ dấu', 'char, có dấu'],
         [f'{t}  →  {strip_marks(t)}' for t in words], OUT / 'pairs.png', label_w=220)


@torch.no_grad()
def fig_trajectory(models, bt, vae, h):
    m = models['nfd']
    w = bt.writers[1]
    texts = ['Nguyễn']
    b, _ = batch_for(bt, w, 1)
    keep = {}
    orig = flow.euler

    def euler_rec(velocity_fn, noise, steps=50):
        x = noise
        ts = torch.linspace(1, 0, steps + 1, device=noise.device)
        for k in range(steps):
            t = ts[k].expand(x.shape[0])
            v = velocity_fn(x, t)
            if k in (0, 10, 20, 30, 40, 45, 49):
                keep[k] = (float(ts[k]), x.clone(), (x - ts[k] * v).clone())   # x_t and the x0 estimate
            x = x + (ts[k + 1] - ts[k]) * v
        keep[steps] = (0.0, x.clone(), x.clone())
        return x

    gen_mod.flow.euler = euler_rec
    try:
        g = torch.Generator(device=DEV).manual_seed(3)
        z, lw = m.sample(texts, b['support'].to(DEV), b['support_width'].to(DEV), h, 50, 3.0, 1.5, generator=g)
    finally:
        gen_mod.flow.euler = orig
    rows_xt, rows_x0, head = [], [], []
    for k in sorted(keep):
        t, xt, x0 = keep[k]
        a, wpx = decode_generated(vae, xt, lw)
        c, _ = decode_generated(vae, x0, lw)
        rows_xt.append(to_img(a[0, 0, :, :int(wpx[0])], 2)); rows_x0.append(to_img(c[0, 0, :, :int(wpx[0])], 2))
        head.append(f't={t:.2f}')
    grid([rows_xt, rows_x0], head, ['x_t (giải mã)', 'ước lượng x₀'], OUT / 'trajectory.png', label_w=190)


@torch.no_grad()
def fig_styles(models, bt, vae, h):
    m = models['nfd']
    texts = ['Hà Nội', 'trường học', 'NGUYỄN']
    rows, labels = [], []
    for w in bt.writers[:8]:
        b, sup = batch_for(bt, w, len(texts))
        g = torch.Generator(device=DEV).manual_seed(0)
        z, lw = m.sample(texts, b['support'].to(DEV), b['support_width'].to(DEV), h, 50, 3.0, 1.5, generator=g)
        img, wpx = decode_generated(vae, z, lw)
        ref = 1 - b['support'][0, 0, :, :int(b['support_width'][0, 0])]
        rows.append([to_img(1 - ref, 2)] + [to_img(img[k, 0, :, :int(wpx[k])], 2) for k in range(len(texts))])
        labels.append(f'writer {w}')
    grid(rows, ['ảnh mẫu (1/15)'] + texts, labels, OUT / 'styles.png', label_w=150)


def main():
    vae, vstate = load_vae(ROOT / 'runs/vae_vn32/final.pt', DEV)
    models = {a: load(a)[0] for a in ('nfd', 'char')}
    st = load('nfd')[1]
    ds = load_vn(str(VATR / 'files/vietnamese/manifest.json'), str(ROOT / 'data/vn'), 'val', apply_holdout=True)
    bt = Batcher(ds, lambda t: True, n_support=15, support_width=st['meta'].get('support_width'), seed=0)
    h = vstate['height']
    fig_pairs(models, bt, vae, h); print('pairs')
    fig_trajectory(models, bt, vae, h); print('trajectory')
    fig_styles(models, bt, vae, h); print('styles')


if __name__ == '__main__':
    main()
