"""Showcase generation for given texts: N candidates per writer, reranked (best-of-N).

Per writer (val split, never seen in training) and text, N candidates with different noise
are generated; each is read by the evaluator OCR (spaces ignored) and embedded with the HWD
VGG16. Ranking: OCR edit distance first, then HWD distance to that writer's 15 support
images. The figure shows, per writer: one support image, the FIRST candidate (= what a
single unselected sample looks like) and the best-of-N pick.

    python scripts/showcase.py --ckpt runs/gen_vn_nfd_v2/snap_200000.pt --texts "Ngô Quyền" "NGÔ QUYỀN" \
        --n 16 --writers 8 --out runs/visual/showcase_ngo_quyen.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_standard import VGG16_URL, hwd_tensor  # noqa: E402
from partscribe.datasets import Batcher  # noqa: E402
from partscribe.evaluator import Codec, align, load_ocr  # noqa: E402
from partscribe.generator import Config, PartScribe  # noqa: E402
from partscribe.runtime import add_data_args, load_split, load_vae  # noqa: E402
from partscribe.vae import FACTOR  # noqa: E402

FONT = '/opt/conda/fonts/DejaVuSans.ttf'


@torch.inference_mode()
def vgg_mean(vgg, inks, dev):
    """Mean HWD feature over a list of ink images (1 = ink)."""
    feats = []
    for ink in inks:
        img = Image.fromarray(((1 - ink.clamp(0, 1).cpu().numpy()) * 255).astype(np.uint8)).convert('RGB')
        x = hwd_tensor(img)
        f = vgg(x[None].to(dev))
        f = f.permute(0, 3, 2, 1).reshape(-1, f.shape[1])[:max(1, x.shape[-1] // 32)]
        feats.append(f.double().cpu())
    return torch.cat(feats).mean(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(ap)
    ap.add_argument('--ckpt', type=Path, required=True)
    ap.add_argument('--vae', type=Path, default=Path('runs/vae_vn32/final.pt'))
    ap.add_argument('--ocr', type=Path, default=Path('runs/eval_ocr_vn/best.pt'))
    ap.add_argument('--texts', nargs='+', required=True)
    ap.add_argument('--split', default='val')
    ap.add_argument('--writers', type=int, default=8)
    ap.add_argument('--n', type=int, default=16)
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--w-content', type=float, default=3.0)
    ap.add_argument('--w-style', type=float, default=1.5)
    ap.add_argument('--w-mark', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.vn_data is None:
        a.vn_data = 'data/vn'
    dev = 'cuda'
    st = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    model = PartScribe(Config(**st['meta']['config']), None).to(dev).eval()
    model.load_state_dict(st['ema'])
    vae, vstate = load_vae(a.vae, dev)
    ds = load_split(a, a.split)
    n_sup = int(st['meta']['args'].get('n_support', 15))
    batcher = Batcher(ds, lambda t: True, n_support=n_sup, support_width=st['meta'].get('support_width'), seed=a.seed)
    ocr, codec = load_ocr(a.ocr, dev), Codec()
    vgg = torchvision.models.vgg16(num_classes=10400)
    vgg.load_state_dict(torch.hub.load_state_dict_from_url(VGG16_URL, map_location='cpu'))
    vgg = vgg.features.to(dev).eval()

    rows, report = [], []
    for w in batcher.writers[:a.writers]:
        i = batcher.targets[w][0]
        sup_idx = batcher.support_for(i)
        b = batcher.collate([i] * a.n, [sup_idx] * a.n)
        sup = b['support'][0]
        sup_inks = [sup[k, :, :int(b['support_width'][0, k])] for k in range(n_sup)]
        ref = vgg_mean(vgg, sup_inks, dev)
        row = {'writer': w, 'support': sup_inks[0], 'texts': []}
        for t in a.texts:
            g = torch.Generator(device=dev).manual_seed(a.seed)
            with torch.no_grad():
                z, lat_w = model.sample([t] * a.n, b['support'].to(dev), b['support_width'].to(dev), vstate['height'],
                                        a.steps, a.w_content, a.w_style, generator=g, w_mark=a.w_mark)
                img = vae.from_latent(z).float().clamp(0, 1)
            cands = [img[k, 0, :, :max(1, int(lat_w[k].item() * FACTOR) - 1)].cpu() for k in range(a.n)]  # drop the VAE edge column
            x = torch.zeros(a.n, 1, cands[0].shape[0], max(c.shape[-1] for c in cands))
            for k, c in enumerate(cands):
                x[k, 0, :, :c.shape[-1]] = c
            with torch.no_grad(), torch.autocast('cuda', torch.bfloat16):
                hyps = codec.decode(ocr(x.to(dev)).float())
            ed = [align(t.replace(' ', ''), h.replace(' ', ''))[1] for h in hyps]
            hw = [torch.dist(vgg_mean(vgg, [c], dev), ref).item() for c in cands]
            best = min(range(a.n), key=lambda k: (ed[k], hw[k]))
            row['texts'].append({'first': cands[0], 'best': cands[best]})
            report.append(f'writer {w} "{t}": first OCR="{hyps[0]}" ed={ed[0]} | best#{best} OCR="{hyps[best]}" '
                          f'ed={ed[best]} hwd={hw[best]:.3f} | correct {sum(e == 0 for e in ed)}/{a.n}')
        rows.append(row)
        print(report[-len(a.texts):], flush=True)

    # figure: per writer row -> support | for each text: first sample, best-of-N
    s, H = 3, 32 * 3
    to_im = lambda ink: Image.fromarray(((1 - ink.numpy()) * 255).astype(np.uint8)).resize((ink.shape[1] * s, H))
    cols = ['Ảnh mẫu style'] + [f'{t} — {k}' for t in a.texts for k in ('mẫu đầu', f'tốt nhất /{a.n}')]
    widths = [max(to_im(r['support']).width for r in rows)] + [
        max(to_im(r['texts'][ti][k]).width for r in rows) for ti in range(len(a.texts)) for k in ('first', 'best')]
    widths = [max(w, 200) + 30 for w in widths]
    font = ImageFont.truetype(FONT, 20)
    out = Image.new('L', (sum(widths) + 90, 50 + len(rows) * (H + 14)), 255)
    d = ImageDraw.Draw(out)
    x = 90
    for c, wd in zip(cols, widths):
        d.text((x, 12), c, fill=0, font=font); x += wd
    for r_i, r in enumerate(rows):
        y = 50 + r_i * (H + 14)
        d.text((5, y + H // 2 - 10), f"w{r['writer']}", fill=0, font=font)
        ims = [to_im(r['support'])] + [to_im(tt[k]) for tt in r['texts'] for k in ('first', 'best')]
        x = 90
        for im, wd in zip(ims, widths):
            out.paste(im, (x, y)); x += wd
        d.line([(0, y + H + 6), (out.width, y + H + 6)], fill=200)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.save(a.out)
    a.out.with_suffix('.txt').write_text('\n'.join(report) + '\n')
    print('saved', a.out)


if __name__ == '__main__':
    main()
