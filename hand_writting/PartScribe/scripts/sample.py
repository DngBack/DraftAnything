"""Generate words in the style of a writer from a held-out split (EMA weights).

    python scripts/sample.py --dataset iam --ckpt runs/gen_iam_parts/last.pt --vae runs/vae_iam64/final.pt \
        --split test --texts "jumping quietly" "Yesterday" --out runs/gen_iam_parts/samples.png

Support images come from the evaluation split and are filtered by the same
holdout rule, so a support never contains a held-out character.
"""
import argparse
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from partscribe.datasets import Batcher  # noqa: E402
from partscribe.generator import Config, PartScribe  # noqa: E402
from partscribe.runtime import add_data_args, load_split, load_vae  # noqa: E402
from partscribe.vae import FACTOR  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_data_args(ap)
    ap.add_argument('--ckpt', type=Path, required=True)
    ap.add_argument('--vae', type=Path, required=True)
    ap.add_argument('--table', type=Path)
    ap.add_argument('--split', default='val')
    ap.add_argument('--texts', nargs='+', required=True)
    ap.add_argument('--n-writers', type=int, default=8)
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--w-content', type=float, default=3.0)
    ap.add_argument('--w-style', type=float, default=1.5)
    ap.add_argument('--w-mark', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--raw', action='store_true', help='use the raw training weights instead of the EMA')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    dev = 'cuda'
    st = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    cfg = Config(**st['meta']['config'])
    table = torch.load(a.table, map_location='cpu', weights_only=False) if cfg.content == 'parts' else None
    model = PartScribe(cfg, table).to(dev).eval()
    model.load_state_dict(st['model'] if a.raw else st['ema'])
    vae, vstate = load_vae(a.vae, dev)
    ds = load_split(a, a.split)
    n_sup = int(st['meta']['args'].get('n_support', 15))
    batcher = Batcher(ds, lambda t: True, n_support=n_sup, support_width=st['meta'].get('support_width'), seed=a.seed)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    rows = []
    for w in batcher.writers[:a.n_writers]:
        i = batcher.targets[w][0]
        sup = batcher.support_for(i)
        b = batcher.collate([i] * len(a.texts), [sup] * len(a.texts))
        z, lat_w = model.sample(a.texts, b['support'].to(dev), b['support_width'].to(dev), vstate['height'],
                                a.steps, a.w_content, a.w_style, generator=gen, w_mark=a.w_mark)
        img = vae.from_latent(z).float().clamp(0, 1)
        # The VAE is only trained on valid columns; blank everything beyond each word's predicted width.
        valid = torch.arange(img.shape[-1], device=dev)[None] < (lat_w * FACTOR)[:, None]
        img = img * valid[:, None, None]
        ref = 1 - b['support'][:1, :1].to(dev)                     # one support image of this writer
        W = max(ref.shape[-1], img.shape[-1])
        pad = lambda t: torch.nn.functional.pad(t, (0, W - t.shape[-1]), value=1)
        rows.append(torch.cat([pad(ref), pad(1 - img)]))
    save_image(torch.cat(rows), a.out, nrow=1, padding=2, pad_value=0.5)
    print(f'saved {a.out}: per writer one support image, then {len(a.texts)} generated words')


if __name__ == '__main__':
    main()
