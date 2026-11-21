"""Freeze a stage-1 tokenizer into a content table for the generator.

Reference fonts = the first --n-fonts *train-split* fonts in a seeded random
order (test fonts stay unseen by every later stage).

    python scripts/build_content_table.py --ckpt runs/tok_slots_strict/best.pt --out data/content_strict.pt
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from partscribe.content import build_content_table, table_digest  # noqa: E402
from partscribe.fonts import load_glyph_bank  # noqa: E402
from partscribe.tokenizer import PartTokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ckpt', type=Path, required=True)
    ap.add_argument('--bank', default='data/glyphs_v1.npz')
    ap.add_argument('--n-fonts', type=int, default=32)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    state = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    assert state['config']['model'] == 'slots', 'content table needs the slot tokenizer'
    net = PartTokenizer()
    net.load_state_dict(state['model'])
    glyphs, chars, fonts = load_glyph_bank(a.bank)
    train = [f['index'] for f in fonts if f['split'] == 'train']
    order = np.random.default_rng(a.seed).permutation(len(train))
    picked = [train[i] for i in order[:a.n_fonts]]
    table = build_content_table(net, glyphs, chars, picked)
    table.update(tokenizer_ckpt=str(a.ckpt), tokenizer_sha256=hashlib.sha256(a.ckpt.read_bytes()).hexdigest(),
                 tokenizer_variant=state['config']['variant'], font_files=[fonts[i]['file'] for i in picked])
    table['digest'] = table_digest(table)
    torch.save(table, a.out)
    v = table['valid'].float().sum(-1)
    print(f'{len(picked)} fonts x {len(chars)} chars, slots/char mean {v.mean():.2f} '
          f'min {v.min():.0f} max {v.max():.0f}; digest {table["digest"][:12]}')


if __name__ == '__main__':
    main()
