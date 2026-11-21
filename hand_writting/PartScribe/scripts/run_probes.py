"""Run the pre-registered stage-1 probes on a trained tokenizer checkpoint.

    python scripts/run_probes.py --ckpt runs/tok_slots_strict/best.pt --out runs/tok_slots_strict/probes.json
    python scripts/run_probes.py --pixels --out runs/probes_pixels.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import partscribe.probes as P  # noqa: E402
from partscribe.fonts import load_glyph_bank  # noqa: E402
from partscribe.tokenizer import GlobalAE, PartTokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ckpt', type=Path)
    ap.add_argument('--pixels', action='store_true', help='raw-pixel baseline, no checkpoint')
    ap.add_argument('--bank', default='data/glyphs_v1.npz')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--probe-epochs', type=int, default=40, help='pre-registered value: 40')
    a = ap.parse_args()
    glyphs, chars, fonts = load_glyph_bank(a.bank)
    P.train_probe.__defaults__ = (a.probe_epochs, 'cuda')
    if a.pixels:
        res = {'kind': 'pixels', 'p1': P.run_p1(None, 'pixels', glyphs, chars, fonts)}
    else:
        state = torch.load(a.ckpt, map_location='cuda', weights_only=False)
        model = state['config']['model']
        net = (PartTokenizer() if model == 'slots' else GlobalAE()).cuda()
        net.load_state_dict(state['model'])
        net.eval()
        kind = 'slots' if model == 'slots' else 'global'
        res = {'kind': kind, 'ckpt': str(a.ckpt), 'step': state['step'], 'variant': state['config']['variant'],
               'p1': P.run_p1(net, kind, glyphs, chars, fonts)}
        if kind == 'slots':
            res['p2'] = P.run_p2(net, glyphs, chars, fonts)
    res['probe_epochs'] = a.probe_epochs
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
