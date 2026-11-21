"""Train the stage-1 part tokenizer (or the GlobalAE baseline) on the font glyph bank.

    python scripts/train_tokenizer.py --model slots --variant strict --out runs/tok_slots_strict
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from partscribe.data import GlyphSampler  # noqa: E402
from partscribe.fonts import load_glyph_bank  # noqa: E402
from partscribe.holdout import held_chars  # noqa: E402
from partscribe.tokenizer import GlobalAE, PartTokenizer, recon_loss  # noqa: E402


def build(model):
    return PartTokenizer() if model == 'slots' else GlobalAE()


@torch.no_grad()
def validate(net, sampler, n_batches=8, bs=256):
    net.eval()
    rng_state = sampler.rng.bit_generator.state
    sampler.rng = np.random.default_rng(1234)  # fixed validation set
    loss, iou, used = 0.0, 0.0, []
    for _ in range(n_batches):
        x, y, s, _ = sampler.sample(bs)
        with torch.autocast('cuda', torch.bfloat16):
            logp, tok, alpha = net(x, s)
        logp = logp.float()
        loss += recon_loss(logp, y).item()
        p, t = logp.exp() > 0.5, y > 0.5
        iou += ((p & t).sum((1, 2)) / (p | t).sum((1, 2)).clamp(min=1)).mean().item()
        if alpha is not None:
            ink = y[:, None]
            share = (alpha.float() * ink).sum((-1, -2)) / ink.sum((-1, -2)).clamp(min=1)
            used.append((share > 0.05).sum(1).float().mean().item())
    sampler.rng.bit_generator.state = rng_state
    net.train()
    return {'val_loss': loss / n_batches, 'val_iou': iou / n_batches,
            'slots_used': float(np.mean(used)) if used else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', choices=['slots', 'global'], required=True)
    ap.add_argument('--variant', choices=['strict', 'full'], required=True)
    ap.add_argument('--bank', default='data/glyphs_v1.npz')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--warmup', type=int, default=1000)
    ap.add_argument('--eval-every', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    dev = 'cuda'
    glyphs, chars, fonts = load_glyph_bank(a.bank)
    train = GlyphSampler(glyphs, chars, fonts, 'train', a.variant, dev, seed=a.seed)
    val = GlyphSampler(glyphs, chars, fonts, 'val', a.variant, dev, seed=a.seed)
    if a.variant == 'strict':
        assert held_chars('vn').isdisjoint(train.chars) and held_chars('vn').isdisjoint(val.chars)
    net = build(a.model).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1, s / a.steps))))
    config = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
              'train_chars': ''.join(train.chars),
              'train_chars_sha256': hashlib.sha256(''.join(train.chars).encode()).hexdigest(),
              'n_train_fonts': len(train.font_ids), 'n_val_fonts': len(val.font_ids),
              'params': sum(p.numel() for p in net.parameters())}
    (a.out / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=1))
    log = open(a.out / 'log.jsonl', 'a')
    best, t0 = float('inf'), time.time()
    for step in range(1, a.steps + 1):
        x, y, s, _ = train.sample(a.bs)
        with torch.autocast('cuda', torch.bfloat16):
            logp, _, _ = net(x, s)
        loss = recon_loss(logp.float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if not torch.isfinite(loss):
            raise RuntimeError(f'non-finite loss at step {step}')
        if step % a.eval_every == 0 or step == a.steps:
            rec = {'step': step, 'loss': loss.item(), 'grad_norm': gn.item(), 'lr': sched.get_last_lr()[0],
                   'elapsed_s': round(time.time() - t0), **validate(net, val)}
            log.write(json.dumps(rec) + '\n')
            log.flush()
            print(json.dumps(rec), flush=True)
            state = {'model': net.state_dict(), 'config': config, 'step': step, 'val': rec}
            torch.save(state, a.out / 'last.pt')
            if rec['val_loss'] < best:
                best = rec['val_loss']
                torch.save(state, a.out / 'best.pt')
    (a.out / 'DONE').write_text(json.dumps({'best_val_loss': best, 'steps': a.steps}))


if __name__ == '__main__':
    main()
