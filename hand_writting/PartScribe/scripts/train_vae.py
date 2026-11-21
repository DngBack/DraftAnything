"""Stage 2: train the f=4 handwriting VAE, then measure the latent scale.

    python scripts/train_vae.py --dataset iam --out runs/vae_iam64
    python scripts/train_vae.py --dataset vn --vn-data /path/new_data --out runs/vae_vn32

Gate (plan, stage 2): reconstruction must keep the text readable; check
recon_examples.png and run the independent evaluator OCR on reconstructions.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from partscribe.datasets import Batcher  # noqa: E402
from partscribe.runtime import add_data_args, load_split, write_json  # noqa: E402
from partscribe.vae import VAE, PatchDisc, col_mask, recon_losses  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_data_args(ap)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--steps', type=int, default=100000)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--z-ch', type=int, default=4)
    ap.add_argument('--w-edge', type=float, default=1.0)
    ap.add_argument('--w-adv', type=float, default=0.05)
    ap.add_argument('--adv-start', type=int, default=20000)
    ap.add_argument('--eval-every', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    dev = 'cuda'
    train = load_split(a, 'train' if a.dataset == 'vn' else 'train_val')
    val = load_split(a, 'val') if a.dataset == 'vn' else None
    batches = iter(Batcher(train, lambda t: True, writers_per_batch=a.bs, per_writer=1, n_support=1, seed=a.seed))
    vae, disc = VAE(a.z_ch).to(dev), PatchDisc().to(dev)
    opt = torch.optim.AdamW(vae.parameters(), lr=a.lr, betas=(0.5, 0.9), weight_decay=0)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=a.lr, betas=(0.5, 0.9), weight_decay=0)
    write_json(a.out / 'config.json', {**{k: str(v) for k, v in vars(a).items()}, 'height': train.height,
                                       'n_train': len(train)})
    log, t0 = open(a.out / 'log.jsonl', 'a'), time.time()
    for step in range(1, a.steps + 1):
        b = next(batches)
        x, wpx = b['image'].to(dev), b['width'].to(dev)
        with torch.autocast('cuda', torch.bfloat16):
            rec, mu, logvar = vae(x)
        L = recon_losses(rec.float(), x, mu.float(), logvar.float(), wpx)
        loss = L['l1'] + a.w_edge * L['edge'] + L['kl']
        adv_on = step >= a.adv_start
        if adv_on:
            m = col_mask(wpx, x.shape[-1], dev)
            with torch.autocast('cuda', torch.bfloat16):
                g_adv = -(disc(rec * m)).float().mean()
            loss = loss + a.w_adv * g_adv
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
        opt.step()
        if adv_on:
            m = col_mask(wpx, x.shape[-1], dev)
            with torch.autocast('cuda', torch.bfloat16):
                d_loss = F.relu(1 - disc(x * m)).float().mean() + F.relu(1 + disc(rec.detach() * m)).float().mean()
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()
        if not math.isfinite(loss.item()):
            raise RuntimeError(f'non-finite loss at step {step}')
        if step % a.eval_every == 0 or step == a.steps:
            rec_ = {'step': step, **{k: v.item() for k, v in L.items()}, 'elapsed_s': round(time.time() - t0)}
            log.write(json.dumps(rec_) + '\n')
            log.flush()
            print(json.dumps(rec_), flush=True)
            save_image(torch.cat([1 - x[:16], 1 - rec[:16].float()]), a.out / 'recon_examples.png', nrow=1)
            torch.save({'model': vae.state_dict(), 'z_ch': a.z_ch, 'height': train.height, 'step': step},
                       a.out / 'last.pt')
    # latent scale: 1 / std of the posterior mean over valid columns
    vae.eval()
    src = val if val is not None else train
    it = iter(Batcher(src, lambda t: True, writers_per_batch=a.bs, per_writer=1, n_support=1, seed=123))
    vals = []
    with torch.no_grad():
        for _ in range(50):
            b = next(it)
            mu, _ = vae.encode(b['image'].to(dev))
            m = (torch.arange(mu.shape[-1], device=dev)[None] < (b['width'].to(dev) / 4).ceil()[:, None])
            vals.append(mu[m[:, None, None, :].expand_as(mu)].float().cpu())
    std = torch.cat(vals).std().item()
    vae.latent_scale.fill_(1.0 / std)
    torch.save({'model': vae.state_dict(), 'z_ch': a.z_ch, 'height': train.height, 'step': a.steps,
                'latent_std': std}, a.out / 'final.pt')
    print(f'latent std {std:.4f}; saved final.pt')


if __name__ == '__main__':
    main()
