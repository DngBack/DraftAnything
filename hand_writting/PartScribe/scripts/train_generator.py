"""Stages 3-4: train the PartScribe generator on a frozen VAE.

    # stage 3, internal baseline (one token per character)
    python scripts/train_generator.py --dataset iam --content char --vae runs/vae_iam64/final.pt --out runs/gen_iam_char
    # stage 4, part tokens
    python scripts/train_generator.py --dataset iam --content parts --table data/content_strict.pt \
        --vae runs/vae_iam64/final.pt --out runs/gen_iam_parts

Holdout protocol v1 is applied by default (images, transcripts; HAT is never
loaded). --no-holdout is only for the full-data IAM parity runs.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from partscribe.content import table_digest  # noqa: E402
from partscribe.datasets import Batcher  # noqa: E402
from partscribe.generator import Config, PartScribe  # noqa: E402
from partscribe.holdout import PROTOCOL_SHA256  # noqa: E402
from partscribe.runtime import EMA, add_data_args, load_split, load_vae, write_json  # noqa: E402
from partscribe.vae import FACTOR  # noqa: E402


def prefetch(it, depth=6):
    """Build batches on a background thread so the GPU does not wait for CPU-side collation
    (numpy/torch release the GIL). Order and content are those of `it` (same seed, same batches)."""
    import queue
    import threading
    q = queue.Queue(depth)

    def work():
        try:
            for b in it:
                for k in ('image', 'support', 'support_width', 'pix_weight'):
                    if k in b:
                        b[k] = b[k].pin_memory()
                q.put(b)
        except BaseException as e:  # surface loader errors in the training thread
            q.put(e)

    threading.Thread(target=work, daemon=True).start()
    while True:
        b = q.get()
        if isinstance(b, BaseException):
            raise b
        yield b


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_data_args(ap)
    ap.add_argument('--content', choices=['parts', 'nfd', 'char'], required=True)
    ap.add_argument('--table', type=Path, help='content table (required for --content parts)')
    ap.add_argument('--vae', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--steps', type=int, default=300000)
    ap.add_argument('--writers-per-batch', type=int, default=16)
    ap.add_argument('--per-writer', type=int, default=4)
    ap.add_argument('--n-support', type=int, default=15)
    ap.add_argument('--support-width', type=int, default=None, help='default 6 x height (VATr-pp: 192 at 32 px)')
    ap.add_argument('--accum', type=int, default=1, help='gradient accumulation (effective batch = batch x accum)')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--warmup', type=int, default=5000)
    ap.add_argument('--ema', type=float, default=0.9999)
    ap.add_argument('--save-every', type=int, default=5000)
    ap.add_argument('--log-every', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--rare-fraction', type=float, default=0.3,
                    help="VATr-pp long-tail sampler: share of rare-weighted target draws (0 = natural frequency)")
    ap.add_argument('--component-mix', type=float, default=0.5, help='VATr-pp ComponentCorpusSampler mix')
    ap.add_argument('--multiword', type=float, default=0.0, help='share of targets built from 2-3 same-writer words')
    ap.add_argument('--mark-weight', type=float, default=0.0, help='extra flow-loss weight on small ink components')
    ap.add_argument('--content-only-per-batch', type=int, default=0,
                    help='VN: HAT2023 words appended to every batch as style-free content targets (VATr-pp HAT task)')
    ap.add_argument('--init-from', type=Path, help='continue from another run checkpoint (model, EMA, optimiser, step)')
    for k, v in Config().__dict__.items():
        if k != 'content':
            ap.add_argument('--' + k.replace('_', '-'), type=type(v) if not isinstance(v, bool) else int, default=v)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    dev = 'cuda'
    cfg = Config(**{k: (bool(getattr(a, k)) if isinstance(v, bool) else getattr(a, k))
                    for k, v in Config().__dict__.items()})
    table = None
    if cfg.content == 'parts':
        assert a.table, '--table required for parts'
        table = torch.load(a.table, map_location='cpu', weights_only=False)
        assert table_digest(table) == table['digest'], 'content table modified'
    vae, vstate = load_vae(a.vae, dev)
    assert vae.z_ch == cfg.z_ch
    model = PartScribe(cfg, table).to(dev)
    train = load_split(a, 'train' if a.dataset == 'vn' else 'train_val')
    assert train.height == vstate['height'], 'VAE trained at a different height'
    assert (train.height // FACTOR) % cfg.patch_h == 0
    batcher = Batcher(train, model.content.supports, a.writers_per_batch, a.per_writer, a.n_support,
                      support_width=a.support_width,
                      width_multiple=FACTOR * cfg.patch_w, max_chars=model.content.max_chars, seed=a.seed,
                      rare_fraction=a.rare_fraction, component_mix=a.component_mix,
                      multiword=a.multiword, mark_weight=a.mark_weight,
                      content_ds=load_split(a, 'hat_ocr') if a.content_only_per_batch else None,
                      content_per_batch=a.content_only_per_batch)
    batches = prefetch(iter(batcher))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / a.warmup))
    ema = EMA(model, a.ema)
    start = 1
    src = a.out / 'last.pt' if a.resume and (a.out / 'last.pt').exists() else a.init_from
    if src:
        st = torch.load(src, map_location='cpu', weights_only=False)
        model.load_state_dict(st['model'])
        ema.shadow.load_state_dict(st['ema'])
        opt.load_state_dict(st['opt'])
        sched.load_state_dict(st['sched'])
        start = st['step'] + 1
        ema.n = st['step']
    meta = {'args': {k: str(v) for k, v in vars(a).items()}, 'config': model.config_dict(),
            'protocol_sha256': None if a.no_holdout else PROTOCOL_SHA256,
            'table_digest': table['digest'] if table else None, 'vae_latent_std': vstate['latent_std'],
            'n_train_images': len(train), 'n_train_writers': len(batcher.writers),
            'writers_per_source': {k: len(v) for k, v in batcher.by_source.items()},
            'support_width': batcher.support_width,
            'params': {n: sum(p.numel() for p in getattr(model, n).parameters())
                       for n in ('content', 'style', 'width', 'dit')}}
    write_json(a.out / 'config.json', meta)
    print(json.dumps(meta['params']), flush=True)
    log, t0, acc = open(a.out / 'log.jsonl', 'a'), time.time(), {}
    for step in range(start, a.steps + 1):
        opt.zero_grad(set_to_none=True)
        parts = {}
        for _ in range(a.accum):
            b = next(batches)
            x = b['image'].to(dev, non_blocking=True)
            with torch.no_grad():
                z0 = vae.to_latent(x)
            lat_w = torch.ceil(b['width'].to(dev).float() / FACTOR)
            z0 = z0 * (torch.arange(z0.shape[-1], device=dev)[None] < lat_w[:, None])[:, None, None]
            with torch.autocast('cuda', torch.bfloat16):
                loss, p = model.loss(z0, lat_w, b['text'], b['support'].to(dev), b['support_width'].to(dev),
                                     b['writer'].to(dev), b['support_text'],
                                     b['pix_weight'].to(dev) if 'pix_weight' in b else None)
            (loss / a.accum).backward()
            for k, v in p.items():
                parts[k] = parts.get(k, 0) + v / a.accum
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        ema.update(model)
        if not math.isfinite(loss.item()):
            raise RuntimeError(f'non-finite loss at step {step}')
        for k, v in parts.items():
            acc[k] = acc.get(k, 0) + v
        if step % a.log_every == 0:
            rec = {'step': step, **{k: v / a.log_every for k, v in acc.items()}, 'grad_norm': gn.item(),
                   'lr': sched.get_last_lr()[0], 'elapsed_s': round(time.time() - t0)}
            acc = {}
            log.write(json.dumps(rec) + '\n')
            log.flush()
            print(json.dumps(rec), flush=True)
        if step % a.save_every == 0 or step == a.steps:
            torch.save({'model': model.state_dict(), 'ema': ema.shadow.state_dict(), 'opt': opt.state_dict(),
                        'sched': sched.state_dict(), 'step': step, 'meta': meta}, a.out / 'last.pt')
    (a.out / 'DONE').write_text(str(a.steps))


if __name__ == '__main__':
    main()
