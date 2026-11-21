"""Train the independent evaluator OCR (never used by any generator).

Vietnamese: VATr-pp train split WITHOUT the holdout filter + HAT2023 OCR split,
sources sampled equally (as VATr-pp's OCR warm-up). Validation = the 30 dev
writers (full val, held-out words included) -> also gives the real-image
upper bound on held-out characters.

    python scripts/train_evaluator.py --dataset vn --vn-data data/vn --out runs/eval_ocr_vn
"""
import argparse
import json
import random
import sys
import time
import unicodedata as ud
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from partscribe.datasets import fit_width, to_ink  # noqa: E402
from partscribe.evaluator import ARCHS, Codec, align, ctc_loss  # noqa: E402
from partscribe.holdout import held_chars  # noqa: E402
from partscribe.runtime import VATR, write_json  # noqa: E402

MAX_W = 512


def load(data_dir, manifest, splits):
    man = json.loads(Path(manifest).read_text())
    with h5py.File(Path(data_dir) / 'train.h5', 'r') as f:
        px, st, wd, wid = f['imgs'][:], f['img_seek_idxs'][:], f['img_lens'][:], f['wids'][:]
        lbs, ls, ll = f['lbs'][:], f['lb_seek_idxs'][:], f['lb_lens'][:]
    items = []
    for sp in splits:
        for i in man['indices'][sp]:
            t = ud.normalize('NFC', ''.join(map(chr, lbs[ls[i]:ls[i] + ll[i]])))
            src = 0 if wid[i] < 106 else (1 if wid[i] < 295 else 2)
            items.append((int(st[i]), int(wd[i]), t, src))
    return px, items


def batch(px, items, codec):
    imgs = [fit_width(px[:, a:a + w], MAX_W) for a, w, _, _ in items]
    W = max(i.shape[1] for i in imgs)
    x = torch.zeros(len(imgs), 1, px.shape[0], W)
    for k, im in enumerate(imgs):
        x[k, 0, :, :im.shape[1]] = to_ink(im)
    texts = [t for _, _, t, _ in items]
    return x, torch.tensor([i.shape[1] for i in imgs]), texts


@torch.no_grad()
def evaluate(net, px, items, codec, dev, bs=256):
    net.eval()
    held = held_chars('vn')
    edits = chars = h_ok = h_n = 0
    for k in range(0, len(items), bs):
        x, w, texts = batch(px, items[k:k + bs], codec)
        with torch.autocast('cuda', torch.bfloat16):
            pred = codec.decode(net(x.to(dev)).float())
        for ref, hyp in zip(texts, pred):
            al, e = align(ref, hyp)
            edits, chars = edits + e, chars + len(ref)
            for c, h in zip(ref, al):
                if c in held:
                    h_n += 1
                    h_ok += h == c
    net.train()
    return {'cer': edits / chars, 'heldout_char_acc': h_ok / max(h_n, 1), 'heldout_n': h_n}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', choices=['vn'], default='vn')
    ap.add_argument('--vn-data', required=True)
    ap.add_argument('--vn-manifest', default=str(VATR / 'files/vietnamese/manifest.json'))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--steps', type=int, default=60000)
    ap.add_argument('--bs', type=int, default=96)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--eval-every', type=int, default=5000)
    ap.add_argument('--arch', choices=list(ARCHS), default='crnn')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    dev = 'cuda'
    random.seed(0)
    torch.manual_seed(0)
    codec = Codec()
    px, train = load(a.vn_data, a.vn_manifest, ['train', 'hat_ocr'])
    _, val = load(a.vn_data, a.vn_manifest, ['val'])
    train = [it for it in train if all(c in codec.index for c in it[2]) and it[2]]
    val = [it for it in val if all(c in codec.index for c in it[2]) and it[2]]
    by_src = {s: [it for it in train if it[3] == s] for s in (0, 1, 2)}
    net = ARCHS[a.arch](px.shape[0]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    write_json(a.out / 'config.json', {**{k: str(v) for k, v in vars(a).items()},
                                       'n_train': {k: len(v) for k, v in by_src.items()}, 'n_val': len(val),
                                       'note': 'full train incl. held-out characters + HAT; never val/test'})
    log, best, t0 = open(a.out / 'log.jsonl', 'a'), float('inf'), time.time()
    for step in range(1, a.steps + 1):
        items = [random.choice(by_src[random.randrange(3)]) for _ in range(a.bs)]
        x, w, texts = batch(px, items, codec)
        tg, tl = codec.encode(texts)
        with torch.autocast('cuda', torch.bfloat16):
            logits = net(x.to(dev))
        loss = ctc_loss(logits, tg, tl, w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        sched.step()
        if step % a.eval_every == 0 or step == a.steps:
            rec = {'step': step, 'loss': loss.item(), **evaluate(net, px, val, codec, dev),
                   'elapsed_s': round(time.time() - t0)}
            log.write(json.dumps(rec) + '\n')
            log.flush()
            print(json.dumps(rec), flush=True)
            state = {'model': net.state_dict(), 'height': px.shape[0], 'arch': a.arch, 'step': step, 'val': rec}
            torch.save(state, a.out / 'last.pt')
            if rec['cer'] < best:
                best = rec['cer']
                torch.save(state, a.out / 'best.pt')
    (a.out / 'DONE').write_text(json.dumps({'best_val_cer': best}))


if __name__ == '__main__':
    main()
