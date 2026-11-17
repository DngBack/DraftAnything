"""Second, independent fidelity metric: a writer-identification embedding.

Trained ONLY on IAM official *train* writers (283, disjoint from the 128 test writers used in H1).
ResNet18 classifier on 32-px word images; the 512-d penultimate feature (L2-normalised) is the embedding.
`python writer_id.py train --train <iam_train> --test <iam_test> --out <ckpt>` also reports real-data
validity on test writers (leave-one-out nearest-centroid writer retrieval).
"""
import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from h1_lib import load_words, ink_crop, to_h32  # noqa: E402

W = 256


def prep(a, train=False):
    a = to_h32(ink_crop(a))
    x = 1 - a.astype(np.float32) / 255  # ink = 1
    if x.shape[1] > W:
        s = random.randint(0, x.shape[1] - W) if train else 0
        x = x[:, s:s + W]
    out = np.zeros((32, W), np.float32)
    off = random.randint(0, W - x.shape[1]) if train else 0
    out[:, off:off + x.shape[1]] = x
    return torch.from_numpy(out)[None]


class Net(nn.Module):
    def __init__(self, n):
        super().__init__()
        m = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        m.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        m.fc = nn.Identity()
        self.body, self.head = m, nn.Linear(512, n)

    def embed(self, x):
        return nn.functional.normalize(self.body(x), dim=1)

    def forward(self, x):
        return self.head(self.body(x))


def load(ckpt, dev):
    s = torch.load(ckpt, map_location='cpu')
    n = Net(s['n']).to(dev).eval()
    n.load_state_dict(s['state'])
    return n


@torch.no_grad()
def embed_arrays(net, arrays, dev, bs=256):
    out = []
    for i in range(0, len(arrays), bs):
        out.append(net.embed(torch.stack([prep(a) for a in arrays[i:i + bs]]).to(dev)).cpu())
    return torch.cat(out)


def validate(net, rows, dev):
    rows = [r for r in rows if r['text'].isalpha() and len(r['text']) >= 3]
    E = embed_arrays(net, [np.array(Image.open(r['path']).convert('L')) for r in rows], dev)
    wr = np.array([r['writer'] for r in rows])
    ws = sorted(set(wr))
    sums = {w: E[wr == w].sum(0) for w in ws}
    cnt = {w: (wr == w).sum() for w in ws}
    hit = 0
    for i, w in enumerate(wr):
        cents = torch.stack([(sums[x] - (E[i] if x == w else 0)) / (cnt[x] - (1 if x == w else 0)) for x in ws])
        hit += ws[int(torch.argmax(cents @ E[i]))] == w
    return hit / len(wr), len(ws)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['train'])
    p.add_argument('--train', required=True)
    p.add_argument('--test', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=8)
    a = p.parse_args()
    random.seed(0)
    torch.manual_seed(0)
    dev = 'cuda'
    rows = [r for r in load_words(a.train) if len(r['text']) >= 2]
    writers = sorted({r['writer'] for r in rows})
    wi = {w: i for i, w in enumerate(writers)}
    imgs = [np.array(Image.open(r['path']).convert('L')) for r in rows]
    ys = torch.tensor([wi[r['writer']] for r in rows])
    net = Net(len(writers)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    steps = a.epochs * (len(rows) // 128)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
    t = 0
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(len(rows))
        tot, acc = 0., 0.
        for b in range(0, len(rows) - 127, 128):
            idx = perm[b:b + 128]
            x = torch.stack([prep(imgs[i], train=True) for i in idx]).to(dev)
            y = ys[idx].to(dev)
            out = net(x)
            loss = nn.functional.cross_entropy(out, y, label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            t += 1
            tot += loss.item()
            acc += (out.argmax(1) == y).float().mean().item()
        nb = len(rows) // 128
        print(f'epoch {ep} loss {tot / nb:.3f} train-acc {acc / nb:.3f}', flush=True)
    net.eval()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(n=len(writers), state=net.state_dict(), writers=writers), a.out)
    top1, nw = validate(net, load_words(a.test), dev)
    print(f'REAL-DATA VALIDITY on unseen test writers: leave-one-out word->writer top-1 = {top1:.3f} '
          f'among {nw} writers (chance {1 / nw:.3f})')


if __name__ == '__main__':
    main()
