"""Shared helpers for the training/sampling scripts."""
import copy
import json
from pathlib import Path

import torch

from .datasets import load_iam, load_vn
from .vae import VAE

VN_TRAIN_SHA = '95d2aa9d849553f94ee99346f1faf7acb1c5f4a10ab73100d5f7c1d08faf99e8'
VN_TEST_SHA = '9814da1acc22bd0ddd21cfee9183bf49a7e62faaf7ea55057b42e883b1c37a28'
VATR = Path(__file__).resolve().parents[2] / 'VATr-pp'


def add_data_args(ap):
    ap.add_argument('--dataset', choices=['vn', 'iam'], required=True)
    ap.add_argument('--vn-data', default=None, help='folder with train.h5/test.h5 (VATr-pp fingerprints)')
    ap.add_argument('--vn-manifest', default=str(VATR / 'files/vietnamese/manifest.json'))
    ap.add_argument('--iam-packed', default='data/iam64.npz')
    ap.add_argument('--no-holdout', action='store_true', help='full-data run (IAM parity only)')


def load_split(a, split):
    if a.dataset == 'vn':
        sha = VN_TEST_SHA if split == 'test' else VN_TRAIN_SHA
        return load_vn(a.vn_manifest, a.vn_data, split, not a.no_holdout, verify_sha=sha)
    return load_iam(a.iam_packed, split, not a.no_holdout)


class EMA:
    """EMA with warm-up: decay_n = min(decay, (1 + n) / (10 + n)).

    Without warm-up a 0.9999 EMA still holds ~37% of the random init after 10k
    steps (found on gen_vn_char, 18/11/2026: EMA samples were noise while the raw
    weights already wrote readable words).
    """

    def __init__(self, model, decay=0.9999, n=0):
        self.decay, self.n = decay, n
        self.shadow = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        ep = list(self.shadow.parameters())
        torch._foreach_lerp_(ep, [p.detach() for p in model.parameters()], 1 - d)
        eb = list(self.shadow.buffers())
        if eb:
            torch._foreach_copy_(eb, list(model.buffers()))


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=1))


def load_vae(path, dev):
    s = torch.load(path, map_location='cpu', weights_only=False)
    assert 'latent_std' in s, 'use final.pt (with measured latent scale)'
    vae = VAE(s['z_ch'])
    vae.load_state_dict(s['model'])
    return vae.to(dev).eval().requires_grad_(False), s
