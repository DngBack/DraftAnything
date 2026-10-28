"""Step 2: frozen encoder features for every mention, plus candidate tables. One pass per split.

cache/{split}.pt: emb [M_total, 768] fp16 (bge-base CLS, L2-normalised), docs: list of candidate tables
with mention indices offset into emb. train is split 85/15 into train/val with a fixed seed;
dev-0 is the test set (test-A labels are not public).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cemkie.data import FIELDS, FTYPE, candidates, context, load_split

CACHE = Path(__file__).resolve().parents[1] / 'cache'
ENCODER = 'BAAI/bge-base-en-v1.5'
N_FEAT = 9


def features(c, text_len, f):
    """Per-candidate scalar features, identical for every variant."""
    fc = c['fields'][f]
    m, s = fc['cand_m'], fc['cand_scale']
    C = len(m)
    if C == 0:
        return np.zeros((0, N_FEAT), np.float32)
    v = np.array([float(x) if FTYPE[f] == 'money' else 0. for x in fc['cand_v']])
    uniq, inv, cnt = np.unique(fc['cand_v'], return_inverse=True, return_counts=True)
    rank = np.zeros(C)
    if FTYPE[f] == 'money':
        rank = np.argsort(np.argsort(v)) / max(C - 1, 1)
    return np.stack([
        c['spans'][m, 0] / max(text_len, 1),
        np.arange(C) / max(C - 1, 1),
        np.log1p(v) / 20,
        s.astype(float),
        c['flags'][m, 0], c['flags'][m, 1],
        np.log1p(cnt[inv]) / 5,
        rank,
        np.full(C, np.log1p(text_len) / 15),
    ], 1).astype(np.float32)


@torch.no_grad()
def encode(texts, tok, model, bs=512):
    out = []
    for i in range(0, len(texts), bs):
        x = tok(texts[i:i + bs], padding=True, truncation=True, max_length=128, return_tensors='pt').to('cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            h = model(**x).last_hidden_state[:, 0]
        out.append(torch.nn.functional.normalize(h.float(), dim=-1).half().cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', nargs='+', default=['train', 'dev-0'])
    args = ap.parse_args()
    CACHE.mkdir(exist_ok=True)
    tok = AutoTokenizer.from_pretrained(ENCODER)
    model = AutoModel.from_pretrained(ENCODER).cuda().eval()
    for split in args.splits:
        t0 = time.time()
        docs = load_split(split)
        tables, ctx, off = [], [], 0
        for d in docs:
            c = candidates(d)
            ctx += [context(d['text'], s, e) for s, e in c['spans']]
            fields = {}
            for f in FIELDS:
                fc = c['fields'][f]
                vals, vid = np.unique(np.array(fc['cand_v'], dtype=object).astype(str), return_inverse=True) \
                    if len(fc['cand_v']) else (np.array([]), np.zeros(0, np.int64))
                fields[f] = dict(m=torch.from_numpy(fc['cand_m'] + off), vid=torch.from_numpy(vid.astype(np.int64)),
                                 values=list(vals), label=torch.from_numpy(fc['label']),
                                 feat=torch.from_numpy(features(c, len(d['text']), f)))
            tables.append(dict(id=d['id'], gold=d['gold'], chars=len(d['text']), n_mentions=len(c['spans']),
                               fields=fields))
            off += len(c['spans'])
        print(split, len(docs), 'docs', off, 'mentions', f'{time.time() - t0:.0f}s candidates', flush=True)
        emb = encode(ctx, tok, model)
        print(split, 'encoded', tuple(emb.shape), f'{time.time() - t0:.0f}s', flush=True)
        torch.save(dict(emb=emb, docs=tables, encoder=ENCODER), CACHE / f'{split}.pt')


if __name__ == '__main__':
    main()
