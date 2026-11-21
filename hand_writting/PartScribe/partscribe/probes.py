"""Stage-1 probes (pre-registered in README, section "Giai đoạn 1").

P1 decomposition: a probe trained on frozen features of *seen* characters in
   train fonts must read (base, mark set) of *held-out* characters in test
   fonts. Slots use a permutation-invariant DeepSets probe; vector features
   (GlobalAE content, raw pixels) use an MLP.
P2 mark binding (slots only): the slot that owns a mark's pixels must be
   mostly that mark (precision >= 0.5) and its FSQ code, looked up in a table
   built on seen characters in train fonts, must name the right mark.
NFD labels are used here only, never in training the tokenizer.
"""
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .charset import MARK_NAMES, composition_chain, parts
from .holdout import held_chars

STYLE_REF_CHAR = 'n'


def label_space(chars):
    letters = [c for c in chars if c.isalpha()]
    bases = sorted({parts(c)[0] for c in letters})
    return letters, bases


def labels(chars_list, bases):
    b = torch.tensor([bases.index(parts(c)[0]) for c in chars_list])
    m = torch.zeros(len(chars_list), len(MARK_NAMES))
    for i, c in enumerate(chars_list):
        for name in parts(c)[1]:
            m[i, MARK_NAMES.index(name)] = 1
    return b, m


@torch.no_grad()
def extract(net, kind, glyphs, font_idx, char_idx, chars, device='cuda', bs=512):
    """Features for every (font, char) pair, fonts-major order."""
    pairs = [(f, c) for f in font_idx for c in char_idx]
    out = []
    for k in range(0, len(pairs), bs):
        chunk = pairs[k:k + bs]
        x = torch.as_tensor(np.stack([glyphs[f, c] for f, c in chunk]), device=device).float().div(255)[:, None]
        if kind == 'pixels':
            out.append(F.avg_pool2d(x, 2).flatten(1).cpu())
        elif kind == 'global':
            out.append(net.content(x).float().cpu())
        else:
            tok = net.tokenize(x)
            out.append(torch.cat([tok['what'], tok['where']], -1).float().cpu())
    return torch.cat(out), pairs


class DeepSets(nn.Module):
    def __init__(self, d_in, n_out, h=128):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(d_in, h), nn.SiLU(), nn.Linear(h, h))
        self.rho = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 256), nn.SiLU(), nn.Linear(256, n_out))

    def forward(self, x):
        return self.rho(self.phi(x).sum(1))


class MLP(nn.Module):
    def __init__(self, d_in, n_out):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, 256), nn.SiLU(), nn.Linear(256, n_out))

    def forward(self, x):
        return self.net(x)


def train_probe(feat, base, marks, n_base, seed, epochs=40, device='cuda'):
    torch.manual_seed(seed)
    probe = (DeepSets(feat.shape[-1], n_base + len(MARK_NAMES)) if feat.dim() == 3
             else MLP(feat.shape[-1], n_base + len(MARK_NAMES))).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    feat, base, marks = feat.to(device), base.to(device), marks.to(device)
    g = torch.Generator(device='cpu').manual_seed(seed)
    for _ in range(epochs):
        for idx in torch.randperm(len(feat), generator=g).split(512):
            idx = idx.to(device)
            out = probe(feat[idx])
            loss = F.cross_entropy(out[:, :n_base], base[idx]) + \
                F.binary_cross_entropy_with_logits(out[:, n_base:], marks[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return probe.eval()


@torch.no_grad()
def score(probe, feat, base, marks, n_base, device='cuda'):
    out = torch.cat([probe(f.to(device)).cpu() for f in feat.split(4096)])
    b_ok = out[:, :n_base].argmax(1) == base
    m_ok = ((out[:, n_base:] > 0).float() == marks).all(1)
    return {'exact': (b_ok & m_ok).float().mean().item(), 'base': b_ok.float().mean().item(),
            'marks': m_ok.float().mean().item(), 'n': len(base)}


def run_p1(net, kind, glyphs, chars, fonts, seeds=(0, 1, 2), device='cuda'):
    letters, bases = label_space(chars)
    held = held_chars('vn')
    seen = [c for c in letters if c not in held]
    tr_fonts = [f['index'] for f in fonts if f['split'] == 'train']
    te_fonts = [f['index'] for f in fonts if f['split'] == 'test']
    ci = lambda cs: [chars.index(c) for c in cs]
    Xtr, ptr = extract(net, kind, glyphs, tr_fonts, ci(seen), chars, device)
    btr, mtr = labels([chars[c] for _, c in ptr], bases)
    evals = {'test_seen': seen, 'test_heldout': sorted(held)}
    from .holdout import vn_categories
    for cat, cs in vn_categories().items():
        evals['test_' + cat] = list(cs)
    feats = {}
    for name, cs in evals.items():
        X, p = extract(net, kind, glyphs, te_fonts, ci(cs), chars, device)
        feats[name] = (X, *labels([chars[c] for _, c in p], bases))
    res = defaultdict(list)
    for s in seeds:
        probe = train_probe(Xtr, btr, mtr, len(bases), s, device=device)
        for name, (X, b, m) in feats.items():
            res[name].append(score(probe, X, b, m, len(bases), device))
    summary = {}
    for name, rs in res.items():
        summary[name] = {k: float(np.mean([r[k] for r in rs])) for k in ('exact', 'base', 'marks')}
        summary[name]['exact_per_seed'] = [r['exact'] for r in rs]
        summary[name]['n'] = rs[0]['n']
    return summary


@torch.no_grad()
def binding_records(net, glyphs, chars, font_idx, char_list, device='cuda'):
    """One record per (font, char, mark step): bound slot code, precision, coverage."""
    recs = []
    sref = chars.index(STYLE_REF_CHAR)
    for f in font_idx:
        cs = [c for c in char_list if parts(c)[1]]
        x = torch.as_tensor(glyphs[f, [chars.index(c) for c in cs]], device=device).float().div(255)[:, None]
        st = torch.as_tensor(glyphs[f, [sref] * len(cs)], device=device).float().div(255)[:, None]
        tok = net.tokenize(x)
        _, alpha, _ = net.decode(tok, st)
        alpha = alpha.float()
        for i, c in enumerate(cs):
            chain, names = composition_chain(c)
            ink = x[i, 0]
            for k, name in enumerate(names):
                cur = torch.as_tensor(glyphs[f, chars.index(chain[k + 1])], device=device).float()
                prev = torch.as_tensor(glyphs[f, chars.index(chain[k])], device=device).float()
                mask = ((cur - prev) / 255 > 0.5).float()
                if mask.sum() < 2:
                    recs.append({'font': f, 'char': c, 'mark': name, 'valid': False})
                    continue
                ov = (alpha[i] * mask).sum((-1, -2))
                s = int(ov.argmax())
                prec = float((alpha[i, s] * ink * mask).sum() / (alpha[i, s] * ink).sum().clamp(min=1e-6))
                cov = float(ov[s] / mask.sum())
                recs.append({'font': f, 'char': c, 'mark': name, 'valid': True, 'slot': s,
                             'code': int(tok['codes'][i, s]), 'precision': prec, 'coverage': cov})
    return recs


def run_p2(net, glyphs, chars, fonts, device='cuda'):
    letters, _ = label_space(chars)
    held = held_chars('vn')
    seen = [c for c in letters if c not in held]
    tr_fonts = [f['index'] for f in fonts if f['split'] == 'train']
    te_fonts = [f['index'] for f in fonts if f['split'] == 'test']
    table = defaultdict(Counter)
    for r in binding_records(net, glyphs, chars, tr_fonts, seen, device):
        if r['valid'] and r['precision'] >= 0.5:
            table[r['code']][r['mark']] += 1
    lookup = {code: cnt.most_common(1)[0][0] for code, cnt in table.items()}
    out = {}
    for name, cs in (('test_seen', seen), ('test_heldout', sorted(held))):
        recs = [r for r in binding_records(net, glyphs, chars, te_fonts, cs, device) if r['valid']]
        ok = [r['precision'] >= 0.5 and lookup.get(r['code']) == r['mark'] for r in recs]
        maj = Counter(r['mark'] for r in recs).most_common(1)[0][1] / len(recs)
        per_mark = defaultdict(list)
        for r, o in zip(recs, ok):
            per_mark[r['mark']].append(o)
        out[name] = {'success': float(np.mean(ok)), 'n': len(recs),
                     'precision_ge_0.5': float(np.mean([r['precision'] >= 0.5 for r in recs])),
                     'mean_coverage': float(np.mean([r['coverage'] for r in recs])),
                     'majority_mark_rate': maj,
                     'per_mark': {k: float(np.mean(v)) for k, v in sorted(per_mark.items())}}
    out['codes_in_table'] = len(lookup)
    return out
