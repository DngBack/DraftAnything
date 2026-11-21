"""Held-out-composition evaluation (report_04 section 2.4) on the Vietnamese dev writers.

Targets: every val transcript containing a held-out character (C1/C2/C3), plus
a control set of val words with only seen characters (same writers, same count).
Support: 15 images of the SAME writer from the holdout-filtered pool, never the
target transcript. Scored by the independent evaluator OCR; real val images
are scored too (upper bound).

    python scripts/eval_holdout.py run --ckpt runs/gen_vn_nfd/last.pt --vae runs/vae_vn32/final.pt \
        --ocr runs/eval_ocr_vn/best.pt --raw --out runs/gen_vn_nfd/eval_val_50k.json
    python scripts/eval_holdout.py compare runs/gen_vn_char/eval_val_50k.json runs/gen_vn_nfd/eval_val_50k.json
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from partscribe.datasets import fit_width, load_vn, to_ink  # noqa: E402
from partscribe.evaluator import Codec, align, load_ocr, sibling  # noqa: E402
from partscribe.generator import Config, PartScribe, strip_marks  # noqa: E402
from partscribe.holdout import held_chars, vn_categories  # noqa: E402
from partscribe.runtime import VATR, load_vae  # noqa: E402
from partscribe.vae import decode_generated  # noqa: E402

CATS = {c: k for k, cs in vn_categories().items() for c in cs}


def ocr_images(ocr, codec, imgs, dev, bs=128):
    out = []
    for k in range(0, len(imgs), bs):
        chunk = imgs[k:k + bs]
        W = max(i.shape[-1] for i in chunk)
        x = torch.zeros(len(chunk), 1, chunk[0].shape[-2], W)
        for j, im in enumerate(chunk):
            x[j, 0, :, :im.shape[-1]] = im
        with torch.no_grad(), torch.autocast('cuda', torch.bfloat16):
            out += codec.decode(ocr(x.to(dev)).float())
    return out


def score(ref, hyp, kind, hyp_plain=None):
    """hyp_plain: OCR of the same word generated WITHOUT marks (same writer, noise, width). With it, a
    held-out position also gets pair_ok = read correctly here AND not read as that character there,
    which an OCR that fills marks in from context cannot fake."""
    al, edits = align(ref, hyp)
    al_plain = align(strip_marks(ref), hyp_plain)[0] if hyp_plain is not None else None
    rec = {'edits': edits, 'len': len(ref), 'pos': []}
    held = held_chars('vn')
    for k, (c, h) in enumerate(zip(ref, al)):
        if kind == 'heldout' and c in held:
            p = {'cat': CATS[c], 'char': c, 'ok': h == c, 'sibling': sibling(c, h)}
            if al_plain is not None:
                p['pair_ok'] = (h == c) and (al_plain[k] != c)
            rec['pos'].append(p)
        elif kind == 'control' and c.isalpha():
            rec['pos'].append({'cat': 'control', 'char': c, 'ok': h == c, 'sibling': sibling(c, h)})
    return rec


def summarise(records, key='gen', n_boot=1000, seed=0):
    """Per-category char accuracy / sibling rate with writer-bootstrap 95% CI."""
    by_cat = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0, 0]))   # cat -> writer -> [ok, sib, n, pair_ok, n_pair]
    for r in records:
        for p in r[key]['pos']:
            for cat in (p['cat'], 'C1+C2' if p['cat'] in ('C1_tone_on_new_base', 'C2_stacked_pair_seen_on_other_base') else None):
                if cat:
                    v = by_cat[cat][r['writer']]
                    v[0] += p['ok']; v[1] += p['sibling']; v[2] += 1
                    if 'pair_ok' in p:
                        v[3] += p['pair_ok']; v[4] += 1
    rng = np.random.default_rng(seed)
    out = {}
    for cat, ws in sorted(by_cat.items()):
        arr = np.array(list(ws.values()), float)
        acc = arr[:, 0].sum() / arr[:, 2].sum()
        boots = []
        for _ in range(n_boot):
            s = arr[rng.integers(len(arr), size=len(arr))]
            boots.append(s[:, 0].sum() / s[:, 2].sum())
        out[cat] = {'acc': acc, 'ci95': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                    'sibling_rate': arr[:, 1].sum() / arr[:, 2].sum(), 'n_pos': int(arr[:, 2].sum()), 'n_writers': len(arr)}
        if arr[:, 4].sum():
            out[cat]['pair_acc'] = arr[:, 3].sum() / arr[:, 4].sum()
    cer = sum(r[key]['edits'] for r in records) / sum(r[key]['len'] for r in records)
    out['word_cer_all'] = cer
    return out


def run(a):
    dev = 'cuda'
    rng = random.Random(a.seed)
    full = load_vn(a.vn_manifest, a.vn_data, a.split, apply_holdout=False)
    clean = load_vn(a.vn_manifest, a.vn_data, a.split, apply_holdout=True)
    st = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    cfg = Config(**st['meta']['config'])
    table = torch.load(a.table, map_location='cpu', weights_only=False) if cfg.content == 'parts' else None
    model = PartScribe(cfg, table).to(dev).eval()
    model.load_state_dict(st['model'] if a.raw else st['ema'])
    vae, vstate = load_vae(a.vae, dev)
    ocr = load_ocr(a.ocr, dev)
    codec = Codec()
    held = held_chars('vn')
    ok = lambda t: model.content.supports(t) and all(c in codec.index for c in t) and 0 < len(t) <= 40
    targets = [i for i, t in enumerate(full.texts) if held & set(t) and ok(t)]
    writers = sorted({int(full.writers[i]) for i in targets})
    ctrl_pool = defaultdict(list)
    for j, t in enumerate(clean.texts):
        if len(t) >= 2 and ok(t) and any(c.isalpha() for c in t):
            ctrl_pool[int(clean.writers[j])].append(j)
    jobs = [('heldout', full, i) for i in targets]
    for i in targets:   # one control word per held-out target, same writer
        w = int(full.writers[i])
        jobs.append(('control', clean, rng.choice(ctrl_pool[w])))
    support_w = st['meta'].get('support_width') or 6 * full.height
    n_sup = int(st['meta']['args'].get('n_support', 15))
    records = []
    for k in range(0, len(jobs), a.bs):
        chunk = jobs[k:k + a.bs]
        texts, sups, sws, reals, metas = [], [], [], [], []
        for kind, ds, i in chunk:
            t, w = ds.texts[i], int(ds.writers[i])
            cand = [j for j in clean.pools[w] if clean.texts[j] != t]
            pick = rng.sample(cand, n_sup) if len(cand) >= n_sup else [rng.choice(cand) for _ in range(n_sup)]
            s = torch.zeros(n_sup, full.height, support_w)
            sw = torch.zeros(n_sup)
            for m, j in enumerate(pick):
                im = fit_width(clean.image(j), support_w)
                s[m, :, :im.shape[1]] = to_ink(im)
                sw[m] = im.shape[1]
            texts.append(t); sups.append(s); sws.append(sw)
            reals.append(to_ink(ds.image(i)))
            metas.append({'kind': kind, 'text': t, 'writer': w, 'index': int(ds.ids[i])})
        gen = torch.Generator(device=dev).manual_seed(a.seed + k)
        z, lat_w = model.sample(texts, torch.stack(sups).to(dev), torch.stack(sws).to(dev), full.height,
                                a.steps, a.w_content, a.w_style, generator=gen, w_mark=a.w_mark)
        img, wpx = decode_generated(vae, z, lat_w)
        gens = [img[j, 0, :, :int(wpx[j])].cpu() for j in range(len(chunk))]
        hyp_gen, hyp_real = ocr_images(ocr, codec, gens, dev), ocr_images(ocr, codec, reals, dev)
        hyp_plain = [None] * len(chunk)
        if a.pairs:
            gen = torch.Generator(device=dev).manual_seed(a.seed + k)
            zp, lwp = model.sample([strip_marks(t) for t in texts], torch.stack(sups).to(dev), torch.stack(sws).to(dev),
                                   full.height, a.steps, a.w_content, a.w_style, widths_override=lat_w,
                                   generator=gen, w_mark=0.0)
            ip, wpp = decode_generated(vae, zp, lwp)
            hyp_plain = ocr_images(ocr, codec, [ip[j, 0, :, :int(wpp[j])].cpu() for j in range(len(chunk))], dev)
        for m, hg, hr, hp in zip(metas, hyp_gen, hyp_real, hyp_plain):
            records.append({**m, 'hyp_gen': hg, 'hyp_real': hr, 'hyp_plain': hp,
                            'gen': score(m['text'], hg, m['kind'], hp if m['kind'] == 'heldout' else None),
                            'real': score(m['text'], hr, m['kind'])})
        if k == 0 and a.panel:
            from torchvision.utils import save_image
            W = max(g.shape[-1] for g in gens[:16])
            pad = lambda g: torch.nn.functional.pad(g, (0, W - g.shape[-1]))
            save_image(1 - torch.stack([pad(g) for p in zip(reals[:16], gens[:16]) for g in p])[:, None],
                       a.panel, nrow=1, padding=2, pad_value=0.5)
        print(f'{len(records)}/{len(jobs)}', flush=True)
    res = {'ckpt': str(a.ckpt), 'step': st['step'], 'weights': 'raw' if a.raw else 'ema', 'content': cfg.content,
           'split': a.split, 'guidance': [a.w_content, a.w_style, a.w_mark], 'sampling_steps': a.steps, 'seed': a.seed,
           'n_targets': len(targets), 'n_writers': len(writers),
           'generated': summarise(records, 'gen'), 'real_upper_bound': summarise(records, 'real')}
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    Path(str(a.out).replace('.json', '_records.jsonl')).write_text(
        '\n'.join(json.dumps(r, ensure_ascii=False) for r in records))
    print(json.dumps({k: res[k] for k in ('step', 'content', 'generated')}, ensure_ascii=False, indent=1))


def compare(a):
    """Paired writer-bootstrap of the difference (B - A) in held-out char accuracy, per category."""
    ra = [json.loads(l) for l in open(a.a.replace('.json', '_records.jsonl'))]
    rb = [json.loads(l) for l in open(a.b.replace('.json', '_records.jsonl'))]

    def per_writer(recs, cat):
        d = defaultdict(lambda: [0, 0])
        for r in recs:
            for p in r['gen']['pos']:
                in_cat = p['cat'] == cat or (cat == 'C1+C2' and p['cat'].startswith(('C1', 'C2')))
                if in_cat and (a.metric == 'ok' or 'pair_ok' in p):
                    d[r['writer']][0] += p[a.metric]; d[r['writer']][1] += 1
        return d
    rng = np.random.default_rng(0)
    out = {}
    for cat in ['C1+C2', 'C1_tone_on_new_base', 'C2_stacked_pair_seen_on_other_base',
                'C3_stacked_pair_never_seen', 'control']:
        da, db = per_writer(ra, cat), per_writer(rb, cat)
        ws = sorted(set(da) & set(db))
        if not ws:
            continue
        A, B = np.array([da[w] for w in ws], float), np.array([db[w] for w in ws], float)
        diff = B[:, 0].sum() / B[:, 1].sum() - A[:, 0].sum() / A[:, 1].sum()
        boots = []
        for _ in range(2000):
            idx = rng.integers(len(ws), size=len(ws))
            boots.append(B[idx, 0].sum() / B[idx, 1].sum() - A[idx, 0].sum() / A[idx, 1].sum())
        out[cat] = {'A': A[:, 0].sum() / A[:, 1].sum(), 'B': B[:, 0].sum() / B[:, 1].sum(), 'diff_B_minus_A': diff,
                    'ci95': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], 'n_writers': len(ws)}
    print(json.dumps({'A': a.a, 'B': a.b, 'metric': a.metric, **out}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--ckpt', type=Path, required=True)
    r.add_argument('--vae', type=Path, required=True)
    r.add_argument('--ocr', type=Path, required=True)
    r.add_argument('--table', type=Path)
    r.add_argument('--vn-data', default='data/vn')
    r.add_argument('--vn-manifest', default=str(VATR / 'files/vietnamese/manifest.json'))
    r.add_argument('--split', default='val', choices=['val'], help='test stays sealed until the final model')
    r.add_argument('--raw', action='store_true')
    r.add_argument('--steps', type=int, default=50)
    r.add_argument('--w-content', type=float, default=3.0)
    r.add_argument('--w-style', type=float, default=1.5)
    r.add_argument('--w-mark', type=float, default=0.0)
    r.add_argument('--pairs', action='store_true', help='also generate each held-out word without marks (pair_acc)')
    r.add_argument('--bs', type=int, default=48)
    r.add_argument('--seed', type=int, default=0)
    r.add_argument('--panel', type=Path, help='optional PNG: real/generated pairs of the first batch')
    r.add_argument('--out', type=Path, required=True)
    c = sub.add_parser('compare')
    c.add_argument('a')
    c.add_argument('b')
    c.add_argument('--metric', choices=['ok', 'pair_ok'], default='ok')
    a = ap.parse_args()
    run(a) if a.cmd == 'run' else compare(a)


if __name__ == '__main__':
    main()
