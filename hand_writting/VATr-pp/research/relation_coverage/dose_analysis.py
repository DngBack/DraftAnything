"""Within-target dose-response of fidelity on support join coverage.

d(gen, real target) = beta * join_cov + gamma * covariates + target fixed effect.
Fixed effects remove everything about the word and writer; beta is identified only from different
supports of the SAME target. Writer-cluster bootstrap CI. Two metrics: HWD, writer-ID (1-cos).
beta is reported per unit of coverage (0 -> 1), and relative to the model's own wrong-writer effect."""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from h1_lib import load_words, HWDFeatures, ink_crop, to_h32  # noqa: E402
import writer_id  # noqa: E402


def fe_beta(rows, ycol, xcols):
    by = defaultdict(list)
    for r in rows:
        by[r['word_id']].append(r)
    X, y = [], []
    for g in by.values():
        if len(g) < 2:
            continue
        gx = np.array([[r[c] for c in xcols] for r in g], float)
        gy = np.array([r[ycol] for r in g], float)
        X.append(gx - gx.mean(0))
        y.append(gy - gy.mean())
    X, y = np.vstack(X), np.concatenate(y)
    keep = X.std(0) > 0
    b = np.zeros(len(xcols))
    b[keep] = np.linalg.lstsq(X[:, keep], y, rcond=None)[0]
    return b


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--records', type=Path, required=True)
    p.add_argument('--wid', required=True)
    p.add_argument('--covariates', nargs='+', required=True)
    p.add_argument('--writer-effect', type=float, nargs=2, default=None, help='OTHER-COV for hwd, wid (scale)')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    dev = 'cuda'
    byid = {r['word_id']: r for r in load_words(a.data)}
    recs = [json.loads(l) for l in open(a.records)]
    recs = [r for r in recs if Path(r['path']).exists()]
    hwd, wid = HWDFeatures(dev), writer_id.load(a.wid, dev)
    targets = sorted({r['word_id'] for r in recs})
    real = [to_h32(ink_crop(np.array(Image.open(byid[t]['path']).convert('L')))) for t in targets]
    RH, RW = hwd(real), writer_id.embed_arrays(wid, real, dev)
    ti = {t: i for i, t in enumerate(targets)}
    for s in range(0, len(recs), 256):
        ch = recs[s:s + 256]
        g = [to_h32(ink_crop(np.array(Image.open(r['path']).convert('L')))) for r in ch]
        GH, GW = hwd(g), writer_id.embed_arrays(wid, g, dev)
        for r, h, w in zip(ch, GH, GW):
            i = ti[r['word_id']]
            r['hwd'] = float((h - RH[i]).norm())
            r['wid'] = float(1 - w @ RW[i])
    xcols = ['join_cov'] + a.covariates
    rng = random.Random(0)
    writers = sorted({r['writer'] for r in recs if 'writer' in r} or {byid[r['word_id']]['writer'] for r in recs})
    for r in recs:
        r.setdefault('writer', byid[r['word_id']]['writer'])
    byw = defaultdict(list)
    for r in recs:
        byw[r['writer']].append(r)
    res = {}
    jc = np.array([r['join_cov'] for r in recs])
    print(f'{a.records.parent.name}: {len(recs)} generations, {len(targets)} targets, {len(writers)} writers; '
          f'join_cov mean {jc.mean():.2f} sd {jc.std():.2f} (within-target sd '
          f'{np.mean([np.std([r["join_cov"] for r in recs if r["word_id"] == t]) for t in targets]):.2f})')
    for k, m in enumerate(('hwd', 'wid')):
        est = fe_beta(recs, m, xcols)
        boots = []
        for _ in range(2000):
            samp = [r for w in (rng.choice(writers) for _ in writers) for r in byw[w]]
            # duplicate writers -> make word ids unique per draw so fixed effects stay per cluster copy
            boots.append(fe_beta(samp, m, xcols)[0])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rel = est[0] / a.writer_effect[k] if a.writer_effect else None
        res[m] = dict(beta_join=float(est[0]), lo=float(lo), hi=float(hi),
                      covariates=dict(zip(a.covariates, map(float, est[1:]))), relative_to_writer_effect=rel)
        print(f'  {m}: beta(join_cov 0->1) = {est[0]:+.4f} [{lo:+.4f}, {hi:+.4f}]'
              + (f'  = {100 * rel:+.1f}% of wrong-writer effect' if rel is not None else '')
              + '  | covariates ' + ', '.join(f'{c} {v:+.4f}' for c, v in zip(a.covariates, est[1:])))
    json.dump(dict(n=len(recs), targets=len(targets), writers=len(writers), results=res), open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
