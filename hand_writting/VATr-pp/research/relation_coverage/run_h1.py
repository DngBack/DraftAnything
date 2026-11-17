"""H1: does VATr++ lose writer fidelity when the target word's letter *joins* are absent from the support?

Within-(writer, word) design. For a real target word t of writer A we build 15-word supports from A's
other words (never containing t itself):
  COV    every bigram of t appears in the support, all letters of t appear
  UNC    no support word contains any bigram of t, all letters of t still appear
  RAND   15 random words of A (natural coverage; observational dose-response)
  OTHER  15 words of a different writer B (floor: wrong writer)
  SELF   COV with one filler replaced by the real image of t (ceiling: copy available)
Fidelity is measured with the HWD backbone (VGG16 on Font², independent of VATr and IAM), never OCR.
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from h1_lib import load_words, load_generator, generate, HWDFeatures, bigrams, letters  # noqa: E402

N = 15


def greedy(pool, need, key, rng):
    """Pick words covering the set `need` under `key` (bigrams or letters)."""
    need, chosen = set(need), []
    cand = pool[:]
    rng.shuffle(cand)
    while need:
        best = max(cand, key=lambda r: len(need & set(key(r['text']))), default=None)
        if best is None or not need & set(key(best['text'])):
            return chosen, need
        chosen.append(best)
        cand.remove(best)
        need -= set(key(best['text']))
    return chosen, need


def fill(chosen, pool, rng):
    rest = [r for r in pool if r not in chosen]
    if len(chosen) > N or len(rest) < N - len(chosen):
        return None
    return chosen + rng.sample(rest, N - len(chosen))


def support_stats(sup, t):
    sb = {b for r in sup for b in bigrams(r['text'])}
    sl = [c for r in sup for c in letters(r['text'])]
    tb = bigrams(t)
    return dict(join_cov=sum(b in sb for b in tb) / len(tb),
                letter_cov=sum(c in set(sl) for c in letters(t)) / len(letters(t)),
                letter_instances=sum(sl.count(c) for c in set(letters(t))),
                mean_len=float(np.mean([len(r['text']) for r in sup])))


def build(byw, writers, rng, max_targets, draws):
    items = []
    for w in writers:
        pool_all = byw[w]
        targets = {}
        for r in pool_all:
            t = r['text']
            if 4 <= len(t) <= 9 and len(set(bigrams(t))) >= 2 and t.lower() not in targets:
                targets[t.lower()] = r
        tlist = list(targets.values())
        rng.shuffle(tlist)
        kept = 0
        for tr in tlist:
            if kept >= max_targets:
                break
            t = tr['text']
            P = [r for r in pool_all if r['text'].lower() != t.lower() and r['width32'] <= 192]
            Bt, Lt = set(bigrams(t)), set(letters(t))
            Pu = [r for r in P if not set(bigrams(r['text'])) & Bt]
            others = [o for o in writers if o != w]
            ok, rec = True, []
            for d in range(draws):
                best = None
                for _ in range(20):  # several fills; keep the pair closest in mean length / letter instances
                    c1, miss_b = greedy(P, Bt, bigrams, rng)
                    c2, miss_l = greedy([r for r in P if r not in c1], Lt - {c for x in c1 for c in letters(x['text'])},
                                        letters, rng)
                    cov = fill(c1 + c2, P, rng) if not miss_b and not miss_l else None
                    u1, miss_u = greedy(Pu, Lt, letters, rng)
                    unc = fill(u1, Pu, rng) if not miss_u else None
                    if cov is None or unc is None:
                        continue
                    sc, su = support_stats(cov, t), support_stats(unc, t)
                    if sc['join_cov'] < 1 or su['join_cov'] > 0:
                        continue
                    gap = abs(sc['mean_len'] - su['mean_len']) + 0.05 * abs(sc['letter_instances'] - su['letter_instances'])
                    if best is None or gap < best[0]:
                        best = (gap, cov, unc)
                if best is None:
                    ok = False
                    break
                _, cov, unc = best
                ow = rng.choice(others)
                op = [r for r in byw[ow] if r['text'].lower() != t.lower() and r['width32'] <= 192]
                o1, _ = greedy(op, Lt, letters, rng)
                oth = fill(o1, op, rng) or rng.sample(op, N)
                rand = rng.sample(P, N)
                self_ = cov[:-1] + [tr]
                rec.append(dict(COV=cov, UNC=unc, RAND=rand, OTHER=oth, SELF=self_, other_writer=ow))
            if not ok:
                continue
            kept += 1
            for d, sups in enumerate(rec):
                items.append(dict(writer=w, target=tr, draw=d, sups=sups))
    return items


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--min-pool', type=int, default=60)
    p.add_argument('--max-targets', type=int, default=25)
    p.add_argument('--draws', type=int, default=2)
    p.add_argument('--seed', type=int, default=742)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    dev = 'cuda'

    rows = load_words(a.data)
    byw = defaultdict(list)
    for r in rows:
        if r['text'].isalpha() and len(r['text']) >= 2:
            byw[r['writer']].append(r)
    writers = sorted(w for w in byw if sum(r['width32'] <= 192 for r in byw[w]) >= a.min_pool)
    items = build(byw, writers, rng, a.max_targets, a.draws)
    print(f'writers {len(writers)}, items {len(items)}, targets {len({(i["writer"], i["target"]["word_id"]) for i in items})}')

    g = load_generator(dev)
    hwd = HWDFeatures(dev)
    conds = ['COV', 'UNC', 'RAND', 'OTHER', 'SELF']
    # writer centroids from all real words of the writer (HWD features), computed once
    feat_cache = {}

    def real_feat(r):
        if r['word_id'] not in feat_cache:
            feat_cache[r['word_id']] = hwd([np.array(Image.open(r['path']).convert('L'))])[0]
        return feat_cache[r['word_id']]

    centroid = {w: torch.stack([real_feat(r) for r in byw[w]]).mean(0) for w in writers}

    records = []
    B = 64
    for c in conds:
        for s in range(0, len(items), B):
            chunk = items[s:s + B]
            fakes = generate(g, [[r['path'] for r in it['sups'][c]] for it in chunk],
                             [it['target']['text'] for it in chunk], dev, seed=a.seed + s)
            f = hwd(fakes)
            for it, fk, ff in zip(chunk, fakes, f):
                st = support_stats(it['sups'][c], it['target']['text'])
                records.append(dict(writer=it['writer'], word_id=it['target']['word_id'], text=it['target']['text'],
                                    draw=it['draw'], cond=c, **st,
                                    d_pair=float((ff - real_feat(it['target'])).norm()),
                                    d_centroid=float((ff - centroid[it['writer']]).norm()),
                                    d_centroid_other=float((ff - centroid[it['sups']['other_writer']]).norm())))
    with open(a.out / 'records.jsonl', 'w') as fh:
        for r in records:
            fh.write(json.dumps(r) + '\n')
    # supports log for audit
    with open(a.out / 'supports.jsonl', 'w') as fh:
        for it in items:
            fh.write(json.dumps(dict(writer=it['writer'], word_id=it['target']['word_id'], draw=it['draw'],
                                     **{c: [r['word_id'] for r in it['sups'][c]] for c in conds})) + '\n')
    # visual panel is rendered by sensitivity_and_panel.py from supports.jsonl


if __name__ == '__main__':
    main()
