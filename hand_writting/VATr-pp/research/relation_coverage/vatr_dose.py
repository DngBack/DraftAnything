"""VATr++ dose-response: K random 15-word supports per H1 target (+ the existing COV/UNC images).
Writes <run>/gen/vatr_dose/<word_id>__r<k>.png and records.jsonl with support coverage.  [.venv-infer]"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from h1_lib import load_words, load_generator, generate  # noqa: E402
from run_h1 import support_stats  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--k', type=int, default=4)
    p.add_argument('--seed', type=int, default=743)
    a = p.parse_args()
    rng = random.Random(a.seed)
    rows = load_words(a.data)
    byid = {r['word_id']: r for r in rows}
    byw = defaultdict(list)
    for r in rows:
        if r['text'].isalpha() and len(r['text']) >= 2 and r['width32'] <= 192:
            byw[r['writer']].append(r)
    sups = [json.loads(l) for l in open(a.run / 'supports.jsonl')]
    sups = [s for s in sups if s['draw'] == 0]
    out = a.run / 'gen' / 'vatr_dose'
    out.mkdir(parents=True, exist_ok=True)
    recs, jobs = [], []
    for s in sups:
        t = byid[s['word_id']]['text']
        P = [r for r in byw[s['writer']] if r['text'].lower() != t.lower()]
        for k in range(a.k):
            sup = rng.sample(P, 15)
            path = out / f"{s['word_id']}__r{k}.png"
            jobs.append((path, [r['path'] for r in sup], t))
            recs.append(dict(path=str(path), word_id=s['word_id'], writer=s['writer'], **support_stats(sup, t)))
        for c in ('COV', 'UNC'):  # already generated in gen/vatr with the logged supports
            sup = [byid[i] for i in s[c]]
            recs.append(dict(path=str(a.run / 'gen' / 'vatr' / c / f"{s['word_id']}.png"), word_id=s['word_id'],
                             writer=s['writer'], **support_stats(sup, t)))
    g = load_generator('cuda')
    for b in range(0, len(jobs), 64):
        ch = jobs[b:b + 64]
        for (path, _, _), im in zip(ch, generate(g, [j[1] for j in ch], [j[2] for j in ch], 'cuda', seed=a.seed + b)):
            Image.fromarray(im).save(path)
    with open(out / 'records.jsonl', 'w') as f:
        for r in recs:
            f.write(json.dumps(r) + '\n')
    print('vatr dose: generated', len(jobs), 'records', len(recs))


if __name__ == '__main__':
    main()
