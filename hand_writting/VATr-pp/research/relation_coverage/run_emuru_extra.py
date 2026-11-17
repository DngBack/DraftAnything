"""Emuru follow-ups for H1.                                           [handwriting/.venv-emuru]

--mode selfm  SELF-masked control: the target's own line with the target word cut out of the image
              (from its box start to the next word's start) and removed from the style text.
              SELF - SELFM = word-copy effect; SELFM - COV = same-writing-session effect.
--mode dose   Dose-response over all H1 targets: K other lines of the writer (never containing t),
              chosen to spread join coverage (lowest / highest / median). Analysed with target fixed
              effects, so the join effect is estimated within target word.
"""
import argparse
import html
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from h1_lib import load_words  # noqa: E402
from gen_saved import EMURU_REV  # noqa: E402
from run_emuru_lines import HW, cov_stats, emuru_batch  # noqa: E402


def to64(im):
    return im.resize((max(im.width * 64 // im.height, 8), 64), Image.BILINEAR)


def line_img(rec):
    g = np.asarray(Image.open(HW / rec.image_path).convert('L')).copy()
    g[g >= rec.threshold] = 255
    return g


def masked_line(rec, k, target, written_tokens):
    """Remove word k (IAM word index) from the line image and its text."""
    g = line_img(rec)
    boxes, words = json.loads(rec.word_boxes), json.loads(rec.word_texts)
    x0 = boxes[k][0]
    if k + 1 < len(boxes):
        x1 = boxes[k + 1][0]
        g = np.concatenate([g[:, :x0], g[:, x1:]], axis=1)
    else:
        prev = boxes[k - 1][2] if k > 0 else 0
        g = g[:, :max(prev + 10, 1)]
    occ = sum(w == target for w in words[:k])  # which occurrence of the word to drop from the text
    toks, seen, out = written_tokens(rec.transcript), 0, []
    dropped = False
    for t in toks:
        core = ''.join(c for c in html.unescape(t) if c.isalpha())  # match only; text kept as fed before
        if not dropped and core == target:
            if seen == occ:
                rest = ''.join(c for c in html.unescape(t) if not c.isalpha())
                if rest:
                    out.append(rest)
                dropped = True
                continue
            seen += 1
        out.append(t)
    if not dropped:
        return None, None
    return to64(Image.fromarray(g)), ' '.join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['selfm', 'dose'], required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--k', type=int, default=3)
    p.add_argument('--batch', type=int, default=12)
    p.add_argument('--seed', type=int, default=742)
    a = p.parse_args()
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    from hwline.segment import written_tokens
    byid = {r['word_id']: r for r in load_words(a.data)}
    man = pd.read_csv(HW / 'data/manifest.csv', dtype={'writer_id': str})
    man = man[man.quality_flag == 'ok'].set_index('line_id', drop=False)
    man['text'] = man.transcript.map(lambda s: ' '.join(written_tokens(s)))

    jobs = []  # (out_path, style_img, style_text, gen_text, record)
    if a.mode == 'selfm':
        out = a.run / 'gen' / 'emuru_lines' / 'SELFM'
        conds = [json.loads(l) for l in open(a.run / 'emuru_lines_conditions.jsonl')]
        for c in conds:
            rec = man.loc[c['lines']['SELF']]
            k = int(c['word_id'].split('-')[3])
            im, txt = masked_line(rec, k, c['text'], written_tokens)
            if im is None:
                continue
            jobs.append((out / f"{c['word_id']}.png", im, txt, c['text'],
                         dict(word_id=c['word_id'], **cov_stats(txt, c['text']))))
    else:
        out = a.run / 'gen' / 'emuru_dose'
        sups = [json.loads(l) for l in open(a.run / 'supports.jsonl')]
        byw = {w: g for w, g in man.groupby('writer_id')}
        for s in (s for s in sups if s['draw'] == 0):
            t = byid[s['word_id']]['text']
            own = '-'.join(s['word_id'].split('-')[:3])
            lines = byw.get(s['writer'])
            if lines is None:
                continue
            cand = lines[(lines.line_id != own) &
                         ~lines.text.str.split().map(lambda ws: any(''.join(ch for ch in html.unescape(w) if ch.isalpha()).lower() == t.lower() for w in ws))]
            if len(cand) < a.k:
                continue
            st = sorted(((cov_stats(tx, t)['join_cov'], rng.random(), lid) for lid, tx in zip(cand.line_id, cand.text)))
            pick = {st[0][2], st[-1][2], st[len(st) // 2][2]}
            while len(pick) < a.k:
                pick.add(rng.choice(st)[2])
            for lid in sorted(pick)[:a.k]:
                rec = man.loc[lid]
                jobs.append((out / f"{s['word_id']}__{lid}.png", to64(Image.fromarray(line_img(rec))), rec.text, t,
                             dict(word_id=s['word_id'], writer=s['writer'], line=lid, line_chars=len(rec.text),
                                  **cov_stats(rec.text, t))))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'records.jsonl', 'w') as f:
        for j in jobs:
            f.write(json.dumps(dict(path=str(j[0]), **j[4])) + '\n')
    print(f'{a.mode}: {len(jobs)} generations', flush=True)

    from transformers import AutoModel
    model = AutoModel.from_pretrained('blowing-up-groundhogs/emuru', revision=EMURU_REV,
                                      trust_remote_code=True).cuda().eval()
    todo = [j for j in jobs if not j[0].exists()]
    todo.sort(key=lambda j: j[1].width)
    fails = 0
    for b in range(0, len(todo), a.batch):
        ch = todo[b:b + a.batch]
        ppc = max(j[1].width / max(len(j[2]), 1) for j in ch)
        max_tok = int(max(len(j[3]) for j in ch) * ppc * 1.6 / 8) + 24
        res = emuru_batch(model, [j[2] for j in ch], [j[3] for j in ch], [j[1] for j in ch], max_tok)
        for j, im in zip(ch, res):
            if im is None or im.width < 8:
                fails += 1
                continue
            im.convert('L').save(j[0])
        print(f'{min(b + a.batch, len(todo))}/{len(todo)}', flush=True)
    print('done; failures', fails)


if __name__ == '__main__':
    main()
