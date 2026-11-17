"""H1 replication on Emuru (CVPR'25; trained only on synthetic fonts, so it cannot have seen IAM writers).

Emuru conditions on ONE real text line (its native protocol); synthetic lines stitched from word crops
make it fail (blank/garbled output), so supports here are real IAM lines of the writer:
  SELF   the line that contains the target word t (copy available)
  COV    another line of the same writer covering every bigram of t, not containing t
  UNC    another line of the same writer covering none of t's bigrams
  OTHER  a line of a different writer
Same H1 target words (supports.jsonl, draw 0). Output: <run>/gen/emuru_lines/<COND>/<word_id>.png,
<run>/emuru_lines_conditions.jsonl (chosen lines + coverage stats).    [handwriting/.venv-emuru]
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from h1_lib import load_words, bigrams, letters  # noqa: E402
from gen_saved import EMURU_REV  # noqa: E402

HW = Path('/home/bachdx2/DraftAnything/handwriting')
CONDS = ['SELF', 'COV', 'UNC', 'OTHER']


def cov_stats(line_text, t):
    lb, ll = set(bigrams(line_text)), set(letters(line_text))
    tb, tl = bigrams(t), letters(t)
    return dict(join_cov=sum(b in lb for b in tb) / len(tb), letter_cov=sum(c in ll for c in tl) / len(tl))


def load_line(rec):
    g = np.asarray(Image.open(HW / rec.image_path).convert('L')).copy()
    g[g >= rec.threshold] = 255
    im = Image.fromarray(g)
    return im.resize((max(im.width * 64 // im.height, 8), 64), Image.BILINEAR)


@torch.inference_mode()
def emuru_batch(model, style_texts, gen_texts, ims, max_tok, after=10, patience=1):
    """Stock generation, but the stop is searched only after the style AND after ink has started."""
    from torchvision.transforms import functional as F
    W = max(i.width for i in ims)
    ts = []
    for im in ims:
        pad = Image.new('L', (W, 64), 255)
        pad.paste(im, (0, 0))
        ts.append(F.normalize(F.to_tensor(pad.convert('RGB')), [0.5], [0.5]))
    widths = [i.width for i in ims]
    imgs, canvas, _ = model._generate(texts=[s + ' ' + g for s, g in zip(style_texts, gen_texts)],
                                      imgs=torch.stack(ts).to(model.device), lengths=widths,
                                      max_new_tokens=max_tok, stopping_criteria='none')
    imgs = (imgs + 1) / 2
    sim = torch.nn.functional.cosine_similarity(canvas, model.padding_token.view(1, 1, -1), dim=-1)
    is_pad = (sim > model.padding_token_threshold).cpu()
    out = []
    for i, w in enumerate(widths):
        s0 = int(np.ceil(w / 8))
        ink = [t for t in range(s0, canvas.size(1)) if not is_pad[i, t]]
        if not ink:
            out.append(None)
            continue
        end = canvas.size(1)
        for t in range(ink[0] + 1, canvas.size(1) - after + 1):
            if is_pad[i, t:t + after].sum() > after - patience:
                end = t
                break
        out.append(F.to_pil_image(imgs[i, ..., ink[0] * 8:end * 8].cpu()))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--seed', type=int, default=742)
    a = p.parse_args()
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    byid = {r['word_id']: r for r in load_words(a.data)}
    sups = [json.loads(l) for l in open(a.run / 'supports.jsonl')]
    sups = [s for s in sups if s['draw'] == 0][:a.limit]
    man = pd.read_csv(HW / 'data/manifest.csv', dtype={'writer_id': str})
    man = man[man.quality_flag == 'ok'].set_index('line_id', drop=False)
    from hwline.segment import written_tokens  # noqa: E402  (handwriting package, same tokens as gen_emuru)
    man['text'] = man.transcript.map(lambda s: ' '.join(written_tokens(s)))
    byw = {w: g for w, g in man.groupby('writer_id')}

    conds = []
    for s in sups:
        t = byid[s['word_id']]['text']
        own = '-'.join(s['word_id'].split('-')[:3])
        if own not in man.index:
            continue
        lines = byw[s['writer']]
        others = lines[(lines.line_id != own) & ~lines.text.str.split().map(lambda ws: any(w.lower() == t.lower() for w in ws))]
        st = {lid: cov_stats(tx, t) for lid, tx in zip(others.line_id, others.text)}
        cov = [l for l, v in st.items() if v['join_cov'] == 1]
        unc = [l for l, v in st.items() if v['join_cov'] == 0]
        if not cov or not unc:
            continue
        # among candidates prefer full letter coverage, then length closest to the other arm
        c = max(cov, key=lambda l: (st[l]['letter_cov'], rng.random()))
        u = max(unc, key=lambda l: (st[l]['letter_cov'], -abs(len(man.at[l, 'text']) - len(man.at[c, 'text']))))
        ow = rng.choice([w for w in sorted({x['writer'] for x in sups}) if w != s['writer'] and w in byw])
        o = rng.choice(list(byw[ow].line_id))
        conds.append(dict(word_id=s['word_id'], writer=s['writer'], text=t,
                          lines=dict(SELF=own, COV=c, UNC=u, OTHER=o),
                          stats={k: cov_stats(man.at[v, 'text'], t) for k, v in dict(SELF=own, COV=c, UNC=u, OTHER=o).items()}))
    with open(a.run / 'emuru_lines_conditions.jsonl', 'w') as f:
        for c in conds:
            f.write(json.dumps(c) + '\n')
    print(f'targets with valid COV/UNC lines: {len(conds)} / {len(sups)}', flush=True)

    from transformers import AutoModel
    model = AutoModel.from_pretrained('blowing-up-groundhogs/emuru', revision=EMURU_REV,
                                      trust_remote_code=True).cuda().eval()
    out = a.run / 'gen' / 'emuru_lines'
    fails = []
    for cname in CONDS:
        (out / cname).mkdir(parents=True, exist_ok=True)
        jobs = [c for c in conds if not (out / cname / f"{c['word_id']}.png").exists()]
        jobs.sort(key=lambda c: len(man.at[c['lines'][cname], 'text']))
        for b in range(0, len(jobs), a.batch):
            ch = jobs[b:b + a.batch]
            recs = [man.loc[c['lines'][cname]] for c in ch]
            ims = [load_line(r) for r in recs]
            ppc = max(im.width / max(len(r.text), 1) for im, r in zip(ims, recs))
            max_tok = int(max(len(c['text']) for c in ch) * ppc * 1.6 / 8) + 24
            res = emuru_batch(model, [r.text for r in recs], [c['text'] for c in ch], ims, max_tok)
            for c, im in zip(ch, res):
                if im is None or im.width < 8:
                    fails.append(dict(cond=cname, word_id=c['word_id']))
                    continue
                im.convert('L').save(out / cname / f"{c['word_id']}.png")
            print(f'{cname} {min(b + a.batch, len(jobs))}/{len(jobs)}', flush=True)
    json.dump(fails, open(out / 'failures.json', 'w'))
    print('done; failures', len(fails))


if __name__ == '__main__':
    main()
