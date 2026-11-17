"""Generate and SAVE images for the H1 conditions with a given model, reusing the exact logged supports.

  --model vatr   VATr++ English (15 support word images, as in run_h1.py)       [.venv-infer]
  --model emuru  Emuru (CVPR'25, trained only on synthetic fonts -> cannot have seen IAM writers).
                 The same 15 support words are concatenated into one 64-px style line with their
                 transcript as style text.                                       [handwriting/.venv-emuru]
Output: <out>/gen/<model>/<COND>/<word_id>.png  (+ failures.json)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from h1_lib import load_words  # noqa: E402

CONDS = ['COV', 'UNC', 'OTHER', 'SELF']
import os
if os.environ.get('H1_CONDS'):
    CONDS = os.environ['H1_CONDS'].split(',')
EMURU_REV = 'c75b78dd6ea0306123ad5a12b18b9ae4a927e149'  # same pinned revision as handwriting/scripts/gen_emuru.py


def style_line(paths, gap=24):
    ims = []
    for p in paths:
        im = Image.open(p).convert('L')
        ims.append(im.resize((max(8, im.width * 64 // im.height), 64), Image.BILINEAR))
    W = sum(i.width for i in ims) + gap * (len(ims) - 1)
    canvas = Image.new('L', (W, 64), 255)
    x = 0
    for i in ims:
        canvas.paste(i, (x, 0))
        x += i.width + gap
    return canvas


def emuru_generate(model, style_texts, gen_texts, style_imgs, widths, max_tok, after=10, patience=1):
    """Emuru generation whose stop search starts AFTER each style image.

    The stock `generate_batch` scans the whole canvas for a run of padding-like latents, so white gaps
    inside a multi-word style line end generation before it starts (empty or mis-cropped outputs).
    Here generation runs with stopping disabled; the stop is the first padding window past the style.
    """
    from torchvision.transforms import functional as F
    texts = [s + ' ' + g for s, g in zip(style_texts, gen_texts)]
    imgs, canvas, _ = model._generate(texts=texts, imgs=style_imgs, lengths=widths, max_new_tokens=max_tok,
                                      stopping_criteria='none')
    imgs = (imgs + 1) / 2
    pad = model.padding_token.view(1, -1)
    sim = torch.nn.functional.cosine_similarity(canvas, pad.unsqueeze(0), dim=-1)  # [B, T]
    is_pad = sim > model.padding_token_threshold
    out = []
    for i, w in enumerate(widths):
        start_tok = int(np.ceil(w / 8))
        end_tok = canvas.size(1)
        for t in range(start_tok + 2, canvas.size(1) - after + 1):  # >=2 tokens of ink before a stop
            if is_pad[i, t:t + after].sum() > after - patience:
                end_tok = t
                break
        out.append(F.to_pil_image(imgs[i, ..., w:end_tok * 8].detach().cpu()))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['vatr', 'emuru'], required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--run', type=Path, required=True, help='folder with supports.jsonl')
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--style-data', default=None, help='same words at native-res height 64 (for emuru)')
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--seed', type=int, default=742)
    p.add_argument('--tag', default=None, help='output subfolder name (default: model)')
    a = p.parse_args()
    byid = {r['word_id']: r for r in load_words(a.data)}
    sups = [json.loads(l) for l in open(a.run / 'supports.jsonl')]
    sups = [s for s in sups if s['draw'] == 0][:a.limit]
    sty = {r['word_id']: r for r in load_words(a.style_data)} if a.style_data else byid
    out = a.run / 'gen' / (a.tag or a.model)
    dev = 'cuda'
    torch.manual_seed(a.seed)
    failures = []

    if a.model == 'vatr':
        from h1_lib import load_generator, generate
        g = load_generator(dev)
        for c in CONDS:
            (out / c).mkdir(parents=True, exist_ok=True)
            for s in range(0, len(sups), 64):
                ch = sups[s:s + 64]
                imgs = generate(g, [[byid[i]['path'] for i in x[c]] for x in ch],
                                [byid[x['word_id']]['text'] for x in ch], dev, seed=a.seed + s)
                for x, im in zip(ch, imgs):
                    Image.fromarray(im).save(out / c / f"{x['word_id']}.png")
    else:
        from transformers import AutoModel
        from torchvision.transforms import functional as F
        model = AutoModel.from_pretrained('blowing-up-groundhogs/emuru', revision=EMURU_REV,
                                          trust_remote_code=True).to(dev).eval()
        for c in CONDS:
            (out / c).mkdir(parents=True, exist_ok=True)
            jobs = []
            for x in sups:
                if (out / c / f"{x['word_id']}.png").exists():
                    continue
                st = style_line([sty[i]['path'] for i in x[c]])
                jobs.append((x['word_id'], st, ' '.join(byid[i]['text'] for i in x[c]), byid[x['word_id']]['text']))
            jobs.sort(key=lambda j: j[1].width)  # less padding per batch
            for b in range(0, len(jobs), a.batch):
                ch = jobs[b:b + a.batch]
                W = max(j[1].width for j in ch)
                ts = []
                for _, st, _, _ in ch:
                    pad = Image.new('L', (W, 64), 255)
                    pad.paste(st, (0, 0))
                    ts.append(F.normalize(F.to_tensor(pad.convert('RGB')), [0.5], [0.5]))
                ppc = max(st.width / max(len(stxt), 1) for _, st, stxt, _ in ch)
                max_tok = int(max(len(gt) for *_, gt in ch) * ppc * 1.6 / 8) + 32
                with torch.inference_mode():
                    res = emuru_generate(model, [j[2] for j in ch], [j[3] for j in ch], torch.stack(ts).to(dev),
                                         [j[1].width for j in ch], max_tok)
                for (wid, *_), im in zip(ch, res):
                    if im.width == 0 or im.height == 0:
                        failures.append(dict(cond=c, word_id=wid))
                        continue
                    im.convert('L').save(out / c / f'{wid}.png')
                print(f'{c} {min(b + a.batch, len(jobs))}/{len(jobs)}', flush=True)
    json.dump(failures, open(out / 'failures.json', 'w'))
    print('done', a.model, 'items', len(sups), 'failures', len(failures))


if __name__ == '__main__':
    main()
