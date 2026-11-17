"""H1 on DiffusionPen (ECCV'24, latent diffusion, 5 style words, IAM-trained on the 339-writer split).

Supports are 5-word prefixes of the logged 15-word H1 supports (coverage words come first in those lists):
  COV = COV[:5] (must still cover every bigram and letter), UNC = UNC[:5] (must still cover every letter),
  SELF = COV[:4] + target image, OTHER = OTHER[:5]; units failing the checks are dropped.
Dose: K random 5-word supports per target (+ COV/UNC) for the within-target regression.
Style preprocessing and DDIM sampling follow the authors' train.py (sampling mode).
Run with PYTHONPATH=<third_party>/pylib:<third_party>/DiffusionPen   [handwriting/.venv-emuru]
"""
import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image, ImageOps
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))
from h1_lib import load_words  # noqa: E402
from run_h1 import support_stats  # noqa: E402

DP_REV = '5636a68365001435ec76553114313c22c6f1ca07'
SD = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
CHARS = ['!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8',
         '9', ':', ';', '?'] + [chr(c) for c in range(65, 91)] + [chr(c) for c in range(97, 123)] + [' ']


def build(dev):
    from huggingface_hub import hf_hub_download
    from diffusers import AutoencoderKL, DDIMScheduler
    from transformers import CanineModel, CanineTokenizer
    from unet import UNetModel
    from feature_extractor import ImageEncoder
    tok = CanineTokenizer.from_pretrained('google/canine-c')
    te = torch.nn.DataParallel(CanineModel.from_pretrained('google/canine-c'), device_ids=[0]).to(dev)
    args = SimpleNamespace(interpolation=False, mix_rate=None, device=dev, latent=True, img_feat=True)
    unet = UNetModel(image_size=(64, 256), in_channels=4, model_channels=320, out_channels=4, num_res_blocks=1,
                     attention_resolutions=(1, 1), channel_mult=(1, 1), num_heads=4, num_classes=339,
                     context_dim=320, vocab_size=len(CHARS), text_encoder=te, args=args)
    unet = torch.nn.DataParallel(unet, device_ids=[0]).to(dev)
    sd = torch.load(hf_hub_download('konnik/DiffusionPen', 'diffusionpen_iam_model_path/models/ema_ckpt.pt',
                                    revision=DP_REV), map_location=dev)
    unet.load_state_dict(sd)
    unet.eval()
    vae = AutoencoderKL.from_pretrained(SD, subfolder='vae').to(dev).eval()
    ddim = DDIMScheduler.from_pretrained(SD, subfolder='scheduler')
    fe = ImageEncoder(model_name='mobilenetv2_100', num_classes=0, pretrained=True, trainable=False)
    st = torch.load(hf_hub_download('konnik/DiffusionPen', 'style_models/iam_style_diffusionpen.pth', revision=DP_REV),
                    map_location=dev)
    md = fe.state_dict()
    loaded = {k: v for k, v in st.items() if k in md and md[k].shape == v.shape}
    assert len(loaded) > 0.9 * len(md), f'style encoder load mismatch {len(loaded)}/{len(md)}'
    md.update(loaded)
    fe.load_state_dict(md)
    fe = fe.to(dev).eval()
    return tok, unet, vae, ddim, fe


def style_tensor(path, tf):
    """Authors' preprocessing: RGB, height 64, pad (centered) or shrink to width 256."""
    from utils.auxilary_functions import image_resize_PIL, centered_PIL
    im = Image.open(path).convert('RGB')
    w, h = im.size
    im = im.resize((int(w * 64 / h), 64))
    w, h = im.size
    if w < 256:
        im = ImageOps.pad(im, size=(256, 64), color='white')
    else:
        while w > 256:
            im = image_resize_PIL(im, width=w - 20)
            w, h = im.size
        im = centered_PIL(im, (64, 256), border_value=255.0)
    return tf(im)


@torch.no_grad()
def sample(models, style_paths, texts, dev, seed):
    tok, unet, vae, ddim, fe = models
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
    n = len(texts)
    st = torch.stack([style_tensor(p, tf) for ps in style_paths for p in ps]).to(dev)  # [n*5,3,64,256]
    feats = fe(st)
    text = tok(texts, padding='max_length', truncation=True, return_tensors='pt', max_length=40).to(dev)
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn((n, 4, 8, 32), generator=g, device=dev)
    ddim.set_timesteps(50)
    labels = torch.zeros(n, dtype=torch.long, device=dev)  # ignored when style features are given
    for t in ddim.timesteps:
        tt = (torch.ones(n) * t.item()).long().to(dev)
        eps = unet(x, tt, text, labels, original_images=st, mix_rate=None, style_extractor=feats)
        x = ddim.step(eps, t, x).prev_sample
    img = vae.decode(x / 0.18215).sample
    img = (img / 2 + 0.5).clamp(0, 1).mean(1)  # grayscale
    return [(i.cpu().numpy() * 255).astype('uint8') for i in img]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='32-px words (texts, coverage)')
    p.add_argument('--style-data', required=True, help='same words at 64-px (style images)')
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--k', type=int, default=4)
    p.add_argument('--batch', type=int, default=48)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--seed', type=int, default=742)
    a = p.parse_args()
    rng = random.Random(a.seed)
    dev = 'cuda'
    byid = {r['word_id']: r for r in load_words(a.data)}
    sty = {r['word_id']: r for r in load_words(a.style_data)}
    sups = [json.loads(l) for l in open(a.run / 'supports.jsonl')]
    sups = [s for s in sups if s['draw'] == 0][:a.limit]
    a.run = a.run.resolve()
    out = a.run / 'gen' / 'diffusionpen'
    dose = a.run / 'gen' / 'diffusionpen_dose'
    jobs, dose_recs, kept = [], [], 0
    pool = {}
    for r in byid.values():
        if r['text'].isalpha() and len(r['text']) >= 2:
            pool.setdefault(r['writer'], []).append(r)
    for s in sups:
        t = byid[s['word_id']]['text']
        S = dict(COV=s['COV'][:5], UNC=s['UNC'][:5], OTHER=s['OTHER'][:5], SELF=s['COV'][:4] + [s['word_id']])
        sc = support_stats([byid[i] for i in S['COV']], t)
        su = support_stats([byid[i] for i in S['UNC']], t)
        if sc['join_cov'] < 1 or sc['letter_cov'] < 1 or su['join_cov'] > 0 or su['letter_cov'] < 1:
            continue
        kept += 1
        for c, ids in S.items():
            jobs.append((out / c / f"{s['word_id']}.png", [sty[i]['path'] for i in ids], t))
        for c in ('COV', 'UNC'):
            dose_recs.append(dict(path=str(out / c / f"{s['word_id']}.png"), word_id=s['word_id'], writer=s['writer'],
                                  **support_stats([byid[i] for i in S[c]], t)))
        P = [r for r in pool[s['writer']] if r['text'].lower() != t.lower()]
        for k in range(a.k):
            sup = rng.sample(P, 5)
            path = dose / f"{s['word_id']}__r{k}.png"
            jobs.append((path, [sty[r['word_id']]['path'] for r in sup], t))
            dose_recs.append(dict(path=str(path), word_id=s['word_id'], writer=s['writer'], **support_stats(sup, t)))
    for d in [out / c for c in ('COV', 'UNC', 'OTHER', 'SELF')] + [dose]:
        d.mkdir(parents=True, exist_ok=True)
    with open(dose / 'records.jsonl', 'w') as f:
        for r in dose_recs:
            f.write(json.dumps(r) + '\n')
    print(f'units kept {kept}/{len(sups)}; generations {len(jobs)}', flush=True)
    models = build(dev)
    todo = [j for j in jobs if not j[0].exists()]
    for b in range(0, len(todo), a.batch):
        ch = todo[b:b + a.batch]
        for (path, _, _), im in zip(ch, sample(models, [j[1] for j in ch], [j[2] for j in ch], dev, a.seed + b)):
            Image.fromarray(im).save(path)
        print(f'{min(b + a.batch, len(todo))}/{len(todo)}', flush=True)
    print('done')


if __name__ == '__main__':
    main()
