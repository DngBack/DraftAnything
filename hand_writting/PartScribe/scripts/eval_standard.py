"""Standard HTG metrics (FID, KID, BFID, HWD, CER) on the Vietnamese val writers, any generator.

Table B of the paper plan: seen characters only (holdout-filtered val), so a model trained
without the protocol (the existing VATr++ Vietnamese checkpoints) is compared fairly.

Every model gets the SAME jobs: target word (text + writer) and the SAME 15 support images
of that writer (never the target transcript). Real target images are the reference set.

    python scripts/eval_standard.py jobs  --out runs/std_vn --per-writer 100
    python scripts/eval_standard.py partscribe --out runs/std_vn --name nfd_160k --ckpt runs/gen_vn_nfd_v2/last.pt \
        --vae runs/vae_vn32/final.pt
    python scripts/eval_standard.py vatr --out runs/std_vn --name vatrpp_175k --ckpt ../VATr-pp/runs/.../baseline_175500.pth
    python scripts/eval_standard.py score --out runs/std_vn --hwd-repo /path/to/HWD
    python scripts/eval_standard.py paired --out runs/std_vn --models vatrpp_comp_250k partscribe_nfd_300k_s0

Metric conventions follow the official HWD repository (aimagelab/HWD, BMVC 2023), whose
code is used directly via --hwd-repo (no package install):
  FID / KID / BFID: first h x h square of each word, resized to 32x32, pytorch-fid InceptionV3 pool features.
  HWD: height 32, VGG16 trained on 10,400 synthetic writers (font_square), per-column features,
       Euclidean distance of the per-writer mean feature, averaged over writers.
CER uses the independent evaluator OCR (trained on real train + HAT images, never on val).
"""
import argparse
import json
import os
import random
import sys
import unicodedata as ud
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from partscribe.datasets import fit_width, load_pnl, load_vn, to_ink  # noqa: E402
from partscribe.runtime import VATR  # noqa: E402

N_SUP = 15
SUP_W = 192


def save_png(arr_u8, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr_u8).save(path)


# ---------------------------------------------------------------- jobs
def eval_pool(a):
    """Target + support pool: the 30 VN val writers, or the 5 held-out PNL uppercase writers (C4)."""
    if a.pool == 'pnl_test':
        return load_pnl(a.pnl_data, 'test')
    return load_vn(a.vn_manifest, a.vn_data, 'val', apply_holdout=True)


def make_jobs(a):
    clean = eval_pool(a)
    vatr_alpha = set(json.loads(Path(a.vn_manifest).read_text())['alphabet'])
    from partscribe.evaluator import Codec
    codec = Codec()
    rng = random.Random(a.seed)
    out = Path(a.out)
    jobs = []
    for w in sorted(clean.pools):
        pool = clean.pools[w]
        cand = [i for i in pool if 1 <= len(clean.texts[i]) <= 20 and ' ' not in clean.texts[i]
                and set(clean.texts[i]) <= vatr_alpha and all(c in codec.index for c in clean.texts[i])
                and any(c.isalpha() for c in clean.texts[i])]
        for i in rng.sample(cand, min(a.per_writer, len(cand))):
            t = clean.texts[i]
            sup = [j for j in pool if clean.texts[j] != t]
            pick = rng.sample(sup, N_SUP) if len(sup) >= N_SUP else [rng.choice(sup) for _ in range(N_SUP)]
            k = len(jobs)
            jobs.append({'k': k, 'writer': int(w), 'text': t, 'target': i, 'support': pick,
                         'file': f'{w}/{k:05d}.png'})
            save_png(clean.image(i), out / 'real' / jobs[-1]['file'])
    (out / 'jobs.json').write_text(json.dumps(jobs, ensure_ascii=False))
    write_transcriptions(out / 'real', jobs)
    print(f'{len(jobs)} jobs, {len({j["writer"] for j in jobs})} writers -> {out}')


def write_transcriptions(folder, jobs):
    (folder / 'transcriptions.json').write_text(json.dumps({j['file']: j['text'] for j in jobs}, ensure_ascii=False))


def load_jobs(a):
    return json.loads((Path(a.out) / 'jobs.json').read_text())


# ---------------------------------------------------------------- generators
def gen_partscribe(a):
    from partscribe.generator import Config, PartScribe
    from partscribe.runtime import load_vae
    from partscribe.vae import decode_generated
    dev = 'cuda'
    jobs = load_jobs(a)
    clean = eval_pool(a)
    st = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    cfg = Config(**st['meta']['config'])
    model = PartScribe(cfg, None).to(dev).eval()
    model.load_state_dict(st['model'] if a.raw else st['ema'])
    vae, vstate = load_vae(a.vae, dev)
    support_w = st['meta'].get('support_width') or 6 * clean.height
    dst = Path(a.out) / a.name
    for b0 in range(0, len(jobs), a.bs):
        chunk = jobs[b0:b0 + a.bs]
        sups, sws = [], []
        for j in chunk:
            s, sw = torch.zeros(N_SUP, clean.height, support_w), torch.zeros(N_SUP)
            for m, i in enumerate(j['support']):
                im = fit_width(clean.image(i), support_w)
                s[m, :, :im.shape[1]] = to_ink(im)
                sw[m] = im.shape[1]
            sups.append(s); sws.append(sw)
        g = torch.Generator(device=dev).manual_seed(a.seed + b0)
        with torch.no_grad():
            z, lat_w = model.sample([j['text'] for j in chunk], torch.stack(sups).to(dev), torch.stack(sws).to(dev),
                                    clean.height, a.steps, a.w_content, a.w_style, generator=g, w_mark=a.w_mark)
            img, wpx = decode_generated(vae, z, lat_w)
        for n, j in enumerate(chunk):
            w = int(wpx[n])
            ink = img[n, 0, :, :w].cpu().numpy()
            save_png(((1 - ink) * 255).round().astype(np.uint8), dst / j['file'])
        print(f'{a.name}: {b0 + len(chunk)}/{len(jobs)}', flush=True)
    write_transcriptions(dst, jobs)
    (dst / 'meta.json').write_text(json.dumps({'ckpt': str(a.ckpt), 'step': st['step'], 'weights': 'raw' if a.raw else 'ema',
                                               'content': cfg.content, 'steps': a.steps,
                                               'guidance': [a.w_content, a.w_style, a.w_mark]}))


def gen_vatr(a):
    import os
    os.chdir(VATR)
    sys.path.insert(0, str(VATR))
    from models.model import Generator
    from models.OCR_network import strLabelConverter
    from data.vietnamese import load_manifest, to_tensor
    from train_vietnamese import model_args, seed_all
    dev = 'cuda'
    jobs = load_jobs(a)
    clean = eval_pool(a)
    seed_all(a.seed)
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    assert ck.get('stage') == 'gan'
    manifest = load_manifest(Path(a.vn_manifest), verify_sources=False)
    assert manifest['alphabet'] == ck['alphabet']
    legacy = ck['config']['legacy_generator_layout']
    cfg = model_args(manifest, dev, not legacy)
    cfg.correct_style_layout = not legacy
    G = Generator(cfg).to(dev).eval()
    G.load_state_dict(ck['model']['G'], strict=True)
    conv = strLabelConverter(ck['alphabet'])
    assert ck['config']['num_examples'] == N_SUP
    dst = Path(a.out) / a.name

    def style_tensor(idx):   # exactly generate_vietnamese.py: 32px images, squeezed to 192 if wider, padded white
        ts = []
        for i in idx:
            im = Image.fromarray(clean.image(i))
            if im.width > SUP_W:
                im = im.resize((SUP_W, 32), Image.Resampling.BICUBIC)
            pad = Image.new('L', (SUP_W, 32), 255)
            pad.paste(im, (0, 0))
            ts.append(to_tensor(pad)[0])
        return torch.stack(ts)

    for b0 in range(0, len(jobs), a.bs):
        chunk = jobs[b0:b0 + a.bs]
        style = torch.stack([style_tensor(j['support']) for j in chunk]).to(dev)
        texts = [ud.normalize('NFC', j['text']) for j in chunk]
        q, _, _ = conv.encode(texts)
        with torch.no_grad():
            fake = G.generate(G.compute_style(style), q.to(dev))
        per_char = fake.shape[-1] // q.shape[1]
        for n, j in enumerate(chunk):
            x = fake[n, 0, :, :per_char * len(texts[n])].clamp(-1, 1).cpu().numpy()
            save_png(((x + 1) * 127.5).round().astype(np.uint8), dst / j['file'])
        print(f'{a.name}: {b0 + len(chunk)}/{len(jobs)}', flush=True)
    write_transcriptions(dst, jobs)
    (dst / 'meta.json').write_text(json.dumps({'ckpt': str(a.ckpt), 'step': ck['step'], 'model': 'VATr++ (VN, native)'}))


# ---------------------------------------------------------------- metrics
def hwd_modules(repo):
    if not repo or not (Path(repo) / 'hwd' / 'metrics' / 'fid' / 'inception.py').exists():
        raise SystemExit('score needs --hwd-repo (or HWD_REPO): git clone https://github.com/aimagelab/HWD')
    """Load the two pure-torch files of the official HWD repo by path (its package __init__
    pulls torchmetrics/transformers, which are not needed for these scores)."""
    import importlib.util
    import types
    base = Path(repo) / 'hwd' / 'metrics' / 'fid'
    pkg = types.ModuleType('hwdfid'); pkg.__path__ = [str(base)]
    sys.modules['hwdfid'] = pkg
    spec = importlib.util.spec_from_file_location('hwdfid.inception', base / 'inception.py')
    m = importlib.util.module_from_spec(spec); sys.modules['hwdfid.inception'] = m
    spec.loader.exec_module(m)
    return m.InceptionV3, frechet_distance


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """pytorch-fid / HWD calculate_frechet_distance (scipy>=1.16 dropped sqrtm(disp=...))."""
    from scipy import linalg
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)


VGG16_URL = 'https://github.com/aimagelab/font_square/releases/download/VGG-16/VGG16_class_10400.pth'
INCEPTION_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'


def otsu(img):
    import cv2
    a = np.array(img.convert('L'))
    _, b = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(b).convert('RGB')


def square32(img):     # HWD FID/KID transform: CropStartSquare -> ResizeSquare(32, NEAREST) -> ToTensor
    w, h = img.size
    img = img.crop((0, 0, h, h)).resize((32, 32), Image.NEAREST)
    return torch.from_numpy(np.asarray(img, np.float32) / 255).permute(2, 0, 1)


def hwd_tensor(img):   # HWD transform: PaddingSquareHeight -> ResizeHeight(32, NEAREST) -> ToTensor
    w, h = img.size
    if w < h:
        full = Image.new('RGB', (h, h), 'white'); full.paste(img, ((h - w) // 2, 0)); img = full
        w = h
    img = img.resize((int(32 * w / h), 32), Image.NEAREST)
    return torch.from_numpy(np.asarray(img, np.float32) / 255).permute(2, 0, 1)


@torch.inference_mode()
def inception_feats(net, imgs, dev, bs=256):
    out = []
    for k in range(0, len(imgs), bs):
        x = torch.stack(imgs[k:k + bs]).to(dev)
        p = net(x)[0]
        if p.shape[2] != 1 or p.shape[3] != 1:
            p = torch.nn.functional.adaptive_avg_pool2d(p, 1)
        out.append(p.flatten(1).double().cpu())
    return torch.cat(out).numpy()


def poly_mmd(fr, ff, degree=3, coef=1.0):   # torchmetrics KID (unbiased MMD, cubic polynomial kernel)
    gamma = 1.0 / fr.shape[1]
    k = lambda x, y: (x @ y.T * gamma + coef) ** degree
    kxx, kyy, kxy = k(fr, fr), k(ff, ff), k(fr, ff)
    m = fr.shape[0]
    return ((kxx.sum() - kxx.diag().sum()) + (kyy.sum() - kyy.diag().sum())) / (m * (m - 1)) - 2 * kxy.sum() / m ** 2


def kid(fr, ff, subsets=100, size=1000, seed=0):
    g = torch.Generator().manual_seed(seed)
    fr, ff = torch.from_numpy(fr), torch.from_numpy(ff)
    n = min(size, len(fr), len(ff))
    vals = [poly_mmd(fr[torch.randperm(len(fr), generator=g)[:n]], ff[torch.randperm(len(ff), generator=g)[:n]])
            for _ in range(subsets)]
    return float(torch.stack(vals).mean())


@torch.inference_mode()
def vgg_writer_means(vgg, imgs, writers, dev):
    """HWD backbone, batch size 1 as in the reference implementation (numerically exact)."""
    sums, counts = {}, defaultdict(int)
    for x, w in zip(imgs, writers):
        f = vgg(x[None].to(dev))                    # (1, C, H', W')
        f = f.permute(0, 3, 2, 1).reshape(-1, f.shape[1])   # AdjustDims: (H'W', C), column-major as in HWD
        n = (x.shape[-1] // 32) * (x.shape[-2] // 32)
        f = f[:n].double().cpu()
        sums[w] = sums.get(w, 0) + f.sum(0)
        counts[w] += f.shape[0]
    return {w: sums[w] / counts[w] for w in sums}


def score(a):
    """Features are cached per model folder (feats.pt), so adding a model only scores that model.
    HWD and CER are per-writer quantities: they get a writer-bootstrap 95% CI and the per-writer
    values are kept for paired comparisons. FID/KID are set-level: use several seeds for spread."""
    import torchvision
    InceptionV3, frechet = hwd_modules(a.hwd_repo)
    from partscribe.evaluator import Codec, align, load_ocr
    dev = 'cuda'
    out = Path(a.out)
    jobs = load_jobs(a)
    models = a.models or sorted(p.name for p in out.iterdir() if p.is_dir() and p.name != 'real' and (p / 'meta.json').exists())
    nets = {}

    def get_nets():
        if not nets:
            nets['inc'] = InceptionV3(url=INCEPTION_URL).to(dev).eval()
            vgg = torchvision.models.vgg16(num_classes=10400)
            vgg.load_state_dict(torch.hub.load_state_dict_from_url(VGG16_URL, map_location='cpu'))
            nets['vgg'] = vgg.features.to(dev).eval()
            nets['ocr'] = load_ocr(a.ocr, dev)
        return nets

    codec = Codec()
    writers = [j['writer'] for j in jobs]
    W = sorted(set(writers))

    def features(name):
        cache = out / name / 'feats.pt'
        if cache.exists() and not a.recompute:
            return torch.load(cache, weights_only=False)
        n = get_nets()
        imgs = [Image.open(out / name / j['file']).convert('RGB') for j in jobs]
        inks = [to_ink(np.array(im.convert('L'))) for im in imgs]
        hyps = []
        for k in range(0, len(inks), 128):
            chunk = inks[k:k + 128]
            x = torch.zeros(len(chunk), 1, chunk[0].shape[-2], max(i.shape[-1] for i in chunk))
            for m, im in enumerate(chunk):
                x[m, 0, :, :im.shape[-1]] = im
            with torch.no_grad(), torch.autocast('cuda', torch.bfloat16):
                hyps += codec.decode(n['ocr'](x.to(dev)).float())
        f = {'inc': inception_feats(n['inc'], [square32(i) for i in imgs], dev),
             'binc': inception_feats(n['inc'], [square32(otsu(i)) for i in imgs], dev),
             'vgg': vgg_writer_means(n['vgg'], [hwd_tensor(i) for i in imgs], writers, dev),
             'edits': [align(j['text'], h)[1] for j, h in zip(jobs, hyps)], 'hyps': hyps}
        torch.save(f, cache)
        return f

    nchar = np.array([len(j['text']) for j in jobs])
    widx = {w: np.array([k for k, x in enumerate(writers) if x == w]) for w in W}
    rng = np.random.default_rng(0)
    boot = [rng.choice(len(W), len(W)) for _ in range(1000)]

    def per_writer(f, ref):
        hwd = np.array([torch.dist(ref['vgg'][w], f['vgg'][w]).item() for w in W])
        e = np.array(f['edits'])
        ed = np.array([e[widx[w]].sum() for w in W]); nc = np.array([nchar[widx[w]].sum() for w in W])
        return hwd, ed, nc

    def ci(vals):
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

    real = features('real')
    mu = lambda x: (x.mean(0), np.cov(x, rowvar=False))
    res = {'n_images': len(jobs), 'n_writers': len(W), 'writers': W, 'conventions': 'HWD repo (aimagelab/HWD)'}
    e = np.array(real['edits'])
    res['real'] = {'CER': float(e.sum() / nchar.sum()), 'word_acc': float(np.mean([h == j['text'] for h, j in zip(real['hyps'], jobs)]))}
    half = np.arange(len(jobs)) % 2 == 0
    res['real_half_vs_half'] = {'FID': float(frechet(*mu(real['inc'][half]), *mu(real['inc'][~half])))}
    for name in models:
        f = features(name)
        hwd, ed, nc = per_writer(f, real)
        res[name] = {
            'meta': json.loads((out / name / 'meta.json').read_text()),
            'FID': float(frechet(*mu(real['inc']), *mu(f['inc']))),
            'BFID': float(frechet(*mu(real['binc']), *mu(f['binc']))),
            'KIDx1e3': 1e3 * kid(real['inc'], f['inc']),
            'HWD': float(hwd.mean()), 'HWD_ci95': ci([hwd[b].mean() for b in boot]),
            'CER': float(ed.sum() / nc.sum()), 'CER_ci95': ci([ed[b].sum() / nc[b].sum() for b in boot]),
            'word_acc': float(np.mean([h == j['text'] for h, j in zip(f['hyps'], jobs)])),
            'per_writer': {'HWD': hwd.tolist(), 'edits': ed.tolist(), 'chars': nc.tolist()},
        }
        print(name, {k: v for k, v in res[name].items() if k not in ('meta', 'per_writer')}, flush=True)
    (out / 'scores.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))


def paired(a):
    """Writer-bootstrap CI of B - A for HWD and CER (same jobs, same writers)."""
    r = json.loads((Path(a.out) / 'scores.json').read_text())
    A, B = r[a.models[0]]['per_writer'], r[a.models[1]]['per_writer']
    rng = np.random.default_rng(0)
    n = len(A['HWD'])
    d_h, d_c = [], []
    for _ in range(2000):
        b = rng.choice(n, n)
        d_h.append(np.mean(np.array(B['HWD'])[b]) - np.mean(np.array(A['HWD'])[b]))
        d_c.append(np.array(B['edits'])[b].sum() / np.array(B['chars'])[b].sum()
                   - np.array(A['edits'])[b].sum() / np.array(A['chars'])[b].sum())
    q = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    print(json.dumps({'A': a.models[0], 'B': a.models[1], 'HWD_B_minus_A_ci95': q(d_h), 'CER_B_minus_A_ci95': q(d_c)}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['jobs', 'partscribe', 'vatr', 'score', 'paired'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--vn-data', default=str(ROOT / 'data/vn'))
    ap.add_argument('--vn-manifest', default=str(VATR / 'files/vietnamese/manifest.json'))
    ap.add_argument('--per-writer', type=int, default=100)
    ap.add_argument('--pool', choices=['vn_val', 'pnl_test'], default='vn_val',
                    help='pnl_test: uppercase C4 set (5 PNL writers never trained on, scripts/prepare_pnl.py)')
    ap.add_argument('--pnl-data', default=str(ROOT / 'data/pnl32.npz'))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--name')
    ap.add_argument('--ckpt', type=Path)
    ap.add_argument('--vae', type=Path, default=ROOT / 'runs/vae_vn32/final.pt')
    ap.add_argument('--raw', action='store_true')
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--w-content', type=float, default=3.0)
    ap.add_argument('--w-style', type=float, default=1.5)
    ap.add_argument('--w-mark', type=float, default=0.0)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--ocr', default=str(ROOT / 'runs/eval_ocr_vn/best.pt'))
    ap.add_argument('--hwd-repo', default=os.environ.get('HWD_REPO'),
                    help='clone of https://github.com/aimagelab/HWD (or set HWD_REPO); needed by score')
    ap.add_argument('--models', nargs='*')
    ap.add_argument('--recompute', action='store_true')
    a = ap.parse_args()
    a.out = str(Path(a.out).resolve())
    if a.ckpt:
        a.ckpt = a.ckpt.resolve()
    {'jobs': make_jobs, 'partscribe': gen_partscribe, 'vatr': gen_vatr, 'score': score, 'paired': paired}[a.cmd](a)


if __name__ == '__main__':
    main()
