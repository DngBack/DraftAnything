"""Shared utilities for the H1 relation-coverage study (VATr++ English on IAM)."""
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torchvision
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.model import Generator  # noqa: E402

ALPHABET = 'Only thewigsofrcvdampbkuq.A-210xT5\'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%'
SPECIAL = 'ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσςΤτΥυΦφΧχΨψΩω'
WEIGHTS = ROOT / 'files/pretrained/vatrpp_english.safetensors'
GLYPHS = ROOT.parent / 'VATr/files/unifont.pickle'
HWD_URL = 'https://github.com/aimagelab/font_square/releases/download/VGG-16/VGG16_class_10400.pth'


def bigrams(text):
    t = text.lower()
    return [t[i:i + 2] for i in range(len(t) - 1) if t[i].isalpha() and t[i + 1].isalpha()]


def letters(text):
    return [c for c in text.lower() if c.isalpha()]


def load_words(folder):
    rows = list(csv.DictReader(open(Path(folder) / 'words.csv')))
    for r in rows:
        r['width32'] = int(r['width32'])
        r['path'] = str(Path(folder) / 'words' / f"{r['word_id']}.png")
    return rows


def load_generator(device, correct_layout=False, query_position=False, add_noise=True):
    from safetensors.torch import load_file
    args = SimpleNamespace(alphabet=ALPHABET, special_alphabet=SPECIAL, device=device, glyph_path=str(GLYPHS),
                           query_input='unifont', vocab_size=len(ALPHABET), resolution=16, add_noise=add_noise,
                           imagenet_weights=False, correct_style_layout=correct_layout,
                           query_position=query_position)
    g = Generator(args)
    raw = load_file(str(WEIGHTS), device='cpu')
    sd = {k[len('model.netG.'):]: v for k, v in raw.items() if k.startswith('model.netG.')}
    missing, unexpected = g.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not m.startswith('query_embed.symbols')]
    if missing:
        raise ValueError(f'missing generator weights: {missing}')
    return g.to(device).eval()


def style_tensor(paths):
    """Replicates FolderDataset.sample_style preprocessing (height 32, aspect kept, right-pad to 192)."""
    out = []
    for p in paths:
        img = np.array(Image.open(p).convert('L'))
        canvas = np.full((32, 192), 255, np.uint8)
        canvas[:, :min(192, img.shape[1])] = img[:, :192]
        out.append(torch.from_numpy(canvas).float().div(255).sub(0.5).div(0.5))
    return torch.stack(out)  # [15, 32, 192]


@torch.no_grad()
def generate(g, supports, texts, device, seed):
    """supports: list of 15-path lists (one per item); texts: one word per item. Returns uint8 arrays."""
    torch.manual_seed(seed)
    st = torch.stack([style_tensor(s) for s in supports]).to(device)
    L = max(len(t) for t in texts)
    q = torch.zeros(len(texts), 1, L, dtype=torch.long)
    for i, t in enumerate(texts):
        q[i, 0, :len(t)] = torch.tensor([ALPHABET.index(c) + 1 for c in t])
    fake = g.evaluate(st, q.to(device))[0]
    res = []
    for i, t in enumerate(texts):
        x = ((fake[i, 0, :, :len(t) * 16] + 1) / 2).clamp(0, 1).cpu().numpy()
        res.append((x * 255).astype(np.uint8))
    return res


class HWDFeatures:
    """HWD backbone (VGG16 trained on Font², BMVC'23): features per 32-px column, averaged per image."""

    def __init__(self, device):
        m = torchvision.models.vgg16(num_classes=10400)
        m.load_state_dict(torch.hub.load_state_dict_from_url(HWD_URL, map_location='cpu'))
        self.net = m.features.to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, arrays):
        feats = []
        for a in arrays:
            img = Image.fromarray(a).convert('RGB')
            w, h = img.size
            if w < h:  # HWD PaddingSquareHeight
                pad = Image.new('RGB', (h, h), 'white')
                pad.paste(img, ((h - w) // 2, 0))
                img = pad
            w, h = img.size
            img = img.resize((max(32, int(32 * w / h)), 32), Image.NEAREST)
            x = torchvision.transforms.functional.to_tensor(img).unsqueeze(0).to(self.device)
            f = self.net(x)  # [1, 512, 1, W/32]
            n = max(1, x.shape[-1] // 32)
            feats.append(f[0, :, 0, :n].mean(-1).cpu())
        return torch.stack(feats)
