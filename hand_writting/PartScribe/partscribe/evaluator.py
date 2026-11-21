"""Independent evaluator OCR (report_04, section 2.4) and held-out scoring helpers.

The evaluator is trained on the FULL training data (held-out characters
included, plus HAT2023 for Vietnamese) and never on val/test. No generator
ever sees it or its gradients.
"""
import unicodedata as ud

import torch
import torch.nn as nn
import torch.nn.functional as F

from .charset import ALPHABET, parts


class CRNN(nn.Module):
    """Conv (height -> 1, width / 4) + 2-layer BiLSTM + CTC head. Blank = index 0."""

    def __init__(self, height=32, n_classes=len(ALPHABET) + 1, ch=(64, 128, 256, 256, 512)):
        super().__init__()
        c = ch
        layers, cin, h = [], 1, height
        plan = [(c[0], (2, 2)), (c[1], (2, 2)), (c[2], (2, 1)), (c[3], (2, 1)), (c[4], (2, 1))]
        for cout, pool in plan:
            layers += [nn.Conv2d(cin, cout, 3, 1, 1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                       nn.Conv2d(cout, cout, 3, 1, 1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                       nn.MaxPool2d(pool)]
            cin, h = cout, h // pool[0]
        self.cnn = nn.Sequential(*layers)
        self.proj = nn.Linear(cin * h, 256)
        self.rnn = nn.LSTM(256, 256, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        self.head = nn.Linear(512, n_classes)

    def forward(self, x):
        f = self.cnn(x)                                   # (B, C, h, W/4)
        f = f.permute(0, 3, 1, 2).flatten(2)
        y, _ = self.rnn(self.proj(f))
        return self.head(y)                               # (B, T, classes)


class LocalOCR(nn.Module):
    """CTC OCR whose output columns see only ~29 px of width (1-2 characters at 32 px height).

    The BiLSTM of `CRNN` learns a word-level language prior: with the dot below
    erased from real images of "mẹ", it still reads "mẹ" (19/32). Here the last
    three conv blocks are height-only (kernel 3x1) and there is no recurrence, so
    a mark has to be in the image to be read. Gate: scripts/check_evaluator.py.
    """

    def __init__(self, height=32, n_classes=len(ALPHABET) + 1, ch=(64, 128, 256, 256, 512)):
        super().__init__()
        layers, cin, h = [], 1, height
        plan = [(ch[0], (2, 2), 3), (ch[1], (2, 2), 3), (ch[2], (2, 1), 1), (ch[3], (2, 1), 1), (ch[4], (2, 1), 1)]
        for cout, pool, kw in plan:
            k, p = (3, kw), (1, kw // 2)
            layers += [nn.Conv2d(cin, cout, k, 1, p), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                       nn.Conv2d(cout, cout, k, 1, p), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                       nn.MaxPool2d(pool)]
            cin, h = cout, h // pool[0]
        self.cnn = nn.Sequential(*layers)
        self.proj = nn.Linear(cin * h, 256)
        self.local = nn.Sequential(nn.Conv1d(256, 512, 3, 1, 1), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
                                   nn.Conv1d(512, 512, 3, 1, 1), nn.BatchNorm1d(512), nn.ReLU(inplace=True))
        self.head = nn.Linear(512, n_classes)

    def forward(self, x):
        f = self.cnn(x).permute(0, 3, 1, 2).flatten(2)
        y = self.local(self.proj(f).transpose(1, 2)).transpose(1, 2)
        return self.head(y)


ARCHS = {'crnn': CRNN, 'local': LocalOCR}


def load_ocr(path, device):
    o = torch.load(path, map_location='cpu', weights_only=False)
    net = ARCHS[o.get('arch', 'crnn')](o['height']).to(device).eval()
    net.load_state_dict(o['model'])
    return net


class Codec:
    def __init__(self, alphabet=ALPHABET):
        self.chars = alphabet
        self.index = {c: i + 1 for i, c in enumerate(alphabet)}

    def encode(self, texts):
        flat = [self.index[c] for t in texts for c in t]
        return torch.tensor(flat), torch.tensor([len(t) for t in texts])

    def decode(self, logits):
        out = []
        for seq in logits.argmax(-1).tolist():
            s, prev = [], 0
            for k in seq:
                if k != prev and k != 0:
                    s.append(self.chars[k - 1])
                prev = k
            out.append(ud.normalize('NFC', ''.join(s)))
        return out


def ctc_loss(logits, targets, target_len, widths_px, down=4):
    lp = logits.float().log_softmax(-1).transpose(0, 1)
    in_len = torch.clamp(torch.ceil(widths_px.float() / down).long(), max=logits.shape[1])
    return F.ctc_loss(lp, targets, in_len, target_len, blank=0, zero_infinity=True)


def align(ref, hyp):
    """Levenshtein alignment; returns for each ref position the aligned hyp char or None."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    out, i, j = [None] * n, n, m
    while i > 0 and j > 0:
        if d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            out[i - 1] = hyp[j - 1]
            i, j = i - 1, j - 1
        elif d[i][j] == d[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return out, d[n][m]


def sibling(a, b):
    """Same base letter (case-insensitive), different marks or case: the error class of report 261118_02."""
    if a is None or b is None or a == b or not a.isalpha() or not b.isalpha():
        return False
    return parts(a)[0].lower() == parts(b)[0].lower()
