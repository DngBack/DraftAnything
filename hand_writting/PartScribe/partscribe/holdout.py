"""Held-out-composition protocol v1 (docs/holdout_protocol_v1.json).

Single source of truth for what a generator, its prompts and its OCR
supervisor must never see. Stage-1 glyph pretraining may see font glyphs of
these characters under the protocol; the strict tokenizer variant excludes
them as well (see README).
"""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

PROTOCOL = Path(__file__).resolve().parents[1] / 'docs' / 'holdout_protocol_v1.json'
PROTOCOL_SHA256 = 'e4729308e8c068751db64bb9c19da6a70dc7a69d8dc3d9173a7c2144cbe3b12b'


@lru_cache(maxsize=1)
def load_protocol(path=PROTOCOL):
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != PROTOCOL_SHA256:
        raise ValueError(f'protocol file changed: {digest}')
    return json.loads(raw)


def vn_categories():
    return dict(load_protocol()['vn_holdout'])


def held_chars(dataset):
    p = load_protocol()
    if dataset == 'vn':
        return frozenset(''.join(p['vn_holdout'].values()))
    if dataset == 'iam':
        return frozenset(p['iam_holdout'])
    raise ValueError(dataset)


def contains_heldout(text, dataset):
    import unicodedata as ud
    return not held_chars(dataset).isdisjoint(ud.normalize('NFC', text))


def filter_texts(texts, dataset):
    """Drop every text containing a held-out character (prompts, corpora, transcripts)."""
    return [t for t in texts if not contains_heldout(t, dataset)]


def removed_vn_indices(split):
    """Indices of images to drop from a VATr-pp manifest split (train/val/test/hat_ocr)."""
    return frozenset(load_protocol()['vn'][split]['indices'])


def removed_iam_ids(split):
    """IAM 'writer,image-id' heads to drop (split: train_val/test)."""
    return frozenset(load_protocol()['iam'][split]['ids'])
