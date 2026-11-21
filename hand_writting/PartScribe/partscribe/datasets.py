"""Word-image datasets with the held-out protocol applied at load time, writer batches and support sets.

Both datasets are packed the same way: one (H, total_width) uint8 array with
white background, plus per-image offset/width/text/writer. Everything is held
in RAM (VN train ~0.4 GB at 32 px; IAM ~1 GB at 64 px).

Ink convention for models: ink = 1 - pixel/255 (background and padding = 0).
"""
import hashlib
import json
import random
import unicodedata as ud
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .holdout import contains_heldout, removed_iam_ids, removed_vn_indices

HAT_FIRST_WRITER = 295   # VATr-pp convention: writer ids >= 295 are HAT pseudo-writers (no style identity)
VNONDB_LAST_WRITER = 105  # VATr-pp: 0-105 VNOnDB, 106-294 ViHTGen (merged, globally unique writer ids)


_SHAPES = {'\u0302': 'circumflex', '\u0306': 'breve', '\u031b': 'horn'}
_TONES = {'\u0300': 'grave', '\u0301': 'acute', '\u0309': 'hook', '\u0303': 'tilde', '\u0323': 'dot_below'}


def factors(c):
    """(base, shape, tone, case) of one character, as VATr-pp research/longtail/protocol.decompose_character."""
    d = ud.normalize('NFD', c)
    base, marks = d[0], d[1:]
    shape, tone = 'none', 'none'
    for m in marks:
        if m in _SHAPES:
            shape = _SHAPES[m]
        elif m in _TONES:
            tone = _TONES[m]
    if base in 'đĐ':
        base, shape = 'd', 'crossbar'
        case = 'upper' if c == 'Đ' else 'lower'
    elif base.isalpha():
        case = 'upper' if base.isupper() else 'lower'
        base = base.lower()
    else:
        case = 'literal'
    return base, shape, tone, case


def interaction_groups(text):
    """VATr-pp 'interaction' groups: base / shape / tone / (shape, tone), each conditioned on case."""
    out = set()
    for c in text:
        b, sh, to, ca = factors(c)
        out.update((f'base:{b}|{ca}', f'shape:{sh}|{ca}', f'tone:{to}|{ca}', f'pair:{sh}|{to}|{ca}'))
    return out


def longtail_weights(texts, component_mix=0.5, cap=20.):
    """Rare-draw weights of VATr-pp's ComponentCorpusSampler (data/vietnamese.py CorpusSampler +
    util/vietnamese_longtail.py), computed over the given transcripts:
      legacy    = min(cap, mean_{c in set(t)} sqrt(N / count_char[c]))
      component = min(cap, mean_{g in groups(t)} sqrt(N / n_texts_with[g]))
      weight    = (1 - mix) * legacy + mix * component"""
    from collections import Counter
    n = len(texts)
    cc = Counter(''.join(texts))
    gs = [interaction_groups(t) for t in texts]
    gc = Counter(g for s in gs for g in s)
    w = []
    for t, s in zip(texts, gs):
        u = set(t)
        legacy = min(cap, sum((n / cc[c]) ** .5 for c in u) / len(u))
        comp = min(cap, sum((n / gc[g]) ** .5 for g in s) / len(s))
        w.append((1 - component_mix) * legacy + component_mix * comp)
    return np.array(w)


def vn_source(writer):
    return 'VNOnDB' if writer <= VNONDB_LAST_WRITER else 'ViHTGen'



class PackedWords:
    def __init__(self, pixels, starts, widths, texts, writers, ids, name, source_of=None):
        self.source_of = source_of or (lambda w: name.split('/')[0])   # writer -> source dataset
        self.pixels, self.starts, self.widths = pixels, np.asarray(starts), np.asarray(widths)
        self.texts, self.writers, self.ids, self.name = list(texts), np.asarray(writers), list(ids), name
        self.height = pixels.shape[0]
        self.pools = defaultdict(list)
        for i, w in enumerate(self.writers):
            self.pools[int(w)].append(i)

    def __len__(self):
        return len(self.texts)

    def image(self, i):
        a, w = int(self.starts[i]), int(self.widths[i])
        return self.pixels[:, a:a + w]

    def assert_clean(self, dataset):
        bad = [t for t in self.texts if contains_heldout(t, dataset)]
        assert not bad, f'{len(bad)} held-out transcripts leaked into {self.name}: {bad[:5]}'

    def filter(self, keep):
        keep = np.asarray(keep, bool)
        idx = np.nonzero(keep)[0]
        return PackedWords(self.pixels, self.starts[idx], self.widths[idx], [self.texts[i] for i in idx],
                           self.writers[idx], [self.ids[i] for i in idx], self.name, self.source_of)


def load_vn(manifest_path, data_dir, split, apply_holdout=True, verify_sha=None):
    """Words of one VATr-pp manifest split from train.h5/test.h5, HAT pseudo-writers excluded.

    Same data policy as VATr-pp (docs/VIETNAMESE.md): VNOnDB + ViHTGen merged with
    globally unique writer ids; 'train' = 265 writers, 'val' = 30 disjoint writers
    (11 VNOnDB + 19 ViHTGen, seed 742); 'test' has its own id space.
    """
    import h5py
    man = json.loads(Path(manifest_path).read_text())
    path = Path(data_dir) / ('test.h5' if split == 'test' else 'train.h5')
    if verify_sha:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 24), b''):
                h.update(chunk)
        assert h.hexdigest() == verify_sha, f'{path} does not match the recorded fingerprint'
    idx = list(man['indices'][split])
    if apply_holdout:
        drop = removed_vn_indices(split)
        idx = [i for i in idx if i not in drop]
    with h5py.File(path, 'r') as f:
        starts, widths, wids = f['img_seek_idxs'][:], f['img_lens'][:], f['wids'][:]
        lbs, ls, ll = f['lbs'][:], f['lb_seek_idxs'][:], f['lb_lens'][:]
        hat = split == 'hat_ocr'
        idx = [i for i in idx if (wids[i] >= HAT_FIRST_WRITER) == hat]
        pixels = f['imgs'][:]
    texts = [ud.normalize('NFC', ''.join(map(chr, lbs[ls[i]:ls[i] + ll[i]]))) for i in idx]
    if hat and apply_holdout:
        # stricter than protocol v1 for this extra source: also drop the capital forms of the
        # held-out characters (NFD letter tokens are shared across case)
        from .holdout import held_chars
        held = set(held_chars('vn')) | {c.upper() for c in held_chars('vn')}
        keep = [k for k, t in enumerate(texts) if held.isdisjoint(t)]
        idx, texts = [idx[k] for k in keep], [texts[k] for k in keep]
    ds = PackedWords(pixels, starts[idx], widths[idx], texts, wids[idx], idx, f'vn/{split}', vn_source)
    if apply_holdout:
        ds.assert_clean('vn')
    return ds


def load_iam(packed_npz, split, apply_holdout=True):
    """IAM words prepared by scripts/prepare_iam.py; split = 'train_val' | 'test' (VATr Groundtruth lists)."""
    z = np.load(packed_npz, allow_pickle=False)
    meta = json.loads(Path(str(packed_npz).replace('.npz', '.json')).read_text())
    sel = [i for i, s in enumerate(meta['split']) if s == split]
    if apply_holdout:
        drop = removed_iam_ids(split)
        sel = [i for i in sel if meta['head'][i] not in drop]
    ds = PackedWords(z['pixels'], z['starts'][sel], z['widths'][sel], [meta['text'][i] for i in sel],
                     np.array([int(meta['writer'][i]) for i in sel]), [meta['head'][i] for i in sel], f'iam/{split}')
    if apply_holdout:
        ds.assert_clean('iam')
    return ds


def to_ink(arr):
    return 1.0 - torch.as_tensor(np.ascontiguousarray(arr), dtype=torch.float32) / 255.0


def ink_normalised(arr):
    """Ink in [0,1] with the background removed: scanned crops (HAT2023) have grey, uneven
    backgrounds. Background = 60th percentile, full ink = 1st percentile; faint residue cut."""
    a = np.asarray(arr, dtype=np.float32)
    bg, fg = np.percentile(a, 60), np.percentile(a, 1)
    if bg - fg < 20:
        return to_ink(arr)
    x = np.clip((bg - a) / (bg - fg), 0, 1)
    x[x < 0.2] = 0
    return torch.from_numpy(np.ascontiguousarray(x))


def fit_width(arr, max_w):
    """Compress horizontally (never crop) when wider than max_w."""
    if arr.shape[1] <= max_w:
        return arr
    return np.asarray(Image.fromarray(arr).resize((max_w, arr.shape[0]), Image.BICUBIC))


class Batcher:
    """Writer-structured batches: `writers_per_batch` writers x `per_writer` targets.

    Writers are drawn as in VATr-pp's GAN sampler: a source dataset uniformly
    (VNOnDB / ViHTGen), then a writer uniformly within it, so the larger source
    does not dominate. IAM has one source.

    Support for each target: `n_support` other images of the same writer, never
    the target itself and never an image with the same transcript (anti-copy).
    Targets wider than `max_width` px or with unsupported text are skipped.

    Long tail (VATr-pp's text sampler, transferred to target images): with
    probability `rare_fraction` a writer slot is a *rare draw*: the anchor image
    is drawn over all targets with `longtail_weights` (VATr-pp defaults: 0.3,
    component mix 0.5) and the slot's other targets from that writer are drawn
    with the same weights. Otherwise: source -> writer -> uniform image.
    rare_fraction=0 reproduces the natural-frequency sampler.

    multiword: probability that a target becomes 2-3 words of the SAME writer joined by a
    blank gap of 0.25-0.6 x height (margins trimmed), text joined with spaces. The source
    data are single words (109 of 96k transcripts contain a space), so without this the
    space token is untrained and phrases such as "Hà Nội" come out with stray strokes.

    mark_weight: the batch carries a pixel weight map 1 + mark_weight on small connected ink
    components (diacritics, dots; area <= 2.5% of height^2 and height <= 30% of height),
    so the flow loss is not dominated by the large strokes.
    """

    def __init__(self, ds, supports_text, writers_per_batch=16, per_writer=4, n_support=15,
                 max_width=512, support_width=None, width_multiple=8, max_chars=40, seed=0,
                 rare_fraction=0.0, component_mix=0.5, multiword=0.0, mark_weight=0.0,
                 content_ds=None, content_per_batch=0):
        self.ds, self.rng = ds, random.Random(seed)
        self.content_ds, self.content_per_batch = content_ds, content_per_batch if content_ds is not None else 0
        if self.content_per_batch:
            assert content_ds.height == ds.height
            self.content_targets = [i for i in range(len(content_ds)) if content_ds.widths[i] <= max_width
                                    and 0 < len(content_ds.texts[i]) <= max_chars and supports_text(content_ds.texts[i])]
        self.P, self.K, self.n_support = writers_per_batch, per_writer, n_support
        # VATr-pp: 15 supports of width 192 at 32 px; keep that aspect (6 x height) at other heights.
        support_width = support_width or 6 * ds.height
        self.max_width, self.support_width, self.mult = max_width, support_width, width_multiple
        self.max_chars = max_chars
        self._clean_cache = {}
        ok = [i for i in range(len(ds)) if ds.widths[i] <= max_width and 0 < len(ds.texts[i]) <= max_chars
              and supports_text(ds.texts[i])]
        self.targets = defaultdict(list)
        for i in ok:
            self.targets[int(ds.writers[i])].append(i)
        self.writers = sorted(w for w, v in self.targets.items() if v and len(ds.pools[w]) > 1)
        self.by_source = defaultdict(list)
        for w in self.writers:
            self.by_source[ds.source_of(w)].append(w)
        self.sources = sorted(self.by_source)
        self.writer_ids = {w: k for k, w in enumerate(sorted(ds.pools))}
        self.rare_fraction = rare_fraction
        self.multiword, self.mark_weight = multiword, mark_weight
        if rare_fraction > 0:
            flat = [i for w in self.writers for i in self.targets[w]]
            wt = longtail_weights([ds.texts[i] for i in flat], component_mix)
            self.weight = dict(zip(flat, wt.tolist()))
            self.flat, self.flat_cum = flat, np.cumsum(wt).tolist()
            self.cum = {w: np.cumsum([self.weight[i] for i in self.targets[w]]).tolist() for w in self.writers}

    def exposure(self, chars):
        """P(a target contains c) per char, under this sampler (natural part: source->writer->uniform)."""
        nat = defaultdict(float)
        for src in self.sources:
            ws = self.by_source[src]
            for w in ws:
                ts = self.targets[w]
                for c in chars:
                    nat[c] += sum(c in self.ds.texts[i] for i in ts) / len(ts) / len(ws) / len(self.sources)
        out = {}
        for c in chars:
            rare = 0.
            if self.rare_fraction > 0:
                tot = self.flat_cum[-1]
                rare = sum(self.weight[i] for i in self.flat if c in self.ds.texts[i]) / tot
            out[c] = (1 - self.rare_fraction) * nat[c] + self.rare_fraction * rare
        return out

    def support_for(self, i):
        members = i if isinstance(i, tuple) else (i,)
        w = int(self.ds.writers[members[0]])
        texts = {self.ds.texts[m] for m in members}
        cand = [j for j in self.ds.pools[w] if j not in members and self.ds.texts[j] not in texts]
        if not cand:
            cand = [j for j in self.ds.pools[w] if j not in members]
        if len(cand) >= self.n_support:
            return self.rng.sample(cand, self.n_support)
        return [self.rng.choice(cand) for _ in range(self.n_support)]

    def _ink(self, item):
        """(ink image (h, w) float tensor, text) for an index or a tuple of same-writer indices."""
        ds = self.ds
        if not isinstance(item, tuple):
            return to_ink(ds.image(item)), ds.texts[item]
        h = ds.height
        ims, cores = [], []
        for m in item:
            im = to_ink(ds.image(m))
            cols = torch.nonzero(im.amax(0) > 0.5).flatten()
            if len(cols):
                im = im[:, max(0, int(cols[0]) - 2):int(cols[-1]) + 3]
            ims.append(im); cores.append(self._core(im))
        # Every word was height-normalised on its own, so pasting them side by side puts each on
        # a different baseline and body size. Like a real line image: bring words without capitals
        # to a common body (x-) height, put every word on one baseline on a canvas tall enough to
        # hold all of them, then scale the whole line back to h (one scale for all words).
        low = [c[1] - c[0] + 1 for c, m in zip(cores, item) if c and not any(ch.isupper() for ch in ds.texts[m])]
        ref_body = float(np.median(low)) if low else None
        placed, texts = [], []
        for im, c, m in zip(ims, cores, item):
            rows = torch.nonzero(im.amax(1) > 0.5).flatten()
            if c is None or not len(rows):
                if not placed:
                    return to_ink(ds.image(item[0])), ds.texts[item[0]]
                break
            s = 1.0
            if ref_body and not any(ch.isupper() for ch in ds.texts[m]):
                s = float(np.clip(ref_body / (c[1] - c[0] + 1), 0.8, 1.25))
            if s != 1.0:
                im = torch.nn.functional.interpolate(im[None, None], scale_factor=s, mode='bilinear',
                                                     align_corners=False)[0, 0]
            rows = torch.nonzero(im.amax(1) > 0.5).flatten()
            base = int(round(c[1] * s))
            placed.append((im[int(rows[0]):int(rows[-1]) + 1], base - int(rows[0])))  # (crop, baseline row)
            texts.append(ds.texts[m])
        up = max(b for _, b in placed)
        down = max(p.shape[0] - b for p, b in placed)
        H = max(h, up + down)
        parts = []
        for k, (p, b) in enumerate(placed):
            col = torch.zeros(H, p.shape[1]); y = up - b + (H - up - down) // 2
            col[y:y + p.shape[0]] = p
            if k:
                parts.append(torch.zeros(H, self.rng.randint(int(0.25 * h), int(0.6 * h))))
            parts.append(col)
        line = torch.cat(parts, 1)
        if H > h:
            line = torch.nn.functional.interpolate(line[None, None], size=(h, max(1, round(line.shape[1] * h / H))),
                                                   mode='area')[0, 0]
            line = (line * (H / h)).clamp(0, 1)  # area averaging thins strokes: restore ink density
        return line, ' '.join(texts)

    def _core(self, ink):
        """(top, bottom) rows of the word body: marks (small components) removed, rows holding
        at least 35% of the peak ink. bottom ~ baseline (descenders are thin, below the band)."""
        from scipy import ndimage
        h = ink.shape[0]
        a = ink.numpy() > 0.5
        lab, n = ndimage.label(a, structure=np.ones((3, 3)))
        if n == 0:
            return None
        objs = ndimage.find_objects(lab)
        area = np.bincount(lab.ravel())[1:]
        small = [k + 1 for k, (o, ar) in enumerate(zip(objs, area))
                 if ar <= 0.025 * h * h and (o[0].stop - o[0].start) <= 0.3 * h]
        body = a & ~np.isin(lab, small)
        p = body.sum(1)
        if p.max() == 0:
            return None
        rows = np.nonzero(p >= 0.35 * p.max())[0]
        return int(rows[0]), int(rows[-1])

    def _mark_map(self, ink):
        from scipy import ndimage
        h = ink.shape[0]
        lab, n = ndimage.label(ink.numpy() > 0.5, structure=np.ones((3, 3)))
        if n == 0:
            return torch.ones_like(ink)
        objs = ndimage.find_objects(lab)
        area = np.bincount(lab.ravel())[1:]
        small = [k + 1 for k, (o, a) in enumerate(zip(objs, area))
                 if a <= 0.025 * h * h and (o[0].stop - o[0].start) <= 0.3 * h]
        m = ndimage.binary_dilation(np.isin(lab, small), iterations=1)
        return 1 + self.mark_weight * torch.from_numpy(m).float()

    def collate(self, items, supports, content=()):
        """content: indices of content-only targets (content_ds), appended after `items`. They carry
        writer -1 and borrow the supports of real targets only to keep tensor shapes; the model
        ignores their style (forced style drop, null g for the width head, no SupCon/anti-content)."""
        ds, h = self.ds, self.ds.height
        n, nc = len(items), len(content)
        inks = [self._ink(i) for i in items]
        inks += [(ink_normalised(self.content_ds.image(j)), self.content_ds.texts[j]) for j in content]
        widths = [im.shape[1] for im, _ in inks]
        W = int(np.ceil(max(widths) / self.mult) * self.mult)
        x = torch.zeros(n + nc, 1, h, W)
        for k, (im, _) in enumerate(inks):
            x[k, 0, :, :widths[k]] = im
        sup = torch.zeros(n + nc, self.n_support, h, self.support_width)
        sup_w = torch.zeros(n + nc, self.n_support)
        for k, js in enumerate(supports):
            for m, j in enumerate(js):
                a = fit_width(ds.image(j), self.support_width)
                sup[k, m, :, :a.shape[1]] = to_ink(a)
                sup_w[k, m] = a.shape[1]
        borrow = [self.rng.randrange(n) for _ in content]
        if nc:
            sup[n:], sup_w[n:] = sup[borrow], sup_w[borrow]
        first = lambda i: i[0] if isinstance(i, tuple) else i
        out = {'image': x, 'width': torch.tensor(widths), 'text': [t for _, t in inks],
               'writer': torch.tensor([self.writer_ids[int(ds.writers[first(i)])] for i in items] + [-1] * nc),
               'support': sup, 'support_width': sup_w,
               'support_text': [[ds.texts[j] for j in js] for js in supports]
                               + [[ds.texts[j] for j in supports[b]] for b in borrow],
               'index': list(items) + [('content', j) for j in content]}
        if self.mark_weight:
            wmap = torch.ones(n + nc, 1, h, W)
            for k, (im, _) in enumerate(inks):
                wmap[k, 0, :, :widths[k]] = self._mark_map(im)
            out['pix_weight'] = wmap
        return out

    def _clean(self, i):
        """White background (scanned grey-background crops would leave visible boxes when joined)."""
        c = self._clean_cache.get(i)
        if c is None:
            im = self.ds.image(i)
            c = self._clean_cache[i] = bool(np.percentile(im, 25) >= 250 and min(im[0].mean(), im[-1].mean(), im[:, 0].mean(), im[:, -1].mean()) >= 245)
        return c

    def _maybe_join(self, i):
        if not self.multiword or self.rng.random() >= self.multiword:
            return i
        w, ds = int(self.ds.writers[i]), self.ds
        if not self._clean(i):
            return i
        members, width, chars = [i], int(ds.widths[i]), len(ds.texts[i])
        for _ in range(self.rng.choice((1, 1, 2))):
            j = self.rng.choice(self.targets[w])
            if j in members or not self._clean(j) or width + int(ds.widths[j]) + ds.height > self.max_width \
                    or chars + len(ds.texts[j]) + 1 > self.max_chars:
                break
            members.append(j); width += int(ds.widths[j]) + ds.height // 2; chars += len(ds.texts[j]) + 1
        return tuple(members) if len(members) > 1 else i

    def __iter__(self):
        while True:
            items = []
            for _ in range(self.P):
                if self.rare_fraction > 0 and self.rng.random() < self.rare_fraction:
                    a = self.rng.choices(self.flat, cum_weights=self.flat_cum)[0]
                    w = int(self.ds.writers[a])
                    items += [a] + self.rng.choices(self.targets[w], cum_weights=self.cum[w], k=self.K - 1)
                else:
                    w = self.rng.choice(self.by_source[self.rng.choice(self.sources)])
                    items += [self.rng.choice(self.targets[w]) for _ in range(self.K)]
            items = [self._maybe_join(i) for i in items]
            content = self.rng.sample(self.content_targets, self.content_per_batch) if self.content_per_batch else ()
            yield self.collate(items, [self.support_for(i) for i in items], content)
