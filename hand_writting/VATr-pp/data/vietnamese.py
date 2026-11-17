"""Lazy, worker-safe HDF5 datasets with audited splits and genuine writer support."""
import json
import hashlib
import itertools
import os
import random
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler


def load_manifest(path, verify_sources=True):
    path = Path(path)
    manifest = json.loads(path.read_text(encoding='utf-8'))
    overlay = os.environ.get('VATRPP_PATHS_JSON')
    if overlay:
        from repro.paths import relocate_manifest
        manifest = relocate_manifest(manifest, path, overlay, verify_sources=verify_sources)
    elif verify_sources:
        for key, filename in [('source_train', 'train.h5'), ('source_test', 'test.h5')]:
            stat = Path(manifest[key]).stat()
            expected = manifest['source_files'][filename]
            if stat.st_size != expected['size'] or stat.st_mtime_ns != expected['mtime_ns']:
                raise ValueError(f'Source changed since audit: {manifest[key]}; rerun preparation')
    if manifest.get('glyph_sha256') and hashlib.sha256(Path(manifest['glyph_path']).read_bytes()).hexdigest() != manifest['glyph_sha256']:
        raise ValueError('Glyph file changed since preparation')
    return manifest


class PackedH5(Dataset):
    def __init__(self, manifest, split='train', include_hat=False, style=False, num_examples=15, style_width=192,
                 include_hat_real=False, hat_support='real-writer', hat_real_fraction=.35):
        self.manifest = load_manifest(manifest) if isinstance(manifest, (str, Path)) else manifest
        self.path = self.manifest['source_test' if split == 'test' else 'source_train']
        self.indices = list(self.manifest['indices'][split])
        if include_hat_real and (not style or split != 'train'):
            raise ValueError('HAT real images require GAN train mode')
        if hat_support not in ('real-writer', 'self-repeat') or not 0 < hat_real_fraction < 1:
            raise ValueError('Invalid HAT support policy or real-image fraction')
        self.include_hat_real = include_hat_real
        self.hat_support = hat_support
        self.hat_real_fraction = hat_real_fraction
        if include_hat:
            if split != 'train' or style:
                raise ValueError('HAT is only valid for train OCR, never style support')
            self.indices += self.manifest['indices']['hat_ocr']
        self.style, self.num_examples, self.style_width = style, num_examples, style_width
        self._h5 = None
        self._pid = None
        with h5py.File(self.path, 'r') as f:
            self.starts = f['img_seek_idxs'][:]
            self.widths = f['img_lens'][:]
            self.writers = f['wids'][:]
            labels, starts, lengths = f['lbs'][:], f['lb_seek_idxs'][:], f['lb_lens'][:]
        self.labels = [unicodedata.normalize('NFC', ''.join(chr(int(c)) for c in labels[int(a):int(a)+int(n)])) for a, n in zip(starts, lengths)]
        self.pools = defaultdict(list)
        for i in self.indices:
            self.pools[int(self.writers[i])].append(i)
        self.writer_map = {wid: j for j, wid in enumerate(sorted(self.pools))}
        if style and split != 'test' and any(w >= 295 for w in self.pools):
            raise ValueError('Pseudo writers found in style dataset')
        if style and any(len(pool) < 2 for pool in self.pools.values()):
            raise ValueError('Every style writer needs multiple images')
        # Build genuine support pools before adding pseudo-writer targets.
        if include_hat_real:
            merged = self.manifest['indices'].get('gan_train')
            expected = self.indices + self.manifest['indices']['hat_ocr']
            if merged != expected or len(set(merged)) != len(merged):
                raise ValueError('Audited gan_train union required for HAT real images')
            self.indices = list(merged)
        self.real_writers = list(self.pools)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_h5'] = None
        state['_pid'] = None
        return state

    def image(self, index):
        if self._h5 is None or self._pid != os.getpid():
            self._h5 = h5py.File(self.path, 'r')
            self._pid = os.getpid()
        a, w = int(self.starts[index]), int(self.widths[index])
        return Image.fromarray(self._h5['imgs'][:, a:a+w])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        image, text = self.image(i), self.labels[i]
        # Natural widths for OCR; matched 16px/codepoint widths for GAN real/fake.
        if self.style:
            image = image.resize((16*len(text), 32), Image.Resampling.BICUBIC)
        item = {'img': to_tensor(image), 'label': text, 'index': i,
                'width': image.width, 'writer': self.writer_map.get(int(self.writers[i]), -1)}
        if self.style:
            is_hat = int(self.writers[i]) >= 295
            if is_hat and self.hat_support == 'self-repeat':
                # Explicit single-image reference, not evidence of writer identity.
                pool = [i]
            elif is_hat:
                pool = self.pools[random.choice(self.real_writers)]
            else:
                pool = self.pools[int(self.writers[i])]
            # Support does not contain the target image; prefer different transcripts.
            candidates = [j for j in pool if self.labels[j] != text]
            if not candidates:
                candidates = [j for j in pool if j != i]
            if not candidates and is_hat and self.hat_support == 'self-repeat':
                candidates = [i]
            chosen = random.sample(candidates, self.num_examples) if len(candidates) >= self.num_examples else random.choices(candidates, k=self.num_examples)
            samples = []
            for j in chosen:
                im = self.image(j)
                if im.width > self.style_width:
                    # Compress entire sample horizontally; never truncate a suffix.
                    im = im.resize((self.style_width, 32), Image.Resampling.BICUBIC)
                padded = Image.new('L', (self.style_width, 32), 255)
                padded.paste(im, (0, 0))
                samples.append(to_tensor(padded)[0])
            item['style'] = torch.stack(samples)
            item['support_indices'] = chosen
        return item

    def sampler(self):
        # OCR: equal source mass; GAN: equal source, then equal genuine writer mass.
        sources = [0 if self.writers[i] < 106 else (1 if self.writers[i] < 295 else 2) for i in self.indices]
        source_counts = Counter(sources)
        writer_counts = Counter(int(self.writers[i]) for i in self.indices)
        source_writers = {s: len({int(self.writers[i]) for i, group in zip(self.indices, sources) if group == s}) for s in source_counts}
        weights = [1.0/(writer_counts[int(self.writers[i])]*source_writers[s]) if self.style else 1.0/source_counts[s] for i, s in zip(self.indices, sources)]
        if self.include_hat_real:
            # Fixed HAT image mass, then equal source and genuine-writer mass.
            real_source_count = sum(s != 2 for s in source_counts)
            weights = [(self.hat_real_fraction/source_counts[2] if s == 2 else
                        (1-self.hat_real_fraction)/(real_source_count*writer_counts[int(self.writers[i])]*source_writers[s]))
                       for i, s in zip(self.indices, sources)]
        return WeightedRandomSampler(weights, len(self), replacement=True)


def to_tensor(image):
    return torch.from_numpy(np.array(image, dtype=np.float32, copy=True)).unsqueeze(0) / 127.5 - 1


def collate(batch):
    widths = torch.tensor([b['width'] for b in batch], dtype=torch.long)
    images = torch.ones(len(batch), 1, 32, max(8, int(widths.max())))
    for k, b in enumerate(batch):
        images[k, :, :, :b['width']] = b['img']
    result = {'img': images, 'widths': widths, 'label': [b['label'] for b in batch],
              'writer': torch.tensor([b['writer'] for b in batch]),
              'indices': [b['index'] for b in batch]}
    if 'style' in batch[0]:
        result['style'] = torch.stack([b['style'] for b in batch])
    return result


class CorpusSampler:
    """Blend empirical transcripts and rare-character-weighted real transcripts."""
    def __init__(self, labels, max_length=20, rare_fraction=.3):
        self.words = [t for t in labels if t.strip() and 1 <= len(t) <= max_length]
        if not self.words:
            raise ValueError('No usable train transcripts')
        counts = Counter(''.join(self.words))
        self.weights = [min(20., sum((len(self.words)/counts[c])**.5 for c in set(t))/len(set(t))) for t in self.words]
        self.cumulative_weights = list(itertools.accumulate(self.weights))
        self.rare_fraction = rare_fraction

    def sample(self, n):
        return [random.choices(self.words, cum_weights=self.cumulative_weights, k=1)[0] if random.random() < self.rare_fraction else random.choice(self.words) for _ in range(n)]
