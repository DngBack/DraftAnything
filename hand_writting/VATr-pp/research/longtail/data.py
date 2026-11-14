"""Paired real-writer data whose targets AND supports obey ExposurePolicy."""
import hashlib
import atexit
from collections import OrderedDict
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from research.longtail.protocol import sampling_weights


_H5_CACHE = OrderedDict()
_CACHE_PID = os.getpid()


def close_h5_cache():
    for handle, _, _ in _H5_CACHE.values():
        handle.close()
    _H5_CACHE.clear()


atexit.register(close_h5_cache)


def packed_reader(path):
    global _CACHE_PID
    if _CACHE_PID != os.getpid():
        close_h5_cache()
        _CACHE_PID = os.getpid()
    key = str(Path(path).resolve())
    if key not in _H5_CACHE:
        import h5py
        handle = h5py.File(key, 'r')
        _H5_CACHE[key] = (handle, handle['img_seek_idxs'][:], handle['img_lens'][:])
        if len(_H5_CACHE) > 4:
            _, (old, _, _) = _H5_CACHE.popitem(last=False)
            old.close()
    _H5_CACHE.move_to_end(key)
    return _H5_CACHE[key]


def read_image(record):
    if 'image_path' in record:
        with Image.open(Path(record['image_path'])) as image:
            array = np.array(image.convert('L'), copy=True)
    else:
        handle, starts, widths = packed_reader(record['h5_path'])
        index = record['h5_index']
        start, width = int(starts[index]), int(widths[index])
        array = np.array(handle['imgs'][:, start:start+width], copy=True)
    actual = hashlib.sha256(str(array.shape).encode()+array.tobytes()).hexdigest()
    if actual != record['pixel_sha256']:
        raise ValueError('Image changed after audit: '+record['sample_id'])
    return Image.fromarray(array)


def tensor(image):
    return torch.from_numpy(np.array(image, dtype=np.float32, copy=True))[None]/127.5-1


class PairedEpisodes(Dataset):
    def __init__(self, policy, split='train', support_count=5, max_length=20,
                 sampler='natural', seed=742, per_writer=None):
        if split not in ('train', 'val', 'test') or support_count < 1 or max_length < 1:
            raise ValueError('Invalid episode configuration')
        self.policy, self.split, self.seed = policy, split, seed
        self.support_count = support_count
        rows = policy.training('image', genuine_only=True) if split == 'train' else [r for r in policy.records.values() if r['split'] == split and r['writer']]
        self.rows, self.pools, self.skipped = [], [], []
        for r in rows:
            pool = policy.supports_for(r, evaluation=split != 'train')
            if len(r['text']) > max_length or not pool:
                self.skipped.append(r['sample_id'])
            else:
                self.rows.append(r)
                self.pools.append(pool)
        if not self.rows:
            raise ValueError(f'No eligible {split} targets with different-transcript clean writer support')
        if per_writer is not None:
            if split == 'train' or per_writer < 1:
                raise ValueError('Positive per-writer panel size is evaluation-only')
            chosen = set()
            for writer in sorted({r['writer'] for r in self.rows}):
                indices = [i for i, r in enumerate(self.rows) if r['writer'] == writer]
                random.Random(seed).shuffle(indices)
                chosen.update(indices[:per_writer])
            # Retain every eligible heldout target in addition to ordinary writer controls.
            chosen.update(i for i, r in enumerate(self.rows) if policy.held.intersection(r['text']))
            self.rows = [r for i, r in enumerate(self.rows) if i in chosen]
            self.pools = [pool for i, pool in enumerate(self.pools) if i in chosen]
        self.weights = sampling_weights(self.rows, sampler)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row, pool = self.rows[index], self.pools[index]
        # Fixed held-out support for comparable panels; stochastic train support.
        rng = random if self.split == 'train' else random.Random(self.seed+index)
        chosen = rng.sample(pool, self.support_count) if len(pool) >= self.support_count else rng.choices(pool, k=self.support_count)
        style = []
        for r in chosen:
            image = read_image(r)
            image = image.resize((min(image.width, 192), 32), Image.Resampling.BICUBIC)
            padded = Image.new('L', (192, 32), 255)
            padded.paste(image, (0, 0))
            style.append(tensor(padded))
        image = read_image(row).resize((len(row['text'])*16, 32), Image.Resampling.BICUBIC)
        return {'image': tensor(image), 'style': torch.stack(style), 'text': row['text'],
                'sample_id': row['sample_id'], 'writer': row['writer'],
                'support_ids': [r['sample_id'] for r in chosen], 'width': image.width}


def collate(rows):
    images = torch.ones(len(rows), 1, 32, max(r['width'] for r in rows))
    for i, r in enumerate(rows):
        images[i, :, :, :r['width']] = r['image']
    return {'image': images, 'style': torch.stack([r['style'] for r in rows]),
            **{k: [r[k] for r in rows] for k in ('text', 'sample_id', 'writer', 'support_ids', 'width')}}
