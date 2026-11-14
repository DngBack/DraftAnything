"""Metadata-first audit and shared exposure policy, independent of legacy trainers.

Records are one JSON object per image. sample_id, split, source, writer, text,
pixel_sha256 and image location are explicit. HAT has writer=null. Counts of
images and writers are intentionally distinct from character occurrences.
"""
from collections import Counter, defaultdict
import hashlib
import json
import random
import unicodedata as ud

from research.vietcompose.orthography import decompose_character


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def validate_records(records):
    if not records:
        raise ValueError('Empty records')
    ids, writer_splits, pixels = set(), defaultdict(set), defaultdict(list)
    for r in records:
        required = {'sample_id', 'split', 'source', 'writer', 'text', 'pixel_sha256'}
        if not required <= r.keys():
            raise ValueError(f'Missing record fields: {required-r.keys()}')
        if not isinstance(r['sample_id'], str) or not r['sample_id'] or r['sample_id'] in ids:
            raise ValueError('Empty, invalid or duplicate sample ID')
        ids.add(r['sample_id'])
        if r['split'] not in ('train', 'val', 'test'):
            raise ValueError('Invalid split')
        if not r['text'].strip() or ud.normalize('NFC', r['text']) != r['text']:
            raise ValueError('Text must be nonempty NFC')
        for c in r['text']:
            decompose_character(c)
        sha = r['pixel_sha256']
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError('A full lowercase pixel SHA256 is required')
        if r['source'] == 'HAT' and r['writer'] is not None:
            raise ValueError('HAT image IDs cannot be writer labels')
        if r['source'] != 'HAT' and (not isinstance(r['writer'], str) or not r['writer']):
            raise ValueError('Genuine writer key must be a nonempty string')
        if r['writer'] is not None:
            writer_splits[r['writer']].add(r['split'])
        pixels[sha].append(r)
    if any(len(s) > 1 for s in writer_splits.values()):
        raise ValueError('Writer leakage across train/val/test')
    # Duplicate pixels inflate image and support evidence even within a split.
    if any(len(group) > 1 for group in pixels.values()):
        raise ValueError('Duplicate pixel hash: deduplicate/quarantine before research')


def groups(text, mode='interaction'):
    result = set()
    for c in text:
        f = decompose_character(c)
        if mode == 'glyph':
            result.add('glyph:'+c)
        elif mode == 'interaction':
            result.update((f'base:{f.base}|{f.case}',
                           f'shape:{f.shape}|{f.case}',
                           f'tone:{f.tone}|{f.case}',
                           f'pair:{f.shape}|{f.tone}|{f.case}'))
        else:
            raise ValueError('Unknown grouping')
    return result


def audit(records):
    validate_records(records)
    result = {}
    for split in ('train', 'val', 'test'):
        for source in sorted({r['source'] for r in records}):
            rows = [r for r in records if r['split'] == split and r['source'] == source]
            if not rows:
                continue
            images, occurrences = Counter(), Counter()
            writers = defaultdict(set)
            for r in rows:
                occurrences.update(r['text'])
                for key in groups(r['text'], 'glyph') | groups(r['text']):
                    images[key] += 1
                    if r['writer'] is not None:
                        writers[key].add(r['writer'])
            result[split+'/'+source] = {
                'images': len(rows), 'genuine_writers': len({r['writer'] for r in rows if r['writer']}),
                'character_occurrences': dict(sorted(occurrences.items())),
                'group_images': dict(sorted(images.items())),
                'group_writers': {k: len(writers[k]) for k in sorted(images)}}
    return result


def make_protocol(records, holdout='ỄễỠỡẴẵ', budget=0, seed=742):
    validate_records(records)
    holdout = ''.join(sorted(set(ud.normalize('NFC', holdout))))
    if not holdout or budget < 0:
        raise ValueError('Nonempty holdout and nonnegative per-glyph image budget required')
    train = [r for r in records if r['split'] == 'train']
    clean = [r['sample_id'] for r in train if not set(holdout).intersection(r['text'])]
    candidates = [r for r in train if r['writer'] is not None and set(holdout).intersection(r['text'])]
    random.Random(seed).shuffle(candidates)
    # A single budget-independent ordering yields nested acquisition sets.
    acquired, acquired_set, counts = [], set(), Counter()
    for level in range(1, budget+1):
        for r in candidates:
            if r['sample_id'] in acquired_set:
                continue
            chars = set(holdout).intersection(r['text'])
            if all(counts[c] < level for c in chars):
                acquired.append(r['sample_id'])
                acquired_set.add(r['sample_id'])
                counts.update(chars)
    selected = set(clean+acquired)
    held_images = {c: sum(c in r['text'] for r in candidates) for c in holdout}
    return {'version': 1, 'records_sha256': digest(records), 'holdout': holdout,
            'seed': seed, 'per_glyph_budget': budget,
            'train_clean': sorted(clean), 'acquired': sorted(acquired),
            'train_allowed': sorted(selected),
            'acquired_images_per_glyph': {c: counts[c] for c in holdout},
            'available_real_images_per_glyph': held_images,
            'evaluation_real_images_per_glyph': {split: {c: sum(c in r['text'] for r in records if r['split'] == split and r['writer'] is not None) for c in holdout} for split in ('val', 'test')},
            'unfilled_budget': {c: budget-counts[c] for c in holdout if counts[c] < budget},
            'canonical_glyph_access': 'Full printed target glyph allowed for glyph baseline; no handwriting checkpoint pretraining.',
            'pretraining': 'scratch-only; legacy exposed checkpoints disallowed',
            'track': 'unseen-handwriting-combination' if budget == 0 else 'few-shot-real-handwriting',
            'note': 'HAT heldout images excluded at every budget. No claim of unseen canonical target glyph.'}


class ExposurePolicy:
    """All training consumers must query this policy, including text-only consumers."""
    def __init__(self, records, protocol):
        validate_records(records)
        if digest(records) != protocol['records_sha256']:
            raise ValueError('Records/protocol fingerprint mismatch')
        # Recompute rather than trusting arbitrary allow-lists in an edited JSON.
        expected = make_protocol(records, protocol['holdout'], protocol['per_glyph_budget'], protocol['seed'])
        if expected != protocol:
            raise ValueError('Protocol differs from deterministic exposure policy')
        self.records = {r['sample_id']: r for r in records}
        self.protocol = protocol
        self.allowed = set(protocol['train_allowed'])
        self.support = set(protocol['train_clean'])
        self.held = set(protocol['holdout'])
        # Index once: scanning all samples for each target makes real-data loading quadratic.
        self._sorted = [r for _, r in sorted(self.records.items())]
        self._train_support = defaultdict(list)
        self._eval_support = defaultdict(list)
        for r in self._sorted:
            if r['writer'] is None:
                continue
            if r['sample_id'] in self.support:
                self._train_support[r['writer']].append(r)
            if r['split'] in ('val', 'test') and not self.held.intersection(r['text']):
                self._eval_support[(r['split'], r['writer'])].append(r)

    def training(self, purpose, genuine_only=False):
        if purpose not in ('image', 'ocr', 'corpus', 'support', 'memory'):
            raise ValueError('Unknown training consumer')
        ids = self.support if purpose in ('support', 'memory') else self.allowed
        return [r for r in self._sorted if r['sample_id'] in ids
                and (not genuine_only or r['writer'] is not None)
                and (purpose not in ('support', 'memory') or r['writer'] is not None)]

    def assert_training_texts(self, texts):
        # A corpus must be constructed from admitted samples, never arbitrary prompts.
        allowed_text = {r['text'] for r in self.training('corpus')}
        if any(ud.normalize('NFC', t) not in allowed_text for t in texts):
            raise ValueError('Text outside exposure-controlled train corpus')

    def supports_for(self, target, evaluation=False):
        if evaluation:
            if target['split'] not in ('val', 'test'):
                raise ValueError('Evaluation target must be held-out writer split')
            pool = self._eval_support.get((target['split'], target['writer']), [])
        else:
            if target['sample_id'] not in self.allowed:
                raise ValueError('Target outside admitted training images')
            pool = self._train_support.get(target['writer'], [])
        return [r for r in pool if r['writer'] is not None and r['writer'] == target['writer']
                and r['sample_id'] != target['sample_id'] and r['text'] != target['text']]


def sampling_weights(records, mode='natural', mixture=.5, beta=.99, cap=20.):
    """Natural image distribution mixed with capped group-balanced image weights.

    Counts use unique image membership. Zero-count groups have no sampling mass.
    This is an ablation of sampling, not a new class-balanced loss theorem.
    """
    if not records or not 0 <= mixture <= 1 or not 0 <= beta < 1 or cap < 1:
        raise ValueError('Invalid sampler inputs')
    if mode == 'natural':
        return [1/len(records)]*len(records)
    keys = [groups(r['text'], mode) for r in records]
    counts = Counter(k for row in keys for k in row)
    # Normalization makes cap a relative multiplier on the balanced arm.
    raw = [sum((1-beta)/(1-beta**counts[k]) for k in row)/len(row) for row in keys]
    avg = sum(raw)/len(raw)
    raw = [min(x/avg, cap) for x in raw]
    total = sum(raw)
    return [(1-mixture)/len(records)+mixture*x/total for x in raw]
