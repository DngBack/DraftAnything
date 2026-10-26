"""Controlled Long-KIE (Tier 3) in token space.

Each page is n tokens; each token is a tuple (kind, field, entity, value) embedded by a
fixed random projection plus noise, standing in for a frozen page encoder. Schema fields
get one reasoning type whose evidence is spread over pages with real dependencies.
"""
from dataclasses import dataclass
import numpy as np
import torch

FILLER, ASSERT, AMEND, REF, DEF, ITEM = range(6)
N_KINDS = 6
TYPES = ['lookup', 'revision', 'link', 'aggregation', 'missing']


@dataclass
class Spec:
    n_tokens: int = 64          # tokens per page
    n_fields: int = 64          # field vocabulary; schema draws F of them
    n_entities: int = 256
    n_values: int = 64
    train_entities: int = 192   # link entities >= this only appear in the unseen-entity test
    d_in: int = 128
    noise: float = 0.5          # per-token noise norm relative to one attribute embedding
    type_probs: tuple = (.2, .2, .25, .15, .2)
    distractor_defs: int = 3    # x F entity definitions unrelated to the schema
    distractor_fields: int = 1  # x F asserts/amends for non-schema fields


class Encoder:
    """Frozen random attribute tables; same seed for train and test."""

    def __init__(self, spec, seed=0):
        g = np.random.default_rng(seed)
        s = spec.d_in ** -.5
        self.tables = [g.normal(0, s, (n, spec.d_in)).astype(np.float32)
                       for n in (N_KINDS, spec.n_fields + 1, spec.n_entities + 1, spec.n_values + 1)]
        self.spec = spec

    def __call__(self, attrs, gen=None):
        """attrs: long tensor [..., 4] -> float [..., d_in], computed on attrs.device."""
        if not hasattr(self, 'dev') or self.dev[0].device != attrs.device:
            self.dev = [torch.from_numpy(t).to(attrs.device) for t in self.tables]
        x = sum(t[attrs[..., i]] for i, t in enumerate(self.dev))
        noise = torch.randn(x.shape, generator=gen, device=x.device)
        return x + noise * (self.spec.noise * self.spec.d_in ** -.5)


def make_doc(spec, T, F, rng, unseen=False):
    n = spec.n_tokens
    kind = np.zeros((T, n), np.int64)
    field = np.zeros((T, n), np.int64)
    ent = rng.integers(1, spec.n_entities + 1, (T, n))
    val = rng.integers(1, spec.n_values + 1, (T, n))
    free = [list(rng.permutation(n)) for _ in range(T)]

    def put(p, k, f, e, v):
        j = free[p].pop()
        kind[p, j], field[p, j], ent[p, j], val[p, j] = k, f, e, v

    def pages(k):
        return np.sort(rng.choice(T, k, replace=False))

    lo, hi = (spec.train_entities, spec.n_entities) if unseen else (0, spec.train_entities)
    used_ent = rng.choice(np.arange(lo, hi) + 1, F * (1 + spec.distractor_defs), replace=False)
    fields = rng.choice(spec.n_fields, F + F * spec.distractor_fields, replace=False) + 1
    schema, others = fields[:F], fields[F:]
    ftype = rng.choice(len(TYPES), F, p=spec.type_probs)
    gold = np.full(F, spec.n_values, np.int64)          # class n_values == null
    evid = np.zeros((F, T), bool)
    for i, (f, ty) in enumerate(zip(schema, ftype)):
        name = TYPES[ty]
        if name == 'lookup':
            p, = pages(1); v = rng.integers(spec.n_values)
            put(p, ASSERT, f, 0, v + 1); gold[i] = v; evid[i, p] = True
        elif name == 'revision':
            ps = pages(min(T, rng.integers(2, 5)))
            vs = rng.choice(spec.n_values, len(ps), replace=False)
            for j, (p, v) in enumerate(zip(ps, vs)):
                put(p, ASSERT if j == 0 else AMEND, f, 0, v + 1)
            gold[i] = vs[-1]; evid[i, ps[-1]] = True
        elif name == 'link':
            pa, pb = rng.choice(T, 2, replace=False)    # ref may come before or after def
            e, v = used_ent[i], rng.integers(spec.n_values)
            put(pa, REF, f, e, 0); put(pb, DEF, 0, e, v + 1)
            gold[i] = v; evid[i, [pa, pb]] = True
        elif name == 'aggregation':
            ps = pages(min(T, rng.integers(2, 4)))
            vs = rng.integers(spec.n_values, size=len(ps))
            for p, v in zip(ps, vs):
                put(p, ITEM, f, 0, v + 1)
            gold[i] = vs.sum() % spec.n_values; evid[i, ps] = True
    for e in used_ent[F:]:
        put(rng.integers(T), DEF, 0, e, rng.integers(spec.n_values) + 1)
    for f in others:
        for j, p in enumerate(pages(min(T, rng.integers(1, 3)))):
            put(p, ASSERT if j == 0 else AMEND, f, 0, rng.integers(spec.n_values) + 1)
    return dict(kind=kind, field=field, ent=ent, val=val, schema=schema, ftype=ftype, gold=gold, evid=evid)


def make_batch(spec, enc, B, T, F, rng, unseen=False, device='cpu'):
    docs = [make_doc(spec, T, F, rng, unseen) for _ in range(B)]
    attrs = np.stack([np.stack([d['kind'], d['field'], d['ent'], d['val']], -1) for d in docs])
    out = {k: np.stack([d[k] for d in docs]) for k in ('schema', 'ftype', 'gold', 'evid')}
    # distance from earliest required evidence to the last page; missing fields -> -1
    has = out['evid'].any(-1)
    out['dist'] = np.where(has, T - 1 - out['evid'].argmax(-1), -1)
    out = {k: torch.from_numpy(v).to(device) for k, v in out.items()}
    gen = torch.Generator(device).manual_seed(int(rng.integers(2**62)))
    out['x'] = enc(torch.from_numpy(attrs).to(device), gen)
    # schema field names go through the same frozen encoder (noise-free), like a text query
    out['svec'] = torch.from_numpy(enc.tables[1][out['schema'].cpu().numpy()]).to(device)
    return out
