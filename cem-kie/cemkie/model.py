"""Evidence resolvers over field candidates. All variants share the input projection and the router.

  point  router only: independent candidate scores (retrieval top-1 / pointer baseline)
  pool   one pooled evidence vector per field (router-weighted mean of all candidates) rescoring top-K
  set    Transformer over [field query; top-K candidates]: shared memory, full attention
  cem    competing evidence slots over top-K (slot attention + discard sink), each slot keeps a pointer
         to its evidence; relation-aware attention between slots, then slot choice -> candidate

A segment is one (document, field) pair. Candidates are flattened over segments; top-K variants
work on a padded [G, K] view. Every variant returns log-probs over its candidate set plus NULL (last).
"""
import math

import torch
from torch import nn
import torch.nn.functional as Fn

D_EMB, N_FEAT = 768, 9


def mlp(d_in, d, d_out):
    return nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d_out))


def padded(seg, G, *xs, fill=0.):
    """Scatter flat per-candidate tensors into [G, Cmax, ...]; returns (mask, idx, *padded)."""
    order = torch.argsort(seg, stable=True)
    cnt = torch.bincount(seg, minlength=G)
    start = torch.cumsum(cnt, 0) - cnt
    pos = torch.empty_like(seg)
    pos[order] = torch.arange(len(seg), device=seg.device) - start[seg[order]]
    C = int(cnt.max()) if len(seg) else 1
    mask = torch.zeros(G, C, dtype=torch.bool, device=seg.device)
    mask[seg, pos] = True
    idx = torch.full((G, C), -1, dtype=torch.long, device=seg.device)
    idx[seg, pos] = torch.arange(len(seg), device=seg.device)
    outs = []
    for x in xs:
        o = torch.full((G, C, *x.shape[1:]), fill, dtype=x.dtype, device=x.device)
        o[seg, pos] = x
        outs.append(o)
    return (mask, idx, *outs)


def masked_log_softmax(logits, mask, null):
    """logits [G, C], mask [G, C], null [G] -> log-probs [G, C + 1] (NULL last)."""
    z = torch.cat([logits.masked_fill(~mask, -1e9), null[:, None]], 1)
    return Fn.log_softmax(z, 1)


class Trunk(nn.Module):
    def __init__(self, n_fields, d):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(D_EMB + N_FEAT, d), nn.LayerNorm(d))
        self.field = nn.Embedding(n_fields, d)
        self.ff = nn.Sequential(nn.LayerNorm(d), mlp(d, 2 * d, d))
        self.router = mlp(d, d, 1)
        self.router_null = nn.Embedding(n_fields, 1)

    def forward(self, x, feat, fid):
        h = self.inp(torch.cat([x, feat], -1)) + self.field(fid)
        h = h + self.ff(h)
        return h, self.router(h).squeeze(-1)


class Resolver(nn.Module):
    """Relation-aware attention between slots. Relation types are latent (no relation labels in Kleister):
    a softmax over n_rel types from slot pair states + provenance pair features, used as attention bias
    and as an additive message embedding."""

    def __init__(self, d, heads=4, n_rel=5, n_pair=7):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.rel = mlp(2 * d + n_pair, d, n_rel)
        self.rel_bias = nn.Linear(n_rel, heads, bias=False)
        self.rel_emb = nn.Linear(n_rel, d, bias=False)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = mlp(d, 4 * d, d)

    def forward(self, z, pair):
        G, R, d = z.shape
        y = self.n1(z)
        zz = torch.cat([y[:, :, None].expand(G, R, R, d), y[:, None].expand(G, R, R, d), pair], -1)
        p_rel = self.rel(zz).softmax(-1)                                   # [G, R, R, n_rel]
        q, k, v = self.qkv(y).view(G, R, 3, self.h, self.dh).unbind(2)
        att = torch.einsum('grhe,gshe->ghrs', q, k) / math.sqrt(self.dh) + self.rel_bias(p_rel).permute(0, 3, 1, 2)
        att = att.softmax(-1)
        msg = torch.einsum('ghrs,gshe->grhe', att, v).reshape(G, R, d) + \
            torch.einsum('ghrs,grsd->grd', att, self.rel_emb(p_rel)) / self.h
        z = z + self.o(msg)
        return z + self.ff(self.n2(z)), p_rel


class Model(nn.Module):
    def __init__(self, kind, n_fields, d=256, K=64, slots=4, iters=3, layers=2, resolver=True, heads=4):
        super().__init__()
        self.kind, self.K, self.R, self.iters = kind, K, slots, iters
        self.trunk = Trunk(n_fields, d)
        if kind == 'pool':
            self.score = mlp(3 * d, d, 1)
            self.null = nn.Linear(d, 1)
        elif kind == 'set':
            self.q = nn.Embedding(n_fields, d)
            layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0., batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, layers)
            self.out, self.null = nn.Linear(d, 1), nn.Linear(d, 1)
        elif kind == 'cem':
            self.q = nn.Embedding(n_fields, d)
            self.slot = nn.Parameter(torch.randn(slots, d) * 0.1)
            self.nx, self.ns = nn.LayerNorm(d), nn.LayerNorm(d)
            self.Wq, self.Wk, self.Wv = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
            self.sink = nn.Linear(d, 1)       # discard competitor: lets a single slot ignore candidates
            self.gru = nn.GRUCell(d, d)
            self.sff = nn.Sequential(nn.LayerNorm(d), mlp(d, 2 * d, d))
            self.Wp, self.Wk2 = nn.Linear(d, d), nn.Linear(d, d)
            self.prov = nn.Linear(3, d)
            self.res = nn.ModuleList(Resolver(d, heads) for _ in range(layers)) if resolver else nn.ModuleList()
            self.nz = nn.LayerNorm(d)
            self.u, self.null = nn.Linear(d, 1), nn.Linear(d, 1)

    def forward(self, emb, b):
        h, r = self.trunk(emb[b['m']].float(), b['feat'], b['fid'])
        G = b['G']
        mask, idx, R_pad = padded(b['seg'], G, r, fill=-1e9)
        out = dict(router=masked_log_softmax(R_pad, mask, self.trunk.router_null(b['seg_fid']).squeeze(-1)),
                   router_idx=idx)
        if self.kind == 'point':
            out.update(logp=out['router'], idx=idx)
            return out
        k = min(self.K, R_pad.shape[1])
        top = R_pad.detach().topk(k, 1).indices
        tmask = mask.gather(1, top)
        tidx = idx.gather(1, top)
        X = h[tidx.clamp(min=0)] * tmask[..., None]
        out['idx'] = tidx
        fid = b['seg_fid']
        if self.kind == 'pool':
            w = R_pad.masked_fill(~mask, -1e9).softmax(1)                     # all candidates
            H = torch.zeros(G, R_pad.shape[1], h.shape[1], device=h.device)
            H[mask] = h[idx[mask]]
            p = torch.einsum('gc,gcd->gd', w, H)
            P = p[:, None].expand_as(X)
            logits = self.score(torch.cat([X, P, X * P], -1)).squeeze(-1)
            out['logp'] = masked_log_softmax(logits, tmask, self.null(p).squeeze(-1))
        elif self.kind == 'set':
            y = self.enc(torch.cat([self.q(fid)[:, None], X], 1),
                         src_key_padding_mask=torch.cat([torch.zeros_like(tmask[:, :1]), ~tmask], 1))
            out['logp'] = masked_log_softmax(self.out(y[:, 1:]).squeeze(-1), tmask, self.null(y[:, 0]).squeeze(-1))
        else:
            out.update(self.cem(X, tmask, tidx, fid, b))
        return out

    def cem(self, X, tmask, tidx, fid, b):
        G, K, d = X.shape
        R = self.R
        x = self.nx(X)
        k, v, sink = self.Wk(x), self.Wv(x), self.sink(x).transpose(1, 2)      # sink [G, 1, K]
        S = self.q(fid)[:, None] + self.slot[None]
        for _ in range(self.iters):
            a = torch.einsum('grd,gkd->grk', self.Wq(self.ns(S)), k) / math.sqrt(d)
            comp = torch.cat([a, sink], 1).softmax(1)[:, :R] * tmask[:, None]   # slots compete for candidates
            w = comp / (comp.sum(-1, keepdim=True) + 1e-6)
            S = self.gru(torch.einsum('grk,gkd->grd', w, v).reshape(-1, d), S.reshape(-1, d)).view(G, R, d)
            S = S + self.sff(S)
        # provenance: each slot points at its evidence candidates
        ptr = torch.einsum('grd,gkd->grk', self.Wp(self.ns(S)), self.Wk2(x)) / math.sqrt(d)
        P = ptr.masked_fill(~tmask[:, None], -1e9).softmax(-1)                # [G, R, K]
        safe = tidx.clamp(min=0)
        pos = b['feat'][safe, 0] * tmask
        logv = b['feat'][safe, 2] * tmask
        conf = comp.sum(-1) / tmask.sum(-1, keepdim=True).clamp(min=1)
        Ep, Ev = torch.einsum('grk,gk->gr', P, pos), torch.einsum('grk,gk->gr', P, logv)
        z = S + self.prov(torch.stack([Ep, Ev, conf], -1))
        vid = b['vid'][safe]
        same = ((vid[:, :, None] == vid[:, None]) & tmask[:, :, None] & tmask[:, None]).float()
        overlap = torch.einsum('grk,gkl,gsl->grs', P, same, P)
        dp, dv = Ep[:, :, None] - Ep[:, None], Ev[:, :, None] - Ev[:, None]
        pair = torch.stack([dp, dp.abs(), dv, dv.abs(), overlap,
                            conf[:, :, None].expand(G, R, R), conf[:, None].expand(G, R, R)], -1)
        for layer in self.res:
            z, _ = layer(z, pair)
        z = self.nz(z)
        slot_logp = Fn.log_softmax(torch.cat([self.u(z).squeeze(-1), self.null(z.mean(1))], 1), 1)
        logp_c = torch.logsumexp(slot_logp[:, :R, None] + torch.log(P + 1e-9), 1).masked_fill(~tmask, -1e9)
        eye = torch.eye(R, device=X.device, dtype=torch.bool)
        div = (torch.einsum('grk,gsk->grs', P, P).masked_fill(eye, 0).sum((1, 2)) / max(R * (R - 1), 1)).mean()
        return dict(logp=torch.cat([logp_c, slot_logp[:, R:]], 1), div=div)


def nll(logp, pos):
    """Multi-positive NLL. logp [G, C+1], pos [G, C] bool; segments without a positive target NULL."""
    pos = torch.cat([pos, ~pos.any(1, keepdim=True)], 1)
    return -torch.logsumexp(logp.masked_fill(~pos, -1e9), 1).mean()


VARIANTS = {
    'point': dict(kind='point'),
    'pool': dict(kind='pool'),
    'set2': dict(kind='set', layers=2),
    'set4': dict(kind='set', layers=4),
    'cem_r1': dict(kind='cem', slots=1),
    'cem_r2': dict(kind='cem', slots=2),
    'cem_r4': dict(kind='cem', slots=4),
    'cem_r8': dict(kind='cem', slots=8),
    'cem_r4_nores': dict(kind='cem', slots=4, resolver=False),
}
