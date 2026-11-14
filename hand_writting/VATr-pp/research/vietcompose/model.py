"""A conservative first architecture: role-aware residual queries for VATr.

This is a query-conditioning prototype. It does NOT yet implement the proposed
pixel-layer/affine mark renderer or an independent factorized image recognizer.
"""
import torch
from torch import nn
from research.vietcompose.orthography import decompose_character,SHAPES,TONES,CASE


class ComponentQuery(nn.Module):
    def __init__(self,glyph,alphabet,mode='factorized',hidden=128):
        super().__init__()
        if mode not in ('factorized','glyph'):raise ValueError(mode)
        self.glyph=glyph;self.mode=mode;self.alphabet=alphabet
        output_dim=glyph.linear.out_features
        base_ids=[0];shape_ids=[0];tone_ids=[0];case_ids=[2]
        for char in alphabet:
            f=decompose_character(char);base=f.base.upper() if f.case=='upper' else f.base
            if base not in alphabet:raise ValueError(f'Missing base glyph {base!r}')
            base_ids.append(alphabet.index(base)+1);shape_ids.append(SHAPES.index(f.shape));tone_ids.append(TONES.index(f.tone));case_ids.append(CASE.index(f.case))
        for name,values in [('base_ids',base_ids),('shape_ids',shape_ids),('tone_ids',tone_ids),('case_ids',case_ids)]:self.register_buffer(name,torch.tensor(values),persistent=False)
        self.down=nn.Linear(output_dim,hidden)
        self.shape=nn.Embedding(len(SHAPES),hidden)
        self.tone=nn.Embedding(len(TONES),hidden)
        self.case=nn.Embedding(len(CASE),hidden)
        self.role=nn.Parameter(torch.randn(3,hidden)*.02)
        # These coordinates are token priors only, not predicted pixel locations.
        self.anchor_projection=nn.Linear(2,hidden,bias=False)
        self.register_buffer('shape_anchors',torch.tensor([[0.,0.],[0.,-.4],[0.,-.4],[.55,-.25],[0.,0.]]),persistent=False)
        self.register_buffer('tone_anchors',torch.tensor([[0.,0.],[0.,-.7],[0.,-.7],[0.,-.7],[0.,-.7],[0.,.7]]),persistent=False)
        layer=nn.TransformerEncoderLayer(hidden,4,hidden*2,dropout=0.,activation='gelu',batch_first=True)
        self.interaction=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.style_down=nn.Linear(output_dim,hidden)
        self.style_attention=nn.MultiheadAttention(hidden,4,dropout=0.,batch_first=True)
        self.fuse=nn.Sequential(nn.LayerNorm(hidden*3),nn.Linear(hidden*3,hidden),nn.GELU())
        self.up=nn.Linear(hidden,output_dim)
        nn.init.zeros_(self.up.weight);nn.init.zeros_(self.up.bias)
        self.last_residual_norm=None

    @property
    def symbols(self):return self.glyph.symbols

    def forward(self,query):
        return self.compose(query,None)

    def compose(self,query,style_memory):
        original=self.glyph(query);batch,length,_=original.shape
        if self.mode=='factorized':
            base=self.down(self.glyph(self.base_ids[query]))+self.case(self.case_ids[query])
            modifier=self.shape(self.shape_ids[query])+self.anchor_projection(self.shape_anchors[self.shape_ids[query]])
            tone=self.tone(self.tone_ids[query])+self.anchor_projection(self.tone_anchors[self.tone_ids[query]])
        else:
            # Same interaction/style/fusion backbone; whole glyph in all slots.
            # Component embeddings remain allocated but unused in this control.
            base=modifier=tone=self.down(original)
        tokens=torch.stack([base,modifier,tone],dim=2)+self.role
        tokens=self.interaction(tokens.reshape(batch*length,3,-1)).reshape(batch,length,3,-1)
        if style_memory is not None:
            # Batch pool before expansion: writer-conditioned role features.
            memory=self.style_down(style_memory.permute(1,0,2))
            queries=tokens.reshape(batch,length*3,-1)
            contextual,_=self.style_attention(queries,memory,memory,need_weights=False)
            tokens=tokens+contextual.reshape_as(tokens)
        residual=self.up(self.fuse(tokens.flatten(2)))
        residual=residual.masked_fill(query.eq(0).unsqueeze(-1),0)
        self.last_residual_norm=residual.detach().square().mean().sqrt()
        return original+residual


def install_composition(generator,mode='factorized',hidden=128):
    """Install after strictly loading a native VATr checkpoint; no key guessing."""
    if isinstance(generator.query_embed,ComponentQuery):raise ValueError('Composition already installed')
    generator.query_embed=ComponentQuery(generator.query_embed,generator.args.alphabet+generator.args.special_alphabet,mode,hidden).to(next(generator.parameters()).device)
    return generator
