"""Unvalidated explicit body/modifier/tone ink compositor.

Unlike the residual-query adapter, this module renders separate layers.
research.longtail ran exploratory real-data variants; they remain unvalidated
and separate from the native pretrained VATr++ continuation. See its STATUS.md.
"""
import string
import unicodedata
import torch
from torch import nn
import torch.nn.functional as F
from research.vietcompose.orthography import decompose_character,SHAPES,TONES,CASE


class LayeredInkComposer(nn.Module):
    def __init__(self,style_dim=512,hidden=128,max_length=64):
        super().__init__();self.max_length=max_length
        self.bases=string.ascii_lowercase+string.digits+string.punctuation+' '
        self.base=nn.Embedding(len(self.bases)+1,hidden,padding_idx=0);self.case=nn.Embedding(3,hidden)
        self.shape=nn.Embedding(len(SHAPES),hidden);self.tone=nn.Embedding(len(TONES),hidden)
        self.position=nn.Embedding(max_length,hidden);self.style=nn.Linear(style_dim,hidden)
        encoder=nn.TransformerEncoderLayer(hidden,4,hidden*2,dropout=0.,batch_first=True,activation='gelu')
        self.body_context=nn.TransformerEncoder(encoder,2,enable_nested_tensor=False)
        self.body_head=nn.Sequential(nn.Linear(hidden,hidden*2),nn.GELU(),nn.Linear(hidden*2,32*16))
        self.shape_head=nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,16*16))
        self.tone_head=nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,16*16))
        self.shape_geometry=nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,4))
        self.tone_geometry=nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,4))
        for head in [self.shape_geometry,self.tone_geometry]:nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)
        # Center x/y and half-size fractions, in normalized canvas coordinates.
        self.register_buffer('shape_prior',torch.tensor([[0.,0.,.6,.3],[0.,-.5,.6,.28],[0.,-.5,.6,.28],[.5,-.25,.35,.4],[0.,0.,.7,.18]]))
        self.register_buffer('tone_prior',torch.tensor([[0.,0.,.4,.2],[0.,-.72,.4,.2],[0.,-.72,.4,.2],[0.,-.72,.4,.2],[0.,-.72,.5,.2],[0.,.75,.2,.12]]))
        self.refine=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.GELU(),nn.Conv2d(16,1,3,padding=1))

    def encode(self,texts,device):
        texts=[unicodedata.normalize('NFC',t) for t in texts]
        if not texts or any(not t or len(t)>self.max_length for t in texts):raise ValueError('Empty or overlong renderer input')
        length=max(map(len,texts));values={k:torch.zeros(len(texts),length,dtype=torch.long,device=device) for k in ['base','case','shape','tone']}
        for i,text in enumerate(texts):
            for j,char in enumerate(text):
                factors=decompose_character(char)
                values['base'][i,j]=self.bases.index(factors.base)+1;values['case'][i,j]=CASE.index(factors.case);values['shape'][i,j]=SHAPES.index(factors.shape);values['tone'][i,j]=TONES.index(factors.tone)
        return values

    @staticmethod
    def _place(patch,prior,delta):
        # Output->input transform is the inverse of desired scale/translation.
        center=prior[:,:2]+.2*delta[:,:2].tanh();scale=prior[:,2:]*torch.exp(.3*delta[:,2:].tanh())
        theta=patch.new_zeros(len(patch),2,3)
        theta[:,0,0]=1/scale[:,0];theta[:,1,1]=1/scale[:,1]
        theta[:,0,2]=-center[:,0]/scale[:,0];theta[:,1,2]=-center[:,1]/scale[:,1]
        grid=F.affine_grid(theta,(len(patch),1,32,16),align_corners=False)
        return F.grid_sample(patch,grid,align_corners=False,padding_mode='zeros')

    def forward(self,texts,style_vector):
        ids=self.encode(texts,style_vector.device);b,l=ids['base'].shape;valid=ids['base'].ne(0)
        # Body receives only stripped base sequence, case, position, and style.
        # Tone and modifier IDs cannot enter this branch.
        body=self.base(ids['base'])+self.case(ids['case'])+self.position(torch.arange(l,device=style_vector.device))[None]+self.style(style_vector)[:,None]
        body=self.body_context(body,src_key_padding_mask=~valid)
        modifier=body+self.shape(ids['shape']);tone=body+self.shape(ids['shape'])+self.tone(ids['tone'])
        body_ink=self.body_head(body).sigmoid().reshape(b*l,1,32,16)
        modifier_ink=self._place(self.shape_head(modifier).sigmoid().reshape(b*l,1,16,16),self.shape_prior[ids['shape'].flatten()],self.shape_geometry(modifier).reshape(b*l,4))
        tone_prior=self.tone_prior[ids['tone'].flatten()].clone()
        stacked=((ids['shape']==1)|(ids['shape']==2))&ids['tone'].gt(0)&ids['tone'].ne(TONES.index('dot_below'))
        tone_prior[:,1]-=.08*stacked.flatten()
        tone_ink=self._place(self.tone_head(tone).sigmoid().reshape(b*l,1,16,16),tone_prior,self.tone_geometry(tone).reshape(b*l,4))
        modifier_ink=modifier_ink*ids['shape'].ne(0).flatten()[:,None,None,None]
        tone_ink=tone_ink*ids['tone'].ne(0).flatten()[:,None,None,None]
        layers=[x*valid.flatten()[:,None,None,None] for x in [body_ink,modifier_ink,tone_ink]]
        ink=1-(1-layers[0])*(1-layers[1])*(1-layers[2])
        def word(patches):return patches.reshape(b,l,1,32,16).permute(0,2,3,1,4).reshape(b,1,32,l*16)
        composed=1-2*word(ink);refined=(composed+.05*self.refine(composed).tanh()).clamp(-1,1)
        mask=valid[:,:,None].expand(b,l,16).reshape(b,1,1,l*16)
        image=torch.where(mask,refined,torch.ones_like(refined))
        return {'image':image,'composed':composed,'base_ink':word(layers[0]),'modifier_ink':word(layers[1]),'tone_ink':word(layers[2])}
