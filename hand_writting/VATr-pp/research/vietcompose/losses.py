"""Component-marginal CTC from the existing frozen character OCR.

This is not an independent recognizer. Marginal streams can disagree in alignment;
we retain character CTC and image supervision rather than trusting this alone.
"""
import torch
from torch import nn
import torch.nn.functional as F
from research.vietcompose.orthography import decompose_character,SHAPES,TONES,CASE


class ComponentCTC(nn.Module):
    def __init__(self,alphabet):
        super().__init__();self.alphabet=alphabet
        factors=[decompose_character(c) for c in alphabet]
        self.vocab={'base':sorted({f.base for f in factors}),'shape':list(SHAPES),'tone':list(TONES),'case':list(CASE)}
        for role,values in self.vocab.items():
            # Blank is 0; the semantic value 'none' is a NONBLANK category.
            self.register_buffer('map_'+role,torch.tensor([0]+[values.index(getattr(f,role))+1 for f in factors]),persistent=False)

    def forward(self,log_probabilities,texts,widths):
        probability=log_probabilities.exp();timesteps=widths.cpu().long()//4+1
        lengths=torch.tensor([len(t) for t in texts],dtype=torch.long)
        components=[[decompose_character(c) for c in t] for t in texts];losses={}
        for role,values in self.vocab.items():
            mapping=getattr(self,'map_'+role)
            merged=probability.new_zeros(*probability.shape[:2],len(values)+1)
            merged.scatter_add_(2,mapping.view(1,1,-1).expand_as(probability),probability)
            sequences=[[values.index(getattr(f,role))+1 for f in fs] for fs in components]
            required=torch.tensor([len(seq)+sum(a==b for a,b in zip(seq,seq[1:])) for seq in sequences])
            if torch.any(required>timesteps):raise ValueError('Invalid component CTC geometry')
            targets=torch.tensor([x for seq in sequences for x in seq],device=probability.device,dtype=torch.long)
            losses[role]=F.ctc_loss(merged.clamp_min(1e-8).log(),targets,timesteps,lengths,blank=0,zero_infinity=False)
        return torch.stack(list(losses.values())).mean(),losses
