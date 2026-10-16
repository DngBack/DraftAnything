"""Real GPU extraction at identical real/blank prefixes, without gold tokens.

One 384px view per base image/wording from VLMBias. Captures the exact input
to get_output_embeddings() rather than assuming hidden-state normalization.
Greedy decision prefixes and a fixed '{' prefix are recorded separately.
Existing paper caches are never modified. See verification protocol.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from transformers import LogitsProcessorList, StoppingCriteriaList

from diag_gen import Stop
from gen_calib import Decide, ILLUSIONS, items
from pilot import MODELS, Runner

ROOT=Path(__file__).resolve().parent


def candidate_ids(tok):
    return [sorted({i for p in ('',' ') for v in (c,c.lower(),c.capitalize())
                    for i in tok.encode(p+v,add_special_tokens=False)
                    if tok.decode([i]).strip().lower()==c.lower()}) for c in ('Yes','No')]


def extend(x,ids):
    out=dict(x);n=x['input_ids'].shape[1]
    for k,v in x.items():
        if k!='input_ids' and torch.is_tensor(v) and v.dim()==2 and v.shape[1]==n:
            pad=(torch.ones if k=='attention_mask' else torch.zeros)(1,ids.shape[1]-n,dtype=v.dtype,device=v.device)
            out[k]=torch.cat([v,pad],1)
    out['input_ids']=ids
    return out


def run(a):
    torch.set_num_threads(4)
    out=a.out/a.model;out.mkdir(parents=True,exist_ok=True)
    its=[x for x in items('vlmbias') if x['group'] in ILLUSIONS]
    seen=set();selected=[]
    for it in its:
        k=(it['key'],it['q'])
        if k not in seen: selected.append(it);seen.add(k)
    if a.limit:selected=selected[:a.limit]
    R=Runner(a.model,['x']);tok=R.p.tokenizer;cids=candidate_ids(tok)
    capture={}
    def pre_hook(module,args):
        capture['h']=args[0][0,-1].detach().float().cpu().numpy()
    hd=R.m.get_output_embeddings().register_forward_pre_hook(pre_hook)

    def inp(it,blank=False):
        im=Image.new('RGB',it['image'].size,(128,128,128)) if blank else it['image']
        msg=[{'role':'user','content':[{'type':'image','image':im},{'type':'text','text':it['q']}]}]
        return R.p.apply_chat_template(msg,add_generation_prompt=True,tokenize=True,
                                      return_dict=True,return_tensors='pt').to('cuda')
    @torch.inference_mode()
    def forward(x):
        o=R.m(**x,logits_to_keep=1,use_cache=False)
        scores=np.array([float(o.logits[0,-1,ids].float().max()) for ids in cids])
        return capture['h'].copy(),scores

    t0=time.monotonic();nodec=0
    try:
        for j,it in enumerate(selected):
            path=out/f'{j:04d}.npz'
            if path.exists():continue
            x=inp(it);xb=inp(it,True)
            assert torch.equal(x['input_ids'],xb['input_ids']), 'Control image changed tokenization'
            n=x['input_ids'].shape[1]
            d=Decide(cids,None,None,())
            d.ready=lambda ids,n=n:tok.decode(ids[0,n:]).strip()=='' or tok.decode(ids[0,n:][-4:]).rstrip().endswith('{')
            with torch.inference_mode():
                seq=R.m.generate(**x,max_new_tokens=a.max_tokens,do_sample=False,
                                 logits_processor=LogitsProcessorList([d]),
                                 stopping_criteria=StoppingCriteriaList([Stop(d)]))
            fixed=torch.cat([x['input_ids'],torch.tensor([tok.encode('{',add_special_tokens=False)],device='cuda')],1)
            hf,sf=forward(extend(x,fixed));bf,sbf=forward(extend(xb,fixed))
            data=dict(y=('Yes','No').index(it['y']),key=it['key'],q=it['q'],group=it['group'],
                      h_fixed=hf,blank_fixed=bf,s_fixed=sf,s_blank_fixed=sbf,
                      fixed_ids=fixed[0].cpu().numpy(),width=it['image'].size[0],
                      generated=tok.decode(seq[0,n:],skip_special_tokens=True),decision=bool(d.done))
            if d.done:
                prefix=seq[:,:-1]
                if torch.equal(prefix,fixed):h,s,b,sb=hf,sf,bf,sbf
                else:
                    h,s=forward(extend(x,prefix));b,sb=forward(extend(xb,prefix))
                data.update(h_generated=h,blank_generated=b,s_generated=s,s_blank_generated=sb,
                            prefix_ids=prefix[0].cpu().numpy(),decision_scores=d.scores,
                            prefix=tok.decode(prefix[0,n:],skip_special_tokens=True),
                            prefix_score_mae=float(np.mean(np.abs(s-d.scores))))
            else:nodec+=1
            np.savez_compressed(path,**data)
            if j%20==0 or j==len(selected)-1:
                print(json.dumps(dict(model=a.model,done=j+1,total=len(selected),nodec=nodec,
                                      elapsed_sec=round(time.monotonic()-t0,1))),flush=True)
    finally:hd.remove()
    (out/'manifest.json').write_text(json.dumps(dict(model=a.model,repo=MODELS[a.model],n=len(selected),
        candidate_ids=cids,max_tokens=a.max_tokens,control='uniform RGB128, same dimensions and exact token prefix',
        capture='get_output_embeddings forward_pre_hook; FP32 copies',source_resolution=384,
        protocol='docs/ver29_verification_protocol.md'),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,choices=MODELS)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--max-tokens',type=int,default=96)
    p.add_argument('--out',type=Path,default=ROOT/'results'/'ver29_controlled')
    run(p.parse_args())
