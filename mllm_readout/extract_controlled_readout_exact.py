"""Capture matched real/blank readout states from the actual generation path.

Unlike extract_controlled_readout.py, this never recomputes a cached multimodal
prefix with stale model-specific position ids. It captures the lm_head input and
candidate logits during generate(), then replays the exact generated prefix on
the blank image through generate() as well.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

from diag_gen import Stop
from gen_calib import Decide, ILLUSIONS, items
from pilot import MODELS, Runner

ROOT=Path(__file__).resolve().parent


def candidate_ids(tok):
    return [sorted({i for p in ('',' ') for v in (c,c.lower(),c.capitalize())
                    for i in tok.encode(p+v,add_special_tokens=False)
                    if tok.decode([i]).strip().lower()==c.lower()}) for c in ('Yes','No')]


def get_scores(logits, cids):
    return np.array([float(logits[0,-1,ids].float().max()) for ids in cids],dtype=np.float32)


class SavedDecide(Decide):
    def __call__(self,input_ids,scores):
        was=self.done
        out=super().__call__(input_ids,scores)
        if self.done and not was:
            self.decision_ids=input_ids[0].detach().cpu().numpy().copy()
        return out


class ForceSuffix(LogitsProcessor):
    def __init__(self,n0,suffix,cids):
        self.n0=int(n0);self.suffix=np.asarray(suffix,dtype=np.int64);self.cids=cids
        self.decision_scores=None;self.decision_ids=None
    def __call__(self,input_ids,scores):
        n=int(input_ids.shape[1]-self.n0)
        if n<len(self.suffix):
            token=int(self.suffix[n]);scores.fill_(float('-inf'));scores[0,token]=0
        elif n==len(self.suffix):
            self.decision_ids=input_ids[0].detach().cpu().numpy().copy()
            self.decision_scores=np.array([float(scores[0,ids].max()) for ids in self.cids],dtype=np.float32)
        return scores


class StopAfterDecision(StoppingCriteria):
    def __init__(self,n0,n_suffix):self.end=int(n0+n_suffix+1)
    def __call__(self,input_ids,scores,**kw):
        return torch.tensor([input_ids.shape[1]>=self.end],device=input_ids.device)


def run(a):
    torch.set_num_threads(4)
    out=a.out/a.model;out.mkdir(parents=True,exist_ok=True)
    selected=[];seen=set()
    for it in items('vlmbias'):
        if it['group'] not in ILLUSIONS:continue
        k=(it['key'],it['q'])
        if k not in seen:selected.append(it);seen.add(k)
    if a.limit:selected=selected[:a.limit]
    R=Runner(a.model,['x']);tok=R.p.tokenizer;cids=candidate_ids(tok)
    marker=np.asarray(tok.encode('{',add_special_tokens=False),dtype=np.int64)
    cap={'h':None,'scores':None}
    def pre_hook(module,args):cap['h']=args[0][0,-1].detach().float().cpu().numpy().copy()
    def out_hook(module,args,result):
        if isinstance(result,tuple):result=result[0]
        cap['scores']=get_scores(result,cids)
    emb=R.m.get_output_embeddings()
    hin=emb.register_forward_pre_hook(pre_hook);hout=emb.register_forward_hook(out_hook)

    def inp(it,blank=False):
        im=Image.new('RGB',it['image'].size,(128,128,128)) if blank else it['image']
        msg=[{'role':'user','content':[{'type':'image','image':im},{'type':'text','text':it['q']}]}]
        return R.p.apply_chat_template(msg,add_generation_prompt=True,tokenize=True,
                                      return_dict=True,return_tensors='pt').to('cuda')

    @torch.inference_mode()
    def decide(x,blank=False):
        n0=x['input_ids'].shape[1];d=SavedDecide(cids,None,None,())
        d.ready=lambda ids,n0=n0:tok.decode(ids[0,n0:]).strip()=='' or tok.decode(ids[0,n0:][-4:]).rstrip().endswith('{')
        seq=R.m.generate(**x,max_new_tokens=a.max_tokens,do_sample=False,
                         logits_processor=LogitsProcessorList([d]),
                         stopping_criteria=StoppingCriteriaList([Stop(d)]))
        if not d.done:return None
        pred=np.asarray(d.scores,dtype=np.float32)
        # The forward hook runs in the model call whose logits are consumed by Decide.
        # Keep the live final capture and verify it against Decide's exact scores.
        hs=cap['h'];actual=cap['scores']
        if hs is None or actual is None:raise RuntimeError('Generation hook did not capture a decision state')
        mae=float(np.mean(np.abs(actual-pred)))
        if mae>1e-3:raise RuntimeError(f'Generation hook/logits mismatch {mae:.6g}')
        suffix=d.decision_ids[n0:].copy()
        return dict(h=hs.copy(),scores=actual.copy(),decision_scores=pred,
                    decision_ids=d.decision_ids.copy(),suffix=suffix,
                    generated=tok.decode(seq[0,n0:],skip_special_tokens=True),
                    prefix=tok.decode(d.decision_ids[n0:],skip_special_tokens=True),
                    prefix_ids=d.decision_ids.copy(),capture_mae=mae)

    @torch.inference_mode()
    def replay(x,suffix):
        n0=x['input_ids'].shape[1];p=ForceSuffix(n0,suffix,cids)
        end=StopAfterDecision(n0,len(suffix))
        _=R.m.generate(**x,max_new_tokens=len(suffix)+1,do_sample=False,
                       logits_processor=LogitsProcessorList([p]),
                       stopping_criteria=StoppingCriteriaList([end]))
        if p.decision_scores is None or p.decision_ids is None:
            raise RuntimeError('Replay did not reach the requested decision prefix')
        if not np.array_equal(p.decision_ids[n0:],suffix):
            raise RuntimeError('Replay prefix differs from original prefix')
        if cap['h'] is None or cap['scores'] is None:raise RuntimeError('Replay hook missed')
        mae=float(np.mean(np.abs(cap['scores']-p.decision_scores)))
        if mae>1e-3:raise RuntimeError(f'Replay hook/logits mismatch {mae:.6g}')
        return cap['h'].copy(),cap['scores'].copy(),mae,p.decision_ids.copy()

    t0=time.monotonic();nodec=0
    try:
        for j,it in enumerate(selected):
            path=out/f'{j:04d}.npz'
            if path.exists():continue
            x=inp(it);xb=inp(it,True)
            assert torch.equal(x['input_ids'],xb['input_ids']),'Control image changed tokenization'
            real=decide(x)
            if real is None:
                nodec+=1;continue
            hblank,sblank,blank_mae,blank_ids=replay(xb,real['suffix'])
            # The fixed-marker baseline is captured through the same cached-generation
            # path, avoiding hand-built position ids. Reuse captures when prefixes match.
            if np.array_equal(real['suffix'],marker):
                hf,sf=real['h'],real['scores'];bf,sbf=hblank,sblank
                fixed_mae=0.0
            else:
                hf,sf,real_fixed_mae,_=replay(x,marker)
                bf,sbf,blank_fixed_mae,_=replay(xb,marker)
                fixed_mae=max(real_fixed_mae,blank_fixed_mae)
            data=dict(y=('Yes','No').index(it['y']),key=it['key'],q=it['q'],group=it['group'],
                      h_fixed=hf,blank_fixed=bf,s_fixed=sf,s_blank_fixed=sbf,
                      h_generated=real['h'],blank_generated=hblank,
                      s_generated=real['scores'],s_blank_generated=sblank,
                      fixed_ids=np.r_[x['input_ids'][0].cpu().numpy(),marker],
                      prefix_ids=real['prefix_ids'],width=it['image'].size[0],
                      generated=real['generated'],prefix=real['prefix'],decision=True,
                      decision_scores=real['decision_scores'],prefix_score_mae=real['capture_mae'],
                      blank_replay_mae=blank_mae,fixed_replay_mae=fixed_mae)
            np.savez_compressed(path,**data)
            if j%20==0 or j==len(selected)-1:
                print(json.dumps(dict(model=a.model,done=j+1,total=len(selected),nodec=nodec,
                                      elapsed_sec=round(time.monotonic()-t0,1),
                                      prefix=real['prefix'],capture_mae=real['capture_mae'],
                                      blank_replay_mae=blank_mae)),flush=True)
    finally:
        hin.remove();hout.remove()
    (out/'manifest.json').write_text(json.dumps(dict(model=a.model,repo=MODELS[a.model],n=len(selected),
        candidate_ids=cids,max_tokens=a.max_tokens,control='uniform RGB128, same dimensions and exact replayed token prefix',
        capture='lm_head input and output captured in generate() cached forward; blank prompt forces exact real prefix',
        source_resolution=384,protocol='docs/ver29_verification_protocol.md'),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,choices=MODELS)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--max-tokens',type=int,default=96)
    p.add_argument('--out',type=Path,default=ROOT/'results'/'ver29_controlled_exact')
    run(p.parse_args())
