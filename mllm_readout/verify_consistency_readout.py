"""Fixed-lambda paired-invariance linear heads on cached states, CPU only.

Target-wording features of training images are available to the consistency
loss, as in Stage A. Compare against same-optimizer source/both-wording heads.
No test accuracy is used to select lambda, epoch or an individual model.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from cached_geometry_pilot import image_folds
from verify_prompt_readout import PATHS,ROOT


def fit(X,y,tr,D,lam,steps=300):
    mu,sd=X[tr].mean(0),X[tr].std(0)+1e-4
    Z=torch.tensor((X[tr]-mu)/sd)
    Y=torch.tensor(y[tr],dtype=torch.float32)
    delta=torch.tensor(D/sd)
    w=torch.zeros(X.shape[1],requires_grad=True);b=torch.zeros((),requires_grad=True)
    opt=torch.optim.Adam([w,b],lr=.01)
    for _ in range(steps):
        opt.zero_grad()
        loss=torch.nn.functional.binary_cross_entropy_with_logits(Z@w+b,Y)+1e-3*w.square().sum()
        if lam:loss=loss+lam*(delta@w).square().mean()
        loss.backward();opt.step()
    w,b=w.detach().numpy(),float(b.detach())
    return lambda A: (((A-mu)/sd)@w+b>=0).astype(int)


def run(name,seed):
    with np.load(PATHS[name]) as d:
        X=d['last'][:,-1].astype(np.float32)
        y,key,group,q=[d[k] for k in ('y','key','group','q')]
    word=np.zeros(len(y),int)
    for g in np.unique(group):
        qs=sorted(np.unique(q[group==g]));word[(group==g)&(q==qs[1])]=1
    fold=image_folds(y,key,group,seed)
    names=['adam_source','adam_both','consistency_1','consistency_10']
    pred={k:np.full(len(y),-1,int) for k in names}
    for f in range(5):
        tr,te=fold!=f,fold==f
        D=np.stack([X[tr&(key==im)&(word==1)].mean(0)-X[tr&(key==im)&(word==0)].mean(0) for im in np.unique(key[tr])])
        pred['adam_both'][te]=fit(X,y,tr,D,0)(X[te])
        for src in (0,1):
            a=tr&(word==src);b=te&(word!=src)
            for method,lam in [('adam_source',0),('consistency_1',1),('consistency_10',10)]:
                pred[method][b]=fit(X,y,a,D,lam)(X[b])
    rows=[dict(model=name,seed=seed,method=k,acc=float((p==y).mean())) for k,p in pred.items()]
    print(json.dumps(rows),flush=True)
    return rows,dict(y=y,key=key,**pred)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',nargs='+',default=['qwen2vl_2b','qwen3vl4b','gemma3_4b'])
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--out',type=Path,default=ROOT/'results'/'ver29_verify_consistency')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);rows=[];torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        for model in a.models:
            for seed in a.seeds:
                rr,pr=run(model,seed);rows.extend(rr)
                np.savez_compressed(a.out/f'{model}_seed{seed}.npz',**pr)
                pd.DataFrame(rows).to_csv(a.out/'metrics.csv',index=False)
