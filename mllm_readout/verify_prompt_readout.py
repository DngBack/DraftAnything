"""Cached prompt-adaptation experiments; see docs/ver29_verification_protocol.md.

All adapted methods see target-prompt features of training images. They do NOT
have zero-shot access to an unseen prompt. Both-wording LR uses exactly the same
unique image labels and is the strongest access-matched simple baseline.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from cached_geometry_pilot import image_folds

ROOT = Path(__file__).resolve().parent
PATHS = {p.name.replace('_illusions_states.npz', ''): p for dr in ('g3', 'g3_heldout')
         for p in (ROOT/'results'/dr).glob('*_illusions_states.npz')}


def lr(X, y, tr):
    sc = StandardScaler().fit(X[tr])
    c = LogisticRegression(C=1, max_iter=1500).fit(sc.transform(X[tr]), y[tr])
    return lambda Z: c.predict(sc.transform(Z))


def run(name, seed, ranks=(1,4)):
    with np.load(PATHS[name]) as d:
        H = d['last'][:, -1].astype(np.float32)
        y, raw, s, key, group, q = [d[k] for k in ('y','ans','scores','key','group','q')]
    word = np.zeros(len(y), int)
    for g in np.unique(group):
        qs = sorted(np.unique(q[group==g])); assert len(qs)==2
        word[(group==g)&(q==qs[1])]=1
    folds = image_folds(y,key,group,seed)
    names = ['raw','margin_cal','source_lr','same_word_lr','both_word_lr','mean_center_lr',
             'random_remove_4']+[f'pair_remove_{k}' for k in ranks]
    predictions={k:np.full((2,len(y)),-1,int) for k in names}
    for f in range(5):
        train, test=folds!=f,folds==f
        assert not set(key[train]) & set(key[test])
        # Same image order within each wording; average the resolutions.
        u=np.unique(key[train])
        D=np.stack([H[train&(key==im)&(word==1)].mean(0)-H[train&(key==im)&(word==0)].mean(0) for im in u])
        _,_,V=np.linalg.svd(D,full_matrices=False)
        r=np.linalg.qr(np.random.default_rng(seed+f+300).normal(size=(H.shape[1],4)))[0].T
        centered=H.copy()
        for g in np.unique(group):
            for w in (0,1):
                mask=(group==g)&(word==w)
                centered[mask]-=H[train&mask].mean(0)
        both=lr(H,y,train)
        for src in (0,1):
            tr=train&(word==src);te=test&(word!=src)
            predictions['raw'][src,te]=raw[te]
            margin=(s[:,0]-s[:,1])[:,None]
            predictions['margin_cal'][src,te]=lr(margin,y,tr)(margin[te])
            predictions['source_lr'][src,te]=lr(H,y,tr)(H[te])
            predictions['same_word_lr'][src,te]=lr(H,y,train&(word!=src))(H[te])
            predictions['both_word_lr'][src,te]=both(H[te])
            predictions['mean_center_lr'][src,te]=lr(centered,y,tr)(centered[te])
            for k,basis in [(f'pair_remove_{k}',V[:k]) for k in ranks]+[('random_remove_4',r)]:
                X=H-(H@basis.T)@basis
                predictions[k][src,te]=lr(X,y,tr)(X[te])
    rows=[]
    for method,pr in predictions.items():
        p=np.maximum(pr[0],pr[1]);assert np.all(p>=0)
        base=raw==y;ok=p==y
        rows.append(dict(model=name,seed=seed,method=method,n=len(y),n_images=len(np.unique(key)),
                         acc=float(ok.mean()),raw=float(base.mean()),fixed=float(ok[~base].mean()),
                         broken=float((~ok[base]).mean())))
    print(json.dumps(dict(model=name,seed=seed,acc={r['method']:round(r['acc'],4) for r in rows})),flush=True)
    return rows,dict(y=y,key=key,fold=folds,**{k:np.maximum(v[0],v[1]) for k,v in predictions.items()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',nargs='+',choices=PATHS,default=['qwen2vl_2b','qwen3vl4b','gemma3_4b'])
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--ranks',nargs='+',type=int,default=[1,4])
    p.add_argument('--out',type=Path,default=ROOT/'results'/'ver29_verify_prompt')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    rows=[]
    with threadpool_limits(limits=2):
        for name in a.models:
            for seed in a.seeds:
                r,pr=run(name,seed,a.ranks);rows.extend(r)
                np.savez_compressed(a.out/f'{name}_seed{seed}.npz',**pr)
                pd.DataFrame(rows).to_csv(a.out/'metrics.csv',index=False)


if __name__=='__main__':main()
