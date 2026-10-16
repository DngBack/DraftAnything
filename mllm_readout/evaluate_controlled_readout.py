"""Evaluate real/blank same-prefix readouts on image-disjoint splits.

Fixed hyperparameters; no method selection using these reported predictions.
The null ablation uses all native Yes/No surface rows. Generated- and fixed-
prefix conditions are reported separately. All classifier fitting is OOF.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from cached_geometry_pilot import image_folds
from readout_geometry_audit import native_rows
from pilot import MODELS

ROOT=Path(__file__).resolve().parent


def load(model,root):
    files=sorted((root/model).glob('[0-9]*.npz'))
    data=[dict(np.load(p)) for p in files]
    assert data
    return {k:np.stack([d[k] for d in data]) for k in data[0]
            if k not in ('fixed_ids','prefix_ids') and all(k in d and np.shape(d[k])==np.shape(data[0][k]) for d in data)}


def fit(X,y,tr):
    c=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1500))
    c.fit(X[tr],y[tr]);return c


def run(model,root,seed):
    d=load(model,root)
    y,key,q,group=[d[k] for k in ('y','key','q','group')]
    word=np.zeros(len(y),int)
    for g in np.unique(group):
        qs=sorted(np.unique(q[group==g]));assert len(qs)==2
        word[(group==g)&(q==qs[1])]=1
    fold=image_folds(y,key,group,seed)
    manifest=json.loads((root/model/'manifest.json').read_text())
    W,_,_=native_rows(MODELS[model],sorted(set(sum(manifest['candidate_ids'],[]))))
    _,sv,V=np.linalg.svd(W,full_matrices=False);V=V[sv>sv[0]*1e-6]
    predictions={};rows=[]
    for condition in ('fixed','generated'):
        if 'h_'+condition not in d:continue
        H,B,S,SB=[d[k+'_'+condition].astype(np.float32) for k in ('h','blank','s','s_blank')]
        diff=H-B
        feats={'hidden':H,'blank':B,'visual_difference':diff,
               'null_difference':diff-(diff@V.T)@V,
               'scores':S,'score_difference':S-SB}
        protocols={'both_word':[(fold!=f,fold==f) for f in range(5)],
                   'cross_word':[((fold!=f)&(word==w),(fold==f)&(word!=w)) for f in range(5) for w in (0,1)],
                   'same_word':[((fold!=f)&(word==w),(fold==f)&(word==w)) for f in range(5) for w in (0,1)],
                   'family_holdout':[(group!=g,group==g) for g in np.unique(group)]}
        raw=S.argmax(1)
        for protocol,splits in protocols.items():
            for feature,X in feats.items():
                pr=np.full(len(y),-1,int)
                for tr,te in splits:
                    assert not set(key[tr])&set(key[te])
                    pr[te]=fit(X,y,tr).predict(X[te])
                assert (pr>=0).all()
                k=f'{condition}__{protocol}__{feature}';predictions[k]=pr
                base=raw==y;ok=pr==y
                row=dict(model=model,seed=seed,condition=condition,protocol=protocol,feature=feature,
                         n=len(y),n_images=len(np.unique(key)),acc=float(ok.mean()),raw=float(base.mean()),
                         fixed=float(ok[~base].mean()),broken=float((~ok[base]).mean()))
                rows.append(row)
            print(json.dumps(dict(model=model,seed=seed,condition=condition,protocol=protocol,
                             acc={r['feature']:round(r['acc'],4) for r in rows if r['condition']==condition and r['protocol']==protocol})),flush=True)
    return rows,dict(y=y,key=key,group=group,q=q,fold=fold,**predictions)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',nargs='+',required=True)
    p.add_argument('--root',type=Path,default=ROOT/'results'/'ver29_controlled')
    p.add_argument('--out',type=Path,default=ROOT/'results'/'ver29_controlled_eval')
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);rows=[]
    with threadpool_limits(limits=2):
        for model in a.models:
            for seed in a.seeds:
                rr,pr=run(model,a.root,seed);rows.extend(rr)
                np.savez_compressed(a.out/f'{model}_seed{seed}.npz',**pr)
                pd.DataFrame(rows).to_csv(a.out/'metrics.csv',index=False)
