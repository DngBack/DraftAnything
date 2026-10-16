"""Exploratory label-budget test with explicit unlabeled prompt-pair access.

All methods have the same labelled base images. Transform estimates use only
outer-training images; no test features or labels enter centering/SVD/scaling.
Ranks and C fixed in protocol addendum. This is prompt adaptation, not unseen-
prompt zero-shot learning. Dataset is the development VLMBias corpus.
"""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd
from threadpoolctl import threadpool_limits
from cached_geometry_pilot import image_folds
from verify_prompt_readout import PATHS,ROOT


def run(model,seed,budgets):
    with np.load(PATHS[model]) as d:
        H=d['last'][:,-1].astype(np.float32)
        y,key,g,q=[d[k] for k in ('y','key','group','q')]
    word=np.zeros(len(y),int)
    for group in np.unique(g):
        qs=sorted(np.unique(q[g==group]));word[(g==group)&(q==qs[1])]=1
    fold=image_folds(y,key,g,seed)
    methods=['source','both_word','both_word_pca32','mean_center','pair64_source','pair64_both']
    pred={(k,m):np.full(len(y),-1,int) for k in budgets for m in methods}
    for f in range(5):
        tr,te=fold!=f,fold==f;u=np.unique(key[tr])
        D=np.stack([H[tr&(key==im)&(word==1)].mean(0)-H[tr&(key==im)&(word==0)].mean(0) for im in u])
        _,_,V=np.linalg.svd(D,full_matrices=False);V=V[:64]
        P=H-(H@V.T)@V
        centered=H.copy()
        for group in np.unique(g):
            for w in (0,1):
                mask=(g==group)&(word==w);centered[mask]-=H[tr&mask].mean(0)
        mu=H[tr].mean(0)
        _,_,B=randomized_svd(H[tr]-mu,n_components=32,n_iter=3,random_state=0)
        pc=(H-mu)@B.T
        label={im:int(y[np.flatnonzero(key==im)[0]]) for im in u}
        rng=np.random.default_rng(seed*101+f)
        ordered={c:rng.permutation([im for im in u if label[im]==c]) for c in (0,1)}
        for src in (0,1):
            test=te&(word!=src)
            features={'source':(H,tr&(word==src)), 'both_word':(H,tr),
                      'both_word_pca32':(pc,tr),'mean_center':(centered,tr&(word==src)),
                      'pair64_source':(P,tr&(word==src)),'pair64_both':(P,tr)}
            for budget in budgets:
                chosen=np.concatenate([ordered[c][:budget] for c in (0,1)]) if budget else u
                lab=np.isin(key,chosen)
                for method,(X,available) in features.items():
                    sc=StandardScaler().fit(X[available])
                    train=available&lab
                    clf=LogisticRegression(C=1,max_iter=1500).fit(sc.transform(X[train]),y[train])
                    pred[budget,method][test]=clf.predict(sc.transform(X[test]))
    rows=[]
    for (budget,method),pp in pred.items():
        assert (pp>=0).all()
        rows.append(dict(model=model,seed=seed,labels_per_class=budget,method=method,acc=float((pp==y).mean())))
    print(json.dumps(dict(model=model,seed=seed,acc={str(k):{r['method']:round(r['acc'],4) for r in rows if r['labels_per_class']==k} for k in budgets})),flush=True)
    return rows,dict(y=y,key=key,**{f'k{k}__{m}':v for (k,m),v in pred.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',nargs='+',default=['qwen2vl_2b','qwen3vl4b','gemma3_4b'])
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--budgets',type=int,nargs='+',default=[2,5,10,20,0])
    p.add_argument('--out',type=Path,default=ROOT/'results'/'ver29_label_efficiency')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);rows=[]
    with threadpool_limits(limits=2):
        for model in a.models:
            for seed in a.seeds:
                rr,pr=run(model,seed,a.budgets);rows.extend(rr)
                np.savez_compressed(a.out/f'{model}_seed{seed}.npz',**pr)
                pd.DataFrame(rows).to_csv(a.out/'metrics.csv',index=False)
