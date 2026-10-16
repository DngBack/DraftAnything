"""Train/evaluate a complete residual + consistency + calibrated gate prototype.

Paired wording states on TRAINING images only. Residual = same-prefix real-minus-
blank projected off candidate rows. Binary linear residual has rank <=1.
The base calibrator is frozen before residual fitting. A generic hidden probe
gets the identical gate-selection procedure and unique-label budget.
"""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from scipy.special import expit
from cached_geometry_pilot import image_folds
from evaluate_controlled_readout import load,ROOT
from readout_geometry_audit import native_rows
from pilot import MODELS


def linear(X,y,tr):
    sc=StandardScaler().fit(X[tr]);c=LogisticRegression(C=1,max_iter=1500).fit(sc.transform(X[tr]),y[tr])
    w=c.coef_[0]/sc.scale_;b=float(c.intercept_[0]-sc.mean_@w)
    return w,b


def train(H,B,S,y,key,word,tr,basis):
    D=H-B;D=D-(D@basis.T)@basis
    cw,cb=linear(S,y,tr)
    mu,sd=D[tr].mean(0),D[tr].std(0)+1e-4
    Z=torch.tensor((D[tr]-mu)/sd);Y=torch.tensor(y[tr],dtype=torch.float32)
    offset=torch.tensor((S[tr]@cw+cb).astype(np.float32))
    pairs=np.stack([D[tr&(key==im)&(word==1)].mean(0)-D[tr&(key==im)&(word==0)].mean(0) for im in np.unique(key[tr])])
    delta=torch.tensor(pairs/sd)
    w=torch.zeros(D.shape[1],requires_grad=True);b=torch.zeros((),requires_grad=True)
    opt=torch.optim.Adam([w,b],lr=.01)
    for _ in range(300):
        opt.zero_grad()
        loss=torch.nn.functional.binary_cross_entropy_with_logits(offset+Z@w+b,Y)+1e-3*w.square().sum()+(delta@w).square().mean()
        loss.backward();opt.step()
    rw=w.detach().numpy()/sd;rb=float(b.detach())-float(mu@rw)
    # Effective raw-space vector, so online evaluation needs only H-B.
    rw=rw-basis.T@(basis@rw)
    hw,hb=linear(H,y,tr)
    return dict(cw=cw,cb=cb,rw=rw,rb=rb,hw=hw,hb=hb)


def probabilities(m,H,B,S):
    return {'residual':expit(S@m['cw']+m['cb']+(H-B)@m['rw']+m['rb']),
            'probe':expit(H@m['hw']+m['hb']),
            'calibration':expit(S@m['cw']+m['cb'])}


def gate(p,raw,y):
    confidence=np.maximum(p,1-p);answer=(p>=.5).astype(int)
    choices=[]
    for t in [.5,.6,.7,.8,.9,.95,.99,1.01]:
        pred=np.where(confidence>=t,answer,raw)
        correct=raw==y
        broken=np.mean(pred[correct]!=y[correct]) if correct.any() else 0
        if broken<=.02:
            choices.append((np.mean(pred==y),t))
    return max(choices,key=lambda a:(a[0],a[1]))[1]


def run(model,root,out,seed):
    d=load(model,root);H,B,S=[d[k].astype(np.float32) for k in ('h_generated','blank_generated','s_generated')]
    y,key,group,q=[d[k] for k in ('y','key','group','q')]
    word=np.zeros(len(y),int)
    for g in np.unique(group):
        qs=sorted(np.unique(q[group==g]));word[(group==g)&(q==qs[1])]=1
    manifest=json.loads((root/model/'manifest.json').read_text())
    W,_,_=native_rows(MODELS[model],sorted(set(sum(manifest['candidate_ids'],[]))))
    _,sv,basis=np.linalg.svd(W,full_matrices=False);basis=basis[sv>sv[0]*1e-6]
    fold=image_folds(y,key,group,seed);raw=S.argmax(1)
    names=['raw','calibration','probe','residual','gated_probe','gated_residual']
    pred={k:np.full(len(y),-1,int) for k in names};pred['raw']=raw
    for f in range(5):
        tr,te=fold!=f,fold==f;u=np.unique(key[tr])
        labels=[y[np.flatnonzero(key==im)[0]] for im in u]
        fitkeys,valkeys=train_test_split(u,test_size=.25,random_state=seed*11+f,stratify=labels)
        a,b=np.isin(key,fitkeys),np.isin(key,valkeys)
        inner=train(H,B,S,y,key,word,a,basis)
        probs=probabilities(inner,H[b],B[b],S[b])
        threshold={k:gate(probs[k],raw[b],y[b]) for k in ('probe','residual')}
        m=train(H,B,S,y,key,word,tr,basis)
        probs=probabilities(m,H[te],B[te],S[te])
        for k in ('probe','residual','calibration'):pred[k][te]=(probs[k]>=.5).astype(int)
        for k in ('probe','residual'):
            p=probs[k];pred['gated_'+k][te]=np.where(np.maximum(p,1-p)>=threshold[k],p>=.5,raw[te])
        np.savez_compressed(out/f'{model}_seed{seed}_fold{f}_weights.npz',**m,
                            threshold_probe=threshold['probe'],threshold_residual=threshold['residual'])
    rows=[]
    for method,p in pred.items():
        correct=raw==y;ok=p==y
        rows.append(dict(model=model,seed=seed,method=method,n=len(y),n_images=len(np.unique(key)),
                         acc=float(ok.mean()),raw=float(correct.mean()),fixed=float(ok[~correct].mean()),
                         broken=float((~ok[correct]).mean()),changed=float((p!=raw).mean())))
    print(json.dumps(rows),flush=True)
    np.savez_compressed(out/f'{model}_seed{seed}_predictions.npz',y=y,key=key,fold=fold,**pred)
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',nargs='+',default=['qwen2vl_2b','qwen3vl4b','gemma3_4b'])
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--root',type=Path,default=ROOT/'results'/'ver29_controlled')
    p.add_argument('--out',type=Path,default=ROOT/'results'/'ver29_residual_repair')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);rows=[];torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        for model in a.models:
            for seed in a.seeds:
                rows.extend(run(model,a.root,a.out,seed))
                pd.DataFrame(rows).to_csv(a.out/'metrics.csv',index=False)
