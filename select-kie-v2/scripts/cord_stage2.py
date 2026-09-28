"""Fixed-capacity evidence ablation and document CV on existing CORD predictions."""
import csv
import json
from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import brier_score_loss

import pilot as P
from evidence import CORD_ALIASES,EVIDENCE,GEOMETRY,support
from common import numbers_in,parse_num,norm_value,aurc,threshold_at_risk

A=P.A; OUT=P.OUT/'stage2'; OUT.mkdir(exist_ok=True)
CONFIGS={
 'fusion':A.FUSION,
 'fusion+binding':A.FUSION+A.BIND,
 'fusion+evidence':A.FUSION+EVIDENCE,
 'all':A.FUSION+A.BIND+EVIDENCE,
 'evidence-no-geometry':A.FUSION+[f for f in EVIDENCE if f not in GEOMETRY],
}

def new_head():
    return HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,min_samples_leaf=40,
                    l2_regularization=1.,early_stopping=False,random_state=20260928)

def enrich(rows,split):
    lookup={d['id']:d for d in P.docs(split)}
    predictions={}
    for r in rows: predictions.setdefault(r['doc'],{})[r['field']]=norm_value(r['pred'])
    for r in rows:
        d=lookup[r['doc']]; width,height=d['image'].size
        # Sanitize all annotation-specific attributes before feature computation.
        lines=[{'text':l['text'],'box':[l['box'][0]/width,l['box'][1]/height,l['box'][2]/width,l['box'][3]/height]} for l in d['lines']]
        vals=[{parse_num(s) for s in numbers_in(l['text'])} for l in lines]
        pv=norm_value(r['pred'])
        matches=[i for i,v in enumerate(vals) if pv is not None and pv in v]
        candidates=[i for i,v in enumerate(vals) if v]
        shared=sum(v==pv for f,v in predictions[r['doc']].items() if f!=r['field']) if pv is not None else 0
        r.update(support(lines,r['field'],matches,candidates,CORD_ALIASES,shared))
    return rows

def main():
    result={}
    for model in P.MODELS:
        print('Processing',model,flush=True)
        A.MD=str(P.ROOT/'selective-kie/outputs'/model)
        tr,va,te=[enrich([r for r in A.rows(s) if r['field'] in A.PRIMARY],s) for s in ['train','validation','test']]
        ev,et=np.array([r['err'] for r in va]),np.array([r['err'] for r in te])
        sv,st,measures={},{},{}
        for name,features in CONFIGS.items():
            xt,xv,xe=[P.matrix(r,features,True) for r in [tr,va,te]]
            yt=np.array([1-r['err'] for r in tr]); groups=np.array([r['doc'] for r in tr])
            oof=np.zeros(len(tr)); fold=[]
            for fit,held in GroupKFold(5).split(xt,yt,groups):
                h=new_head().fit(xt[fit],yt[fit]); oof[held]=h.predict_proba(xt[held])[:,1]
                fold.append(aurc(oof[held],1-yt[held]))
            h=new_head().fit(xt,yt)
            sv[name]=h.predict_proba(xv)[:,1];st[name]=h.predict_proba(xe)[:,1]
            joblib.dump({'head':h,'features':features,'field_order':A.PRIMARY},OUT/f'{model}.{name}.joblib')
            measures[name]={'aurc':aurc(st[name],et),'brier':float(brier_score_loss(1-et,st[name])),
                            'cv_fold_aurc':fold,'cv_mean_aurc':float(np.mean(fold)),
                            'targets':{str(a):P.metric(st[name],et,threshold_at_risk(sv[name],ev,a)) for a in [.02,.05,.10]}}
            print(name,measures[name],flush=True)
        contrasts={a+' minus '+b:P.bootstrap(va,te,sv,st,(a,b),1000) for a,b in [
            ('fusion+evidence','fusion'),('all','fusion+binding'),('all','fusion+evidence'),('fusion+evidence','evidence-no-geometry')]}
        total={r['doc']:norm_value(r['gold']) for r in va+te if r['field']=='total'}
        def keep(r): return not(r['field']=='subtotal' and norm_value(r['gold']) is None and norm_value(r['pred'])==total.get(r['doc']))
        iv=np.array([keep(r) for r in va]);it=np.array([keep(r) for r in te])
        sensitivity={n:P.metric(st[n][it],et[it],threshold_at_risk(sv[n][iv],ev[iv],.05)) for n in CONFIGS}
        result[model]={'measures':measures,'contrasts':contrasts,'sensitivity':sensitivity}
        with (OUT/f'cord-scores-{model}.csv').open('w') as f:
            w=csv.writer(f); w.writerow(['doc','field','error']+list(CONFIGS))
            for i,r in enumerate(te): w.writerow([r['doc'],r['field'],r['err']]+[st[n][i] for n in CONFIGS])
    (OUT/'cord-metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__': main()
