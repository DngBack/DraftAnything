"""Fit identical target-domain calibrators; freeze before fresh confirmation."""
import hashlib,json
from pathlib import Path
import joblib,numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from vrdu_evaluate import FIELDS
import sys
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/stage7';PREV=ROOT/'results/stage5';sys.path.insert(0,str(ROOT.parent/'selective-kie'))
from common import threshold_at_risk
PAIRS=[(i,j) for i in range(4) for j in range(i+1,4)]
def inputs(score,field,isnull,variant):
    logit=np.log(np.clip(score,1e-6,1-1e-6)/np.clip(1-score,1e-6,1-1e-6))[:,None]
    if variant=='platt':return logit
    one=np.array([[int(f==k) for k in FIELDS] for f in field],float);n=isnull[:,None]
    return np.concatenate([logit,one,n,one*n],1)
def apply_map(score,field,isnull,m):
    if m['variant']=='raw':return score
    v=inputs(score,field,isnull,m['variant']);v=(v-np.array(m['mean']))/np.array(m['scale']);logit=v@np.array(m['coef'])+m['intercept'];return 1/(1+np.exp(-np.clip(logit,-60,60)))
def main():
    assert not list(OUT.glob('*predictions.jsonl')),'Do not refit maps after confirmation starts'
    data=np.load(PREV/'features.npz');z=data['nodes'];raw=np.concatenate([data['scalars'],np.stack([(z[:,i]*z[:,j]).sum(1) for i,j in PAIRS],1)],1);train=data['partition']=='train';dev=data['partition']=='development';scaler=StandardScaler().fit(raw[train]);xx=np.concatenate([z.reshape(len(z),-1),np.clip(scaler.transform(raw),-10,10)],1)
    tree=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,min_samples_leaf=40,l2_regularization=1,random_state=20260928,early_stopping=True,n_iter_no_change=15,tol=1e-5,validation_fraction=None)
    tree.fit(xx[train],data['y'][train],X_val=xx[dev],y_val=data['y'][dev]);devscore=tree.predict_proba(xx)[:,1];joblib.dump({'head':tree,'scaler':scaler},OUT/'semantic-hgb-dev.joblib')
    stored=np.load(PREV/'scores.npz');sem=np.load(PREV/'semantic-hgb-scores.npz');scores={'generic':stored['ensemble_generic'],'relation':stored['ensemble_relation'],'semantic-HGB':sem['source'],'semantic-HGB-dev':devscore}
    names=sorted(json.loads((ROOT/'results/stage4/holdout-manifest.json').read_text())['filenames'],key=lambda n:hashlib.sha256(n.encode()).hexdigest());stage=data['partition']=='stage4-qwen9b';fit=stage&np.isin(data['doc'],names[:50]);cal=stage&np.isin(data['doc'],names[50:]);assert fit.sum()==cal.sum()==300 and not (fit&cal).any()
    scalar_names=json.loads((PREV/'feature-manifest.json').read_text())['scalars'];isnull=data['scalars'][:,scalar_names.index('is_null')];field=data['field'];y=data['y'];result={}
    for name,score in scores.items():
        variants={}
        for variant in ['raw','platt','field-null']:
            m={'variant':variant}
            if variant!='raw':
                inp=inputs(score[fit],field[fit],isnull[fit],variant);st=StandardScaler().fit(inp);lr=LogisticRegression(C=1.,solver='lbfgs',max_iter=1000,tol=1e-8,random_state=20260928).fit(st.transform(inp),y[fit]);m.update(mean=st.mean_.tolist(),scale=st.scale_.tolist(),coef=lr.coef_[0].tolist(),intercept=float(lr.intercept_[0]))
            sc=apply_map(score,field,isnull,m);th=threshold_at_risk(sc[cal],1-y[cal],.05);m['threshold']=float(th) if np.isfinite(th) else None;variants[variant]=m
        ft={}
        for f in FIELDS:
            ix=cal&(field==f);th=threshold_at_risk(score[ix],1-y[ix],.05);ft[f]=float(th) if np.isfinite(th) else None
        variants['raw-field']={'variant':'raw','field_thresholds':ft};result[name]=variants
    paths=list(PREV.glob('*.pt'))+[PREV/'scalar-scaler.joblib',PREV/'semantic-hgb.joblib',OUT/'semantic-hgb-dev.joblib',ROOT/'scripts/relation_features.py',ROOT/'scripts/relation_train.py',ROOT/'scripts/stage7_calibrate.py']
    manifest={'maps':result,'calibration_fit_docs':names[:50],'threshold_docs':names[50:],'HGB_dev_iterations':int(tree.n_iter_),'field_order':FIELDS,'isnull_feature':'model predicted JSON null; no gold-presence input','frozen_files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'protocol_sha256':hashlib.sha256((ROOT/'docs/stage7-protocol.md').read_bytes()).hexdigest()}
    (OUT/'calibration-manifest.json').write_text(json.dumps(manifest,indent=2));print('Frozen target-model calibration:4 heads ×4 variants; HGB dev trees',tree.n_iter_)
if __name__=='__main__':main()
