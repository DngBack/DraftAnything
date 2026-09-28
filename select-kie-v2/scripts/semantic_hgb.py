"""Post-confirmation stronger generic baseline; fixed config, no target fitting."""
import json,hashlib,sys
from pathlib import Path
import numpy as np,joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/stage5';sys.path.insert(0,str(ROOT.parent/'selective-kie'))
from common import aurc,threshold_at_risk,apply_threshold
PAIRS=[(i,j) for i in range(4) for j in range(i+1,4)]
def rawside(z,s):return np.concatenate([s,np.stack([(z[:,i]*z[:,j]).sum(1) for i,j in PAIRS],1)],1)
def metric(sc,err,th,ne):
    acc=sc>=th;c,r=apply_threshold(sc,err,th)
    return {'aurc':aurc(sc,err),'normalized_aurc':aurc(sc,ne),'brier':float(brier_score_loss(1-err,sc)),'coverage':float(c),'risk':float(r) if np.isfinite(r) else None,'accepted':int(acc.sum()),'errors':int(err[acc].sum()),'normalized_errors':int(ne[acc].sum()),'threshold':float(th) if np.isfinite(th) else None}
def main():
    d=np.load(OUT/'features.npz');train=d['partition']=='train';cal=d['partition']=='calibration';z=d['nodes'];raw=rawside(z,d['scalars']);scaler=StandardScaler().fit(raw[train]);inputs=np.concatenate([z.reshape(len(z),-1),np.clip(scaler.transform(raw),-10,10)],1)
    config={'max_iter':100,'max_leaf_nodes':7,'min_samples_leaf':40,'l2_regularization':1.,'random_state':20260928,'early_stopping':False}
    head=HistGradientBoostingClassifier(**config).fit(inputs[train],d['y'][train]);score=head.predict_proba(inputs)[:,1];err=1-d['y'];ne=1-d['normalized_y'];th=threshold_at_risk(score[cal],err[cal],.05);measures={}
    for target in ['test','stage4-qwen2b','stage4-qwen9b']:
        ix=d['partition']==target;measures[target]=metric(score[ix],err[ix],th,ne[ix])
    fresh=np.load(ROOT/'results/stage6/scored-features.npz');fz=fresh['nodes'];fs=rawside(fz,fresh['scalars']);fx=np.concatenate([fz.reshape(len(fz),-1),np.clip(scaler.transform(fs),-10,10)],1);fscore=head.predict_proba(fx)[:,1];fe=1-fresh['y'];measures['stage6']=metric(fscore,fe,th,1-fresh['normalized_y'])
    relation=fresh['relation'];generic=fresh['generic'];rng=np.random.default_rng(20260928);groups=np.arange(600).reshape(100,6);delta=[];ndelta=[]
    for _ in range(2000):
        ix=groups[rng.integers(100,size=100)].ravel();delta.append(aurc(relation[ix],fe[ix])-aurc(fscore[ix],fe[ix]));ndelta.append(aurc(relation[ix],(1-fresh['normalized_y'])[ix])-aurc(fscore[ix],(1-fresh['normalized_y'])[ix]))
    result={'status':'post-confirmation diagnostic; chosen after stage6 neural comparison; not preregistered primary','information':'same four raw384d nodes, six raw cosine similarities, stage2 scalar features and field one-hot; no extra labels','config':config,'measures':measures,'stage6_relation_minus_semantic_hgb_aurc':{'difference':aurc(relation,fe)-aurc(fscore,fe),'ci95':np.percentile(delta,[2.5,97.5]).tolist()},'stage6_normalized_difference':{'difference':aurc(relation,1-fresh['normalized_y'])-aurc(fscore,1-fresh['normalized_y']),'ci95':np.percentile(ndelta,[2.5,97.5]).tolist()}}
    joblib.dump({'head':head,'scaler':scaler},OUT/'semantic-hgb.joblib');np.savez_compressed(OUT/'semantic-hgb-scores.npz',source=score,stage6=fscore);(OUT/'semantic-hgb-metrics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
