"""Matched calibration on fresh89; co-primary comparisons fixed beforehand."""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np,joblib
from sklearn.metrics import brier_score_loss
from stage7_calibrate import apply_map,PAIRS
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/stage7';PREV=ROOT/'results/stage5';sys.path.insert(0,str(ROOT.parent/'selective-kie'))
from common import aurc

def main():
    frozen=json.loads((OUT/'calibration-manifest.json').read_text())
    for p,sha in frozen['frozen_files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha,p
    d=np.load(OUT/'scored-features.npz');rows=list(csv.DictReader((OUT/'scores.csv').open()));assert len(rows)==534;z=d['nodes'];side=np.concatenate([d['scalars'],np.stack([(z[:,i]*z[:,j]).sum(1) for i,j in PAIRS],1)],1);raw={n:d[n] for n in ['generic','relation']}
    for name,path in [('semantic-HGB',PREV/'semantic-hgb.joblib'),('semantic-HGB-dev',OUT/'semantic-hgb-dev.joblib')]:
        obj=joblib.load(path);inp=np.concatenate([z.reshape(len(z),-1),np.clip(obj['scaler'].transform(side),-10,10)],1);raw[name]=obj['head'].predict_proba(inp)[:,1]
    names=json.loads((PREV/'feature-manifest.json').read_text())['scalars'];isnull=d['scalars'][:,names.index('is_null')];fields=np.array([r['field'] for r in rows]);present=np.array([int(r['gold_present']) for r in rows],bool);err=1-d['y'];ne=1-d['normalized_y'];scores={};measures={}
    for name,v in frozen['maps'].items():
        for variant,m in v.items():
            score=apply_map(raw[name],fields,isnull,m);key=name+'/'+variant;scores[key]=score
            if variant=='raw-field':threshold=np.array([m['field_thresholds'][f] if m['field_thresholds'][f] is not None else np.inf for f in fields])
            else:threshold=m['threshold'] if m['threshold'] is not None else np.inf
            accept=score>=threshold;ap=accept&present;aa=accept&~present
            measures[key]={'aurc':aurc(score,err),'normalized_aurc':aurc(score,ne),'present_aurc':aurc(score[present],err[present]),'normalized_present_aurc':aurc(score[present],ne[present]),'brier':float(brier_score_loss(1-err,score)),'coverage':float(accept.mean()),'accepted':int(accept.sum()),'errors':int(err[accept].sum()),'risk':float(err[accept].mean()) if accept.any() else None,'accepted_present':int(ap.sum()),'errors_present':int(err[ap].sum()),'accepted_absent':int(aa.sum()),'errors_absent':int(err[aa].sum())}
            for i,row in enumerate(rows):row[key+'_score']=float(score[i]);row[key+'_accepted']=int(accept[i])
    a=scores['relation/field-null'];b=scores['semantic-HGB-dev/field-null'];groups=np.arange(534).reshape(89,6);rng=np.random.default_rng(20260928);samples={'official':[],'normalized':[]}
    for _ in range(2000):
        ix=groups[rng.integers(89,size=89)].ravel();ix=ix[present[ix]]
        for metric,e in [('official',err),('normalized',ne)]:samples[metric].append(aurc(a[ix],e[ix])-aurc(b[ix],e[ix]))
    primary={}
    for metric,e in [('official',err),('normalized',ne)]:
        ci=np.percentile(samples[metric],[1.25,98.75]).tolist();primary[metric]={'difference':aurc(a[present],e[present])-aurc(b[present],e[present]),'ci97_5':ci}
    result={'documents':89,'decisions':534,'present_decisions':int(present.sum()),'measures':measures,'co_primary':primary,'co_primary_gate_passed':all(v['ci97_5'][1]<0 for v in primary.values()),'primary_comparator':'relation vs dev-stopped semantic HGB, field/null calibration; present-only','bootstrap':2000,'calibration_fit_docs':50,'threshold_docs':50,'human_audit':'pending; user has no reviewers currently','limitations':'same dataset; annotation OCR; no risk certificate; document bootstrap, not registrant-independent'}
    (OUT/'metrics.json').write_text(json.dumps(result,indent=2));np.savez_compressed(OUT/'calibrated-scores.npz',**scores)
    with (OUT/'all-scores.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({'co_primary':primary,'gate':result['co_primary_gate_passed'],'field-null':{k:v for k,v in measures.items() if k.endswith('/field-null')}},indent=2))
if __name__=='__main__':main()
