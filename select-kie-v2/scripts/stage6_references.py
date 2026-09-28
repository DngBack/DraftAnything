"""Frozen scalar-only references; use the matching sklearn1.9.1 CPU env."""
import csv,json,sys
from pathlib import Path
import numpy as np,joblib,sklearn
from sklearn.metrics import brier_score_loss
from vrdu_evaluate import x
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/stage6';sys.path.insert(0,str(ROOT.parent/'selective-kie'))
from common import aurc,apply_threshold
assert sklearn.__version__=='1.9.1'
def main():
    data=np.load(OUT/'scored-features.npz');features=json.loads((ROOT/'results/stage5/feature-manifest.json').read_text())['scalars'];rows=list(csv.DictReader((OUT/'scores.csv').open()));vf=[dict(zip(features,v)) for v in data['scalars']]
    for r,v in zip(rows,vf):v['field']=r['field']
    frozen=json.loads((OUT/'frozen-head-manifest.json').read_text());err=1-data['y'];ne=1-data['normalized_y'];reference={}
    for name in ['HGB-fusion','HGB-evidence']:
        obj=joblib.load(ROOT/f'results/stage2/vrdu-qwen35-predictions.{name}.joblib');sc=obj['head'].predict_proba(x(vf,obj['features']))[:,1];th=frozen['thresholds'][name];acc=sc>=th;c,risk=apply_threshold(sc,err,th);reference[name]={'aurc':aurc(sc,err),'normalized_aurc':aurc(sc,ne),'brier':float(brier_score_loss(1-err,sc)),'coverage':float(c),'risk':float(risk) if np.isfinite(risk) else None,'accepted':int(acc.sum()),'errors':int(err[acc].sum()),'threshold':th,'sklearn':sklearn.__version__}
        for i,row in enumerate(rows):row[name+'_score']=float(sc[i]);row[name+'_accepted']=int(acc[i])
    m=json.loads((OUT/'metrics.json').read_text());m['scalar_only_references']=reference;(OUT/'metrics.json').write_text(json.dumps(m,indent=2))
    with (OUT/'scores.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(reference,indent=2))
if __name__=='__main__':main()
