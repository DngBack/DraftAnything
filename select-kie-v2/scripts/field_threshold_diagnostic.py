"""Posthoc per-field empirical thresholds expose pooled-calibration confounds."""
import csv,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];PREV=ROOT/'results/stage5';OUT=ROOT/'results/stage6';sys.path.insert(0,str(ROOT.parent/'selective-kie'))
from common import threshold_at_risk

def main():
    source=np.load(PREV/'features.npz');cal=source['partition']=='calibration';se=1-source['y'];sf=source['field'];old=np.load(PREV/'scores.npz');fresh=np.load(OUT/'scored-features.npz');rows=list(csv.DictReader((OUT/'scores.csv').open()));ff=np.array([r['field'] for r in rows]);fe=1-fresh['y'];pairs={n:(old['ensemble_'+n],fresh[n]) for n in ['generic','relation']};sem=np.load(PREV/'semantic-hgb-scores.npz');pairs['semantic-HGB']=(sem['source'],sem['stage6']);results={}
    for name,(sc,score) in pairs.items():
        accepted=np.zeros(600,bool);thresholds={};perfield={}
        for field in sorted(set(sf)):
            ix=cal&(sf==field);th=threshold_at_risk(sc[ix],se[ix],.05);thresholds[str(field)]=float(th) if np.isfinite(th) else None;fi=ff==field;acc=fi&(score>=th);accepted|=acc;perfield[str(field)]={'accepted':int(acc.sum()),'errors':int(fe[acc].sum()),'risk':float(fe[acc].mean()) if acc.any() else None}
        results[name]={'coverage':float(accepted.mean()),'accepted':int(accepted.sum()),'errors':int(fe[accepted].sum()),'risk':float(fe[accepted].mean()) if accepted.any() else None,'thresholds':thresholds,'per_field':perfield}
    result={'status':'posthoc after observing ABSENT contribution; no fresh confirmation for this calibration variant','fit_data':'old stage2 calibration50 only; no stage6 labels used to select thresholds','criterion':'same empirical alpha5% independently per field; no formal risk certificate','measures':results}
    (OUT/'field-threshold-diagnostic.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:{f:v[f] for f in ['coverage','accepted','errors','risk']} for k,v in results.items()},indent=2))
if __name__=='__main__':main()
