"""Report predefined normalization and posthoc present-only diagnostics."""
import csv,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/stage6';sys.path.insert(0,str(ROOT.parent/'selective-kie'))
from common import aurc

def main():
    d=np.load(OUT/'scored-features.npz');rows=list(csv.DictReader((OUT/'scores.csv').open()));present=np.array([int(r['gold_present']) for r in rows],bool);err=1-d['y'];ne=1-d['normalized_y'];groups=np.arange(600).reshape(100,6);rng=np.random.default_rng(20260928);normdiff=[];pdiff=[]
    for _ in range(2000):
        ix=groups[rng.integers(100,size=100)].ravel();normdiff.append(aurc(d['relation'][ix],ne[ix])-aurc(d['generic'][ix],ne[ix]));pi=ix[present[ix]];pdiff.append(aurc(d['relation'][pi],err[pi])-aurc(d['generic'][pi],err[pi]))
    perfield={}
    for field in sorted({r['field'] for r in rows}):
        ix=np.array([r['field']==field for r in rows]);ms={}
        for name in ['generic','relation']:
            acc=np.array([int(r[name+'_accepted']) for r in rows],bool)&ix;ms[name]={'accepted':int(acc.sum()),'errors':int(err[acc].sum()),'coverage':float(acc.sum()/ix.sum()),'risk':float(err[acc].mean()) if acc.any() else None}
        perfield[field]=ms
    result={'normalized_secondary':{'difference':aurc(d['relation'],ne)-aurc(d['generic'],ne),'ci95':np.percentile(normdiff,[2.5,97.5]).tolist()},'present_only_posthoc':{'decisions':int(present.sum()),'generic_aurc':aurc(d['generic'][present],err[present]),'relation_aurc':aurc(d['relation'][present],err[present]),'difference':aurc(d['relation'][present],err[present])-aurc(d['generic'][present],err[present]),'ci95':np.percentile(pdiff,[2.5,97.5]).tolist()},'per_field_secondary':perfield,'interpretation':'pooled alpha5% does not establish per-field risk control; coverage gain includes ABSENT decisions'}
    (OUT/'sensitivity.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
