"""Fixed confidence heads on official unseen-template VRDU split. No tuning on test."""
import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor'))
sys.path.insert(0,str(ROOT.parent/'selective-kie'))
import vrdu_match_utils as M
from common import aurc,threshold_at_risk,apply_threshold
from evidence import VRDU_ALIASES,EVIDENCE,GEOMETRY,support,clean_text

FIELDS=list(VRDU_ALIASES)
MATCHERS={'registration_num':M.NumericalStringMatch,'registrant_name':M.NameMatch,
          'foreign_principle_name':M.NameMatch,'file_date':M.DateMatch,'signer_name':M.NameMatch,'signer_title':M.GeneralStringMatch}
BASE=['lp_mean','lp_min','lp_first','n_tok','is_null','missing_key','on_page','n_occ','pred_len','numeric','n_lines']
CONFIGS={'LR-fusion':(BASE,'lr'),'HGB-fusion':(BASE,'hgb'),
         'HGB-evidence':(BASE+EVIDENCE,'hgb'),'HGB-no-geometry':(BASE+[f for f in EVIDENCE if f not in GEOMETRY],'hgb')}

def x(rows,features):
    return np.array([[r[k] for k in features]+[int(r['field']==f) for f in FIELDS] for r in rows],float)

def metric(s,e,th):
    c,r=apply_threshold(s,e,th); accepted=s>=th
    return {'coverage':float(c),'risk':float(r) if np.isfinite(r) else None,'accepted':int(accepted.sum()),
            'errors':int(e[accepted].sum()),'threshold':float(th) if np.isfinite(th) else None}

def group_index(rows):
    d={}
    for i,r in enumerate(rows):d.setdefault(r['doc'],[]).append(i)
    return list(d.values())

def bootstrap(v,t,sv,st,a,b,B=1000):
    rng=np.random.default_rng(20260928);gv,gt=group_index(v),group_index(t)
    ev=np.array([r['err'] for r in v]);et=np.array([r['err'] for r in t]);diff=[];adiff=[];risks={a:[],b:[]};empty={a:0,b:0}
    for _ in range(B):
        iv=np.concatenate([gv[j] for j in rng.integers(len(gv),size=len(gv))]);it=np.concatenate([gt[j] for j in rng.integers(len(gt),size=len(gt))])
        cov={}
        for n in [a,b]:
            th=threshold_at_risk(sv[n][iv],ev[iv],.05); c,r=apply_threshold(st[n][it],et[it],th);cov[n]=c
            if np.isfinite(r):risks[n].append(r>.05)
            else:empty[n]+=1
        diff.append(cov[a]-cov[b]);adiff.append(aurc(st[a][it],et[it])-aurc(st[b][it],et[it]))
    return {'coverage_diff_ci95':np.percentile(diff,[2.5,97.5]).tolist(),
            'aurc_diff_ci95':np.percentile(adiff,[2.5,97.5]).tolist(),
            'risk_exceedance_given_nonempty':{n:float(np.mean(v)) if v else None for n,v in risks.items()},
            'empty_fraction':{n:v/B for n,v in empty.items()}}

def main():
    p=argparse.ArgumentParser();p.add_argument('--predictions',default='vrdu-predictions.jsonl');args=p.parse_args()
    out=ROOT/'results/stage2'; records=[json.loads(l) for l in (out/args.predictions).read_text().splitlines()]
    recs={r['id']:r for r in records};assert len(recs)==len(records)==600,'Require complete, unique 600 docs before evaluation'
    split=json.loads((ROOT/'data/vrdu/split.json').read_text())
    assert set(recs)==set(split['train']+split['valid']+split['test'])
    valid=sorted(split['valid'],key=lambda f:hashlib.sha256(f.encode()).hexdigest()); dev=set(valid[:50]);cal=set(valid[50:])
    rows={'train':[],'development':[],'calibration':[],'test':[]};excluded=Counter();audit=[];counts=Counter();lengths=[]
    with gzip.open(ROOT/'data/vrdu/fara.jsonl.gz','rt') as f:
        for line in f:
            d=json.loads(line);r=recs.get(d['filename'])
            if r is None:continue
            assert r['ocr_sha256']==hashlib.sha256(d['ocr']['text'].encode()).hexdigest()
            partition='development' if r['id'] in dev else 'calibration' if r['id'] in cal else r['split']
            counts[partition]+=1;lengths.append(r['input_tokens'])
            gold=dict(d['annotations']);text=clean_text(d['ocr']['text'])
            lines=[{'text':l['text'],'box':[l['bbox'][1],l['bbox'][2]+page['page_id'],l['bbox'][3],l['bbox'][4]+page['page_id']]}
                   for page in d['ocr']['pages'] for l in page['lines']]
            canon=[clean_text(l['text']) for l in lines]
            for field in FIELDS:
                gs=gold.get(field,[])
                if len({clean_text(g[0]) for g in gs})>1:excluded[partition+':'+field]+=1;continue
                pred=r['fields'][field]['pred'];pred=None if pred is None or not str(pred).strip() else str(pred)
                pv=clean_text(pred)
                ok=(not gs) if pred is None else bool(gs and MATCHERS[field].match((pred,None,None),gs))
                present=bool(pv and pv in text)
                if ok:etype='correct'
                elif pred is None:etype='miss'
                elif not gs:etype='absent-on-page' if present else 'absent-hallucination'
                elif present:etype='on-page-wrong'  # do not equate presence with annotated binding error
                else:etype='ungrounded-value'
                lp=r['fields'][field]['lps'];v=np.array(lp or [0.])
                matches=[i for i,t in enumerate(canon) if pv and pv in t]
                candidates=[i for i,l in enumerate(lines) if re.search(r'\d',l['text'])] if field=='registration_num' else list(range(len(lines)))
                shared=sum(pv==clean_text(r['fields'][f]['pred']) for f in FIELDS if f!=field) if pv else 0
                features=support(lines,field,matches,candidates,VRDU_ALIASES,shared)
                row={'doc':r['id'],'field':field,'err':int(not ok),'etype':etype,'pred':pred,'gold':[g[0] for g in gs],
                     'lp_mean':float(v.mean()),'lp_min':float(v.min()),'lp_first':float(v[0]),'n_tok':len(lp),'is_null':int(pred is None),
                     'missing_key':int(r['fields'][field]['missing_key']),'on_page':int(present),'n_occ':text.count(pv) if pv else 0,
                     'pred_len':len(pv),'numeric':int(bool(pv and pv.isdigit())),'n_lines':len(lines),**features}
                rows[partition].append(row)
    tr,va,te=rows['train'],rows['calibration'],rows['test'];sv,st,measures={},{},{}
    ev=np.array([r['err'] for r in va]);et=np.array([r['err'] for r in te])
    for name,(features,kind) in CONFIGS.items():
        h=(make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=5000)) if kind=='lr' else
           HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,min_samples_leaf=40,l2_regularization=1.,early_stopping=False,random_state=20260928))
        h.fit(x(tr,features),[1-r['err'] for r in tr]);sv[name]=h.predict_proba(x(va,features))[:,1];st[name]=h.predict_proba(x(te,features))[:,1]
        measures[name]={'aurc':aurc(st[name],et),'brier':float(brier_score_loss(1-et,st[name])),
                        'targets':{str(a):metric(st[name],et,threshold_at_risk(sv[name],ev,a)) for a in [.02,.05,.10]}}
        joblib.dump({'head':h,'features':features,'field_order':FIELDS},out/f'{Path(args.predictions).stem}.{name}.joblib')
        print(name,measures[name],flush=True)
    contrasts={a+' minus '+b:bootstrap(va,te,sv,st,a,b) for a,b in [('HGB-evidence','HGB-fusion'),('HGB-evidence','HGB-no-geometry')]}
    # Secondary: same head, source-template vs target-template calibration.
    train_names=sorted(split['train'],key=lambda f:hashlib.sha256(f.encode()).hexdigest())
    source_cal=set(train_names[-50:]); secondary_tr=[r for r in tr if r['doc'] not in source_cal]
    secondary_va=[r for r in tr if r['doc'] in source_cal]; secondary={}
    for name in ['HGB-fusion','HGB-evidence']:
        features=CONFIGS[name][0]
        h=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,min_samples_leaf=40,l2_regularization=1.,early_stopping=False,random_state=20260928)
        h.fit(x(secondary_tr,features),[1-r['err'] for r in secondary_tr])
        scores=h.predict_proba(x(te,features))[:,1]; source_scores=h.predict_proba(x(secondary_va,features))[:,1]
        target_scores=h.predict_proba(x(va,features))[:,1]
        secondary[name]={'train_docs':150,'source_calibration_docs':50,'target_calibration_docs':50,
                         'aurc':aurc(scores,et),
                         'source_calibration':metric(scores,et,threshold_at_risk(source_scores,[r['err'] for r in secondary_va],.05)),
                         'target_calibration':metric(scores,et,threshold_at_risk(target_scores,ev,.05))}
    result={'counts_docs':dict(counts),'counts_decisions':{k:len(v) for k,v in rows.items()},'excluded':dict(excluded),
            'accept_all_error':{k:float(np.mean([r['err'] for r in v])) for k,v in rows.items()},
            'taxonomy_test':dict(Counter(r['etype'] for r in te if r['err'])),
            'per_field_test':{f:{'decisions':sum(r['field']==f for r in te),'error_rate':float(np.mean([r['err'] for r in te if r['field']==f]))} for f in FIELDS},
            'measures':measures,'contrasts':contrasts,'calibration_shift_secondary':secondary,'max_input_tokens':max(lengths),
            'truncated_inputs':sum(r['truncated_input'] for r in recs.values()),'capped_outputs':sum(r['output_capped'] for r in recs.values()),
            'calibration_filenames':sorted(cal),'development_filenames':sorted(dev),
            'matcher_sha256':hashlib.sha256((ROOT/'vendor/vrdu_match_utils.py').read_bytes()).hexdigest()}
    stem=Path(args.predictions).stem
    (out/f'{stem}-metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    with (out/f'{stem}-test.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['doc','field','pred','gold','err','etype']+list(CONFIGS))
        for i,r in enumerate(te):w.writerow([r['doc'],r['field'],r['pred'],' | '.join(r['gold']),r['err'],r['etype']]+[st[n][i] for n in CONFIGS])
    print('Saved',stem,dict(counts),flush=True)

if __name__=='__main__':main()
