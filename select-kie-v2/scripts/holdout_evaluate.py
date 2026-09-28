"""Confirmation-only analysis of the frozen 2x2 holdout experiment."""
import csv,gzip,hashlib,json,re,sys,time
from pathlib import Path
from collections import Counter
import numpy as np
import joblib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor'))
import vrdu_match_utils as M
from vrdu_evaluate import FIELDS,MATCHERS,x
from evidence import clean_text,vrdu_features

def correctness(pred,gs,field,normalized=False):
    if pred is None:return not gs
    if not gs:return False
    if normalized:
        if field=='file_date':
            cleaned=[(re.split(r'\(?\s*date of signature',g[0],flags=re.I)[0].strip(),None,None) for g in gs]
            return M.DateMatch.match((pred,None,None),cleaned)
        if field in ['registrant_name','foreign_principle_name','signer_name','signer_title']:
            return clean_text(pred) in {clean_text(g[0]) for g in gs}
    return MATCHERS[field].match((pred,None,None),gs)

def main():
    out=ROOT/'results/stage3';manifest=json.loads((out/'holdout-manifest.json').read_text());names=manifest['filenames']
    records={};conditions=[]
    for modality in ['ocr','image']:
        items=[json.loads(l) for l in (out/f'{modality}-predictions.jsonl').read_text().splitlines()]
        assert len(items)==200
        assert {(r['id'],r['schema']) for r in items}=={(n,s) for n in names for s in ['baseline','clarified']}
        for schema in ['baseline','clarified']:
            key=modality+'-'+schema;conditions.append(key);records[key]={r['id']:r for r in items if r['schema']==schema}
    docs={}
    with gzip.open(ROOT/'data/vrdu/fara.jsonl.gz','rt') as f:
        for l in f:
            d=json.loads(l)
            if d['filename'] in names:docs[d['filename']]=d
    measures={};rows=[];arrays={};excluded=Counter();gold_present=[]
    frozen_metrics=json.loads((ROOT/'results/stage2/vrdu-qwen35-predictions-metrics.json').read_text())
    frozen_heads={n:joblib.load(ROOT/f'results/stage2/vrdu-qwen35-predictions.{n}.joblib') for n in ['HGB-fusion','HGB-evidence']}
    for key in conditions:
        correct=[];norm_correct=[];present_masks=[];stats=Counter();perfield={};elapsed=[];mem=[];feature_rows=[];feature_seconds=0.
        for name in names:
            d=docs[name];rec=records[key][name];gold=dict(d['annotations']);text=clean_text(d['ocr']['text']);c=[];nc=[];present=[]
            elapsed.append(rec['batch_seconds']/rec['batch_size']);mem.append(rec['peak_memory_gib'])
            for field in FIELDS:
                gs=gold.get(field,[])
                assert len({clean_text(g[0]) for g in gs})<=1,'Multi-value scope needs explicit exclusion'
                raw=rec['fields'][field];pred=raw['pred'];pred=None if pred is None or not str(pred).strip() else str(pred)
                feature_start=time.perf_counter();feature_rows.append(vrdu_features(d['ocr'],rec,field));feature_seconds+=time.perf_counter()-feature_start
                ok=correctness(pred,gs,field);nok=correctness(pred,gs,field,True);c.append(int(ok));nc.append(int(nok));present.append(int(bool(gs)))
                lp=np.array(raw['lps'] or [0.]);hc=bool(len(raw['lps']) and lp.mean()>np.log(.99));pv=clean_text(pred);on_page=bool(pv and pv in text)
                if hc:stats['high_confidence_decisions']+=1
                if hc and not ok:stats['high_confidence_errors']+=1
                if not ok:
                    typ=('miss' if pred is None else 'absent-on-page' if not gs and on_page else 'absent-hallucination' if not gs else 'on-page-wrong' if on_page else 'ungrounded-value')
                    stats[typ]+=1
                pf=perfield.setdefault(field,Counter());pf['decisions']+=1;pf['correct']+=int(ok);pf['normalized_correct']+=int(nok)
                rows.append({'condition':key,'doc':name,'field':field,'pred':pred,'gold':' | '.join(g[0] for g in gs),
                             'correct':int(ok),'normalized_correct':int(nok),'gold_present':int(bool(gs)),
                             'on_page':int(on_page),'mean_token_lp':float(lp.mean()),'human_type':'','notes':''})
            correct.append(c);norm_correct.append(nc);present_masks.append(present)
        c=np.array(correct);nc=np.array(norm_correct);present=np.array(present_masks,bool)
        arrays[key]={'correct':c,'normalized_correct':nc};gold_present=present
        transfer={}
        for name,obj in frozen_heads.items():
            head_start=time.perf_counter()
            scores=obj['head'].predict_proba(x(feature_rows,obj['features']))[:,1]
            head_seconds=time.perf_counter()-head_start
            th=frozen_metrics['measures'][name]['targets']['0.05']['threshold']
            acc=scores>=th if th is not None else np.zeros(len(scores),bool)
            error=1-c.ravel();ne=1-nc.ravel()
            transfer[name]={'threshold_from_stage2':th,'coverage':float(acc.mean()),'accepted':int(acc.sum()),
                            'official_errors':int(error[acc].sum()),'normalized_errors':int(ne[acc].sum()),
                            'official_risk':float(error[acc].mean()) if acc.any() else None,
                            'normalized_risk':float(ne[acc].mean()) if acc.any() else None,
                            'mean_head_seconds_per_doc':head_seconds/100}
        measures[key]={'decisions':int(c.size),'accuracy':float(c.mean()),'normalized_accuracy':float(nc.mean()),
                       'present_accuracy':float(c[present].mean()),'absent_accuracy':float(c[~present].mean()),
                       'stats':dict(stats),'per_field':{f:dict(v) for f,v in perfield.items()},
                       'mean_amortized_generate_seconds':float(np.mean(elapsed)), 'max_peak_memory_gib':max(mem),
                       'capped_outputs':sum(r['output_capped'] for r in records[key].values()),
                       'truncated_inputs':sum(r['truncated_input'] for r in records[key].values()),'frozen_selector_transfer':transfer,
                       'mean_evidence_feature_seconds_per_doc':feature_seconds/100}
        print(key,measures[key]['accuracy'],measures[key]['normalized_accuracy'],measures[key]['stats'],flush=True)
    contrasts={};rng=np.random.default_rng(20260928);ix=rng.integers(100,size=(2000,100))
    for a,b in [('image-baseline','ocr-baseline'),('image-clarified','ocr-clarified'),
                ('ocr-clarified','ocr-baseline'),('image-clarified','image-baseline')]:
        contrast={}
        for metric in ['correct','normalized_correct']:
            delta=arrays[a][metric]-arrays[b][metric];samples=delta[ix].mean(axis=(1,2))
            contrast[metric]={'accuracy_difference':float(delta.mean()),'ci95':np.percentile(samples,[2.5,97.5]).tolist()}
        contrasts[a+' minus '+b]=contrast
    result={'documents':100,'gold_present_decisions':int(gold_present.sum()),'measures':measures,'contrasts':contrasts,
            'bootstrap':2000,'seed':20260928,'model':'Qwen3.5-2B','limitations':'same dataset; document holdout, not registrant/person holdout; schema clarification is not architecture novelty',
            'inputs_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [out/'holdout-manifest.json',out/'ocr-predictions.jsonl',out/'image-predictions.jsonl']}}
    (out/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    with (out/'audit.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

if __name__=='__main__':main()
