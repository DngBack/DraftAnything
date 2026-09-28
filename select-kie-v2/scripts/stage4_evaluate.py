"""Locked paired analysis and blinded human audit preparation; no head fitting."""
import csv,gzip,hashlib,html,json
from pathlib import Path
import numpy as np
from holdout_evaluate import correctness,FIELDS
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/stage4'
def main():
    names=json.loads((OUT/'holdout-manifest.json').read_text())['filenames'];docs={}
    with gzip.open(ROOT/'data/vrdu/fara.jsonl.gz','rt') as f:
        for line in f:
            d=json.loads(line)
            if d['filename'] in names:docs[d['filename']]=d
    arrays={};measures={};rows=[]
    for tag in ['qwen2b','qwen9b']:
        records=[json.loads(l) for l in (OUT/f'{tag}-image-predictions.jsonl').read_text().splitlines()]
        assert len(records)==300 and len({(r['id'],r['schema']) for r in records})==300
        for schema in ['baseline','length_control','clarified']:
            recs={r['id']:r for r in records if r['schema']==schema};assert set(recs)==set(names)
            key=tag+'-'+schema;c=[];nc=[];present=[];hc_errors=hc_total=missing=0
            for name in names:
                gold=dict(docs[name]['annotations']);cc=[];nn=[];pp=[]
                for field in FIELDS:
                    gs=gold.get(field,[]);raw=recs[name]['fields'][field];pred=raw['pred']
                    pred=None if pred is None or not str(pred).strip() else str(pred)
                    ok=correctness(pred,gs,field);nok=correctness(pred,gs,field,True)
                    cc.append(ok);nn.append(nok);pp.append(bool(gs));missing+=raw['missing_key']
                    lp=raw['lps'];hc=bool(lp and np.mean(lp)>np.log(.99));hc_total+=hc;hc_errors+=hc and not ok
                    rows.append({'condition':key,'doc':name,'field':field,'prediction':pred,'gold':' | '.join(g[0] for g in gs),'correct':int(ok),'normalized_correct':int(nok),'high_confidence':int(hc),'human_type':'','notes':''})
                c.append(cc);nc.append(nn);present.append(pp)
            a=np.array(c,int);n=np.array(nc,int);p=np.array(present,bool);arrays[key]=(a,n)
            measures[key]={'accuracy':float(a.mean()),'normalized_accuracy':float(n.mean()),'present_accuracy':float(a[p].mean()),'absent_accuracy':float(a[~p].mean()),'high_confidence_decisions':int(hc_total),'high_confidence_errors':int(hc_errors),'missing_keys':int(missing),'output_caps':sum(r['output_capped'] for r in recs.values()),'mean_generate_seconds':float(np.mean([r['batch_seconds']/r['batch_size'] for r in recs.values()])),'peak_memory_gib':max(r['peak_memory_gib'] for r in recs.values()),'per_field_accuracy':dict(zip(FIELDS,a.mean(axis=0).tolist()))}
    rng=np.random.default_rng(20260928);ix=rng.integers(len(names),size=(2000,len(names)));contrasts={}
    pairs=[(t+'-clarified',t+'-length_control') for t in ['qwen2b','qwen9b']]+[(t+'-length_control',t+'-baseline') for t in ['qwen2b','qwen9b']]+[('qwen9b-'+s,'qwen2b-'+s) for s in ['baseline','length_control','clarified']]
    for a,b in pairs:
        contrasts[a+' minus '+b]={}
        for j,metric in enumerate(['official','normalized']):
            delta=arrays[a][j]-arrays[b][j];samples=delta[ix].mean(axis=(1,2))
            contrasts[a+' minus '+b][metric]={'difference':float(delta.mean()),'ci95':np.percentile(samples,[2.5,97.5]).tolist()}
    result={'documents':len(names),'decisions_per_condition':len(names)*6,'measures':measures,'contrasts':contrasts,'human_audit_status':'pending','inputs_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*predictions.jsonl')}}
    (OUT/'metrics.json').write_text(json.dumps(result,indent=2))
    with (OUT/'all-decisions.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    population=[i for i,r in enumerate(rows) if r['condition']=='qwen9b-clarified']
    random_indices=rng.choice(population,100,replace=False).tolist()
    targeted=[]
    for condition in measures:
        errors=[i for i,r in enumerate(rows) if r['condition']==condition and not r['correct'] and i not in random_indices]
        targeted.extend(rng.choice(errors,min(16,len(errors)),replace=False).tolist())
    remaining=[i for i,r in enumerate(rows) if not r['correct'] and i not in random_indices+targeted]
    targeted.extend(rng.choice(remaining,min(100-len(targeted),len(remaining)),replace=False).tolist())
    selected=random_indices+targeted;rng.shuffle(selected)
    audit=[];mapping=[]
    for j,i in enumerate(selected):
        r=rows[i];aid=f'A{j+1:03d}';audit.append({'audit_id':aid,'doc':r['doc'],'field':r['field'],'prediction':r['prediction'],'gold':r['gold'],'human_type':'','notes':''});mapping.append({'audit_id':aid,**r,'stratum':'random' if i in random_indices else 'targeted_error'})
    for fn,data in [('audit-reviewer1.csv',audit),('audit-reviewer2-overlap.csv',audit[:20]),('audit-private-mapping.csv',mapping)]:
        with (OUT/fn).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    cards=[]
    for r in audit:
        pdf='../../data/vrdu/pdfs/'+hashlib.sha256(r['doc'].encode()).hexdigest()+'.pdf'
        cards.append('<article><h3>'+r['audit_id']+' — '+html.escape(r['field'])+'</h3><p>Prediction: '+html.escape(str(r['prediction']))+'</p><p>Gold: '+html.escape(r['gold'])+'</p><a href="'+pdf+'">Open all PDF pages</a><details><summary>OCR reference</summary><pre>'+html.escape(docs[r['doc']]['ocr']['text'])+'</pre></details></article>')
    (OUT/'audit-viewer.html').write_text('<!doctype html><meta charset="utf-8"><title>Blinded binding audit</title><style>body{max-width:1000px;margin:auto;font-family:sans-serif}article{border-bottom:1px solid #ccc;padding:20px}pre{white-space:pre-wrap}</style><h1>200-decision human audit</h1><p>Labels: correct, binding, miss, hallucination, partial-value, transcription, gold/schema-ambiguity, other. Verify PDF, not only OCR. Model and confidence hidden. Save labels in reviewer CSV.</p>'+''.join(cards))
    report=['# Stage 4 results','', '| Condition | Official accuracy | Normalized | High-confidence errors/decisions |','|---|---:|---:|---:|']
    for k,m in measures.items():report.append(f"| {k} | {m['accuracy']:.2%} | {m['normalized_accuracy']:.2%} | {m['high_confidence_errors']}/{m['high_confidence_decisions']} |")
    report+=['','Paired document bootstrap95% intervals:','']
    for k,v in contrasts.items():
        d=v['official'];report.append(f"- {k}: {100*d['difference']:+.2f} percentage points; CI [{100*d['ci95'][0]:+.2f}, {100*d['ci95'][1]:+.2f}].")
    report+=['','Human audit remains pending. 9B batch4 encountered a multi-page memory failure and resumed batch2 without reducing pages or resolution; failed attempts, loading and rendering are excluded from generate-only timings. The targeted stratum cannot estimate population binding prevalence; use the random100 separately. These are scale and schema diagnostics within one model family, not a new relational architecture or evidence of a guaranteed CVPR score.']
    (ROOT/'docs/stage4-results.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(measures,indent=2))
if __name__=='__main__':main()
