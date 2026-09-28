"""Fresh-document scoring with immutable stage5 heads and thresholds."""
import csv,gzip,hashlib,json,time
from pathlib import Path
import numpy as np,joblib,torch
from transformers import AutoModel,AutoTokenizer
from relation_features import nodes,SCALARS,FIELDS
from evidence import vrdu_features
from holdout_evaluate import correctness
from relation_train import Generic,Relation,PAIRS,metrics,aurc
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/stage7';PREV=ROOT/'results/stage5'
def main():
    torch.set_num_threads(4);frozen=json.loads((OUT/'calibration-manifest.json').read_text())
    for path,sha in frozen['frozen_files'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
    names=json.loads((OUT/'holdout-manifest.json').read_text())['filenames'];recs=[json.loads(l) for l in (OUT/'qwen9b-image-predictions.jsonl').read_text().splitlines()];assert len(recs)==len(names)==89;lookup={r['id']:r for r in recs};assert set(lookup)==set(names)
    docs={}
    with gzip.open(ROOT/'data/vrdu/fara.jsonl.gz','rt') as f:
        for line in f:
            d=json.loads(line)
            if d['filename'] in lookup:docs[d['filename']]=d
    texts=[];ti={};indices=[];side=[];ys=[];nys=[];rows=[];hc=[];present=[]
    for name in names:
        rec=lookup[name];d=docs[name];gold=dict(d['annotations'])
        for field in FIELDS:
            ns=nodes(d['ocr'],rec,field);ix=[]
            for text in ns:
                if text not in ti:ti[text]=len(texts);texts.append(text)
                ix.append(ti[text])
            indices.append(ix);vf=vrdu_features(d['ocr'],rec,field);side.append([vf[k] for k in SCALARS]+[int(field==f) for f in FIELDS])
            pred=rec['fields'][field]['pred'];pred=None if pred is None or not str(pred).strip() else str(pred);gs=gold.get(field,[]);ok=int(correctness(pred,gs,field));nok=int(correctness(pred,gs,field,True));ys.append(ok);nys.append(nok);present.append(bool(gs));lp=rec['fields'][field]['lps'];hc.append(bool(lp and np.mean(lp)>np.log(.99)))
            rows.append({'doc':name,'field':field,'prediction':pred,'gold':' | '.join(g[0] for g in gs),'correct':ok,'normalized_correct':nok,'high_confidence':int(hc[-1]),'gold_present':int(bool(gs))})
    enc=json.loads((PREV/'feature-manifest.json').read_text())['encoder_snapshot'];tokenizer=AutoTokenizer.from_pretrained(enc,local_files_only=True);model=AutoModel.from_pretrained(enc,local_files_only=True).cuda().eval();vectors=[];truncated=0;start=time.perf_counter()
    for off in range(0,len(texts),128):
        chunk=texts[off:off+128];truncated+=sum(len(t)>256 for t in tokenizer(chunk,add_special_tokens=True)['input_ids']);inp=tokenizer(chunk,padding=True,truncation=True,max_length=256,return_tensors='pt').to('cuda')
        with torch.inference_mode():
            h=model(**inp).last_hidden_state;mask=inp.attention_mask.unsqueeze(-1);pool=(h*mask).sum(1)/mask.sum(1).clamp(min=1);vectors.append(torch.nn.functional.normalize(pool,p=2,dim=1).cpu().numpy())
    z=np.concatenate(vectors)[np.array(indices)];side=np.array(side,np.float32);rawcos=np.stack([(z[:,i]*z[:,j]).sum(1) for i,j in PAIRS],1);scaler=joblib.load(PREV/'scalar-scaler.joblib');s=np.clip(scaler.transform(np.concatenate([side,rawcos],1)),-10,10).astype(np.float32);zz=torch.from_numpy(z);ss=torch.from_numpy(s);scores={}
    for name in ['generic','relation']:
        seed_scores=[]
        for seed in range(20260928,20260933):
            obj=torch.load(PREV/f'{name}-{seed}.pt',map_location='cpu',weights_only=True);head=Generic(s.shape[1],obj['generic_width']) if name=='generic' else Relation(s.shape[1]);head.load_state_dict(obj['state_dict']);head.eval()
            with torch.inference_mode():seed_scores.append(head(zz,ss).sigmoid().numpy())
        scores[name]=np.stack(seed_scores).mean(0)
    y=np.array(ys);ny=np.array(nys)
    for name,score in scores.items():
        for i,row in enumerate(rows):row[name+'_score']=float(score[i])
    np.savez_compressed(OUT/'scored-features.npz',nodes=z,scalars=side,y=y,normalized_y=ny,**scores)
    with (OUT/'scores.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    m={'documents':len(names),'decisions':len(rows),'extractor_accuracy':float(y.mean()),'normalized_extractor_accuracy':float(ny.mean()),'high_confidence_decisions':sum(hc),'high_confidence_errors':int((1-y)[np.array(hc)].sum()),'output_caps':sum(r['output_capped'] for r in recs),'missing_keys':sum(v['missing_key'] for r in recs for v in r['fields'].values()),'truncated_unique_encoder_texts':truncated,'frozen_files':'verified'}
    (OUT/'raw-metrics.json').write_text(json.dumps(m,indent=2));print(json.dumps(m,indent=2))
if __name__=='__main__':main()
