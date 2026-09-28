"""Frozen semantic nodes from OCR/predictions only; labels added separately."""
import argparse,gzip,hashlib,json,re,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel,AutoTokenizer
from evidence import clean_text,vrdu_features,VRDU_ALIASES,EVIDENCE
from vrdu_evaluate import BASE,FIELDS
from holdout_extract import CLARIFIED
from holdout_evaluate import correctness
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/stage5';SCALARS=BASE+EVIDENCE

def nodes(ocr,rec,field):
    pred=rec['fields'][field]['pred'];pv=clean_text(pred)
    lines=[{'text':l['text'],'box':[l['bbox'][1],l['bbox'][2]+p['page_id'],l['bbox'][3],l['bbox'][4]+p['page_id']]} for p in ocr['pages'] for l in p['lines']]
    matches=[i for i,l in enumerate(lines) if pv and pv in clean_text(l['text'])]
    keys=[i for i,l in enumerate(lines) if re.search(VRDU_ALIASES[field],l['text'],re.I)]
    def dist(i,j):
        a,b=lines[i]['box'],lines[j]['box'];return 4*abs((a[1]+a[3]-b[1]-b[3])/2)+abs((a[0]+a[2]-b[0]-b[2])/2)
    if matches and keys:_,ki,vi=min((dist(k,v),k,v) for k in keys for v in matches)
    else:ki=keys[0] if keys else None;vi=matches[0] if matches else None
    def context(i):
        if i is None:return 'NO OCR EVIDENCE'
        return ' | '.join(l['text'] for l in lines[max(0,i-2):i+2])[:600]
    return [CLARIFIED[field],str(pred) if pred is not None else 'ABSENT',context(vi),context(ki)]

def main():
    p=argparse.ArgumentParser();p.add_argument('--encoder',required=True);args=p.parse_args();OUT.mkdir(exist_ok=True)
    source=[json.loads(l) for l in (ROOT/'results/stage2/vrdu-qwen35-predictions.jsonl').read_text().splitlines()]
    split=json.loads((ROOT/'data/vrdu/split.json').read_text());valid=sorted(split['valid'],key=lambda n:hashlib.sha256(n.encode()).hexdigest());dev=set(valid[:50]);cal=set(valid[50:])
    tasks=[(r,'development' if r['id'] in dev else 'calibration' if r['id'] in cal else r['split']) for r in source]
    for tag in ['qwen2b','qwen9b']:
        tasks.extend((json.loads(l),'stage4-'+tag) for l in (ROOT/f'results/stage4/{tag}-image-predictions.jsonl').read_text().splitlines() if json.loads(l)['schema']=='clarified')
    needed={r['id'] for r,_ in tasks};docs={}
    with gzip.open(ROOT/'data/vrdu/fara.jsonl.gz','rt') as f:
        for line in f:
            d=json.loads(line)
            if d['filename'] in needed:docs[d['filename']]=d
    texts=[];text_index={};indices=[];scalars=[];labels=[];normalized=[];parts=[];docids=[];fields=[]
    for rec,part in tasks:
        d=docs[rec['id']];gold=dict(d['annotations'])
        for field in FIELDS:
            ns=nodes(d['ocr'],rec,field);ix=[]
            for text in ns:
                if text not in text_index:text_index[text]=len(texts);texts.append(text)
                ix.append(text_index[text])
            indices.append(ix);vf=vrdu_features(d['ocr'],rec,field);scalars.append([vf[k] for k in SCALARS]+[int(field==f) for f in FIELDS])
            pred=rec['fields'][field]['pred'];pred=None if pred is None or not str(pred).strip() else str(pred);gs=gold.get(field,[])
            labels.append(int(correctness(pred,gs,field)));normalized.append(int(correctness(pred,gs,field,True)));parts.append(part);docids.append(rec['id']);fields.append(field)
    tokenizer=AutoTokenizer.from_pretrained(args.encoder,local_files_only=True);model=AutoModel.from_pretrained(args.encoder,local_files_only=True).cuda().eval();embs=[];truncated=0;start=time.perf_counter()
    for off in range(0,len(texts),128):
        chunk=texts[off:off+128];truncated+=sum(len(x)>256 for x in tokenizer(chunk,add_special_tokens=True)['input_ids'])
        inputs=tokenizer(chunk,padding=True,truncation=True,max_length=256,return_tensors='pt').to('cuda')
        with torch.inference_mode():
            hidden=model(**inputs).last_hidden_state;mask=inputs.attention_mask.unsqueeze(-1);pool=(hidden*mask).sum(1)/mask.sum(1).clamp(min=1);embs.append(torch.nn.functional.normalize(pool,p=2,dim=1).cpu().numpy())
        if off%1280==0:print('encoded',off,'/',len(texts),flush=True)
    z=np.concatenate(embs)[np.array(indices)];assert z.shape==(4800,4,384)
    np.savez_compressed(OUT/'features.npz',nodes=z,scalars=np.array(scalars,np.float32),y=np.array(labels,np.float32),normalized_y=np.array(normalized,np.float32),partition=np.array(parts),doc=np.array(docids),field=np.array(fields))
    (OUT/'feature-manifest.json').write_text(json.dumps({'encoder_snapshot':args.encoder,'encoder_source':'https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2','pooling':'attention-mask mean; L2 normalization','max_tokens':256,'rows':len(labels),'nodes_per_row':4,'unique_texts':len(texts),'truncated_unique_texts':truncated,'encode_seconds':time.perf_counter()-start,'scalars':SCALARS+['field:'+f for f in FIELDS],'features_sha256':hashlib.sha256((OUT/'features.npz').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2));print('saved',z.shape,flush=True)
if __name__=='__main__':main()
