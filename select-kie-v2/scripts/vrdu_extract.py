"""Frozen text extractor on genuine VRDU OCR, resumable, with no gold inputs."""
import argparse
import gzip
import hashlib
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, AutoConfig

ROOT = Path(__file__).resolve().parents[1]
FIELDS = {
    'registration_num': 'FARA registration number of the registrant',
    'registrant_name': 'name of the registrant',
    'foreign_principle_name': 'name of the foreign principal represented',
    'file_date': 'date of filing/signing this form, not the generic printed form revision date',
    'signer_name': 'name of the individual signing this form',
    'signer_title': 'title of the individual signing this form',
}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model', default='/home/bachdx2/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct')
    p.add_argument('--limit', type=int)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--output',default='vrdu-predictions.jsonl')
    args=p.parse_args()
    path=Path(args.model)
    if (path/'snapshots').exists(): path=next((path/'snapshots').iterdir())
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    tokenizer.padding_side='left'
    config=AutoConfig.from_pretrained(path,local_files_only=True)
    cls=AutoModelForImageTextToText if config.model_type=='qwen3_5' else AutoModelForCausalLM
    model=cls.from_pretrained(path,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
    split=json.loads((ROOT/'data/vrdu/split.json').read_text())
    names={f:s for s in ['train','valid','test'] for f in split[s]}
    assert len(names)==600, 'split overlaps'
    docs=[]
    with gzip.open(ROOT/'data/vrdu/fara.jsonl.gz','rt') as f:
        for line in f:
            d=json.loads(line)
            if d['filename'] in names:
                docs.append({'id':d['filename'],'split':names[d['filename']], 'text':d['ocr']['text']})
    assert len(docs)==600
    docs.sort(key=lambda d: (['train','valid','test'].index(d['split']), hashlib.sha256(d['id'].encode()).hexdigest()))
    if args.limit: docs=docs[:args.limit]
    out=ROOT/'results/stage2'/args.output; out.parent.mkdir(parents=True,exist_ok=True)
    done={json.loads(l)['id'] for l in out.read_text().splitlines()} if out.exists() else set()
    docs=[d for d in docs if d['id'] not in done]
    instruction='Extract the following fields from the document OCR. Return ONLY one JSON object. Copy each single value exactly as printed, preserving spelling and case. Use null if the field is not printed. Do not invent a value.\n'+'\n'.join(f'{f}: {desc}' for f,desc in FIELDS.items())
    print('Model',path,'remaining',len(docs),flush=True)
    for offset in range(0,len(docs),args.batch_size):
        batch=docs[offset:offset+args.batch_size]; prompts=[]; lengths=[]
        for d in batch:
            ids=tokenizer.encode(d['text'],add_special_tokens=False); lengths.append(len(ids))
            text=d['text'] if len(ids)<=7600 else tokenizer.decode(ids[:3800])+'\n[OCR middle omitted for context budget]\n'+tokenizer.decode(ids[-3800:])
            prompts.append(tokenizer.apply_chat_template([{'role':'system','content':instruction}, {'role':'user','content':'Document OCR:\n'+text}],tokenize=False,add_generation_prompt=True,enable_thinking=False))
        inputs=tokenizer(prompts,padding=True,return_tensors='pt').to('cuda')
        start=time.perf_counter()
        with torch.inference_mode():
            generated=model.generate(**inputs,max_new_tokens=320,do_sample=False,output_logits=True,return_dict_in_generate=True,
                                     pad_token_id=tokenizer.pad_token_id)
            ids=generated.sequences[:,inputs.input_ids.shape[1]:]
            lps=torch.stack([l.float().log_softmax(-1).gather(-1,ids[:,i:i+1])[:,0] for i,l in enumerate(generated.logits)],1).cpu().tolist()
        elapsed=time.perf_counter()-start
        with out.open('a') as f:
            for d, tokens, probs, input_n in zip(batch,ids.cpu().tolist(),lps,lengths):
                n=next((i for i,t in enumerate(tokens) if t in [tokenizer.eos_token_id,tokenizer.pad_token_id]),len(tokens))
                tokens=tokens[:n]; probs=probs[:n]; raw=tokenizer.decode(tokens,skip_special_tokens=True)
                ends=[len(tokenizer.decode(tokens[:i+1],skip_special_tokens=True)) for i in range(len(tokens))]; starts=[0]+ends[:-1]
                fields={}
                for name in FIELDS:
                    m=re.search(r'"'+name+r'"\s*:\s*(null|"((?:[^"\\]|\\.)*)"|-?[\d.]+)',raw)
                    pred=None; pl=[]
                    if m:
                        group=2 if m.group(2) is not None else 1
                        value=m.group(group)
                        if m.group(1)!='null':
                            try: pred=json.loads('"'+value+'"') if group==2 else value
                            except json.JSONDecodeError: pred=value
                        a,b=m.span(group)
                        pl=[v for s,e,v in zip(starts,ends,probs) if s<b and e>a]
                    fields[name]={'pred':pred,'lps':pl,'missing_key':m is None}
                record={'id':d['id'],'split':d['split'],'raw':raw,'fields':fields,'input_tokens':input_n,
                        'truncated_input':input_n>7600,'output_capped':n==320,'batch_seconds':elapsed,
                        'batch_size':len(batch),'ocr_sha256':hashlib.sha256(d['text'].encode()).hexdigest()}
                f.write(json.dumps(record,ensure_ascii=False)+'\n')
        del generated,inputs,ids
        print(f'{offset+len(batch)}/{len(docs)} batch={elapsed:.2f}s',flush=True)
    manifest={'model_snapshot':str(path),'torch':torch.__version__,'max_new_tokens':320,'ocr_budget':7600,
              'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'data/vrdu/fara.jsonl.gz',ROOT/'data/vrdu/split.json']},
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out.parent/(out.stem+'-manifest.json')).write_text(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
