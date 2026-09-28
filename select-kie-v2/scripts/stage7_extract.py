"""Frozen 2x2 image/OCR and schema experiment on new VRDU documents."""
import argparse,gzip,hashlib,json,re,time
from pathlib import Path
import fitz
from PIL import Image
import torch
from transformers import AutoModelForImageTextToText,AutoProcessor,AutoTokenizer
from vrdu_extract import FIELDS

ROOT=Path(__file__).resolve().parents[1]
CLARIFIED={
 'registration_num':'numeric registration identifier written beside Registration No./Registration Number. Copy the digits. Not FARA, an OMB number, phone number, or date',
 'registrant_name':'name of the FARA-registered organization/entity. On a Short Form use the registrant FOR WHOM THE INDIVIDUAL IS ACTING, not the individual named at the top',
 'foreign_principle_name':'name of the foreign principal represented by that registrant, not the individual or registrant organization',
 'file_date':'date of signature in the EXECUTION/signature section. Do not use the received-by-registration-unit stamp or form revision/expiry date',
 'signer_name':'printed name of the person signing the EXECUTION/signature section. Copy a printed name if shown there; do not guess from an illegible signature',
 'signer_title':'explicitly printed title of the person signing the EXECUTION/signature section. Return null if no title is explicitly printed there',
}

def parse(tokenizer,tokens,probs):
    raw=tokenizer.decode(tokens,skip_special_tokens=True)
    ends=[len(tokenizer.decode(tokens[:i+1],skip_special_tokens=True)) for i in range(len(tokens))];starts=[0]+ends[:-1]
    result={}
    for name in FIELDS:
        m=re.search(r'"'+name+r'"\s*:\s*(null|"((?:[^"\\]|\\.)*)"|-?[\d.]+)',raw);pred=None;lp=[]
        if m:
            g=2 if m.group(2) is not None else 1;value=m.group(g)
            if m.group(1)!='null':
                try:pred=json.loads('"'+value+'"') if g==2 else value
                except json.JSONDecodeError:pred=value
            a,b=m.span(g);lp=[v for s,e,v in zip(starts,ends,probs) if s<b and e>a]
        result[name]={'pred':pred,'lps':lp,'missing_key':m is None}
    return raw,result

def main():
    p=argparse.ArgumentParser();p.add_argument('--modality',choices=['image','ocr'],required=True);p.add_argument('--batch-size',type=int,default=4);p.add_argument('--limit',type=int);p.add_argument('--model',required=True);p.add_argument('--tag',required=True);p.add_argument('--shard',type=int,choices=[0,1],required=True);args=p.parse_args()
    out=ROOT/'results/stage7';manifest=json.loads((out/'holdout-manifest.json').read_text());names=manifest['filenames']
    lookup={}
    with gzip.open(ROOT/'data/vrdu/fara.jsonl.gz','rt') as f:
        for l in f:
            d=json.loads(l)
            if d['filename'] in names:lookup[d['filename']]=d['ocr']['text']
    model_path=Path(args.model)
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True);tokenizer.padding_side='left'
    processor=AutoProcessor.from_pretrained(model_path,local_files_only=True) if args.modality=='image' else None
    if processor:processor.tokenizer.padding_side='left'
    model=AutoModelForImageTextToText.from_pretrained(model_path,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
    seed_path=out/f'{args.tag}-{args.modality}-predictions.jsonl'
    path=out/f'{args.tag}-{args.modality}-shard{args.shard}-predictions.jsonl'
    done={(r['id'],r['schema']) for r in [json.loads(l) for l in path.read_text().splitlines()]} if path.exists() else set()
    if seed_path.exists():done.update((r['id'],r['schema']) for r in [json.loads(l) for l in seed_path.read_text().splitlines()])
    selected=names[:args.limit] if args.limit else names
    selected=[n for n in selected if int(hashlib.sha256(n.encode()).hexdigest(),16)%2==args.shard]
    common='Extract the following fields from this document. Return ONLY one JSON object. Copy each single value exactly as printed, preserving spelling and case. Use null if the field is not printed. Do not invent a value.\n'
    instructions={k:common+'\n'.join(f'{f}: {v}' for f,v in fs.items()) for k,fs in [('baseline',FIELDS),('clarified',CLARIFIED)]}
    target=len(tokenizer.encode(instructions['clarified'],add_special_tokens=False))
    control=instructions['baseline']+'\nGeneral output guidance:'
    while len(tokenizer.encode(control,add_special_tokens=False))<target:
        control+=' Copy faithfully.'
    ids=tokenizer.encode(control,add_special_tokens=False)[:target]
    control=tokenizer.decode(ids)
    assert len(tokenizer.encode(control,add_special_tokens=False))==target
    instructions['length_control']=control
    (out/f'{args.tag}-prompts.json').write_text(json.dumps({'instructions':instructions,'token_counts':{k:len(tokenizer.encode(v,add_special_tokens=False)) for k,v in instructions.items()}},indent=2))
    tasks=[(n,s) for s in ['clarified'] for n in selected if (n,s) not in done]
    print(args.modality,'remaining',len(tasks),flush=True)
    for offset in range(0,len(tasks),args.batch_size):
        batch=tasks[offset:offset+args.batch_size];prompts=[];images=[];pages=[];truncated=[]
        for name,schema in batch:
            instruction=instructions[schema]
            if args.modality=='image':
                pdf=ROOT/'data/vrdu/pdfs'/(hashlib.sha256(name.encode()).hexdigest()+'.pdf');doc=fitz.open(pdf);ims=[]
                for page in doc:
                    scale=1280/max(page.rect.width,page.rect.height);pix=page.get_pixmap(matrix=fitz.Matrix(scale,scale),alpha=False)
                    ims.append(Image.frombytes('RGB',[pix.width,pix.height],pix.samples))
                doc.close();pages.append(len(ims));images.extend(ims);truncated.append(False)
                content=[{'type':'image','image':im} for im in ims]+[{'type':'text','text':'Document pages.'}]
                prompts.append(processor.apply_chat_template([{'role':'system','content':instruction},{'role':'user','content':content}],tokenize=False,add_generation_prompt=True,enable_thinking=False))
            else:
                ids=tokenizer.encode(lookup[name],add_special_tokens=False);cut=len(ids)>7600;truncated.append(cut);pages.append(None)
                text=lookup[name] if not cut else tokenizer.decode(ids[:3800])+'\n[OCR middle omitted]\n'+tokenizer.decode(ids[-3800:])
                prompts.append(tokenizer.apply_chat_template([{'role':'system','content':instruction},{'role':'user','content':'Document OCR:\n'+text}],tokenize=False,add_generation_prompt=True,enable_thinking=False))
        inputs=(processor(text=prompts,images=images,padding=True,return_tensors='pt') if processor else tokenizer(prompts,padding=True,return_tensors='pt')).to('cuda')
        started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            result=model.generate(**inputs,max_new_tokens=320,do_sample=False,output_logits=True,return_dict_in_generate=True,pad_token_id=tokenizer.pad_token_id)
            generated=result.sequences[:,inputs.input_ids.shape[1]:]
            probs=torch.stack([l.float().log_softmax(-1).gather(-1,generated[:,i:i+1])[:,0] for i,l in enumerate(result.logits)],1).cpu().tolist()
        elapsed=time.perf_counter()-started;memory=torch.cuda.max_memory_allocated()/2**30
        with path.open('a') as f:
            for (name,schema),tokens,lps,n_pages,cut in zip(batch,generated.cpu().tolist(),probs,pages,truncated):
                n=next((i for i,t in enumerate(tokens) if t in [tokenizer.eos_token_id,tokenizer.pad_token_id]),len(tokens))
                raw,fields=parse(tokenizer,tokens[:n],lps[:n]);r={'id':name,'schema':schema,'model_tag':args.tag,'modality':args.modality,'raw':raw,'fields':fields,
                    'pages':n_pages,'truncated_input':cut,'output_capped':n==320,'batch_seconds':elapsed,'batch_size':len(batch),'peak_memory_gib':memory}
                f.write(json.dumps(r,ensure_ascii=False)+'\n')
        del result,inputs,generated
        print(f'{offset+len(batch)}/{len(tasks)} batch={elapsed:.2f}s mem={memory:.2f}GiB',flush=True)
    (out/f'{args.tag}-{args.modality}-shard{args.shard}-manifest.json').write_text(json.dumps({'shard':args.shard,'model':str(model_path),'torch':torch.__version__,
       'schemas':{'baseline':FIELDS,'clarified':CLARIFIED},'image_max_side':1280,'max_new_tokens':320,
       'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))

if __name__=='__main__':main()
