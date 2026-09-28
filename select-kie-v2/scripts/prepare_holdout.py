"""Freeze new documents before looking at labels; download public PDFs."""
from pathlib import Path
import concurrent.futures
import gzip,hashlib,json,urllib.parse,urllib.request

ROOT=Path(__file__).resolve().parents[1]
data=ROOT/'data/vrdu'; out=ROOT/'results/stage3'; out.mkdir(exist_ok=True)
split=json.loads((data/'split.json').read_text());used=set(split['train']+split['valid']+split['test'])
available=[]
with gzip.open(data/'fara.jsonl.gz','rt') as f:
    for l in f:
        d=json.loads(l);n=d['filename']
        if n.endswith('_Short-Form.pdf') and n not in used:available.append(n)
names=sorted(available,key=lambda n:hashlib.sha256(n.encode()).hexdigest())[:100]
assert len(names)==100 and not(set(names)&used)
revision=json.loads((data/'revision.json').read_text())['sha']
manifest={'filenames':names,'excluded_stage2_count':len(used),'available_count':len(available),
          'repository_revision':revision,'selection':'first 100 by SHA256(filename), no gold filtering'}
(out/'holdout-manifest.json').write_text(json.dumps(manifest,indent=2))
pdfs=data/'pdfs';pdfs.mkdir(exist_ok=True)
def download(name):
    target=pdfs/(hashlib.sha256(name.encode()).hexdigest()+'.pdf')
    if not target.exists():
        url='https://raw.githubusercontent.com/google-research-datasets/vrdu/'+revision+'/registration-form/main/pdfs/'+urllib.parse.quote(name,safe='')
        blob=urllib.request.urlopen(url,timeout=45).read();assert blob[:4]==b'%PDF',name
        target.write_bytes(blob)
    return name,hashlib.sha256(target.read_bytes()).hexdigest()
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    manifest['pdf_sha256']=dict(pool.map(download,names))
(out/'holdout-manifest.json').write_text(json.dumps(manifest,indent=2))
print('Frozen and downloaded',len(names),'new documents',flush=True)
