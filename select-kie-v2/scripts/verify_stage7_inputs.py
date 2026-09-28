"""Check document/text separation without inspecting annotations."""
import gzip,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
split=json.loads((ROOT/'data/vrdu/split.json').read_text())
old=set(split['train']+split['valid']+split['test'])|set(json.loads((ROOT/'results/stage3/holdout-manifest.json').read_text())['filenames'])
old.update(json.loads((ROOT/'results/stage4/holdout-manifest.json').read_text())['filenames'])
old.update(json.loads((ROOT/'results/stage6/holdout-manifest.json').read_text())['filenames'])
m=json.loads((ROOT/'results/stage7/holdout-manifest.json').read_text());new=set(m['filenames']);assert len(new)==89 and not old&new
hashes={}
with gzip.open(ROOT/'data/vrdu/fara.jsonl.gz','rt') as f:
    for l in f:
        d=json.loads(l)
        if d['filename'] in old|new:hashes[d['filename']]=hashlib.sha256(d['ocr']['text'].encode()).hexdigest()
oldhash={hashes[n] for n in old};newhash=[hashes[n] for n in new]
result={'new_docs':89,'old_docs':len(old),'filename_overlap':0,'ocr_exact_cross_stage_overlap':len(set(newhash)&oldhash),'ocr_exact_within_stage_duplicates':89-len(set(newhash))}
for name in new:
    p=ROOT/'data/vrdu/pdfs'/(hashlib.sha256(name.encode()).hexdigest()+'.pdf');assert hashlib.sha256(p.read_bytes()).hexdigest()==m['pdf_sha256'][name]
assert result['ocr_exact_cross_stage_overlap']==0 and result['ocr_exact_within_stage_duplicates']==0
(ROOT/'results/stage7/input-verification.json').write_text(json.dumps(result,indent=2));print(result)
