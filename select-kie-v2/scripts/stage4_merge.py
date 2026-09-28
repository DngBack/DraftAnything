"""Merge disjoint resumed9B shards, requiring the complete frozen task set."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/stage4'
def main():
    dest=OUT/'qwen9b-image-predictions.jsonl';seed=OUT/'qwen9b-image-seed.jsonl'
    if not seed.exists():seed.write_bytes(dest.read_bytes())
    paths=[seed]+sorted(OUT.glob('qwen9b-image-shard*-predictions.jsonl'));rows=[]
    for p in paths:rows.extend(json.loads(l) for l in p.read_text().splitlines())
    keys=[(r['id'],r['schema']) for r in rows];assert len(keys)==len(set(keys))==300,(len(keys),len(set(keys)))
    names=json.loads((OUT/'holdout-manifest.json').read_text())['filenames'];assert set(keys)=={(n,s) for n in names for s in ['baseline','length_control','clarified']}
    dest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    (OUT/'merge-manifest.json').write_text(json.dumps({'records':len(rows),'parts':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'merged_sha256':hashlib.sha256(dest.read_bytes()).hexdigest()},indent=2));print('Merged300 unique tasks')
if __name__=='__main__':main()
