"""Freeze before confirmation, or verify an existing freeze without overwriting it."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/stage6';PREV=ROOT/'results/stage5'
def main():
    path=OUT/'frozen-head-manifest.json';OUT.mkdir(exist_ok=True)
    if path.exists():
        m=json.loads(path.read_text())
        for file,sha in m['heads_and_pipeline_sha256'].items():assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==sha,file
        print('Existing frozen heads verified; manifest preserved');return
    assert not list(OUT.glob('*predictions.jsonl')),'Freeze before confirmation predictions'
    m=json.loads((PREV/'metrics.json').read_text());files=list(PREV.glob('generic-*.pt'))+list(PREV.glob('relation-*.pt'))+[PREV/'scalar-scaler.joblib',PREV/'metrics.json',PREV/'feature-manifest.json',ROOT/'scripts/relation_features.py',ROOT/'scripts/relation_train.py'];assert len(list(PREV.glob('*.pt')))==10
    hashes={str(p.relative_to(ROOT.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    path.write_text(json.dumps({'heads_and_pipeline_sha256':hashes,'thresholds':{k:v['stage4-qwen9b']['threshold'] for k,v in m['measures'].items()},'status':'locked before stage6 inference'},indent=2));print('Frozen10 heads and original calibration thresholds')
if __name__=='__main__':main()
