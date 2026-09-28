"""Read frozen HGB estimators in the matching CPU sklearn environment."""
import hashlib,json
from pathlib import Path
import joblib,numpy as np,sklearn
from vrdu_evaluate import x
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/stage5'
def main():
    assert sklearn.__version__=='1.9.1','Use the original CPU environment for these archived estimators'
    data=np.load(OUT/'features.npz');features=json.loads((OUT/'feature-manifest.json').read_text())['scalars'];rows=[dict(zip(features,v)) for v in data['scalars']]
    for r,f in zip(rows,data['field']):r['field']=str(f)
    scores={}
    for name in ['HGB-fusion','HGB-evidence']:
        obj=joblib.load(ROOT/f'results/stage2/vrdu-qwen35-predictions.{name}.joblib');scores[name]=obj['head'].predict_proba(x(rows,obj['features']))[:,1]
    np.savez_compressed(OUT/'hgb-references.npz',**scores)
    (OUT/'hgb-reference-manifest.json').write_text(json.dumps({'sklearn':sklearn.__version__,'scores_sha256':hashlib.sha256((OUT/'hgb-references.npz').read_bytes()).hexdigest()},indent=2));print('Frozen reference scores generated in sklearn',sklearn.__version__)
if __name__=='__main__':main()
