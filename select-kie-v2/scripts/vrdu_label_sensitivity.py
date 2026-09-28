"""Post-hoc diagnostic only: fixed scores, alternate operational normalization."""
import csv,gzip,json,hashlib,re,sys
from pathlib import Path
import numpy as np
import joblib

from vrdu_evaluate import CONFIGS,x,MATCHERS,FIELDS
from evidence import clean_text,VRDU_ALIASES,support
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'selective-kie'))
from common import aurc,threshold_at_risk,apply_threshold
from holdout_evaluate import correctness

# Analyze exported test decisions only, never refit scores or select a threshold from test.
ROOT=Path(__file__).resolve().parents[1]
def main():
    results={}
    for stem in ['vrdu-predictions','vrdu-qwen35-predictions']:
        metrics=json.loads((ROOT/f'results/stage2/{stem}-metrics.json').read_text())
        rows=list(csv.DictReader((ROOT/f'results/stage2/{stem}-test.csv').open()))
        docs={r['doc'] for r in rows};lookup={}
        with gzip.open(ROOT/'data/vrdu/fara.jsonl.gz','rt') as f:
            for l in f:
                d=json.loads(l)
                if d['filename'] in docs:lookup[d['filename']]=dict(d['annotations'])
        official=np.array([int(r['err']) for r in rows]); normalized=[];examples=[]
        for r in rows:
            pred=r['pred'] or None;gs=lookup[r['doc']].get(r['field'],[])
            e=int(not correctness(pred,gs,r['field'],True));normalized.append(e)
            if int(r['err'])!=e:examples.append({k:r[k] for k in ['doc','field','pred','gold']})
        normalized=np.array(normalized);summary={}
        for name,m in metrics['measures'].items():
            scores=np.array([float(r[name]) for r in rows]);th=m['targets']['0.05']['threshold'];accepted=scores>=th if th is not None else np.zeros(len(rows),bool)
            summary[name]={'official_aurc':aurc(scores,official),'normalized_aurc':aurc(scores,normalized),
                           'accepted_at_original_threshold':int(accepted.sum()),
                           'official_accepted_errors':int(official[accepted].sum()),'normalized_accepted_errors':int(normalized[accepted].sum())}
        results[stem]={'post_hoc':True,'rule':'case/punctuation-insensitive alphanumeric names/title; remove Date of signature label from gold date; fixed original scores and thresholds',
                       'official_error_rate':float(official.mean()),'normalized_error_rate':float(normalized.mean()),
                       'changed_decisions':len(examples),'changed_examples_first20':examples[:20],'scores':summary}
    (ROOT/'results/stage2/label-sensitivity.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
