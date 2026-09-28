"""Summarize actual human labels; missing labels stay explicitly pending."""
import csv,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import cohen_kappa_score
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/stage4'
LABELS={'correct','binding','miss','hallucination','partial-value','transcription','gold/schema-ambiguity','other'}
def read(path):
    rows=list(csv.DictReader(path.open()));assert len({r['audit_id'] for r in rows})==len(rows)
    for r in rows:assert not r['human_type'] or r['human_type'] in LABELS,(r['audit_id'],r['human_type'])
    return {r['audit_id']:r for r in rows}
def main():
    one=read(OUT/'audit-reviewer1.csv');two=read(OUT/'audit-reviewer2-overlap.csv');mapping=read(OUT/'audit-private-mapping.csv')
    assert set(one)==set(mapping) and set(two)<=set(one)
    result={'reviewer1_completed':sum(bool(r['human_type']) for r in one.values()),'reviewer1_total':len(one),'reviewer2_completed':sum(bool(r['human_type']) for r in two.values()),'human_audit_status':'pending'}
    random={i:r for i,r in one.items() if mapping[i]['stratum']=='random'};done=[r for r in random.values() if r['human_type']]
    result['random_counts_completed']=dict(Counter(r['human_type'] for r in done));result['random_completed']=len(done);result['random_total']=len(random)
    if len(done)==len(random):
        groups=defaultdict(list)
        for r in done:groups[r['doc']].append(int(r['human_type']=='binding'))
        values=list(groups.values());rng=np.random.default_rng(20260928);draws=[]
        for _ in range(2000):
            sampled=[values[i] for i in rng.integers(len(values),size=len(values))];draws.append(sum(map(sum,sampled))/sum(map(len,sampled)))
        result['9b_clarified_binding_prevalence']=float(np.mean([r['human_type']=='binding' for r in done]));result['binding_cluster_bootstrap_ci95']=np.percentile(draws,[2.5,97.5]).tolist();result['random_unique_documents']=len(values)
        if sum(bool(r['human_type']) for r in one.values())==len(one):result['human_audit_status']='reviewer1_complete; overlap/adjudication may remain'
    paired=[i for i in two if one[i]['human_type'] and two[i]['human_type']]
    result['overlap_completed']=len(paired)
    if paired:
        a=[one[i]['human_type'] for i in paired];b=[two[i]['human_type'] for i in paired];result['agreement']=sum(x==y for x,y in zip(a,b))/len(a)
        result['cohen_kappa']=float(cohen_kappa_score(a,b)) if len(set(a+b))>1 else None
    (OUT/'human-audit-summary.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
