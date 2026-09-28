"""Equal-information neural comparison; stage4 evaluation is exploratory."""
import copy,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss
import joblib
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/stage5';sys.path.insert(0,str(ROOT.parent/'selective-kie'))
from common import aurc,threshold_at_risk,apply_threshold
from vrdu_evaluate import BASE,x
from evidence import EVIDENCE
PAIRS=[(i,j) for i in range(4) for j in range(i+1,4)]
class Generic(nn.Module):
    def __init__(self,d,width):
        super().__init__();self.net=nn.Sequential(nn.Linear(1536+d,width),nn.ReLU(),nn.Dropout(.1),nn.Linear(width,width),nn.ReLU(),nn.Dropout(.1),nn.Linear(width,1))
    def forward(self,z,s):return self.net(torch.cat([z.flatten(1),s],1)).squeeze(1)
class Relation(nn.Module):
    def __init__(self,d):
        super().__init__();self.project=nn.Linear(384,32);self.net=nn.Sequential(nn.Linear(128+192+6+d,64),nn.ReLU(),nn.Dropout(.1),nn.Linear(64,32),nn.ReLU(),nn.Dropout(.1),nn.Linear(32,1))
    def forward(self,z,s):
        h=torch.nn.functional.normalize(torch.tanh(self.project(z)),p=2,dim=-1)
        cos=torch.stack([(h[:,i]*h[:,j]).sum(1) for i,j in PAIRS],1)
        interactions=torch.cat([v for j in [1,2,3] for v in [(h[:,0]-h[:,j]).abs(),h[:,0]*h[:,j]]],1)
        return self.net(torch.cat([h.flatten(1),interactions,cos,s],1)).squeeze(1)
def count(m):return sum(p.numel() for p in m.parameters())
def metrics(score,err,threshold,normalized):
    accepted=score>=threshold;c,r=apply_threshold(score,err,threshold)
    return {'aurc':aurc(score,err),'normalized_aurc':aurc(score,normalized),'brier':float(brier_score_loss(1-err,score)),'coverage':float(c),'risk':float(r) if np.isfinite(r) else None,'accepted':int(accepted.sum()),'errors':int(err[accepted].sum()),'normalized_errors':int(normalized[accepted].sum()),'threshold':float(threshold) if np.isfinite(threshold) else None}
def main():
    torch.set_num_threads(4);data=np.load(OUT/'features.npz');part=data['partition'];train=part=='train';dev=part=='development';cal=part=='calibration';assert (train.sum(),dev.sum(),cal.sum())==(1200,300,300)
    z=data['nodes'].astype(np.float32);raw=data['scalars'];rawcos=np.stack([(z[:,i]*z[:,j]).sum(1) for i,j in PAIRS],1);raw=np.concatenate([raw,rawcos],1)
    scaler=StandardScaler().fit(raw[train]);s=np.clip(scaler.transform(raw),-10,10).astype(np.float32);joblib.dump(scaler,OUT/'scalar-scaler.joblib')
    zz=torch.from_numpy(z);ss=torch.from_numpy(s);yy=torch.from_numpy(data['y']);models={};traces=[];params=count(Relation(s.shape[1]));width=min(range(8,65),key=lambda w:abs(count(Generic(s.shape[1],w))-params));generic_params=count(Generic(s.shape[1],width));assert abs(generic_params-params)/params<.03
    train_idx=np.flatnonzero(train);dev_idx=np.flatnonzero(dev);scores={}
    for name in ['generic','relation']:
        seed_scores=[]
        for seed in range(20260928,20260933):
            torch.manual_seed(seed);rng=np.random.default_rng(seed);model=Generic(s.shape[1],width) if name=='generic' else Relation(s.shape[1]);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001);best=float('inf');patience=0;trace=[];best_epoch=0
            for epoch in range(100):
                model.train();perm=rng.permutation(train_idx)
                for off in range(0,len(perm),128):
                    ix=perm[off:off+128];optimizer.zero_grad();loss=nn.functional.binary_cross_entropy_with_logits(model(zz[ix],ss[ix]),yy[ix]);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
                model.eval()
                with torch.inference_mode():loss=float(nn.functional.binary_cross_entropy_with_logits(model(zz[dev_idx],ss[dev_idx]),yy[dev_idx]))
                trace.append(loss)
                if loss<best-1e-5:best=loss;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch+1;patience=0
                else:patience+=1
                if patience>=15:break
            model.load_state_dict(best_state);model.eval()
            with torch.inference_mode():score=model(zz,ss).sigmoid().numpy()
            seed_scores.append(score);torch.save({'state_dict':best_state,'architecture':name,'generic_width':width,'scalar_dim':s.shape[1],'seed':seed,'best_epoch':best_epoch},OUT/f'{name}-{seed}.pt')
            traces.append({'architecture':name,'seed':seed,'epochs':len(trace),'best_epoch':best_epoch,'development_bce':best,'loss_trace':trace});print(name,seed,'epoch',best_epoch,'dev BCE',best,flush=True)
        scores[name]=np.stack(seed_scores)
    err=1-data['y'];normalized=1-data['normalized_y'];measures={};ensembles={name:sc.mean(0) for name,sc in scores.items()}
    for name,sc in ensembles.items():
        th=threshold_at_risk(sc[cal],err[cal],.05);ms={}
        for target in ['test','stage4-qwen2b','stage4-qwen9b']:
            ix=part==target;ms[target]=metrics(sc[ix],err[ix],th,normalized[ix]);ms[target]['seed_aurcs']=[aurc(v[ix],err[ix]) for v in scores[name]]
        measures[name]=ms
    # Frozen scalar-only references are not equal-information architecture baselines.
    reference=np.load(OUT/'hgb-references.npz')
    for name in ['HGB-fusion','HGB-evidence']:
        sc=reference[name];th=threshold_at_risk(sc[cal],err[cal],.05);ensembles[name]=sc;measures[name]={t:metrics(sc[part==t],err[part==t],th,normalized[part==t]) for t in ['test','stage4-qwen2b','stage4-qwen9b']}
    contrasts={};rng=np.random.default_rng(20260928)
    for target in ['test','stage4-qwen2b','stage4-qwen9b']:
        inds=np.flatnonzero(part==target);docs=data['doc'][inds];groups=[inds[docs==n] for n in sorted(set(docs))];diff=[]
        for _ in range(2000):
            ix=np.concatenate([groups[j] for j in rng.integers(len(groups),size=len(groups))]);diff.append(aurc(ensembles['relation'][ix],err[ix])-aurc(ensembles['generic'][ix],err[ix]))
        contrasts[target]={'relation_minus_generic_aurc':measures['relation'][target]['aurc']-measures['generic'][target]['aurc'],'ci95':np.percentile(diff,[2.5,97.5]).tolist()}
    result={'status':'exploratory; stage4 already viewed before method design','parameter_counts':{'relation':params,'generic':generic_params},'generic_width':width,'seeds':list(range(20260928,20260933)),'measures':measures,'contrasts':contrasts,'threshold_calibration':'stage2 target-template50; empirical alpha5%; no risk guarantee','feature_sha256':hashlib.sha256((OUT/'features.npz').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'metrics.json').write_text(json.dumps(result,indent=2));(OUT/'training-traces.json').write_text(json.dumps(traces,indent=2))
    np.savez_compressed(OUT/'scores.npz',**scores,**{'ensemble_'+k:v for k,v in ensembles.items()})
    with (OUT/'scores.csv').open('w') as f:
        columns=['partition','doc','field','correct','normalized_correct']+list(ensembles);w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
        for i in range(len(part)):w.writerow({'partition':part[i],'doc':data['doc'][i],'field':data['field'][i],'correct':data['y'][i],'normalized_correct':data['normalized_y'][i],**{k:v[i] for k,v in ensembles.items()}})
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
