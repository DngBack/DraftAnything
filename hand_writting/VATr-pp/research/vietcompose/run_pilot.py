"""Launch matched warm-start representation pilots sequentially; not zero-shot."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/vietcompose_pilot_v1'


def write(path,obj):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));temp.replace(path)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=os.open(OUT/'queue.lock',os.O_CREAT|os.O_RDWR,0o600);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    parser=argparse.ArgumentParser();parser.add_argument('--resume-queue',action='store_true');args=parser.parse_args()
    if (OUT/'plan.json').exists():
        if not args.resume_queue:raise RuntimeError('Existing pilot; pass --resume-queue after inspecting status')
        plan=json.loads((OUT/'status.json').read_text());trials=plan['trials']
    else:
        trials=[]
        for mode in ['factorized','glyph']:
            folder=OUT/mode
            if folder.exists():raise RuntimeError(f'Output exists: {folder}')
            folder.mkdir()
            cmd=[sys.executable,'-u','-m','research.vietcompose.train_pilot','--source',str(ROOT/'runs/continuation_200k_v1/best.pth'),'--mode',mode,'--output',str(folder),'--steps','1000','--batch-size','8','--eval-every','250','--lr','0.0001']
            trials.append({'mode':mode,'state':'pending','command':cmd})
        plan={'architecture':'vietcompose-query-v1','source_step':175500,'adapter_updates_per_trial':1000,'trials':trials,'warning':'Previously exposed warm-start adaptation; not a clean compositional zero-shot evaluation; no automatic promotion.'}
        write(OUT/'plan.json',plan)
    for trial in trials:
        if trial['state']=='completed':continue
        if trial['state']!='pending':raise RuntimeError('Inspect running/failed trial before retry')
        folder=OUT/trial['mode'];print(datetime.datetime.now(datetime.timezone.utc).isoformat(),'START',trial['mode'],flush=True)
        with (folder/'train.log').open('ab') as log:
            p=subprocess.Popen(trial['command'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            trial.update(state='running',pid=p.pid);write(OUT/'status.json',plan)
            code=p.wait()
        trial.update(state='completed' if code==0 else 'failed',exit_code=code);write(OUT/'status.json',plan)
        summaries=[]
        for t in trials:
            d=OUT/t['mode'];files=[f for f in sorted((d/'expanded').glob('benchmark_*.json')) if f.stem.split('_')[-1].isdigit()]
            if not files:continue
            initial=json.loads(files[0].read_text());last=json.loads(files[-1].read_text());step=last['step']
            v=d/'visual'/f'visual_{step:06d}.json';c=d/f'components_{step:06d}.json'
            summaries.append({'mode':t['mode'],'adapter_step':step,'initial_cer':initial['guidance_ocr_cer'],'last_cer':last['guidance_ocr_cer'],'last_exact':last['exact_transcript_rate'],'prompts':{k:last['per_prompt'][k] for k in ['NGUYỄN','Ễ','Ắ','Ỡ']},'visual':{k:val for k,val in json.loads(v.read_text()).items() if k!='predictions'} if v.exists() else None,'components':json.loads(c.read_text()) if c.exists() else None})
        write(OUT/'comparison.json',summaries)
        print(datetime.datetime.now(datetime.timezone.utc).isoformat(),'END',trial['mode'],code,flush=True)
        if code:raise RuntimeError('Trial failed; queue stopped')


if __name__=='__main__':main()
