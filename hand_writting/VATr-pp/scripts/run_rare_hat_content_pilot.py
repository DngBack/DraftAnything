"""Controlled 5k-step pilot: auxiliary HAT single-image reconstruction versus control."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/rare_hat_Ễ_pilot_v1'
BASE=ROOT/'runs/ablation_100k_v1/A_control'


def write(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(path)


def comparison():
    report=[]
    for name in ['auxiliary','control']:
        folder=OUT/name
        paths=[f for f in sorted((folder/'expanded').glob('benchmark_*.json')) if f.stem.split('_')[-1].isdigit()]
        for path in paths:
            d=json.loads(path.read_text());prompt=d['per_prompt']['Ễ']
            report.append({'branch':name,'step':d['step'],'overall_cer':d['guidance_ocr_cer'],'overall_exact':d['exact_transcript_rate'],'Ễ':prompt,'NGUYỄN':d['per_prompt']['NGUYỄN'],'warning':'Supervisor OCR proxy on genuine held-out writer supports; auxiliary self-reconstruction is evaluated separately.'})
    write(OUT/'comparison.json',report)


def main():
    import fcntl
    OUT.mkdir(parents=True,exist_ok=True)
    lock=os.open(OUT/'queue.lock',os.O_CREAT|os.O_RDWR,0o600);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'plan.json').exists():raise RuntimeError('Pilot already planned; inspect status instead of overwriting')
    origin=json.loads((ROOT/'runs/ablation_100k_v1/plan.json').read_text())['trials'][0]['command']
    trials=[]
    for name,weight in [('auxiliary','0.2'),('control','0')]:
        folder=OUT/name;folder.mkdir()
        shutil.copy2(BASE/'benchmark_ocr.pth',folder/'benchmark_ocr.pth')
        cmd=list(origin)
        for flag,value in {'--resume':str(BASE/'best.pth'),'--steps':'112500','--output':str(folder)}.items():cmd[cmd.index(flag)+1]=value
        cmd += ['--hat-content-weight',weight,'--hat-content-every','4','--hat-content-batch-size','2','--hat-content-chars','Ễ']
        trials.append({'branch':name,'command':cmd,'state':'pending'})
    with (BASE/'best.pth').open('rb') as handle:digest=hashlib.file_digest(handle,'sha256').hexdigest()
    plan={'start_step':107500,'target_step':112500,'source':str(BASE/'best.pth'),'source_sha256':digest,'trials':trials,'warning':'Experimental self-reference reconstruction; no HAT pseudo-writer targets. No automatic promotion into the 150k run.'}
    write(OUT/'plan.json',plan)
    for trial in trials:
        folder=OUT/trial['branch']
        with (folder/'train.log').open('ab') as log:
            proc=subprocess.Popen(trial['command'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            trial.update(state='running',pid=proc.pid,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());write(OUT/'status.json',plan);write(folder/'process.json',trial)
            code=proc.wait()
        trial.update(state='completed' if code==0 else 'failed',exit_code=code);write(OUT/'status.json',plan);comparison()
        if code:raise RuntimeError(f"{trial['branch']} failed; queue stopped")
    print('Pilot complete: see comparison.json. No automatic promotion.',flush=True)


if __name__=='__main__':main()
