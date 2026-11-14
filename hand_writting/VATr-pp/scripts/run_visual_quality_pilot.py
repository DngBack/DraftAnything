"""Three isolated 150k->155k trials; no automatic promotion by OCR alone."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/continuation_150k_v1/training'
OUT=ROOT/'runs/visual_quality_pilot_v1'


def write(path,obj):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));temp.replace(path)


def summarize():
    summaries=[]
    for name in ['feature_only','feature_pixel','control']:
        folder=OUT/name
        files=sorted((folder/'visual').glob('visual_*.json'))
        if not files:continue
        initial=json.loads(files[0].read_text());last=json.loads(files[-1].read_text())
        entry={'branch':name,'step':last['step'],'initial_feature_distance':initial['mean_feature_distance'],'last_feature_distance':last['mean_feature_distance'],'last_pixel_l1':last['mean_pixel_l1'],'last_paired_cer':last['guidance_ocr_cer'],'samples':last['samples'],'warning':'Fixed source-D proxy, no human aesthetic calibration; compare paired panels and expanded CER before promoting.'}
        expanded=folder/'expanded'/f"benchmark_{last['step']:06d}.json"
        if expanded.exists():
            d=json.loads(expanded.read_text());entry.update(expanded_cer=d['guidance_ocr_cer'],expanded_exact=d['exact_transcript_rate'],prompts={k:d['per_prompt'][k] for k in ['NGUYỄN','Ễ']})
        summaries.append(entry)
    write(OUT/'comparison.json',summaries)


def main():
    import fcntl
    OUT.mkdir(parents=True,exist_ok=True)
    lock=os.open(OUT/'queue.lock',os.O_CREAT|os.O_RDWR,0o600);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'plan.json').exists():raise RuntimeError('Existing pilot; inspect rather than overwrite')
    original=json.loads((ROOT/'runs/continuation_150k_v1/status.json').read_text())
    if original['state']!='completed':raise RuntimeError('Source 150k run has not completed')
    trials=[]
    for name,feature,pixel in [('feature_only','1','0'),('feature_pixel','1','0.1'),('control','0','0')]:
        folder=OUT/name;folder.mkdir()
        shutil.copy2(BASE/'benchmark_ocr.pth',folder/'benchmark_ocr.pth')
        cmd=list(original['command'])
        for flag,value in {'--resume':str(BASE/'last.pth'),'--steps':'155000','--output':str(folder)}.items():cmd[cmd.index(flag)+1]=value
        cmd += ['--visual-benchmark','--paired-feature-weight',feature,'--paired-pixel-weight',pixel,'--paired-every','4']
        trials.append({'branch':name,'state':'pending','command':cmd})
    with (BASE/'last.pth').open('rb') as handle:digest=hashlib.file_digest(handle,'sha256').hexdigest()
    plan={'source':str(BASE/'last.pth'),'source_sha256':digest,'start_step':150000,'target_step':155000,'trials':trials,'promotion':'No automatic promotion; fixed-feature distances are proxies and sample review is required.'}
    write(OUT/'plan.json',plan)
    for trial in trials:
        folder=OUT/trial['branch']
        with (folder/'train.log').open('ab') as log:
            p=subprocess.Popen(trial['command'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            trial.update(state='running',pid=p.pid,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());write(OUT/'status.json',plan);write(folder/'process.json',trial)
            code=p.wait()
        trial.update(state='completed' if code==0 else 'failed',exit_code=code);write(OUT/'status.json',plan);summarize()
        if code:raise RuntimeError(f"{trial['branch']} failed; queue stopped")
    print('Completed. Review visual panels and comparison.json; no automatic promotion.',flush=True)


if __name__=='__main__':main()
