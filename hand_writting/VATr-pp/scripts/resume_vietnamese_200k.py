"""Resume the completed 150k baseline once its original data/runtime are accessible."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'runs/continuation_150k_v1/training'
OUT=ROOT/'runs/continuation_200k_v1'


def write(path,obj):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));temp.replace(path)


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=os.open(OUT/'launcher.lock',os.O_CREAT|os.O_RDWR,0o600);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'process.json').exists() or (OUT/'last.pth').exists():
        raise RuntimeError('200k run already launched; inspect existing process/checkpoint before resuming')
    prior=json.loads((ROOT/'runs/continuation_150k_v1/status.json').read_text())
    if prior['state']!='completed':raise RuntimeError('150k source run is not complete')
    manifest=json.loads((ROOT/'files/vietnamese/manifest.json').read_text())
    command=list(prior['command'])
    for flag,value in {'--resume':str(SOURCE/'last.pth'),'--steps':'200000','--output':str(OUT)}.items():command[command.index(flag)+1]=value
    requirements=[Path(command[0]),Path(manifest['source_train']),Path(manifest['source_test'])]
    write(OUT/'plan.json',{'source_checkpoint':str(SOURCE/'last.pth'),'source_step':150000,'target_step':200000,'command':command,'configuration':'Baseline; HAT reconstruction and paired feature/pixel auxiliary losses disabled.'})
    while True:
        inaccessible=[]
        for path in requirements:
            try:
                with path.open('rb') as handle:handle.read(1)
                if path==requirements[0] and not os.access(path,os.X_OK):raise PermissionError('Not executable')
            except OSError as e:inaccessible.append({'path':str(path),'error':str(e)})
        if not inaccessible:break
        write(OUT/'status.json',{'state':'waiting_for_data_and_runtime_access','checked_utc':utc(),'launcher_pid':os.getpid(),'target_step':200000,'inaccessible':inaccessible})
        time.sleep(30)
    # Validate actual source step using the original training runtime before launch.
    check=subprocess.run([command[0],'-c',"import torch,sys; s=torch.load(sys.argv[1],map_location='cpu',weights_only=False); assert s['step']==150000 and s['stage']=='gan'; print(s['step'])",str(SOURCE/'last.pth')],cwd=ROOT,capture_output=True,text=True)
    if check.returncode:
        write(OUT/'status.json',{'state':'failed_source_validation','stderr':check.stderr[-4000:],'checked_utc':utc()});return
    shutil.copy2(SOURCE/'benchmark_ocr.pth',OUT/'benchmark_ocr.pth')
    with (OUT/'train.log').open('ab') as log:
        process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        state={'state':'running','pid':process.pid,'launcher_pid':os.getpid(),'source_step':150000,'target_step':200000,'started_utc':utc(),'command':command,'log':str(OUT/'train.log')}
        write(OUT/'process.json',state);write(OUT/'status.json',state)
        code=process.wait()
    state.update(state='completed' if code==0 else 'failed',exit_code=code,finished_utc=utc());write(OUT/'status.json',state)


if __name__=='__main__':main()
