"""Wait for A/B/C, apply predeclared proxy gates, and continue a qualifying best checkpoint."""
import datetime
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT/'runs/ablation_100k_v1'
OUT = ROOT/'runs/continuation_150k_v1'


def write_json(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    tmp.replace(path)


def records(folder):
    return [json.loads(p.read_text()) for p in sorted((folder/'expanded').glob('benchmark_*.json')) if p.stem.split('_')[-1].isdigit()]


def stats(rs):
    return {'mean_cer': statistics.mean(r['guidance_ocr_cer'] for r in rs),
            'std_cer': statistics.pstdev(r['guidance_ocr_cer'] for r in rs),
            'mean_exact': statistics.mean(r['exact_transcript_rate'] for r in rs),
            'uppercase_cer': statistics.mean(r['groups']['uppercase']['cer'] for r in rs)}


def select(plan):
    summaries = []
    control = stats(records(EXPERIMENT/'A_control')[-5:])
    for trial in plan['trials']:
        folder = EXPERIMENT/trial['branch']
        rs = records(folder)
        if len(rs) < 6 or rs[-1]['step'] != 110000:
            raise RuntimeError(f"Incomplete evaluation: {trial['branch']}")
        initial, recent = rs[0], stats(rs[-5:])
        selection = json.loads((folder/'best_selection.json').read_text())
        selected_record = json.loads((folder/'expanded'/f"benchmark_{selection['step']:06d}.json").read_text())
        checks = {
            'recent_mean_cer_improves_initial_by_10pct': recent['mean_cer'] <= .9*initial['guidance_ocr_cer'],
            'recent_mean_exact_gains_3pp': recent['mean_exact'] >= initial['exact_transcript_rate']+.03,
            'recent_uppercase_not_worse_than_initial': recent['uppercase_cer'] <= initial['groups']['uppercase']['cer'],
            'last_checkpoint_not_worse_than_initial': rs[-1]['guidance_ocr_cer'] <= initial['guidance_ocr_cer'],
            'selected_checkpoint_not_worse_than_initial': selected_record['guidance_ocr_cer'] <= .9*initial['guidance_ocr_cer'],
        }
        if trial['branch'] != 'A_control':
            checks.update({
                'recent_mean_cer_beats_control_by_5pct': recent['mean_cer'] <= .95*control['mean_cer'],
                'recent_exact_not_worse_than_control': recent['mean_exact'] >= control['mean_exact'],
                'recent_uppercase_not_worse_than_control': recent['uppercase_cer'] <= control['uppercase_cer'],
                'recent_variability_within_control_tolerance': recent['std_cer'] <= max(.005,1.25*control['std_cer']),
            })
        summaries.append({'branch': trial['branch'], 'recent': recent, 'selected_step': selection['step'], 'checks': checks, 'qualifies': all(checks.values()), 'command': trial['command']})
    eligible = [s for s in summaries if s['qualifies']]
    winner = min(eligible, key=lambda s:s['recent']['mean_cer']) if eligible else None
    return summaries, winner


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = os.open(OUT/'watcher.lock', os.O_CREAT|os.O_RDWR, 0o600)
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'decision.json').exists():
        raise RuntimeError('Decision already exists; inspect rather than relaunch')
    write_json(OUT/'status.json', {'state':'waiting_for_abc', 'target_total_steps':150000, 'warning':'Automatic gates use supervisor OCR proxy, not independent style/human evaluation.'})
    while True:
        plan=json.loads((EXPERIMENT/'status.json').read_text())
        if any(t['state']=='failed' for t in plan['trials']):
            write_json(OUT/'status.json', {'state':'blocked_trial_failed'});return
        if all(t['state']=='completed' for t in plan['trials']):break
        running=[t for t in plan['trials'] if t['state']=='running']
        if running and not Path(f"/proc/{running[0]['pid']}/cmdline").exists():
            # Allow the queue a short interval to commit completion state.
            time.sleep(5)
            latest=json.loads((EXPERIMENT/'status.json').read_text())
            if latest==plan:
                write_json(OUT/'status.json', {'state':'blocked_queue_not_running'});return
        time.sleep(30)
    summaries,winner=select(plan)
    decision={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'summaries':summaries,'winner':winner['branch'] if winner else None,'warning':'Proxy-only automatic selection; style and independent OCR improvements are not established.'}
    write_json(OUT/'decision.json',decision)
    if winner is None:
        write_json(OUT/'status.json',{'state':'stopped_no_qualifying_improvement','target_total_steps':150000});return
    source=EXPERIMENT/winner['branch']
    output=OUT/'training';output.mkdir()
    shutil.copy2(source/'benchmark_ocr.pth',output/'benchmark_ocr.pth')
    command=list(winner['command'])
    for flag,value in {'--resume':str(source/'best.pth'),'--output':str(output),'--steps':'150000'}.items():command[command.index(flag)+1]=value
    with (output/'train.log').open('ab') as log:
        p=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        state={'state':'running_to_150000','pid':p.pid,'source_branch':winner['branch'],'source_step':winner['selected_step'],'target_total_steps':150000,'command':command}
        write_json(OUT/'status.json',state);write_json(output/'process.json',state)
        code=p.wait()
    state.update(state='completed' if code==0 else 'failed',exit_code=code)
    write_json(OUT/'status.json',state)


if __name__=='__main__':main()
