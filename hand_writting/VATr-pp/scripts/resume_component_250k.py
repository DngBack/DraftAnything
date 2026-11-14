"""Continue the completed component-sampling pilot to 250,000 total steps.

Runs as a detached supervisor, with an exclusive lock and periodic status updates.
Preserves source weights, Adam rates, batch/support and all sampling/loss flags.
"""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'runs/pretrained_longtail_v1/component'
OUT = ROOT/'runs/component_continuation_250k_v1'
TARGET = 250000


def write(path, value):
    temp = path.with_suffix('.partial')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def continuation_command(command, source, output):
    command = list(command)
    if command[command.index('--text-sampler')+1] != 'component':
        raise ValueError('Continuation must use component sampling')
    for flag, value in {'--resume': str(source), '--steps': str(TARGET), '--output': str(output),
                        '--benchmark-every': '1000', '--eval-every': '1000', '--save-every': '1000',
                        '--keep-checkpoints': '3'}.items():
        command[command.index(flag)+1] = value
    return command


def last_progress():
    path = OUT/'metrics.jsonl'
    if not path.exists():
        return {}
    # Read a bounded tail rather than rereading a growing training log.
    with path.open('rb') as handle:
        handle.seek(max(0, path.stat().st_size-32768))
        lines = handle.read().decode('utf-8', errors='replace').splitlines()
    for line in reversed(lines):
        try:
            row = json.loads(line)
            return {'last_logged_step': row['step'], 'training_seconds': row['seconds']}
        except (ValueError, KeyError):
            pass
    return {}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = os.open(OUT/'launcher.lock', os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUT/'plan.json').exists() or (OUT/'last.pth').exists():
        raise RuntimeError('Continuation already prepared/launched; inspect status before restarting')
    prior = json.loads((ROOT/'runs/pretrained_longtail_v1/status.json').read_text())
    if prior['state'] != 'completed':
        raise ValueError('Source pilot not complete')
    trial = next(t for t in prior['trials'] if t['sampler'] == 'component')
    if trial['state'] != 'completed':
        raise ValueError('Component source trial not complete')
    source = SOURCE/'last.pth'
    command = continuation_command(trial['command'], source, OUT)
    check = subprocess.run([command[0], '-c',
        "import torch,sys,json; x=torch.load(sys.argv[1],map_location='cpu',weights_only=False); "
        "assert x['step']==176500 and x['stage']=='gan' and x['config']['text_sampler']=='component'; "
        "print(json.dumps({'step':x['step'],'torch':torch.__version__,'freeze_ocr':x['config']['freeze_ocr']}))",
        str(source)], cwd=ROOT, text=True, capture_output=True)
    if check.returncode:
        write(OUT/'status.json', {'state': 'failed_source_validation', 'stderr': check.stderr, 'updated_utc': utc()})
        raise RuntimeError(check.stderr)
    verified = json.loads(check.stdout.strip().splitlines()[-1])
    source_hash = sha256(source)
    plan = {'state': 'prepared', 'source_checkpoint': str(source), 'source_sha256': source_hash,
            'source_step': verified['step'], 'target_step': TARGET, 'additional_outer_steps': TARGET-verified['step'],
            'torch': verified['torch'], 'command': command, 'launcher_pid': os.getpid(),
            'created_utc': utc(), 'method': 'Native pretrained VATr++ with component-aware rare transcript sampling; no architecture changes.',
            'checkpoint_interval': 1000, 'benchmark_interval': 1000, 'numbered_checkpoints_to_keep': 3,
            'note': 'G/D/W/R and Adam/RNG restored. Data iterator restarts; not bit-exact uninterrupted training. Supervisor OCR metrics are quality proxies.'}
    shutil.copy2(SOURCE/'benchmark_ocr.pth', OUT/'benchmark_ocr.pth')
    for relative in ['train_vietnamese.py', 'data/vietnamese.py', 'util/vietnamese_longtail.py',
                     'research/longtail/protocol.py', 'models/model.py', 'util/vietnamese_benchmark.py',
                     'scripts/resume_component_250k.py']:
        destination = OUT/'code'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/relative, destination)
    write(OUT/'plan.json', plan)
    write(OUT/'status.json', plan)
    with (OUT/'train.log').open('ab') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        plan.update(state='running', pid=process.pid, started_utc=utc(), log=str(OUT/'train.log'))
        write(OUT/'process.json', plan)
        write(OUT/'status.json', plan)
        def stop(signum, frame):
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
            plan.update(state='interrupted', updated_utc=utc(), **last_progress())
            write(OUT/'status.json', plan)
            raise SystemExit(130)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        while process.poll() is None:
            plan.update(updated_utc=utc(), **last_progress())
            write(OUT/'status.json', plan)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
        code = process.returncode
    plan.update(state='completed' if code == 0 else 'failed', exit_code=code,
                finished_utc=utc(), **last_progress())
    if code == 0:
        profile = json.loads((OUT/'profile.json').read_text())
        plan['completed_step'] = profile['completed_step']
        plan['source_unchanged'] = sha256(source) == source_hash
        if profile['completed_step'] != TARGET or not plan['source_unchanged']:
            plan['state'] = 'failed_completion_validation'
    write(OUT/'status.json', plan)
    if plan['state'] == 'completed':
        result = subprocess.run([command[0], str(ROOT/'scripts/report_component_250k.py')], cwd=ROOT, capture_output=True, text=True)
        plan['report_exit_code'] = result.returncode
        plan['report_status'] = 'completed' if result.returncode == 0 else 'failed'
        (OUT/'report.log').write_text(result.stdout+result.stderr)
        write(OUT/'status.json', plan)


if __name__ == '__main__':
    main()
