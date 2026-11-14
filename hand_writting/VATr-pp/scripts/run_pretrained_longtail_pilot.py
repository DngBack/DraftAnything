"""Matched continuation from the recommended pretrained VATr++ checkpoint.

Only transcript rare-draw weighting differs. Original G/D/W/R and Adam states
are restored by train_vietnamese.py; this is exposed warm-start, not zero-shot.
"""
import argparse
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
import torch

ROOT = Path(__file__).resolve().parents[1]


def write(path, obj):
    temp = path.with_suffix('.partial')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str))
    temp.replace(path)


def sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b''): value.update(chunk)
    return value.hexdigest()


def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT/'runs/continuation_200k_v1/best.pth')
    p.add_argument('--output', type=Path, default=ROOT/'runs/pretrained_longtail_v1')
    p.add_argument('--additional-steps', type=int, default=1000)
    p.add_argument('--parallel', type=int, default=2)
    args = p.parse_args()
    if args.additional_steps < 1 or args.parallel not in (1, 2): p.error('Invalid budget/parallelism')
    args.output = args.output.resolve(); args.source = args.source.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = os.open(args.output/'launcher.lock', os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (args.output/'plan.json').exists(): p.error('Existing experiment: inspect status; do not overwrite')
    state = torch.load(args.source, map_location='cpu', weights_only=False)
    if state['stage'] != 'gan' or not {'G', 'D', 'W', 'R'} <= state['model'].keys():
        raise ValueError('Complete pretrained GAN checkpoint required')
    start, cfg = state['step'], state['config']
    if not cfg['freeze_ocr']: raise ValueError('Pilot expects frozen source OCR')
    del state
    import gc
    gc.collect()
    source_hash = sha256(args.source)
    trials = []
    target = start+args.additional_steps
    for sampler in ('legacy', 'component'):
        folder = args.output/sampler
        if folder.exists(): raise ValueError('Trial output already exists')
        folder.mkdir()
        shutil.copy2(args.source.parent/'benchmark_ocr.pth', folder/'benchmark_ocr.pth')
        command = [sys.executable, '-u', str(ROOT/'train_vietnamese.py'), '--stage', 'gan',
                   '--manifest', str(ROOT/'files/vietnamese/manifest.json'), '--resume', str(args.source),
                   '--output', str(folder), '--steps', str(target), '--text-sampler', sampler,
                   '--batch-size', str(cfg['batch_size']), '--num-examples', str(cfg['num_examples']),
                   '--workers', '2', '--lr', str(cfg['lr']), '--encoder-lr', str(cfg['encoder_lr']),
                   '--d-lr', str(cfg['d_lr']), '--writer-lr', str(cfg.get('writer_lr') or cfg['lr']),
                   '--ocr-weight', str(cfg['ocr_weight']), '--writer-weight', str(cfg['writer_weight']),
                   '--hat-fraction', str(cfg['hat_fraction']), '--rare-fraction', str(cfg['rare_fraction']),
                   '--freeze-ocr', '--expanded-benchmark', '--benchmark-every', '250',
                   '--eval-every', '250', '--save-every', '250', '--log-every', '50',
                   '--keep-checkpoints', '2', '--best-window', '3', '--gpu-memory-gb', '12']
        if not cfg['legacy_generator_layout']: command.append('--adapt-generator-layout')
        if not cfg['gradient_balance']: command.append('--no-gradient-balance')
        if not cfg['crop_discriminator']: command.append('--no-crop-discriminator')
        if cfg['no_hat']: command.append('--no-hat')
        trials.append({'sampler': sampler, 'state': 'pending', 'command': command, 'folder': str(folder)})
    plan = {'state': 'prepared', 'created_utc': utc(), 'source': str(args.source),
            'source_sha256': source_hash, 'source_step': start, 'target_step': target,
            'additional_outer_steps_per_trial': args.additional_steps, 'torch': torch.__version__,
            'python': sys.version, 'trials': trials,
            'note': 'Warm-start from native pretrained G/D/W/R plus optimizer/RNG. Only text sampling differs. No zero-shot claim, no automatic promotion.'}
    write(args.output/'plan.json', plan)
    # Archive source code before starting children.
    for relative in ['train_vietnamese.py', 'data/vietnamese.py', 'util/vietnamese_longtail.py',
                     'research/longtail/protocol.py', 'models/model.py', 'util/vietnamese_benchmark.py']:
        target_path = args.output/'code'/relative; target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/relative, target_path)
    active = []
    def stop(signum, frame):
        for trial, process, handle in active:
            if process.poll() is None: process.send_signal(signal.SIGINT)
        plan['state'] = 'interrupted'; write(args.output/'status.json', plan)
        raise SystemExit(130)
    signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)
    pending = list(trials)
    while pending or active:
        while pending and len(active) < args.parallel:
            trial = pending.pop(0); folder = Path(trial['folder'])
            handle = (folder/'train.log').open('ab')
            process = subprocess.Popen(trial['command'], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)
            trial.update(state='running', pid=process.pid, started_utc=utc())
            active.append((trial, process, handle)); plan['state'] = 'running'
            write(args.output/'status.json', plan)
            print(utc(), 'START', trial['sampler'], start, '->', target, flush=True)
        for item in list(active):
            trial, process, handle = item
            code = process.poll()
            if code is None: continue
            handle.close(); active.remove(item)
            trial.update(state='completed' if code == 0 else 'failed', exit_code=code, finished_utc=utc())
            write(args.output/'status.json', plan)
            print(utc(), 'END', trial['sampler'], code, flush=True)
            if code:
                for other, child, log in active:
                    if child.poll() is None: child.send_signal(signal.SIGINT)
                plan['state'] = 'failed'; write(args.output/'status.json', plan)
                raise SystemExit(code)
        if pending or active: time.sleep(5)
    if sha256(args.source) != source_hash: raise RuntimeError('Source checkpoint changed unexpectedly')
    plan.update(state='completed', finished_utc=utc(), source_unchanged=True)
    write(args.output/'status.json', plan)


if __name__ == '__main__': main()
