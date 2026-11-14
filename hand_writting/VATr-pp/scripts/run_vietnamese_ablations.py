"""Run isolated 100k->110k A/B/C trials sequentially; never overwrite the source."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'runs/vatrpp_vn_finetune_100k'
OUT = ROOT/'runs/ablation_100k_v1'


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def summarize():
    result = []
    for name in ['A_control', 'B_half_g_lr', 'C_half_ocr_weight']:
        folder = OUT/name
        records = []
        for path in sorted((folder/'expanded').glob('benchmark_*.json')):
            if path.stem.split('_')[-1].isdigit():
                records.append(json.loads(path.read_text()))
        if records:
            best_path = folder/'best_selection.json'
            result.append({'branch': name, 'last_step': records[-1]['step'], 'initial_cer': records[0]['guidance_ocr_cer'], 'last_cer': records[-1]['guidance_ocr_cer'], 'last_exact': records[-1]['exact_transcript_rate'], 'best_selection': json.loads(best_path.read_text()) if best_path.exists() else None, 'warning': 'Supervisor OCR proxy; no independent OCR/style metric yet.'})
    write_json(OUT/'comparison.json', result)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = os.open(OUT/'queue.lock', os.O_CREAT | os.O_RDWR, 0o600)
    import fcntl
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan_path = OUT/'plan.json'
    if plan_path.exists():
        raise RuntimeError('Existing experiment plan: inspect state rather than silently restart')
    digest = hashlib.file_digest((BASE/'last.pth').open('rb'), 'sha256').hexdigest()
    common = [sys.executable, '-u', 'train_vietnamese.py', '--stage', 'gan', '--resume', str(BASE/'last.pth'), '--steps', '110000', '--batch-size', '8', '--num-examples', '15', '--workers', '2', '--freeze-ocr', '--encoder-lr', '0.000005', '--lr', '0.00002', '--writer-lr', '0.00002', '--d-lr', '0.00001', '--ocr-weight', '4', '--writer-weight', '0.1', '--gpu-memory-gb', '10', '--save-every', '500', '--eval-every', '500', '--benchmark-every', '500', '--log-every', '50', '--expanded-benchmark', '--keep-checkpoints', '3', '--best-window', '3', '--override-resume-lr']
    trials = []
    for name, changes in [('A_control', {}), ('B_half_g_lr', {'--encoder-lr': '0.0000025', '--lr': '0.00001'}), ('C_half_ocr_weight', {'--ocr-weight': '2'})]:
        command = list(common)
        for flag, value in changes.items():
            command[command.index(flag)+1] = value
        folder = OUT/name
        if folder.exists():
            raise RuntimeError(f'Output already exists: {folder}')
        folder.mkdir()
        shutil.copy2(BASE/'benchmark_ocr.pth', folder/'benchmark_ocr.pth')
        command.extend(['--output', str(folder)])
        trials.append({'branch': name, 'command': command, 'state': 'pending'})
    plan = {'source': str(BASE/'last.pth'), 'source_sha256': digest, 'start_step': 100000, 'target_step': 110000, 'selection': '3-evaluation rolling expanded supervisor OCR CER; proxy only', 'trials': trials}
    write_json(plan_path, plan)
    for trial in trials:
        folder = OUT/trial['branch']
        trial['state'] = 'running'
        trial['started_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with (folder/'train.log').open('ab') as log:
            process = subprocess.Popen(trial['command'], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            trial['pid'] = process.pid
            write_json(OUT/'status.json', plan)
            write_json(folder/'process.json', trial)
            code = process.wait()
        trial['exit_code'] = code
        trial['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        trial['state'] = 'completed' if code == 0 else 'failed'
        write_json(OUT/'status.json', plan)
        summarize()
        if code:
            raise RuntimeError(f"{trial['branch']} failed with exit code {code}; queue stopped")
    print('All trials completed. See comparison.json.', flush=True)


if __name__ == '__main__':
    main()
