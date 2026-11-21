"""Snapshot generator checkpoints at milestones and evaluate char vs nfd at matched steps.

* Every 5 min: when runs/gen_vn_{arm}/last.pt reaches a multiple of --every steps
  (and the file has been stable for 60 s), copy it to snap_{step}.pt.
* When both arms have snap_{S}.pt and the evaluator OCR is finished, run
  eval_holdout (raw weights, val split) for both and the paired comparison.
  Results: runs/eval/vn_{S}/{char,nfd}.json, compare.json, panels.

    python scripts/eval_loop.py
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('char', 'nfd')
EXTRA = []
TAG = ''   # --tag _lt -> runs/gen_vn_char_lt, runs/eval/vn_lt_<step>


def log(*a):
    print(time.strftime('[%F %T]'), *a, flush=True)


def snapshot(arm, every):
    last = ROOT / f'runs/gen_vn_{arm}{TAG}/last.pt'
    if not last.exists() or time.time() - last.stat().st_mtime < 60:
        return
    try:
        step = torch.load(last, map_location='cpu', weights_only=False)['step']
    except Exception as e:  # being written
        log('skip unreadable', last, e)
        return
    snap = last.with_name(f'snap_{step}.pt')
    if step % every == 0 and not snap.exists():
        shutil.copy2(last, snap.with_suffix('.tmp'))
        snap.with_suffix('.tmp').rename(snap)
        log('snapshot', snap)


def evaluate(step, ocr):
    out = ROOT / f'runs/eval/vn{TAG}_{step}'
    if (out / 'compare.json').exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        res = out / f'{arm}.json'
        if not res.exists():
            cmd = [sys.executable, 'scripts/eval_holdout.py', 'run', '--ckpt', f'runs/gen_vn_{arm}{TAG}/snap_{step}.pt',
                   '--vae', 'runs/vae_vn32/final.pt', '--ocr', str(ocr), '--raw',
                   '--panel', str(out / f'{arm}_panel.png'), '--out', str(res)] + EXTRA
            log('eval', ' '.join(cmd))
            subprocess.run(cmd, cwd=ROOT, check=True)
    cmp = subprocess.run([sys.executable, 'scripts/eval_holdout.py', 'compare', str(out / 'char.json'),
                          str(out / 'nfd.json')], cwd=ROOT, check=True, capture_output=True, text=True).stdout
    if '--pairs' in EXTRA:
        pair = subprocess.run([sys.executable, 'scripts/eval_holdout.py', 'compare', str(out / 'char.json'),
                               str(out / 'nfd.json'), '--metric', 'pair_ok'], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout
        (out / 'compare_pairs.json').write_text(pair)
    (out / 'compare.json').write_text(cmp)
    log(f'step {step} char(A) vs nfd(B):\n{cmp}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--every', type=int, default=25000)
    ap.add_argument('--ocr', type=Path, default=ROOT / 'runs/eval_ocr_vn/best.pt')
    ap.add_argument('--tag', default='')
    ap.add_argument('--ocr-done', type=Path, default=None, help='marker that the OCR finished (default: <ocr dir>/DONE)')
    ap.add_argument('--pairs', action='store_true')
    ap.add_argument('--w-mark', type=float, default=0.0)
    a = ap.parse_args()
    global TAG, EXTRA
    TAG = a.tag
    EXTRA = (['--pairs'] if a.pairs else []) + ['--w-mark', str(a.w_mark)]
    while True:
        for arm in ARMS:
            snapshot(arm, a.every)
        if (a.ocr_done or a.ocr.parent / 'DONE').exists():
            steps = [{int(p.stem.split('_')[1]) for p in (ROOT / f'runs/gen_vn_{arm}{TAG}').glob('snap_*.pt')} for arm in ARMS]
            for s in sorted(steps[0] & steps[1]):
                evaluate(s, a.ocr)
        if all((ROOT / f'runs/gen_vn_{arm}{TAG}/DONE').exists() or (ROOT / f'runs/gen_vn_{arm}{TAG}/STOPPED').exists()
               for arm in ARMS) and (a.ocr_done or a.ocr.parent / 'DONE').exists():
            # the final last.pt may be younger than snapshot()'s 60 s stability window: wait, then
            # snapshot and evaluate once more so the final step is never skipped
            time.sleep(65)
            for arm in ARMS:
                snapshot(arm, a.every)
            steps = [{int(p.stem.split('_')[1]) for p in (ROOT / f'runs/gen_vn_{arm}{TAG}').glob('snap_*.pt')} for arm in ARMS]
            for s in sorted(steps[0] & steps[1]):
                evaluate(s, a.ocr)
            log('both arms finished; last evaluations done')
            break
        time.sleep(300)


if __name__ == '__main__':
    main()
