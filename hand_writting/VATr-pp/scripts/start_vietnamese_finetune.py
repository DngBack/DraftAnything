"""Start an authorized persistent VATr++ fine-tune after checking full-development OCR quality."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ocr-run', type=Path, default=Path('runs/vatrpp_vn_ocr_train'))
    p.add_argument('--output', type=Path, default=Path('runs/vatrpp_vn_finetune'))
    p.add_argument('--steps', type=int, default=50000)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--ocr-weight', type=float, default=4., help='Content priority after the initial panel showed insufficient diacritic learning')
    p.add_argument('--max-ocr-cer', type=float, default=.20)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    os.chdir(root)
    evaluations=[]
    for path in args.ocr_run.glob('validation_*.json'):
        result=json.loads(path.read_text())
        if result['samples']==11786:
            evaluations.append((result['cer'],path,result))
    if not evaluations:
        raise ValueError('No full-development OCR evaluation found')
    cer,path,result=min(evaluations,key=lambda r:r[0])
    if cer>args.max_ocr_cer:
        raise ValueError(f'OCR quality gate failed: {cer:.4f} > {args.max_ocr_cer:.4f}')
    checkpoint=args.ocr_run/'best.pth'
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    args.output.mkdir(parents=True,exist_ok=True)
    pid_file=args.output/'process.json'
    if pid_file.exists():
        old=json.loads(pid_file.read_text())
        try:
            os.kill(old['pid'],0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError(f"Job already running: {old['pid']}")
    command=[sys.executable,'-u','train_vietnamese.py','--stage','gan',
             '--output',str(args.output),'--steps',str(args.steps),'--batch-size',str(args.batch_size),
             '--num-examples','15','--workers','2','--freeze-ocr',
             '--lr','0.00002','--encoder-lr','0.000005','--d-lr','0.00001',
             '--writer-weight','0.1','--ocr-weight',str(args.ocr_weight),'--gpu-memory-gb','10',
             '--save-every','500','--eval-every','500','--benchmark-every','500','--log-every','50']
    if (args.output/'last.pth').exists():
        command += ['--resume',str(args.output/'last.pth')]
    else:
        command += ['--ocr-checkpoint',str(checkpoint),
                    '--generator-checkpoint','files/pretrained/vatrpp_english.safetensors']
    with (args.output/'train.log').open('a') as log:
        process=subprocess.Popen(command,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    metadata=dict(pid=process.pid,command=command,ocr_gate_cer=cer,ocr_validation=str(path),
                  ocr_samples=result['samples'],gate_note='Engineering launch threshold, not a claim that OCR is fully accurate.',
                  log=str((args.output/'train.log').resolve()))
    pid_file.write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2),flush=True)


if __name__=='__main__':
    main()
