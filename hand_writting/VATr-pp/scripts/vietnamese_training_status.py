"""Report live VATr++ progress from process metadata, metrics and fixed benchmarks."""
import json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1]
    ocr=root/'runs/vatrpp_vn_ocr_train'
    gan=root/'runs/vatrpp_vn_finetune'
    result={}
    evaluations=[]
    for path in ocr.glob('validation_*.json'):
        record=json.loads(path.read_text())
        if record.get('samples')==11786:
            evaluations.append((record['cer'],path,record))
    if evaluations:
        cer,path,record=min(evaluations,key=lambda r:r[0])
        result['ocr']={'best_step':int(path.stem.split('_')[-1]),'cer':cer,'cer_without_diacritics':record['cer_without_diacritics'], 'samples':record['samples']}
    process_path=gan/'process.json'
    if process_path.exists():
        process=json.loads(process_path.read_text())
        pid=process['pid']
        try:
            command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
            running='train_vietnamese.py' in command
        except (FileNotFoundError,PermissionError):
            running=False
        result['gan']={'pid':pid,'running':running,'target_steps':int(process['command'][process['command'].index('--steps')+1])}
    metrics_path=gan/'metrics.jsonl'
    if metrics_path.exists():
        rows=[]
        for line in metrics_path.read_text().splitlines():
            try: rows.append(json.loads(line))
            except json.JSONDecodeError: pass
        if rows:
            segment=[]
            for record in rows:
                if segment and (record['seconds'] < segment[-1]['seconds'] or record['step'] <= segment[-1]['step']):
                    segment=[]
                segment.append(record)
            rows=segment
            result.setdefault('gan',{})['logged_step']=rows[-1]['step']
            if len(rows)>2:
                sample=rows[-min(10,len(rows)-1):]
                rate=(sample[-1]['seconds']-sample[0]['seconds'])/(sample[-1]['step']-sample[0]['step'])
                result['gan']['seconds_per_outer_step']=rate
                if rate>0 and result['gan'].get('running'):
                    result['gan']['estimated_hours_remaining']=(result['gan']['target_steps']-rows[-1]['step'])*rate/3600
    checkpoints=sorted(gan.glob('benchmark_*.json'))
    checkpoints=[p for p in checkpoints if p.stem.removeprefix('benchmark_').isdigit()]
    if checkpoints:
        last=json.loads(checkpoints[-1].read_text())
        result['last_fixed_panel']={k:v for k,v in last.items() if k!='predictions'}
    result['paths']={'log':str(gan/'train.log'),'metrics':str(metrics_path),'checkpoint':str(gan/'last.pth')}
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
