"""Compare uppercase Vietnamese inference using identical genuine-writer references."""
import argparse
import csv
import gc
import json
from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont

from data.vietnamese import to_tensor
from models.model import Generator
from models.OCR_network import CRNN,strLabelConverter
from train_vietnamese import model_args,decoded,edit_distance,seed_all

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'runs/uppercase_comparison_v1');args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);seed_all(742)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    manifest=json.loads((ROOT/'files/vietnamese/manifest.json').read_text())
    words=['DƯƠNG','HƯỜNG','HƯƠNG','ĐƯỜNG','TRƯỜNG','NGUYỄN','THƯỜNG','HOÀNG']
    converter=strLabelConverter(manifest['alphabet']);converter.encode(words)
    # Inference only needs saved references, not the original private HDF5 files.
    reference_root=ROOT/'runs/continuation_150k_v1/training'
    reference_metadata=json.loads((reference_root/'benchmark_references.json').read_text())
    reference_sheet=Image.open(reference_root/'benchmark_references.png').convert('L')
    styles={};provenance={}
    for column,record in enumerate(reference_metadata['styles']):
        writer=record['writer'];chosen=record['references'];tensors=[]
        if any(r['text'] in words for r in chosen):raise ValueError('Target transcript present in saved style references')
        references=Image.new('L',(192,len(chosen)*36),255)
        for row,reference in enumerate(chosen):
            padded=reference_sheet.crop((column*192,row*36,column*192+192,row*36+32))
            tensors.append(to_tensor(padded)[0]);references.paste(padded,(0,row*36))
        styles[writer]=torch.stack(tensors).unsqueeze(0).to(device)
        provenance[writer]=chosen
        references.save(args.output/f'writer_{writer}_references.png')
    ocr=CRNN(model_args(manifest,device)).to(device).eval()
    ocr.load_state_dict(torch.load(ROOT/'runs/continuation_150k_v1/training/benchmark_ocr.pth',map_location='cpu',weights_only=True))
    font=ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',17)
    checkpoints=[('baseline_150k',ROOT/'runs/continuation_150k_v1/training/last.pth'),('control_155k',ROOT/'runs/visual_quality_pilot_v1/control/last.pth'),('feature_only_155k',ROOT/'runs/visual_quality_pilot_v1/feature_only/last.pth'),('feature_pixel_155k',ROOT/'runs/visual_quality_pilot_v1/feature_pixel/last.pth')]
    cell_width,cell_height=260,120
    overview=Image.new('RGB',(cell_width*len(words),4*(cell_height*4+35)),'white');draw_all=ImageDraw.Draw(overview)
    records=[]
    for model_index,(name,path) in enumerate(checkpoints):
        state=torch.load(path,map_location='cpu',weights_only=False);step=state['step'];config=state['config']
        if state['alphabet']!=manifest['alphabet']:raise ValueError('Alphabet mismatch')
        cfg=model_args(manifest,device,not config['legacy_generator_layout']);cfg.correct_style_layout=not config['legacy_generator_layout']
        model=Generator(cfg).to(device).eval();model.load_state_dict(state['model']['G']);del state;gc.collect()
        panel=Image.new('RGB',(cell_width*len(words),cell_height*4+35),'white');draw=ImageDraw.Draw(panel)
        draw.text((5,4),f'{name}: same references, generated height 32 px; enlarged 2x for viewing',font=font,fill='black')
        with torch.no_grad():
            for row,(writer,style) in enumerate(styles.items()):
                memory=model.compute_style(style)
                for column,text in enumerate(words):
                    query,_,_=converter.encode([text]);fake=model.generate(memory,query.to(device))
                    prediction=decoded(ocr(fake),[16*len(text)],manifest['alphabet'])[0]
                    image=Image.fromarray(((fake[0,0].clamp(-1,1).cpu().numpy()+1)*127.5).astype('uint8'))
                    output_name=f'{name}_writer{writer}_{text}.png';image.save(args.output/output_name)
                    record={'model':name,'checkpoint':str(path),'step':step,'writer':writer,'text':text,'ocr_prediction':prediction,'exact':text==prediction,'cer':edit_distance(text,prediction)/len(text),'file':output_name};records.append(record)
                    enlarged=image.resize((image.width*2,64),Image.Resampling.NEAREST)
                    x=column*cell_width+5;y=row*cell_height+35
                    panel.paste(enlarged,(x,y+22));draw.text((x,y),f'w{writer}: {text}',font=font,fill='black');draw.text((x,y+88),f'OCR: {prediction}',font=font,fill='black')
        panel.save(args.output/f'{name}_panel.png');overview.paste(panel,(0,model_index*(4*cell_height+35)))
        subset=[r for r in records if r['model']==name];print(name,'exact',sum(r['exact'] for r in subset),'/',len(subset),flush=True)
        del model;gc.collect()
        if device=='cuda':torch.cuda.empty_cache()
    overview.save(args.output/'comparison_all_models.png')
    (args.output/'results.json').write_text(json.dumps({'torch':torch.__version__,'reference_source':str(reference_root),'seed':742,'reference_seed':reference_metadata['seed'],'split':'val','prompts':words,'references':provenance,'warning':'OCR proxy uses the same frozen content supervisor. Inspect raw generated images; OCR correctness does not imply visual quality.','records':records},ensure_ascii=False,indent=2))
    with (args.output/'results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    report=['# Inference chữ hoa tiếng Việt','','Cùng 15 support cố định cho mỗi writer development; cùng seed giữa checkpoint. Ảnh gốc cao 32px, panel phóng 2x. OCR nội bộ chỉ để tham khảo.','','| Model | Đúng DƯƠNG | Đúng HƯỜNG | Đúng toàn bộ 8 từ × 4 writer |','| --- | --- | --- | --- |']
    for name,_ in checkpoints:
        rs=[r for r in records if r['model']==name];report.append(f"| {name} | {sum(r['exact'] for r in rs if r['text']=='DƯƠNG')}/4 | {sum(r['exact'] for r in rs if r['text']=='HƯỜNG')}/4 | {sum(r['exact'] for r in rs)}/32 |")
    for name,_ in checkpoints:report += ['',f'## {name}','',f'![{name}]({name}_panel.png)']
    (args.output/'REPORT_VI.md').write_text('\n'.join(report)+'\n')


if __name__=='__main__':main()
