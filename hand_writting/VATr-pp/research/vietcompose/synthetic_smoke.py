"""Train the v2 ink compositor on one canonical font as an engineering check.

NOT handwriting, NOT writer generalization, NOT a fair comparison to VATr, and
NOT evidence that the real-data problem has been solved.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image,ImageDraw,ImageFont

from research.vietcompose.renderer import LayeredInkComposer


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=300);parser.add_argument('--output',type=Path,default=Path('runs/vietcompose_renderer_synthetic_smoke_v1'));args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'renderer.pth').exists():raise RuntimeError('Existing output; do not overwrite')
    torch.manual_seed(742);torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    path=Path('files/vietnamese/unifont.pickle')
    with path.open('rb') as f:glyphs=pickle.load(f)
    matrices={chr(x['idx'][0]):x['mat'] for x in glyphs};alphabet=json.loads(Path('files/vietnamese/manifest.json').read_text())['alphabet']
    held=list('ỄễỠỡẴẵ');train=[c for c in alphabet if c not in held and c!=' ']
    def target(chars):
        masks=torch.tensor(np.stack([matrices[c] for c in chars]),dtype=torch.float32,device=device).unsqueeze(1)
        return F.interpolate(masks,size=(32,16),mode='nearest')
    model=LayeredInkComposer(hidden=64).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    history=[]
    @torch.no_grad()
    def evaluate(step):
        model.eval();result=model(held,torch.zeros(len(held),512,device=device));truth=target(held);pred=(1-result['image'])/2
        metric={'update':step,'heldout_font_pixel_l1':float((pred-truth).abs().mean()),'warning':'One-font synthetic engineering smoke, no handwriting/style/generalization claim.'};history.append(metric)
        print(json.dumps(metric),flush=True);return result,truth
    evaluate(0)
    for step in range(1,args.steps+1):
        model.train();indices=torch.randint(len(train),(32,)).tolist();chars=[train[i] for i in indices]
        output=model(chars,torch.zeros(32,512,device=device));truth=target(chars);pred=((1-output['image'])/2).clamp(1e-5,1-1e-5)
        loss=(F.binary_cross_entropy(pred,truth,reduction='none')*(1+truth)).mean()
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step()
        if step%50==0 or step==args.steps:
            evaluate(step);history[-1]['train_loss']=float(loss.detach())
    result,truth=evaluate(args.steps)
    panel=Image.new('RGB',(6*150,210),'white');draw=ImageDraw.Draw(panel);font=ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',16)
    draw.text((4,0),'Synthetic font targets / untrained-start compositor after smoke training',font=font,fill='black')
    for i,char in enumerate(held):
        draw.text((i*150+5,25),char,font=font,fill='black')
        for row,array in enumerate([1-truth[i,0].cpu().numpy(),(result['image'][i,0].cpu().numpy()+1)/2]):
            image=Image.fromarray((array*255).astype('uint8')).resize((48,96),Image.Resampling.NEAREST)
            # Fit both target and prediction side by side under each label.
            panel.paste(image,(i*150+5+row*65,60))
    panel.save(args.output/'synthetic_only_panel.png')
    torch.save({'architecture':'layered-ink-v2-unvalidated','model':model.state_dict(),'steps':args.steps},args.output/'renderer.pth')
    (args.output/'results.json').write_text(json.dumps({'history':history,'train_characters':train,'holdout_characters':held,'font_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'torch':torch.__version__,'warning':'Toy engineering run only. No writer styles, no real handwriting, no claim of superiority or novelty.'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
