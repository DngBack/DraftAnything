"""Warm-start adaptation pilot, NOT a clean zero-shot composition experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.vietnamese import PackedH5,collate,load_manifest
from models.model import Generator
from models.OCR_network import CRNN,strLabelConverter
from models.BigGAN_networks import Discriminator
from research.vietcompose.model import install_composition
from research.vietcompose.losses import ComponentCTC
from research.vietcompose.orthography import component_errors
from train_vietnamese import model_args,seed_all,infinite,ocr_loss
from util.vietnamese_visual import feature_matching_loss,paired_pixel_loss,VisualBenchmark
from util.vietnamese_benchmark import FixedBenchmark


def write(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str));temp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('runs/continuation_200k_v1/best.pth'));p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['glyph','factorized'],required=True);p.add_argument('--steps',type=int,default=1000,help='New adapter updates, not legacy GAN outer steps');p.add_argument('--batch-size',type=int,default=8);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--eval-every',type=int,default=250);p.add_argument('--seed',type=int,default=742);p.add_argument('--device',default='cuda');p.add_argument('--resume',type=Path);args=p.parse_args()
    if min(args.steps,args.batch_size,args.eval_every)<1:p.error('Positive steps/batch/eval interval required')
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'adapter_last.pth').exists() and not args.resume:raise RuntimeError('Output already contains checkpoint; specify resume')
    torch.set_num_threads(4);seed_all(args.seed)
    if args.device.startswith('cuda'):
        total=torch.cuda.get_device_properties(0).total_memory;torch.cuda.set_per_process_memory_fraction(min(12*1024**3/total,1.))
    manifest=load_manifest(Path('files/vietnamese/manifest.json'));manifest_hash=hashlib.sha256(Path('files/vietnamese/manifest.json').read_bytes()).hexdigest()
    with args.source.open('rb') as handle:source_hash=hashlib.file_digest(handle,'sha256').hexdigest()
    source=torch.load(args.source,map_location='cpu',weights_only=False)
    if source['manifest_sha256']!=manifest_hash:raise ValueError('Source manifest mismatch')
    cfg=model_args(manifest,args.device,not source['config']['legacy_generator_layout']);cfg.correct_style_layout=not source['config']['legacy_generator_layout']
    generator=Generator(cfg).to(args.device);generator.load_state_dict(source['model']['G'],strict=True)
    for parameter in generator.parameters():parameter.requires_grad_(False)
    install_composition(generator,args.mode);generator.eval()
    recognizer=CRNN(cfg).to(args.device).eval();recognizer.load_state_dict(source['model']['R'])
    discriminator=Discriminator(resolution=16,crop_size=None).to(args.device).eval();discriminator.load_state_dict(source['model']['D'])
    for net in [recognizer,discriminator]:
        for parameter in net.parameters():parameter.requires_grad_(False)
    source_step=source['step'];del source
    converter=strLabelConverter(manifest['alphabet']);component_ctc=ComponentCTC(manifest['alphabet']).to(args.device)
    parameters=[parameter for parameter in generator.parameters() if parameter.requires_grad];optimizer=torch.optim.Adam(parameters,lr=args.lr)
    dataset=PackedH5(manifest,style=True,num_examples=15)
    loader=DataLoader(dataset,batch_size=args.batch_size,sampler=dataset.sampler(),num_workers=2,collate_fn=collate,drop_last=True)
    start=0
    if args.resume:
        state=torch.load(args.resume,map_location='cpu',weights_only=False)
        if state['source_sha256']!=source_hash or state['mode']!=args.mode:raise ValueError('Resume source/mode mismatch')
        result=generator.query_embed.load_state_dict(state['adapter'],strict=False)
        if result.unexpected_keys or any(not k.startswith('glyph.') for k in result.missing_keys):raise ValueError(result)
        optimizer.load_state_dict(state['optimizer']);start=state['adapter_step'];random.setstate(state['rng']['python']);np.random.set_state(state['rng']['numpy']);torch.set_rng_state(state['rng']['torch']);torch.cuda.set_rng_state_all(state['rng']['cuda'])
    config={**vars(args),'source':str(args.source.resolve()),'source_sha256':source_hash,'source_step':source_step,'manifest_sha256':manifest_hash,'torch':torch.__version__,'trainable_parameters':sum(p.numel() for p in parameters),'loss_weights':{'pixel':1.,'fixed_d_feature':.2,'character_ctc':.05,'component_ctc':.05},'warning':'Warm-start, previously exposed characters. Not zero-shot; no GAN/D/W updates; only new query adapter is trained. Glyph control shares interaction backbone but has unused component embeddings.'}
    write(args.output/'config.json',config)
    expanded=FixedBenchmark(manifest,15,args.output/'expanded',args.device,expanded=True)
    visual=VisualBenchmark(manifest,15,args.output/'visual',args.device,discriminator)
    def evaluate(step):
        a=expanded.render(generator,recognizer,converter,step);b=visual.evaluate(generator,recognizer,converter,step);counts={}
        for row in a['predictions']:
            for key,value in component_errors(row['reference'],row['prediction']).items():counts[key]=counts.get(key,0)+value
        write(args.output/f'components_{step:06d}.json',{'adapter_step':step,'counts':counts,'rates':{k:v/max(counts['reference_characters'],1) for k,v in counts.items() if k.endswith('_errors')},'warning':'Components derived from supervisor OCR outputs; not independent evidence of mark placement or style.'})
        return a,b
    a,b=evaluate(start);best_score=b['mean_feature_distance'];baseline_cer=a['guidance_ocr_cer'];batches=infinite(loader);begun=time.monotonic()
    def save(step,name):
        state={'version':1,'architecture':'vietcompose-query-v1','adapter_step':step,'source_step':source_step,'source':str(args.source.resolve()),'source_sha256':source_hash,'mode':args.mode,'adapter':{k:v.detach().cpu() for k,v in generator.query_embed.state_dict().items() if not k.startswith('glyph.')},'optimizer':optimizer.state_dict(),'config':config,'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}}
        temp=args.output/(name+'.partial');torch.save(state,temp);temp.replace(args.output/name)
    for step in range(start+1,args.steps+1):
        batch=next(batches);target=batch['img'].to(args.device);style=batch['style'].to(args.device);widths=batch['widths'].to(args.device);texts=batch['label']
        # Eval freezes original BN/dropout; the adapter itself has dropout=0.
        generator.eval();query,_,_=converter.encode(texts)
        with torch.no_grad():memory=generator.compute_style(style)
        fake=generator.generate(memory,query.to(args.device));valid=torch.arange(fake.shape[-1],device=args.device)[None,None,None,:]<widths[:,None,None,None];fake=torch.where(valid,fake,torch.ones_like(fake))
        pixel=paired_pixel_loss(fake,target,widths);feature=feature_matching_loss(discriminator,fake,target,widths)
        character=ocr_loss(recognizer,converter,fake,texts,widths)
        component,role_losses=component_ctc(recognizer(fake),texts,widths)
        loss=pixel+.2*feature+.05*character+.05*component
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite adaptation loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,5,error_if_nonfinite=True);optimizer.step()
        if step==start+1 or step%25==0:
            record={'adapter_step':step,'source_step':source_step,'seconds':time.monotonic()-begun,'loss':float(loss.detach()),'pixel':float(pixel.detach()),'feature':float(feature.detach()),'character_ctc':float(character.detach()),'component_ctc':float(component.detach()),'grad_norm':float(norm),'query_residual_rms':float(generator.query_embed.last_residual_norm),'roles':{k:float(v.detach()) for k,v in role_losses.items()}}
            print(json.dumps(record),flush=True)
            with (args.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        if step%args.eval_every==0 or step==args.steps:
            a,b=evaluate(step);save(step,'adapter_last.pth')
            if b['mean_feature_distance']<best_score and a['guidance_ocr_cer']<=baseline_cer+.02:
                best_score=b['mean_feature_distance'];save(step,'adapter_best.pth');write(args.output/'best_selection.json',{'adapter_step':step,'feature_distance':best_score,'expanded_cer':a['guidance_ocr_cer'],'warning':'Proxy-only best. No automatic promotion or novelty/generalization claim.'})
    write(args.output/'profile.json',{'completed_adapter_step':args.steps,'seconds':time.monotonic()-begun,'torch':torch.__version__,'source_step':source_step})


if __name__=='__main__':main()
