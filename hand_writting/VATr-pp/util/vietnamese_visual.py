"""Paired visual supervision and a fixed genuine-writer development panel.

Feature distances are proxies, not calibrated human aesthetic scores.
"""
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from data.vietnamese import PackedH5, to_tensor


def feature_matching_loss(discriminator, fake, real, widths):
    """Same transcript/writer; frozen D features, padding masked at each scale."""
    with torch.no_grad():
        reference = discriminator.return_features(real)
    generated = discriminator.return_features(fake)
    pixel_mask = (torch.arange(real.shape[-1], device=real.device)[None, None, None, :] < widths[:, None, None, None]).float()
    losses = []
    for target, prediction in zip(reference[-3:], generated[-3:]):
        mask = F.interpolate(pixel_mask, size=target.shape[-2:], mode='nearest')
        denominator = (mask.sum()*target.shape[1]).clamp_min(1)
        scale = ((target.abs()*mask).sum()/denominator).clamp_min(.1).detach()
        losses.append(((prediction-target.detach()).abs()*mask).sum()/denominator/scale)
    return torch.stack(losses).mean()


def paired_pixel_loss(fake, real, widths):
    """Low-weight paired image constraint; same writer need not mean identical strokes."""
    mask = (torch.arange(real.shape[-1], device=real.device)[None, None, None, :] < widths[:, None, None, None]).expand_as(real)
    return ((fake-real).abs()*mask).sum()/mask.sum().clamp_min(1)


class VisualBenchmark:
    def __init__(self, manifest, num_examples, output, device, fixed_discriminator):
        self.data = PackedH5(manifest, split='val')
        self.num_examples, self.device = num_examples, device
        self.output = Path(output);self.output.mkdir(parents=True, exist_ok=True)
        self.discriminator = fixed_discriminator.eval()
        for parameter in self.discriminator.parameters():parameter.requires_grad_(False)
        rng = random.Random(74913)
        self.records = []
        for writer in sorted(self.data.pools):
            pool = self.data.pools[writer]
            targets = [i for i in pool if len(self.data.labels[i]) >= 2]
            for index in rng.sample(targets, min(2,len(targets))):
                candidates = [i for i in pool if i != index and self.data.labels[i] != self.data.labels[index]]
                if not candidates:raise ValueError('No distinct-transcript support for visual evaluation')
                refs = rng.sample(candidates, num_examples) if len(candidates)>=num_examples else rng.choices(candidates,k=num_examples)
                self.records.append({'writer':writer,'target':index,'text':self.data.labels[index],'support':refs})
        path = self.output/'references.json'
        if path.exists() and json.loads(path.read_text())!=self.records:raise ValueError('Visual references changed')
        path.write_text(json.dumps(self.records,ensure_ascii=False,indent=2))

    def batch(self, record):
        text = record['text'];width=16*len(text)
        target=to_tensor(self.data.image(record['target']).resize((width,32),Image.Resampling.BICUBIC)).unsqueeze(0).to(self.device)
        supports=[]
        for index in record['support']:
            image=self.data.image(index)
            if image.width>192:image=image.resize((192,32),Image.Resampling.BICUBIC)
            padded=Image.new('L',(192,32),255);padded.paste(image,(0,0));supports.append(to_tensor(padded)[0])
        return torch.stack(supports).unsqueeze(0).to(self.device),target,torch.tensor([width],device=self.device)

    @torch.no_grad()
    def evaluate(self, generator, ocr, converter, step):
        from train_vietnamese import decoded,edit_distance
        previous=generator.training;generator.eval()
        rows=[];panel=Image.new('RGB',(1000,110*len(self.records)),'white');draw=ImageDraw.Draw(panel)
        font=ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',15)
        try:
            for j,record in enumerate(self.records):
                support,real,widths=self.batch(record);query,_,_=converter.encode([record['text']])
                fake=generator.generate(generator.compute_style(support),query.to(self.device))
                feature=float(feature_matching_loss(self.discriminator,fake,real,widths));pixel=float(paired_pixel_loss(fake,real,widths))
                prediction=decoded(ocr(fake),widths,self.data.manifest['alphabet'])[0]
                rows.append({**record,'prediction':prediction,'feature_distance':feature,'pixel_l1':pixel})
                draw.text((5,j*110),f"writer {record['writer']} | {record['text']} | OCR {prediction} | real / generated / support",font=font,fill='black')
                for column,tensor in enumerate([real,fake,support[:,0:1,:,:]]):
                    image=Image.fromarray(((tensor[0,0].clamp(-1,1).cpu().numpy()+1)*127.5).astype('uint8'))
                    image=image.resize((min(320,image.width*2),64),Image.Resampling.NEAREST);panel.paste(image,(column*333+5,j*110+25))
        finally:generator.train(previous)
        result={'step':step,'samples':len(rows),'mean_feature_distance':sum(r['feature_distance'] for r in rows)/len(rows),'mean_pixel_l1':sum(r['pixel_l1'] for r in rows)/len(rows),'guidance_ocr_cer':sum(edit_distance(r['text'],r['prediction']) for r in rows)/sum(len(r['text']) for r in rows),'warning':'Fixed source discriminator feature proxy on held-out genuine writers; not a human quality score. OCR remains the training supervisor. No target image or transcript in style support.','predictions':rows}
        (self.output/f'visual_{step:06d}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));panel.save(self.output/f'visual_{step:06d}.png')
        print(json.dumps({k:v for k,v in result.items() if k!='predictions'}),flush=True)
        return result
