"""Explicit single-image HAT reconstruction, without pseudo-writer supervision.

This auxiliary task intentionally uses the target image as its own style reference.
It can teach glyph appearance but does not establish transfer to unseen writer styles.
"""
import hashlib
import random

import numpy as np
import torch
from PIL import Image

from data.vietnamese import PackedH5, to_tensor


class RareHatContent:
    def __init__(self, manifest, chars='Ễ', seed=742):
        self.data = PackedH5(manifest, include_hat=True)
        self.chars = chars
        self.seed = seed
        self.train_indices, self.eval_indices = [], []
        for index in manifest['indices']['hat_ocr']:
            label = self.data.labels[index]
            if not any(c in label for c in chars):
                continue
            # Group by transcript. This holdout is only for the new auxiliary loss;
            # these images/transcripts may already have appeared in OCR warmup.
            bucket = int(hashlib.sha256(f'{seed}:{label}'.encode()).hexdigest()[:8], 16)%5
            (self.eval_indices if bucket == 0 else self.train_indices).append(index)
        if not self.train_indices or not self.eval_indices:
            raise ValueError('Rare HAT content requires nonempty transcript-grouped train/eval subsets')

    def image(self, index):
        array = np.asarray(self.data.image(index), dtype=np.float32)
        low, high = np.percentile(array, [5, 95])
        if high-low < 1:
            raise ValueError(f'Insufficient contrast in HAT image {index}')
        # Normalize gray HAT backgrounds without changing vertical crop/diacritics.
        array = np.clip((array-low)/(high-low), 0, 1)*255
        return Image.fromarray(array.astype('uint8'))

    def batch(self, batch_size, num_examples, device, indices=None):
        chosen = list(indices) if indices is not None else random.choices(self.train_indices, k=batch_size)
        texts = [self.data.labels[i] for i in chosen]
        widths = [16*len(t) for t in texts]
        target = torch.ones(len(chosen), 1, 32, max(widths))
        support = []
        for j, (index, width) in enumerate(zip(chosen, widths)):
            image = self.image(index)
            target[j, :, :, :width] = to_tensor(image.resize((width, 32), Image.Resampling.BICUBIC))
            if image.width > 192:
                image = image.resize((192, 32), Image.Resampling.BICUBIC)
            padded = Image.new('L', (192, 32), 255)
            padded.paste(image, (0, 0))
            # Repetition is an architectural input shape requirement, not 15
            # distinct examples or an assertion of a shared writer identity.
            support.append(to_tensor(padded)[0].unsqueeze(0).repeat(num_examples, 1, 1))
        return {'target': target.to(device), 'style': torch.stack(support).to(device),
                'texts': texts, 'widths': torch.tensor(widths, device=device), 'indices': chosen}


def rare_reconstruction_loss(fake, target, widths):
    if fake.shape != target.shape:
        raise ValueError(f'Reconstruction shape mismatch: {fake.shape} vs {target.shape}')
    valid = torch.arange(fake.shape[-1], device=fake.device)[None, None, None, :] < widths[:, None, None, None]
    weights = valid*(1+4*(target < .5).to(target.dtype))
    return ((fake-target).abs()*weights).sum()/weights.sum().clamp_min(1)


@torch.no_grad()
def evaluate_rare_reconstruction(content, generator, ocr, converter, output, step, num_examples, device):
    """Monitor auxiliary holdout separately from genuine-writer transfer benchmark."""
    import json
    from pathlib import Path
    from PIL import ImageDraw, ImageFont
    from train_vietnamese import decoded, edit_distance
    previous = generator.training
    generator.eval()
    records, total_loss = [], 0.
    panel = Image.new('RGB', (700, 110*len(content.eval_indices)), 'white')
    draw = ImageDraw.Draw(panel)
    font = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf', 15)
    try:
        for row, index in enumerate(content.eval_indices):
            batch = content.batch(1, num_examples, device, indices=[index])
            query, _, _ = converter.encode(batch['texts'])
            fake = generator.generate(generator.compute_style(batch['style']), query.to(device))
            loss = float(rare_reconstruction_loss(fake, batch['target'], batch['widths']))
            pred = decoded(ocr(fake), batch['widths'], content.data.manifest['alphabet'])[0]
            truth = batch['texts'][0]
            total_loss += loss
            records.append({'index': index, 'reference': truth, 'prediction': pred, 'reconstruction_loss': loss})
            draw.text((5, row*110), f'{truth} | OCR: {pred} | target left / reconstruction right', font=font, fill='black')
            for column, tensor in enumerate([batch['target'], fake]):
                im = Image.fromarray(((tensor[0, 0].clamp(-1, 1).cpu().numpy()+1)*127.5).astype('uint8'))
                im = im.resize((min(335, 2*im.width),64), Image.Resampling.NEAREST)
                panel.paste(im, (column*350+5,row*110+25))
    finally:
        generator.train(previous)
    result = {'step': step, 'samples': len(records), 'mean_reconstruction_loss': total_loss/len(records), 'guidance_ocr_cer': sum(edit_distance(r['reference'], r['prediction']) for r in records)/sum(len(r['reference']) for r in records), 'warning': 'Target image is also self-reference. This is not evidence of transfer to unseen writer styles or an independent evaluation.', 'predictions': records}
    output = Path(output);output.mkdir(parents=True, exist_ok=True)
    (output/f'reconstruction_{step:06d}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    panel.save(output/f'reconstruction_{step:06d}.png')
    return result
