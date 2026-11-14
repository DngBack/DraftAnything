"""Fixed development writer prompts; guidance OCR scores are a proxy, not independent quality."""
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from data.vietnamese import PackedH5, to_tensor


class FixedBenchmark:
    def __init__(self, manifest, num_examples, output, device, expanded=False):
        self.output, self.device = Path(output), device
        self.output.mkdir(parents=True, exist_ok=True)
        self.alphabet = manifest['alphabet']
        self.prompts = ['Tiếng', 'Việt', 'rất', 'đẹp', 'ma', 'má', 'mà', 'mả', 'mã', 'mạ', 'NGUYỄN', 'Ỹ']
        if expanded:
            self.prompts += ['người', 'trường', 'học', 'đường', 'Ắ', 'Ễ', 'Ỡ', 'Ự']
        if set(''.join(self.prompts)) - set(self.alphabet):
            raise ValueError('Benchmark has unsupported characters')
        dataset = PackedH5(manifest, split='val')
        writers = sorted(dataset.pools)
        self.writers = [w for group in ([w for w in writers if w < 106], [w for w in writers if 106 <= w < 295]) for w in group[:2]]
        if expanded:
            self.writers = writers
        rng = random.Random(20261111)
        self.styles, records = [], []
        refs = Image.new('L', (192*len(self.writers), 36*num_examples), 255)
        for column, writer in enumerate(self.writers):
            candidates = [i for i in dataset.pools[writer] if dataset.labels[i] not in self.prompts]
            chosen = rng.sample(candidates, num_examples) if len(candidates) >= num_examples else rng.choices(candidates, k=num_examples)
            style = []
            for row, index in enumerate(chosen):
                im = dataset.image(index)
                if im.width > 192:
                    im = im.resize((192, 32), Image.Resampling.BICUBIC)
                padded = Image.new('L', (192, 32), 255); padded.paste(im, (0, 0))
                style.append(to_tensor(padded)[0])
                refs.paste(padded, (column*192, row*36))
            self.styles.append(torch.stack(style).unsqueeze(0).to(device))
            records.append({'writer': writer, 'references': [{'index': i, 'text': dataset.labels[i]} for i in chosen]})
        metadata = {'seed': 20261111, 'split': 'val', 'prompts': self.prompts, 'styles': records}
        path = self.output/'benchmark_references.json'
        if path.exists() and json.loads(path.read_text()) != metadata:
            raise ValueError('Fixed benchmark references changed on resume')
        path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        refs.save(self.output/'benchmark_references.png')

    @torch.no_grad()
    def render(self, generator, ocr, converter, step):
        # Evaluation must not advance the training random streams.
        python_state, numpy_state = random.getstate(), np.random.get_state()
        torch_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        previous_mode = generator.training
        try:
            return self._render(generator, ocr, converter, step)
        finally:
            generator.train(previous_mode)
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)

    @torch.no_grad()
    def _render(self, generator, ocr, converter, step):
        from train_vietnamese import decoded, edit_distance, unaccent
        previous_mode = generator.training
        generator.eval()
        font = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf', 15)
        cell_width, cell_height = 224, 116
        panel = Image.new('RGB', (cell_width*len(self.prompts), 30+cell_height*len(self.styles)), 'white')
        draw = ImageDraw.Draw(panel)
        predictions, errors, chars, plain_errors, plain_chars = [], 0, 0, 0, 0
        for column, text in enumerate(self.prompts):
            draw.text((column*cell_width+5, 5), text, fill='black', font=font)
        for row, (writer, style) in enumerate(zip(self.writers, self.styles)):
            memory = generator.compute_style(style)
            for column, text in enumerate(self.prompts):
                query, _, _ = converter.encode([text])
                fake = generator.generate(memory, query.to(self.device))
                prediction = decoded(ocr(fake), [16*len(text)], self.alphabet)[0]
                errors += edit_distance(text, prediction); chars += len(text)
                plain_errors += edit_distance(unaccent(text), unaccent(prediction)); plain_chars += len(unaccent(text))
                im = Image.fromarray(((fake[0, 0].clamp(-1, 1).cpu().numpy()+1)*127.5).astype('uint8'))
                im = im.resize((min(im.width*2, cell_width-8), 64), Image.Resampling.NEAREST)
                x, y = column*cell_width+4, row*cell_height+30
                panel.paste(im, (x, y))
                draw.text((x, y+68), f'w{writer}: {prediction}', fill='black', font=font)
                predictions.append({'writer': writer, 'reference': text, 'prediction': prediction})
        metrics = dict(step=step, samples=len(predictions), guidance_ocr_cer=errors/chars,
                       guidance_ocr_cer_without_diacritics=plain_errors/plain_chars,
                       exact_transcript_rate=sum(r['reference']==r['prediction'] for r in predictions)/len(predictions),
                       warning='Fixed warmup OCR is also the content supervisor; this is not an independent quality metric.', predictions=predictions)
        groups = {'uppercase': lambda s: s.isupper(), 'other': lambda s: not s.isupper()}
        metrics['groups'] = {}
        for name, predicate in groups.items():
            records = [r for r in predictions if predicate(r['reference'])]
            denominator = sum(len(r['reference']) for r in records)
            metrics['groups'][name] = {'samples': len(records), 'cer': sum(edit_distance(r['reference'], r['prediction']) for r in records)/denominator if denominator else None}
        metrics['per_writer'] = {}
        for writer in self.writers:
            records = [r for r in predictions if r['writer'] == writer]
            metrics['per_writer'][str(writer)] = {'cer': sum(edit_distance(r['reference'], r['prediction']) for r in records)/sum(len(r['reference']) for r in records), 'exact_transcript_rate': sum(r['reference'] == r['prediction'] for r in records)/len(records)}
        metrics['per_prompt'] = {}
        for prompt in self.prompts:
            records = [r for r in predictions if r['reference'] == prompt]
            metrics['per_prompt'][prompt] = {'samples': len(records), 'cer': sum(edit_distance(prompt, r['prediction']) for r in records)/(len(prompt)*len(records)), 'exact_transcript_rate': sum(prompt == r['prediction'] for r in records)/len(records)}
        panel.save(self.output/f'benchmark_{step:06d}.png')
        (self.output/f'benchmark_{step:06d}.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({k:v for k,v in metrics.items() if k != 'predictions'}), flush=True)
        generator.train(previous_mode)
        return metrics
