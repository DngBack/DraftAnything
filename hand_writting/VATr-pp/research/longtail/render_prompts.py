"""Fixed clean development writer supports for unpaired uppercase prompt evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import torch
from PIL import Image, ImageDraw, ImageFont

from research.longtail.data import read_image, tensor
from research.longtail.models import PairedGenerator, load_glyphs
from research.longtail.prepare import read_records, write_json
from research.longtail.protocol import ExposurePolicy, digest
from research.longtail.panels import font as get_font


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--records', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.output.exists(): p.error('Fresh output required')
    torch.set_num_threads(2)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg, protocol = state['config'], state['protocol']
    records = read_records(args.records); policy = ExposurePolicy(records, protocol)
    if digest(records) != cfg['records_sha256']: raise ValueError('Checkpoint/data mismatch')
    glyphs = load_glyphs(cfg['glyphs']) if cfg['mode'] == 'glyph' else None
    model = PairedGenerator(cfg['mode'], cfg['hidden'], cfg['max_length'], glyphs).to(args.device).eval()
    model.load_state_dict(state['model'], strict=True)
    prompts = ['E', 'Ê', 'Ẽ', 'Ễ', 'Ỡ', 'Ẵ', 'NGUYỄN', 'ễ', 'ỡ', 'ẵ']
    writers = sorted({r['writer'] for r in records if r['split'] == 'val' and r['writer']})
    rows, supports = [], []
    font = get_font(14)
    panel = Image.new('RGB', (len(prompts)*130+170, len(writers)*70+35), 'white')
    draw = ImageDraw.Draw(panel)
    for j, text in enumerate(prompts): draw.text((170+j*130, 5), text, font=font, fill='black')
    args.output.mkdir(parents=True)
    for i, writer in enumerate(writers):
        pool = [r for r in records if r['split'] == 'val' and r['writer'] == writer
                and not policy.held.intersection(r['text']) and r['text'] not in prompts]
        if not pool: raise ValueError('No clean prompt support for '+writer)
        rng = random.Random(742+i)
        chosen = rng.sample(pool, cfg['support_count']) if len(pool) >= cfg['support_count'] else rng.choices(pool, k=cfg['support_count'])
        images = []
        for r in chosen:
            source = read_image(r); source = source.resize((min(source.width, 192), 32), Image.Resampling.BICUBIC)
            image = Image.new('L', (192, 32), 255); image.paste(source, (0, 0)); images.append(tensor(image))
        support = torch.stack(images)[None].to(args.device).expand(len(prompts), -1, -1, -1, -1)
        fake = model(prompts, support)['image'].cpu()
        draw.text((3, 40+i*70), writer, font=font, fill='black')
        supports.append({'writer': writer, 'support_ids': [r['sample_id'] for r in chosen]})
        for j, text in enumerate(prompts):
            width = len(text)*16
            array = ((fake[j, 0, :, :width].clamp(-1, 1)+1)*127.5).numpy().round().astype('uint8')
            image = Image.fromarray(array)
            path = args.output/f'{i:02d}_{j:02d}.png'; image.save(path)
            panel.paste(image.resize((max(width, 32), 48)), (170+j*130, 35+i*70))
            rows.append({'sample_id': f'prompt:{writer}:{text}', 'reference': text, 'writer': writer,
                         'image_path': str(path.resolve()), 'support_ids': supports[-1]['support_ids'],
                         'paired_real_ground_truth': False})
    panel.save(args.output/'panel.png')
    with (args.output/'references.jsonl').open('x', encoding='utf-8') as handle:
        for row in rows: handle.write(json.dumps(row, ensure_ascii=False)+'\n')
    write_json(args.output/'supports.json', supports)
    write_json(args.output/'config.json', {'checkpoint': str(args.checkpoint), 'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
               'checkpoint_step': state['step'], 'prompts': prompts, 'writer_count': len(writers),
               'note': 'Unpaired prompt-only evaluation. Known reference transcript supports legibility scoring, not paired glyph/style fidelity. No target-prompt handwriting enters supports.'})


if __name__ == '__main__': main()
