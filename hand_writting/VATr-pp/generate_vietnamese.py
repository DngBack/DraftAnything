"""Generate NFC Vietnamese text from a native Vietnamese VATr checkpoint."""
import argparse
import json
import random
import unicodedata
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data.vietnamese import PackedH5, load_manifest, to_tensor
from models.model import Generator
from models.OCR_network import strLabelConverter
from train_vietnamese import model_args, seed_all


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--text', required=True)
    p.add_argument('--output', type=Path, required=True)
    style = p.add_mutually_exclusive_group(required=True)
    style.add_argument('--writer', type=int, help='Original HDF5 writer ID within selected split')
    style.add_argument('--style-folder', type=Path)
    p.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    p.add_argument('--seed', type=int, default=742)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    torch.set_num_threads(4)
    seed_all(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint.get('stage') != 'gan':
        p.error('Expected a GAN checkpoint, not an OCR checkpoint')
    original_manifest = Path(checkpoint['config']['manifest'])
    manifest_path = args.manifest or (original_manifest if original_manifest.exists() else Path('files/vietnamese/manifest.json'))
    manifest = load_manifest(manifest_path, verify_sources=not bool(args.style_folder))
    if manifest['alphabet'] != checkpoint['alphabet']:
        raise ValueError('Checkpoint and manifest alphabets differ')
    cfg = model_args(manifest, args.device, not checkpoint['config']['legacy_generator_layout'])
    cfg.correct_style_layout = not checkpoint['config']['legacy_generator_layout']
    model = Generator(cfg).to(args.device).eval()
    model.load_state_dict(checkpoint['model']['G'], strict=True)
    converter = strLabelConverter(checkpoint['alphabet'])
    text = unicodedata.normalize('NFC', args.text)
    words = text.split()
    if not words:
        raise ValueError('Text is empty')
    converter.encode(words)  # Validate before loading style; never silently drop accents.
    n = checkpoint['config']['num_examples']
    if args.style_folder:
        paths = sorted(p for p in args.style_folder.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
        if not paths:
            raise ValueError('No images in style folder')
        chosen = random.sample(paths, n) if len(paths) >= n else random.choices(paths, k=n)
        images = []
        for path in chosen:
            with Image.open(path) as im:
                im = im.convert('L')
                images.append(im.resize((max(1, round(im.width*32/im.height)), 32), Image.Resampling.BICUBIC))
        references = [str(p.resolve()) for p in chosen]
    else:
        dataset = PackedH5(manifest, split=args.split)
        candidates = [i for i in dataset.pools.get(args.writer, []) if dataset.labels[i] not in words]
        if not candidates:
            raise ValueError('Writer absent or no references with distinct target transcripts')
        chosen = random.sample(candidates, n) if len(candidates) >= n else random.choices(candidates, k=n)
        images = [dataset.image(i) for i in chosen]
        references = [{'index': i, 'label': dataset.labels[i]} for i in chosen]
    tensors = []
    reference_page = Image.new('L', (192, n*36), 255)
    for i, im in enumerate(images):
        if im.width > 192:
            im = im.resize((192, 32), Image.Resampling.BICUBIC)
        padded = Image.new('L', (192, 32), 255)
        padded.paste(im, (0, 0))
        tensors.append(to_tensor(padded)[0])
        reference_page.paste(padded, (0, i*36))
    style_images = torch.stack(tensors).unsqueeze(0).to(args.device)
    results = []
    with torch.no_grad():
        memory = model.compute_style(style_images)
        for word in words:
            query, _, _ = converter.encode([word])
            fake = model.generate(memory, query.to(args.device))
            results.append(Image.fromarray(((fake[0, 0].clamp(-1, 1).cpu().numpy()+1)*127.5).astype('uint8')))
    output = Image.new('L', (sum(im.width for im in results)+16*(len(results)-1), 32), 255)
    x = 0
    for im in results:
        output.paste(im, (x, 0)); x += im.width+16
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.save(args.output)
    reference_page.save(args.output.with_suffix('.references.png'))
    args.output.with_suffix('.json').write_text(json.dumps({'text': text, 'words': words, 'references': references,
        'writer': args.writer, 'split': args.split, 'seed': args.seed, 'checkpoint': str(args.checkpoint.resolve()),
        'checkpoint_step': checkpoint['step']}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
