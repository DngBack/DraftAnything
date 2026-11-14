"""Scratch paired training: glyph control vs explicit component renderer.

No legacy checkpoint, OCR gradient, HAT writer labels, local memory or GAN is
used. This implements the first data/representation ablation, not the full
proposed method or a fair VATr reproduction. Test split is never evaluated here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from PIL import Image

from research.longtail.data import PairedEpisodes, collate
from research.longtail.models import PairedGenerator, load_glyphs
from research.longtail.prepare import read_records, write_json
from research.longtail.protocol import ExposurePolicy, digest


def paired_loss(fake, target, widths):
    widths = torch.as_tensor(widths, device=fake.device)
    mask = (torch.arange(fake.shape[-1], device=fake.device)[None, None, None, :]
            < widths[:, None, None, None]).expand_as(fake)
    return ((fake-target).abs()*mask).sum()/mask.sum().clamp_min(1)


@torch.no_grad()
def evaluate(model, loader, device, output, step):
    model.eval()
    rows, absolute, pixels = [], 0., 0
    folder = output/f'dev_{step:06d}'
    folder.mkdir(parents=True, exist_ok=False)
    for batch in loader:
        fake = model(batch['text'], batch['style'].to(device))['image'].cpu()
        for i, text in enumerate(batch['text']):
            width = batch['width'][i]
            pred = fake[i, 0, :, :width]
            real = batch['image'][i, 0, :, :width]
            error = (pred-real).abs().sum().item()
            absolute += error
            pixels += real.numel()
            image_path = folder/f'{len(rows):06d}.png'
            Image.fromarray(((pred.clamp(-1, 1)+1)*127.5).numpy().round().astype('uint8')).save(image_path)
            # References live beside output, never fed as predictions.
            rows.append({'sample_id': batch['sample_id'][i], 'writer': batch['writer'][i],
                         'reference': text, 'image_path': str(image_path.resolve()),
                         'support_ids': batch['support_ids'][i], 'pixel_l1': error/real.numel()})
    metrics = {'step': step, 'dev_pixel_l1': absolute/pixels, 'samples': len(rows),
               'warning': 'Pixel proxy only; no readability, mark correctness or writer-quality claim.'}
    write_json(folder/'metrics.json', metrics)
    with (folder/'references.jsonl').open('x', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False)+'\n')
    return metrics


def run(args):
    if min(args.steps, args.batch_size, args.eval_every, args.hidden, args.support_count, args.max_length) < 1 or args.lr <= 0:
        raise ValueError('Positive training parameters required')
    if args.output.exists():
        raise ValueError('Output exists; use a fresh output directory')
    if getattr(args, 'foreground_weight', 4.) < 0 or getattr(args, 'ctc_weight', 0.) < 0:
        raise ValueError('Nonnegative loss weights required')
    if getattr(args, 'adversarial_weight', 0.) < 0 or getattr(args, 'feature_weight', 0.) < 0:
        raise ValueError('Nonnegative GAN weights required')
    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    records = read_records(args.records)
    protocol = json.loads(args.protocol.read_text())
    policy = ExposurePolicy(records, protocol)
    # Build and read an episode before creating run output: missing data fails early.
    train = PairedEpisodes(policy, support_count=args.support_count, max_length=args.max_length,
                           sampler=args.sampler, seed=args.seed)
    dev = PairedEpisodes(policy, split='val', support_count=args.support_count,
                         max_length=args.max_length, seed=742, per_writer=getattr(args, 'dev_per_writer', None))
    train[0]
    dev[0]
    glyphs = load_glyphs(args.glyphs) if args.mode == 'glyph' else None
    model = PairedGenerator(args.mode, args.hidden, args.max_length, glyphs).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    recognizer = None
    if getattr(args, 'ocr_checkpoint', None):
        from research.longtail.recognizer import FactorizedCTC
        state = torch.load(args.ocr_checkpoint, map_location='cpu', weights_only=False)
        if state['architecture'] != 'factorized-joint-ctc-v1' or state['config']['protocol_sha256'] != digest(protocol):
            raise ValueError('Recognizer must come from exactly this clean protocol')
        if state['config']['records_sha256'] != digest(records):
            raise ValueError('Recognizer data provenance mismatch')
        if getattr(args, 'ctc_weight', 0.) <= 0:
            raise ValueError('Positive ctc-weight required with recognizer')
        recognizer = FactorizedCTC(state['config']['alphabet'], state['config']['hidden']).to(args.device)
        recognizer.load_state_dict(state['model'], strict=True)
        recognizer.eval()
        for parameter in recognizer.parameters(): parameter.requires_grad_(False)
        del state
    elif getattr(args, 'ctc_weight', 0.) != 0:
        raise ValueError('ctc-weight requires a clean recognizer checkpoint')
    discriminator = d_optimizer = None
    if getattr(args, 'adversarial_weight', 0.) or getattr(args, 'feature_weight', 0.):
        from research.longtail.discriminator import PatchDiscriminator
        discriminator = PatchDiscriminator().to(args.device)
        d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=.0002, betas=(0., .999))
    start_step = 0
    if getattr(args, 'resume', None):
        source = torch.load(args.resume, map_location='cpu', weights_only=False)
        if source['config']['records_sha256'] != digest(records):
            raise ValueError('Resume data mismatch')
        if source['config'].get('ocr_checkpoint') != (str(args.ocr_checkpoint) if getattr(args, 'ocr_checkpoint', None) else None):
            raise ValueError('Resume guidance checkpoint path mismatch')
        if source['protocol'] != protocol:
            raise ValueError('Resume exposure policy mismatch')
        for key in ('mode', 'hidden', 'support_count', 'max_length', 'reconstruction', 'foreground_weight', 'ctc_weight'):
            if source['config'].get(key) != getattr(args, key, None):
                raise ValueError('Resume configuration mismatch: '+key)
        model.load_state_dict(source['model'], strict=True)
        optimizer.load_state_dict(source['optimizer'])
        start_step = source['step']
        if args.steps <= start_step:
            raise ValueError('Total steps must exceed resumed step')
        if discriminator is not None and 'discriminator' in source:
            discriminator.load_state_dict(source['discriminator'])
            d_optimizer.load_state_dict(source['d_optimizer'])
        del source
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(train.weights, max(len(train), args.batch_size), replacement=True, generator=generator)
    loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler, collate_fn=collate, num_workers=0)
    dev_loader = DataLoader(dev, batch_size=args.batch_size, collate_fn=collate, num_workers=0)
    args.output.mkdir(parents=True, exist_ok=False)
    config = {**vars(args), 'records_sha256': digest(records), 'protocol_sha256': digest(protocol),
              'torch': torch.__version__, 'source_sha256': {str(path.resolve().relative_to(Path.cwd().resolve())): hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(__file__), Path('research/longtail/models.py'), Path('research/longtail/recognizer.py'), Path('research/longtail/data.py'), Path('research/longtail/protocol.py'), Path('research/longtail/discriminator.py'), Path('research/vietcompose/renderer.py')]}, 'train_targets': len(train), 'dev_targets': len(dev),
              'train_skipped': train.skipped, 'dev_skipped': dev.skipped,
              'dev_panel': 'all eligible holdout targets plus fixed per-writer controls' if getattr(args, 'dev_per_writer', None) else 'full eligible development',
              'dev_holdout_images': {c: sum(c in r['text'] for r in dev.rows) for c in protocol['holdout']},
              'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'discriminator_parameters': sum(p.numel() for p in discriminator.parameters()) if discriminator else 0,
              'start_step': start_step, 'resume_note': 'Model/optimizer continuation; sampler/support RNG restarted, not bit-exact. New GAN modules may initialize from scratch.' if start_step else None,
              'warning': 'Scratch paired image ablation; parameters/compute not matched; no OCR or style-quality metric.'}
    config = json.loads(json.dumps(config, default=str))
    if args.mode == 'glyph':
        config['glyph_sha256'] = hashlib.sha256(args.glyphs.read_bytes()).hexdigest()
    if recognizer is not None:
        config['ocr_checkpoint_sha256'] = hashlib.sha256(args.ocr_checkpoint.read_bytes()).hexdigest()
        config['warning'] = 'Scratch paired reconstruction plus clean frozen factorized-CTC guidance; guidance OCR cannot be called independent evaluation.'
    write_json(args.output/'config.json', config)
    write_json(args.output/'protocol.json', protocol)
    # Preserve exact source bytes for subsequent runs, not just hashes.
    for source_path in config['source_sha256']:
        source_path = Path(source_path)
        destination = args.output/'code'/source_path.resolve().relative_to(Path.cwd().resolve())
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source_path.read_bytes())
    baseline = evaluate(model, dev_loader, args.device, args.output, start_step)
    iterator = iter(loader)
    begun, best = time.monotonic(), baseline['dev_pixel_l1']
    for step in range(start_step+1, args.steps+1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        model.train()
        fake = model(batch['text'], batch['style'].to(args.device))['image']
        target = batch['image'].to(args.device)
        if getattr(args, 'reconstruction', 'l1') == 'balanced-bce':
            ink = (1-target)/2
            prediction = ((1-fake)/2).clamp(1e-5, 1-1e-5)
            valid = (torch.arange(fake.shape[-1], device=args.device)[None, None, None, :]
                     < torch.tensor(batch['width'], device=args.device)[:, None, None, None]).expand_as(fake)
            weight = valid*(1+args.foreground_weight*ink)
            reconstruction = (F.binary_cross_entropy(prediction, ink, reduction='none')*weight).sum()/weight.sum().clamp_min(1)
        else:
            reconstruction = paired_loss(fake, target, batch['width'])
        content = fake.sum()*0
        if recognizer is not None:
            from research.longtail.recognizer import ctc_loss
            content = ctc_loss(recognizer, fake, batch['text'], batch['width'])
        adversarial = feature = fake.sum()*0
        loss_d = fake.sum()*0
        if discriminator is not None:
            discriminator.train()
            for parameter in discriminator.parameters(): parameter.requires_grad_(True)
            real_score = discriminator(target)
            fake_score = discriminator(fake.detach())
            loss_d = F.relu(1-real_score).mean()+F.relu(1+fake_score).mean()
            d_optimizer.zero_grad(set_to_none=True); loss_d.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 5., error_if_nonfinite=True)
            d_optimizer.step()
            discriminator.eval()
            for parameter in discriminator.parameters(): parameter.requires_grad_(False)
            fake_score, fake_features = discriminator(fake, features=True)
            with torch.no_grad(): _, real_features = discriminator(target, features=True)
            from research.longtail.discriminator import feature_loss
            adversarial = -fake_score.mean()
            feature = feature_loss(fake_features, real_features, batch['width'], fake.shape[-1])
        loss = (reconstruction+getattr(args, 'ctc_weight', 0.)*content
                +getattr(args, 'adversarial_weight', 0.)*adversarial+getattr(args, 'feature_weight', 0.)*feature)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite paired loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        metric = {'step': step, 'loss': loss.detach().item(), 'grad_norm': float(norm),
                  'seconds': time.monotonic()-begun, 'reconstruction': reconstruction.detach().item(), 'ctc': content.detach().item(), 'adversarial': adversarial.detach().item(), 'feature': feature.detach().item(), 'discriminator': loss_d.detach().item()}
        with (args.output/'train.jsonl').open('a') as handle:
            handle.write(json.dumps(metric)+'\n')
        if step % args.eval_every == 0 or step == args.steps:
            metric.update(evaluate(model, dev_loader, args.device, args.output, step))
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'step': step, 'config': config, 'protocol': protocol,
                     'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                             'torch': torch.get_rng_state(), 'sampler': generator.get_state(),
                             'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}}
            if discriminator is not None:
                state['discriminator'] = discriminator.state_dict()
                state['d_optimizer'] = d_optimizer.state_dict()
            temp = args.output/'last.pth.partial'
            torch.save(state, temp)
            temp.replace(args.output/'last.pth')
            if metric['dev_pixel_l1'] < best:
                best = metric['dev_pixel_l1']
                torch.save(state, args.output/'best_pixel_proxy.pth')
            print(json.dumps(metric), flush=True)
    write_json(args.output/'completion.json', {'completed_updates': args.steps,
               'seconds': time.monotonic()-begun, 'best_dev_pixel_proxy': best,
               'status': 'engineering-run-complete; scientific evaluation outstanding'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['glyph', 'factorized'], required=True)
    parser.add_argument('--sampler', choices=['natural', 'glyph', 'interaction'], default='natural')
    parser.add_argument('--glyphs', type=Path, default=Path('files/vietnamese/unifont.pickle'))
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--adversarial-weight', type=float, default=0.)
    parser.add_argument('--feature-weight', type=float, default=0.)
    parser.add_argument('--reconstruction', choices=['l1', 'balanced-bce'], default='l1')
    parser.add_argument('--foreground-weight', type=float, default=4.)
    parser.add_argument('--ocr-checkpoint', type=Path)
    parser.add_argument('--ctc-weight', type=float, default=0.)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--eval-every', type=int, default=250)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--support-count', type=int, default=5)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--max-length', type=int, default=20)
    parser.add_argument('--seed', type=int, default=742)
    parser.add_argument('--dev-per-writer', type=int, default=2, help='Fixed development panel plus all heldout targets; not full development')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default='cpu')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
