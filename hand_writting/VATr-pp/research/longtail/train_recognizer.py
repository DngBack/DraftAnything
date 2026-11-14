"""Train factorized CTC only on exposure-admitted images, never heldout labels."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import time
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image

from research.longtail.data import read_image, tensor
from research.longtail.prepare import read_records, write_json
from research.longtail.protocol import ExposurePolicy, digest
from research.longtail.recognizer import FactorizedCTC, WholeCharacterCTC, GlyphPrototypeCTC, ctc_loss, decode
from research.longtail.evaluate import score


class OCRImages(Dataset):
    def __init__(self, policy, evaluation=False, per_writer=5):
        rows = ([r for r in policy.records.values() if r['split'] == 'val' and r['writer']]
                if evaluation else policy.training('ocr'))
        self.rows = rows
        if evaluation:
            selected = set()
            for writer in sorted({r['writer'] for r in rows}):
                ids = [i for i, r in enumerate(rows) if r['writer'] == writer]
                random.Random(742).shuffle(ids); selected.update(ids[:per_writer])
            selected.update(i for i, r in enumerate(rows) if policy.held.intersection(r['text']))
            self.rows = [r for i, r in enumerate(rows) if i in selected]
        if not self.rows: raise ValueError('Empty OCR dataset')

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]; image = read_image(row)
        minimum = len(row['text'])+sum(a == b for a, b in zip(row['text'], row['text'][1:]))
        width = max(image.width, 4*minimum, 8)
        repaired = width != image.width
        image = image.resize((width, 32), Image.Resampling.BICUBIC)
        return {'image': tensor(image), 'width': width, 'text': row['text'], 'repaired': repaired,
                'sample_id': row['sample_id'], 'writer': row['writer']}


def collate(rows):
    images = torch.ones(len(rows), 1, 32, max(r['width'] for r in rows))
    for i, r in enumerate(rows): images[i, :, :, :r['width']] = r['image']
    return {'image': images, **{k: [r[k] for r in rows] for k in ('width', 'text', 'repaired', 'sample_id', 'writer')}}


@torch.no_grad()
def evaluate(network, loader, device, held):
    network.eval(); rows = []
    for batch in loader:
        predictions = decode(network(batch['image'].to(device)), batch['width'], network.alphabet)
        rows.extend({'reference': text, 'prediction': pred, 'writer': writer, 'sample_id': key}
                    for text, pred, writer, key in zip(batch['text'], predictions, batch['writer'], batch['sample_id']))
    metric = score(rows, held)
    seen = [r for r in rows if not set(held).intersection(r['reference'])]
    unseen = [r for r in rows if set(held).intersection(r['reference'])]
    metric['seen_combination_cer'] = score(seen, held)['cer'] if seen else None
    metric['unseen_combination_cer'] = score(unseen, held)['cer'] if unseen else None
    return metric, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records', type=Path, required=True); p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--manifest', type=Path, default=Path('files/vietnamese/manifest.json'))
    p.add_argument('--output', type=Path, required=True); p.add_argument('--steps', type=int, default=5000)
    p.add_argument('--batch-size', type=int, default=64); p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--eval-every', type=int, default=500); p.add_argument('--seed', type=int, default=742)
    p.add_argument('--mode', choices=['factorized', 'whole', 'glyph'], default='factorized')
    p.add_argument('--glyphs', type=Path, default=Path('files/vietnamese/unifont.pickle'))
    p.add_argument('--device', default='cuda'); args = p.parse_args()
    if args.output.exists(): p.error('Fresh output required')
    if min(args.steps, args.batch_size, args.hidden, args.eval_every) < 1: p.error('Positive parameters required')
    torch.set_num_threads(2); random.seed(args.seed); torch.manual_seed(args.seed)
    records = read_records(args.records); protocol = json.loads(args.protocol.read_text())
    policy = ExposurePolicy(records, protocol); alphabet = json.loads(args.manifest.read_text())['alphabet']
    train, dev = OCRImages(policy), OCRImages(policy, evaluation=True)
    source_counts = Counter(r['source'] for r in train.rows)
    weights = [1/source_counts[r['source']] for r in train.rows]
    sampler = WeightedRandomSampler(weights, max(len(train), args.batch_size), replacement=True,
                                    generator=torch.Generator().manual_seed(args.seed))
    loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler, collate_fn=collate)
    dev_loader = DataLoader(dev, batch_size=args.batch_size, collate_fn=collate)
    if args.mode == 'glyph':
        from research.longtail.models import load_glyphs
        network = GlyphPrototypeCTC(alphabet, load_glyphs(args.glyphs), args.hidden).to(args.device)
    else:
        network = (FactorizedCTC if args.mode == 'factorized' else WholeCharacterCTC)(alphabet, args.hidden).to(args.device)
    optimizer = torch.optim.Adam(network.parameters(), lr=.001)
    args.output.mkdir(parents=True, exist_ok=False)
    config = {**vars(args), 'alphabet': alphabet, 'protocol_sha256': digest(protocol), 'records_sha256': digest(records),
              'trainable_parameters': sum(p.numel() for p in network.parameters()), 'train_images': len(train), 'dev_panel_images': len(dev), 'source_images': dict(source_counts),
              'warning': 'Clean training recognizer. Development used for diagnostics/selection; never independent evaluation if its gradients later supervise G. Natural widths expanded only when needed for CTC.'}
    write_json(args.output/'config.json', json.loads(json.dumps(config, default=str)))
    batches = iter(loader); start = time.monotonic(); best = float('inf')
    for step in range(1, args.steps+1):
        try: batch = next(batches)
        except StopIteration: batches = iter(loader); batch = next(batches)
        network.train(); loss = ctc_loss(network, batch['image'].to(args.device), batch['text'], batch['width'])
        optimizer.zero_grad(set_to_none=True); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(network.parameters(), 5., error_if_nonfinite=True); optimizer.step()
        if step % 50 == 0:
            row = {'step': step, 'loss': loss.detach().item(), 'grad_norm': float(norm), 'seconds': time.monotonic()-start,
                   'ctc_width_repairs_in_batch': sum(batch['repaired'])}
            with (args.output/'train.jsonl').open('a') as handle: handle.write(json.dumps(row)+'\n')
        if step % args.eval_every == 0 or step == args.steps:
            metric, rows = evaluate(network, dev_loader, args.device, protocol['holdout'])
            metric.update(step=step, seconds=time.monotonic()-start)
            write_json(args.output/f'validation_{step:06d}.json', metric)
            write_json(args.output/f'predictions_{step:06d}.json', rows)
            state = {'architecture': ({'factorized': 'factorized-joint-ctc-v1', 'whole': 'whole-joint-ctc-v1', 'glyph': 'glyph-prototype-ctc-v1'}[args.mode]), 'model': network.state_dict(), 'optimizer': optimizer.state_dict(),
                     'config': json.loads(json.dumps(config, default=str)), 'step': step, 'protocol': protocol}
            torch.save(state, args.output/'last.pth')
            # Choose on seen-combination CER, not on the heldout-combination diagnostic.
            select = metric['seen_combination_cer']
            if select is not None and select < best:
                best = select; torch.save(state, args.output/'best_seen_cer.pth')
            print(json.dumps(metric, ensure_ascii=False), flush=True)
    write_json(args.output/'completion.json', {'updates': args.steps, 'best_seen_dev_cer': best, 'seconds': time.monotonic()-start})


if __name__ == '__main__': main()
