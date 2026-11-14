"""Evaluate generated images with a separate legacy real-image OCR checkpoint.

The evaluator may know heldout glyphs; it is never used for new G training.
Outputs remain OCR proxies, particularly for standalone uppercase glyphs.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch

from models.OCR_network import CRNN
from research.longtail.prepare import read_records, write_json
from research.longtail.evaluate import evaluate, score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--references', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, default=Path('runs/vatrpp_vn_ocr_train/best.pth'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--targets', default='ỄễỠỡẴẵ')
    p.add_argument('--device', default='cuda'); p.add_argument('--batch-size', type=int, default=32)
    args = p.parse_args()
    if args.output.exists(): p.error('Fresh output required')
    torch.set_num_threads(2)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    alphabet = state['alphabet']
    network = CRNN(SimpleNamespace(alphabet=alphabet, special_alphabet='', resolution=16, device=args.device)).to(args.device).eval()
    network.load_state_dict(state['model']['R'], strict=True)
    references = read_records(args.references); predictions = []
    with torch.no_grad():
        for start in range(0, len(references), args.batch_size):
            rows = references[start:start+args.batch_size]
            arrays = []
            for row in rows:
                with Image.open(row['image_path']) as image:
                    if image.height != 32: raise ValueError('Expected 32px generated image')
                    arrays.append(np.array(image.convert('L'), dtype=np.float32, copy=True))
            batch = torch.ones(len(arrays), 1, 32, max(8, max(a.shape[1] for a in arrays)), device=args.device)
            for i, array in enumerate(arrays): batch[i, 0, :, :array.shape[1]] = torch.from_numpy(array).to(args.device)/127.5-1
            paths = network(batch).argmax(-1).cpu()
            for i, (row, array) in enumerate(zip(rows, arrays)):
                previous, text = None, []
                for token in paths[:array.shape[1]//4+1, i].tolist():
                    if token and token != previous: text.append(alphabet[token-1])
                    previous = token
                predictions.append({'sample_id': row['sample_id'], 'prediction': ''.join(text)})
    args.output.mkdir(parents=True)
    with (args.output/'predictions.jsonl').open('x', encoding='utf-8') as handle:
        for row in predictions: handle.write(json.dumps(row, ensure_ascii=False)+'\n')
    metrics = evaluate(references, predictions, args.targets)
    by_id = {r['sample_id']: r['prediction'] for r in predictions}
    texts = sorted({r['reference'] for r in references})
    if len(texts) <= 20:
        metrics['per_prompt'] = {text: score([{'reference': r['reference'], 'prediction': by_id[r['sample_id']]} for r in references if r['reference'] == text], args.targets) for text in texts}
    metrics['evaluator'] = {'checkpoint': str(args.checkpoint), 'sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                            'step': state['step'], 'training_relation': 'Legacy full-data OCR; evaluation-only, no gradient to new G. May know target handwriting. Selected on legacy development; not a blinded human judgment.'}
    write_json(args.output/'metrics.json', metrics)
    print(json.dumps({k: metrics[k] for k in ('cer', 'word_exact', 'tail_macro_f1', 'samples')}, ensure_ascii=False))


if __name__ == '__main__': main()
