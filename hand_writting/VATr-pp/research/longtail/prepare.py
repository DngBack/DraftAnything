"""Export pixel-audited image records, create protocols, and check run readiness."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import unicodedata

from research.longtail.protocol import audit, make_protocol, ExposurePolicy, digest


def read_records(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    tmp.replace(path)


def export_records(manifest, train_path=None, test_path=None):
    import h5py
    import numpy as np
    result = []
    for file_group, splits, override in [('train', ('train', 'hat_ocr', 'val'), train_path),
                                         ('test', ('test',), test_path)]:
        path = Path(override or manifest['source_'+file_group]).resolve()
        with h5py.File(path, 'r') as handle:
            labels = handle['lbs'][:]
            starts, lengths = handle['lb_seek_idxs'][:], handle['lb_lens'][:]
            image_starts, widths, writers = handle['img_seek_idxs'][:], handle['img_lens'][:], handle['wids'][:]
            for split in splits:
                for index in manifest['indices'].get(split, []):
                    index = int(index)
                    writer = int(writers[index])
                    text = ''.join(chr(int(c)) for c in labels[int(starts[index]):int(starts[index]+lengths[index])])
                    text = unicodedata.normalize('NFC', text)
                    image = np.asarray(handle['imgs'][:, int(image_starts[index]):int(image_starts[index]+widths[index])])
                    # Hash dimensions as well as bytes, consistently across both files.
                    sha = hashlib.sha256(str(image.shape).encode()+image.tobytes()).hexdigest()
                    is_hat = split == 'hat_ocr'
                    if file_group == 'train' and (writer >= 295) != is_hat:
                        raise ValueError('Manifest/source HAT membership mismatch')
                    source = 'HAT' if is_hat else ('test_unknown' if file_group == 'test' else 'VNOnDB' if writer < 106 else 'ViHTGen')
                    result.append({'sample_id': f'{file_group}:{index}',
                                   'split': 'train' if is_hat else split, 'source': source,
                                   'writer': None if is_hat else f'{file_group}:{source}:{writer}',
                                   'text': text, 'pixel_sha256': sha,
                                   'h5_path': str(path), 'h5_index': index})
    return result


def preflight(manifest_path, records_path=None, protocol_path=None):
    checks = []
    def check(name, operation):
        try:
            detail = operation()
            checks.append({'name': name, 'ok': True, 'detail': str(detail)})
        except Exception as exc:
            checks.append({'name': name, 'ok': False, 'detail': f'{type(exc).__name__}: {exc}'})
    manifest = json.loads(Path(manifest_path).read_text())
    def inspect_h5(path):
        import h5py
        with h5py.File(path, 'r') as handle:
            for key in ('imgs', 'lbs', 'wids', 'img_seek_idxs', 'img_lens', 'lb_seek_idxs', 'lb_lens'):
                if key not in handle:
                    raise ValueError('Missing dataset '+key)
            return f'{len(handle["wids"])} rows'
    for key in ('source_train', 'source_test'):
        check(key, lambda key=key: inspect_h5(manifest[key]))
    if records_path is not None and protocol_path is not None:
        check('exposure_policy', lambda: ExposurePolicy(read_records(records_path), json.loads(Path(protocol_path).read_text())).protocol['track'])
    def environment():
        import torch
        return {'torch': torch.__version__, 'cuda_available': torch.cuda.is_available()}
    check('torch', environment)
    return {'ready': all(row['ok'] for row in checks), 'checks': checks,
            'note': 'Preflight is not a completed training experiment.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    export = sub.add_parser('export')
    export.add_argument('--manifest', type=Path, required=True)
    export.add_argument('--train-h5', type=Path)
    export.add_argument('--test-h5', type=Path)
    export.add_argument('--output', type=Path, required=True)
    plan = sub.add_parser('protocol')
    plan.add_argument('--records', type=Path, required=True)
    plan.add_argument('--output', type=Path, required=True)
    plan.add_argument('--holdout', default='ỄễỠỡẴẵ')
    plan.add_argument('--budget', type=int, default=0)
    plan.add_argument('--seed', type=int, default=742)
    ready = sub.add_parser('preflight')
    ready.add_argument('--manifest', type=Path, required=True)
    ready.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'export':
        if args.output.exists():
            parser.error('Output exists; use a new path')
        manifest = json.loads(args.manifest.read_text())
        records = export_records(manifest, args.train_h5, args.test_h5)
        summary = audit(records)  # Fail before writing if duplicate or leaked.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as handle:
            for row in records:
                handle.write(json.dumps(row, ensure_ascii=False)+'\n')
        write_json(args.output.with_suffix('.audit.json'), {'records_sha256': digest(records), 'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(), 'audit': summary})
    elif args.command == 'protocol':
        if args.output.exists():
            parser.error('Output directory exists; use a new path')
        records = read_records(args.records)
        protocol = make_protocol(records, args.holdout, args.budget, args.seed)
        write_json(args.output/'protocol.json', protocol)
        write_json(args.output/'audit.json', audit(records))
    else:
        result = preflight(args.manifest)
        write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result['ready']:
            sys.exit(2)


if __name__ == '__main__':
    main()
