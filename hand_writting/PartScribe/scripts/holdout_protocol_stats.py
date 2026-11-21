"""Freeze the held-out-composition protocol (plan 261118_plan_03, stage 0).

Counts, per split, the images whose transcript contains a held-out character,
and checks that every Unicode part (NFD base/mark) of each held-out character
still occurs in the remaining training images. NFD is used only to design and
audit the protocol; the model never receives it.
"""
import argparse, hashlib, json, unicodedata as ud
from collections import Counter, defaultdict
from pathlib import Path

VN_HOLDOUT = {
    'C1_tone_on_new_base': 'èéẻẹỷ',
    'C2_stacked_pair_seen_on_other_base': 'ễỡử',
    'C3_stacked_pair_never_seen': 'ẳ',
}
IAM_HOLDOUT = 'jqJY'


def vn_stats(manifest_path, data_dir):
    import h5py
    m = json.loads(Path(manifest_path).read_text())
    held = set(''.join(VN_HOLDOUT.values()))
    cache, out, remain = {}, {}, Counter()
    for split, idx in m['indices'].items():
        path = Path(data_dir) / ('test.h5' if split == 'test' else 'train.h5')
        if path not in cache:
            with h5py.File(path, 'r') as f:
                cache[path] = (f['lbs'][:], f['lb_seek_idxs'][:], f['lb_lens'][:], f['wids'][:])
        lbs, st, ln, w = cache[path]
        removed, writers, per = [], set(), Counter()
        for i in idx:
            text = ud.normalize('NFC', ''.join(map(chr, lbs[st[i]:st[i] + ln[i]])))
            hit = held & set(text)
            if hit:
                removed.append(int(i)); writers.add(int(w[i])); per.update(hit)
            elif split == 'train':
                remain.update(set(text))
        out[split] = {'images': len(idx), 'with_holdout': len(removed), 'writers': len(writers),
                      'per_char': dict(per),
                      'index_sha256': hashlib.sha256(json.dumps(removed).encode()).hexdigest(),
                      'indices': removed}
    parts = {}
    for c in sorted(held):
        d = ud.normalize('NFD', c)
        marks = list(d[1:])
        parts[c] = {'base': d[0],
                    'mark_images_elsewhere': {f'U+{ord(x):04X}': sum(v for k, v in remain.items() if x in ud.normalize('NFD', k)) for x in marks}}
        if len(marks) == 2:
            parts[c]['pair_images_elsewhere'] = sum(v for k, v in remain.items() if all(x in ud.normalize('NFD', k) for x in marks))
    return out, parts


def iam_stats(gt_dir):
    held, out = set(IAM_HOLDOUT), {}
    for split, name in [('train_val', 'gan.iam.tr_va.gt.filter27'), ('test', 'gan.iam.test.gt.filter27')]:
        removed, writers, per, n = [], set(), Counter(), 0
        for line in (Path(gt_dir) / name).read_text(encoding='utf-8').splitlines():
            head, _, word = line.partition(' ')
            n += 1
            hit = held & set(word)
            if hit:
                removed.append(head); writers.add(head.split(',')[0]); per.update(hit)
        out[split] = {'images': n, 'with_holdout': len(removed), 'writers': len(writers), 'per_char': dict(per),
                      'ids_sha256': hashlib.sha256(json.dumps(removed).encode()).hexdigest(), 'ids': removed}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', default='files/vietnamese/manifest.json')
    ap.add_argument('--data-dir', required=True, help='folder with train.h5/test.h5 matching repro/data_fingerprints.json')
    ap.add_argument('--groundtruth', default='Groundtruth')
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    vn, parts = vn_stats(a.manifest, a.data_dir)
    result = {'version': 1, 'vn_holdout': VN_HOLDOUT, 'iam_holdout': IAM_HOLDOUT,
              'vn': vn, 'vn_parts_in_remaining_train': parts, 'iam': iam_stats(a.groundtruth)}
    Path(a.output).write_text(json.dumps(result, ensure_ascii=False, indent=1))
    for k in ('vn', 'iam'):
        for s, v in result[k].items():
            print(k, s, v['images'], v['with_holdout'], v['writers'], v['per_char'])
    print(json.dumps(parts, ensure_ascii=False))


if __name__ == '__main__':
    main()
