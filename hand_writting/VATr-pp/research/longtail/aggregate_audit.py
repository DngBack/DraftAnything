"""Reproducible coarse audit from legacy coverage; no inferred writer counts."""
import argparse
import json
from pathlib import Path
from research.longtail.prepare import write_json
from research.longtail.protocol import groups
from research.vietcompose.orthography import decompose_character


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--coverage', type=Path, default=Path('docs/vietcompose/protocol/coverage.json'))
    p.add_argument('--output', type=Path, default=Path('docs/acl_longtail/aggregate_audit.json'))
    p.add_argument('--targets', default='ỄỠẴẮ')
    args = p.parse_args()
    data = json.loads(args.coverage.read_text())['coverage']
    output = {'source': str(args.coverage), 'unit': 'character occurrences, NOT images or writers',
              'note': 'Split counts before clean composition filtering; source artifact not raw data re-audit.', 'targets': {}}
    for c in args.targets:
        f = decompose_character(c)
        row = {'factors': f.as_dict(), 'occurrences': {}, 'real_transfer_evidence': {}}
        for split, info in data.items(): row['occurrences'][split] = info['characters'].get(c, 0)
        train = data['train']['characters']
        for key in sorted(groups(c)):
            examples = {char: count for char, count in train.items() if key in groups(char)}
            row['real_transfer_evidence'][key] = {'occurrences': sum(examples.values()), 'characters': examples}
        row['zero_real_train'] = row['occurrences']['train'] == 0
        row['missing_real_dev_ground_truth'] = row['occurrences']['val'] == 0
        output['targets'][c] = row
    write_json(args.output, output)
    print(json.dumps({c: {'occurrences': r['occurrences'], 'missing_real_dev_ground_truth': r['missing_real_dev_ground_truth']} for c, r in output['targets'].items()}, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
