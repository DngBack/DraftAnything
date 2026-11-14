"""Plan or sequentially execute the scratch representation/sampling ablations.

Execution requires real evaluation examples for every requested primary glyph.
No test selection, legacy checkpoint, background queue or automatic promotion.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from research.longtail.prepare import read_records, write_json
from research.longtail.protocol import make_protocol, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--holdout', default='ỄễỠỡẴẵ')
    p.add_argument('--primary', default='ỄỠẴ')
    p.add_argument('--budgets', nargs='+', type=int, default=[0, 1, 5, 20])
    p.add_argument('--seeds', nargs='+', type=int, default=[742, 743, 744])
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--device', default='cpu')
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    if args.output.exists(): p.error('Fresh suite output required')
    if args.steps < 1 or any(b < 0 for b in args.budgets): p.error('Invalid steps/budget')
    if len(set(args.budgets)) != len(args.budgets) or len(set(args.seeds)) != len(args.seeds):
        p.error('Duplicate budget or seed creates colliding output')
    if not args.primary or not set(args.primary) <= set(args.holdout):
        p.error('Primary glyphs must be a nonempty subset of holdout')
    records = read_records(args.records)
    jobs, limitations = [], []
    for budget in args.budgets:
        # Acquisition seed is fixed across optimization seeds so data is comparable.
        protocol = make_protocol(records, args.holdout, budget, 742)
        folder = args.output/f'budget_{budget}'
        protocol_path = folder/'protocol.json'
        write_json(protocol_path, protocol)
        missing = [c for c in args.primary if protocol['evaluation_real_images_per_glyph']['val'][c] == 0]
        if missing: limitations.append({'budget': budget, 'missing_dev_primary_glyphs': missing})
        if protocol['unfilled_budget']: limitations.append({'budget': budget, 'unfilled_real_image_budget': protocol['unfilled_budget']})
        for seed in args.seeds:
            for mode in ('glyph', 'factorized'):
                for sampler in ('natural', 'glyph', 'interaction'):
                    output = folder/f'{mode}_{sampler}_seed{seed}'
                    argv = [sys.executable, '-m', 'research.longtail.train', '--records', str(args.records.resolve()),
                            '--protocol', str(protocol_path.resolve()), '--output', str(output.resolve()),
                            '--mode', mode, '--sampler', sampler, '--seed', str(seed),
                            '--steps', str(args.steps), '--eval-every', str(min(250, args.steps)), '--device', args.device]
                    jobs.append({'argv': argv, 'output': str(output), 'budget': budget, 'seed': seed, 'mode': mode, 'sampler': sampler})
    plan = {'records_sha256': digest(records), 'jobs': jobs, 'limitations': limitations,
            'scientific_ready': not limitations,
            'status': 'planned', 'note': 'Scratch controls; real VATr/WriteViT/CONSTANT and independent evaluation still required.'}
    write_json(args.output/'plan.json', plan)
    if args.execute:
        if limitations:
            plan['status'] = 'not-executed-incomplete-data'
            write_json(args.output/'plan.json', plan)
            raise SystemExit('Incomplete primary evaluation or unfilled budgets; inspect plan.json. No training launched.')
        for job in jobs:
            plan['status'] = 'running'
            plan['current_output'] = job['output']
            write_json(args.output/'plan.json', plan)
            result = subprocess.run(job['argv'], check=False)
            job['returncode'] = result.returncode
            write_json(args.output/'plan.json', plan)
            if result.returncode:
                plan['status'] = 'stopped-after-failed-job'
                write_json(args.output/'plan.json', plan)
                raise SystemExit(result.returncode)
        plan['status'] = 'training-complete-independent-evaluation-pending'
        write_json(args.output/'plan.json', plan)
    print(json.dumps({'jobs': len(jobs), 'scientific_ready': not limitations, 'status': plan['status']}))


if __name__ == '__main__': main()
