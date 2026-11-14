"""Run a portable native recipe; --execute starts training, default prints argv."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from repro.paths import ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    recipes = json.loads((ROOT/'repro/recipes.json').read_text())['recipes']
    if args.recipe not in recipes:
        p.error('Known recipes: '+', '.join(sorted(recipes)))
    if args.output.exists():
        p.error('Use a fresh output directory; do not overwrite released checkpoints')
    if not os.environ.get('VATRPP_PATHS_JSON'):
        p.error('First run python -m repro.configure and export VATRPP_PATHS_JSON')
    config = recipes[args.recipe]
    command = [sys.executable, '-u', str(ROOT/'train_vietnamese.py')]
    command += [str(value).replace('{root}', str(ROOT)).replace('{output}', str(args.output.resolve())) for value in config['argv']]
    command += ['--device', args.device]
    print(json.dumps({'recipe': args.recipe, 'command': command, 'note': config['note']}, indent=2))
    if args.execute:
        # Frozen panel OCR is copied before trainer startup, as in the originals.
        import shutil
        benchmark = ROOT/config['benchmark_ocr'] if config.get('benchmark_ocr') else None
        if benchmark:
            if not benchmark.exists():
                raise FileNotFoundError('Fetch benchmark_ocr artifact first')
            args.output.mkdir(parents=True)
            shutil.copy2(benchmark, args.output/'benchmark_ocr.pth')
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
