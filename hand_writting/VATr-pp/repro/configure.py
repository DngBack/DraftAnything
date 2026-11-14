"""Create a machine-local overlay; never rewrite the original checkpoint manifest."""
import argparse
import json
from pathlib import Path
import shlex
from repro.paths import ROOT, digest_file, relocate_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=ROOT/'files/vietnamese/manifest.json')
    p.add_argument('--glyphs', type=Path, default=ROOT/'files/vietnamese/unifont.pickle')
    p.add_argument('--data-dir', type=Path, help='Directory containing the original train.h5 and test.h5; omit for style-folder inference only')
    p.add_argument('--output', type=Path, default=ROOT/'.local/repro_paths.json')
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text())
    overlay = {'manifest_sha256': digest_file(args.manifest), 'glyph_path': str(args.glyphs.resolve())}
    if args.data_dir:
        overlay.update(source_train=str((args.data_dir/'train.h5').resolve()), source_test=str((args.data_dir/'test.h5').resolve()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.pending.json')
    temporary.write_text(json.dumps(overlay, indent=2)+'\n')
    try:
        relocate_manifest(manifest, args.manifest, temporary, verify_sources=args.data_dir is not None)
        temporary.replace(args.output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print('export VATRPP_PATHS_JSON='+shlex.quote(str(args.output.resolve())))
    print('Validated original data content.' if args.data_dir else 'Configured style-folder inference only; training still requires --data-dir.')


if __name__ == '__main__':
    main()
