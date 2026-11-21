"""Download a pinned, Vietnamese-capable font set from google/fonts (OFL only).

Selection is deterministic: every OFL family whose METADATA.pb lists the
``vietnamese`` subset, pinned to one git commit of google/fonts. Each file is
checked with fontTools to cover the full PartScribe charset; files that do not
are recorded and skipped. Fonts are untrusted data: they are only parsed by
fontTools/Pillow and never executed.

    python scripts/fetch_fonts.py --out data/fonts --manifest data/fonts_manifest.json
"""
import argparse
import hashlib
import json
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from partscribe.charset import ALPHABET  # noqa: E402

COMMIT = '7af0d90b4d532ffa4b041937b560c6df4e7b1e95'
RAW = f'https://raw.githubusercontent.com/google/fonts/{COMMIT}/'
TREE = f'https://api.github.com/repos/google/fonts/git/trees/{COMMIT}?recursive=1'


def get(url, tries=3):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'font-fetch'}), timeout=60) as r:
                return r.read()
        except Exception:
            if k == tries - 1:
                raise


def parse_metadata(text):
    cat = re.findall(r'^category:\s*"([A-Z_]+)"', text, re.M)
    return {'name': (re.findall(r'^name:\s*"([^"]+)"', text, re.M) or [''])[0],
            'category': cat[0] if cat else 'UNKNOWN',
            'subsets': re.findall(r'^subsets:\s*"([^"]+)"', text, re.M)}


def covers(path):
    from fontTools.ttLib import TTFont
    cmap = TTFont(str(path), lazy=True).getBestCmap()
    return [c for c in ALPHABET if ord(c) not in cmap]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--tree-cache', type=Path, help='optional cached JSON of the git tree')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    tree = json.loads(a.tree_cache.read_text() if a.tree_cache else get(TREE))
    assert tree['sha'] == COMMIT and not tree.get('truncated')
    paths = [x['path'] for x in tree['tree']]
    families = sorted({p.split('/')[1] for p in paths if p.startswith('ofl/') and p.endswith('/METADATA.pb')})
    with ThreadPoolExecutor(16) as ex:
        metas = dict(zip(families, ex.map(lambda f: parse_metadata(get(f'{RAW}ofl/{f}/METADATA.pb').decode('utf-8', 'replace')), families)))
    chosen = [f for f in families if 'vietnamese' in metas[f]['subsets']]
    files = [p for p in paths if p.endswith('.ttf') and p.split('/')[0] == 'ofl' and p.split('/')[1] in chosen
             and p.count('/') == 2]

    def fetch(p):
        dst = a.out / p.split('/', 1)[1].replace('/', '__')
        if not dst.exists():
            dst.write_bytes(get(RAW + p))
        return p, dst

    with ThreadPoolExecutor(16) as ex:
        got = list(ex.map(fetch, files))
    records = []
    for p, dst in got:
        fam = p.split('/')[1]
        try:
            missing = covers(dst)
        except Exception as e:  # unreadable font
            missing = [f'ERROR {type(e).__name__}']
        records.append({'family': fam, 'name': metas[fam]['name'], 'category': metas[fam]['category'],
                        'source': RAW + p, 'file': dst.name, 'sha256': hashlib.sha256(dst.read_bytes()).hexdigest(),
                        'full_coverage': not missing, 'missing': ''.join(missing)[:50]})
    a.manifest.write_text(json.dumps({'commit': COMMIT, 'families_with_vietnamese': len(chosen),
                                      'fonts': records}, ensure_ascii=False, indent=1))
    ok = [r for r in records if r['full_coverage']]
    print(f'{len(chosen)} VN families, {len(records)} files, {len(ok)} with full coverage')


if __name__ == '__main__':
    main()
