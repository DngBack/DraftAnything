"""Download/import release parts, verify SHA-256, and restore canonical paths.

Drive assets are account-controlled unless the owner changes their sharing.
For those assets, download the folder with your own Drive login, then pass
--downloads /path/to/downloaded/files. Public URLs can use --download.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def verify(path, expected):
    if path.stat().st_size != expected['bytes'] or sha256(path) != expected['sha256']:
        raise ValueError('Size/SHA-256 mismatch: '+str(path))


def safe_destination(root, relative):
    root = Path(root).resolve()
    path = (root/relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError('Artifact path escapes installation root: '+relative)
    return path


def restore(asset, source_dir, root):
    destination = safe_destination(root, asset['install_path'])
    if destination.exists():
        verify(destination, asset)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name+'.partial')
    try:
        with temporary.open('wb') as output:
            for part in asset['parts']:
                source = safe_destination(source_dir, part['file_name'])
                verify(source, part)
                with source.open('rb') as handle:
                    shutil.copyfileobj(handle, output, length=8*1024*1024)
        verify(temporary, asset)
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def unpack(bundle, root):
    with tarfile.open(bundle, 'r:*') as archive:
        members = archive.getmembers()
        # Validate the entire member list before extracting anything.
        for member in members:
            safe_destination(root, member.name)
            if not (member.isfile() or member.isdir()):
                raise ValueError('Bundle may contain only regular files/directories')
        for member in members:
            path = safe_destination(root, member.name)
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            data = source.read()
            if path.exists():
                if path.read_bytes() != data:
                    raise ValueError('Refusing to overwrite different local file: '+str(path))
            else:
                path.write_bytes(data)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog', type=Path, default=ROOT/'repro/artifacts.json')
    p.add_argument('--id', nargs='+', help='Artifact IDs; omit with --list to inspect')
    p.add_argument('--list', action='store_true')
    p.add_argument('--downloads', type=Path, help='Directory with manually downloaded Drive parts')
    p.add_argument('--download', action='store_true', help='Fetch URLs; private Drive links still require access')
    p.add_argument('--root', type=Path, default=ROOT)
    args = p.parse_args()
    catalog = json.loads(args.catalog.read_text())
    lookup = {a['id']: a for a in catalog['assets']}
    if args.list:
        for key, asset in lookup.items():
            print(key, asset['bytes'], asset.get('access', 'unknown'), asset['description'])
        return
    if not args.id or any(key not in lookup for key in args.id):
        p.error('Provide known --id values; use --list')
    cache = args.downloads or args.root/'artifacts/downloads'
    cache.mkdir(parents=True, exist_ok=True)
    for key in args.id:
        asset = lookup[key]
        for part in asset['parts']:
            path = safe_destination(cache, part['file_name'])
            if not path.exists():
                if not args.download:
                    raise FileNotFoundError('Download this part from the catalog with your Drive account: '+part['file_name'])
                if asset.get('access') == 'local-only':
                    raise RuntimeError('This checkpoint is awaiting user-managed upload. Provide the whole file via --downloads; see WEIGHTS.md.')
                if asset.get('access') == 'account-controlled':
                    raise RuntimeError('Account-controlled Drive asset: use Drive UI to download, then --downloads. No authentication token is requested or stored by this script.')
                if part.get('drive_id'):
                    import gdown
                    gdown.download(id=part['drive_id'], output=str(path), quiet=False)
                else:
                    with urllib.request.urlopen(part['download_url']) as source, path.open('wb') as output:
                        shutil.copyfileobj(source, output, length=8*1024*1024)
            verify(path, part)
        destination = restore(asset, cache, args.root)
        if asset['kind'] == 'bundle':
            unpack(destination, args.root)
        print(key, 'verified and restored:', destination)


if __name__ == '__main__':
    main()
