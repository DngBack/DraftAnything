"""Relocate data paths without changing the byte identity of a legacy manifest."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=32)
def _checked_sha256(filename, size, mtime_ns):
    value = hashlib.sha256()
    with Path(filename).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def digest_file(path):
    path = Path(path).resolve()
    stat = path.stat()
    return _checked_sha256(str(path), stat.st_size, stat.st_mtime_ns)


def relocate_manifest(manifest, manifest_path, overlay_path, verify_sources=True):
    fingerprints = json.loads((ROOT/'repro/data_fingerprints.json').read_text())
    overlay = json.loads(Path(overlay_path).read_text())
    actual = digest_file(manifest_path)
    if actual != fingerprints['manifest_sha256'] or overlay['manifest_sha256'] != actual:
        raise ValueError('Path overlay must match the published original manifest exactly')
    relocated = dict(manifest)
    for key, filename in [('source_train', 'train.h5'), ('source_test', 'test.h5')]:
        candidate = overlay.get(key)
        if verify_sources:
            if not candidate:
                raise ValueError('Data paths not configured; rerun python -m repro.configure --data-dir ...')
            path = Path(candidate).resolve()
            expected = fingerprints['source_files'][filename]
            if path.stat().st_size != expected['bytes'] or digest_file(path) != expected['sha256']:
                raise ValueError('Data content differs from published fingerprint: '+filename)
        if candidate:
            relocated[key] = str(Path(candidate).resolve())
    glyph = Path(overlay['glyph_path']).resolve()
    if digest_file(glyph) != manifest['glyph_sha256']:
        raise ValueError('Relocated Unifont subset differs from checkpoint manifest')
    relocated['glyph_path'] = str(glyph)
    return relocated
