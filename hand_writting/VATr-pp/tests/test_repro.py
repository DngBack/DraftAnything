"""Portability and integrity tests for release assets and path relocation."""
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from data.vietnamese import load_manifest
from repro import paths
from repro.fetch import restore, unpack, sha256


class PortablePathsTests(unittest.TestCase):
    def fixture(self, root):
        (root/'repro').mkdir()
        data = root/'new_data'; data.mkdir()
        for name in ['train.h5', 'test.h5']:
            (data/name).write_bytes(('sample-'+name).encode())
        glyph = root/'glyph.pickle'; glyph.write_bytes(b'test glyph bytes')
        manifest = {'source_train': '/unavailable/original/train.h5', 'source_test': '/unavailable/original/test.h5',
                    'glyph_path': '/unavailable/original/unifont.pickle', 'glyph_sha256': sha256(glyph),
                    'source_files': {name: {'size': 1, 'mtime_ns': 0} for name in ['train.h5','test.h5']},
                    'alphabet': 'aA', 'indices': {}}
        file = root/'manifest.json'; file.write_text(json.dumps(manifest))
        fingerprints = {'manifest_sha256': sha256(file), 'source_files': {name: {'bytes': (data/name).stat().st_size, 'sha256': sha256(data/name)} for name in ['train.h5','test.h5']}}
        (root/'repro/data_fingerprints.json').write_text(json.dumps(fingerprints))
        overlay = {'manifest_sha256': sha256(file), 'glyph_path': str(glyph),
                   'source_train': str(data/'train.h5'), 'source_test': str(data/'test.h5')}
        overlay_path = root/'overlay.json'; overlay_path.write_text(json.dumps(overlay))
        return file, overlay_path, data

    def test_relocation_preserves_manifest_hash_and_checks_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); file, overlay, data = self.fixture(root); before = sha256(file)
            with patch.object(paths, 'ROOT', root), patch.dict(os.environ, {'VATRPP_PATHS_JSON': str(overlay)}):
                manifest = load_manifest(file)
                self.assertEqual(manifest['source_train'], str(data/'train.h5'))
                self.assertEqual(sha256(file), before)
                (data/'train.h5').write_bytes(b'wrong data')
                with self.assertRaises(ValueError): load_manifest(file)

    def test_style_only_inference_does_not_require_hdf5(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); file, overlay, data = self.fixture(root)
            content = json.loads(overlay.read_text()); content.pop('source_train'); content.pop('source_test'); overlay.write_text(json.dumps(content))
            with patch.object(paths, 'ROOT', root), patch.dict(os.environ, {'VATRPP_PATHS_JSON': str(overlay)}):
                self.assertEqual(load_manifest(file, verify_sources=False)['alphabet'], 'aA')
                with self.assertRaises(ValueError): load_manifest(file)

    def test_modified_manifest_is_rejected_even_if_alphabet_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); file, overlay, data = self.fixture(root)
            file.write_text(file.read_text()+' ')
            with patch.object(paths, 'ROOT', root), patch.dict(os.environ, {'VATRPP_PATHS_JSON': str(overlay)}):
                with self.assertRaises(ValueError): load_manifest(file)


class RestoreTests(unittest.TestCase):
    def test_verified_parts_assemble_and_bad_part_leaves_no_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); downloads = root/'downloads'; downloads.mkdir()
            parts = []
            for i, data in enumerate([b'checkpoint-', b'payload']):
                path = downloads/f'model.part{i}'; path.write_bytes(data)
                parts.append({'file_name':path.name,'bytes':len(data),'sha256':sha256(path)})
            whole = b'checkpoint-payload'
            asset = {'install_path':'runs/model.pth','bytes':len(whole),'sha256':hashlib.sha256(whole).hexdigest(),'parts':parts}
            output = restore(asset, downloads, root/'install')
            self.assertEqual(output.read_bytes(), whole)
            output.unlink(); (downloads/'model.part1').write_bytes(b'corrupt')
            with self.assertRaises(ValueError): restore(asset, downloads, root/'install')
            self.assertFalse(output.exists()); self.assertFalse(output.with_name(output.name+'.partial').exists())

    def test_bundle_traversal_is_rejected_before_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); bundle = root/'evil.tar'
            with tarfile.open(bundle,'w') as archive:
                entry = tarfile.TarInfo('../outside.txt'); entry.size=1; archive.addfile(entry, io.BytesIO(b'x'))
            with self.assertRaises(ValueError): unpack(bundle, root/'install')
            self.assertFalse((root/'outside.txt').exists())


if __name__ == '__main__': unittest.main()
