"""Download the pinned AirGapAgent-R files and record their checksums."""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

from .config import HF_REPO_ID, HF_REVISION, RAW_DIR, SPLITS, hf_url

MANIFEST = "MANIFEST.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_all(dest: Path = RAW_DIR, force: bool = False) -> dict:
    """Fetch every split into `dest` (skipping existing files) and write a manifest."""
    dest.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, filename in SPLITS.items():
        path = dest / filename
        if force or not path.exists():
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(hf_url(filename), tmp)
            json.loads(tmp.read_text())  # fail loudly on an HTML error page
            tmp.replace(path)
        files[split] = {"file": filename, "bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {"repo_id": HF_REPO_ID, "revision": HF_REVISION, "files": files}
    (dest / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
