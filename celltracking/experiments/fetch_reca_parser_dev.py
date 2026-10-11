"""Fetch only the predeclared recA parser-development colony via HTTP ranges.

This script does not read or extract the nine reserved recA test colonies.
Run from celltracking/: python experiments/fetch_reca_parser_dev.py
"""

import hashlib
import io
import sys
import time
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.family_set import ROOT

URL = "https://zenodo.org/api/records/268921/files/recA.zip/content"
COLONY = "151027-05"
DEST = ROOT / "data/raw/reca_parser_dev" / COLONY


class HTTPRangeReader(io.RawIOBase):
    def __init__(self, url):
        self.session = requests.Session()
        self.url = url
        response = self.session.head(url, timeout=30)
        response.raise_for_status()
        self.size = int(response.headers["Content-Length"])
        self.position = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = (offset if whence == 0 else
                         self.position + offset if whence == 1 else
                         self.size + offset)
        return self.position

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.position
        if n == 0:
            return b""
        end = min(self.position + n - 1, self.size - 1)
        for attempt in range(8):
            response = self.session.get(self.url, headers={"Range":
                                        f"bytes={self.position}-{end}"}, timeout=90)
            if response.status_code != 429:
                break
            delay = min(5 * (2 ** attempt), 60)
            print(f"Zenodo rate limit; retry after {delay}s", flush=True)
            time.sleep(delay)
        if response.status_code != 206:
            raise RuntimeError(f"Expected partial content, got {response.status_code}")
        data = response.content
        self.position += len(data)
        return data


def main():
    prefix = f"recA/{COLONY}/"
    with zipfile.ZipFile(HTTPRangeReader(URL)) as archive:
        selected = [info for info in archive.infolist()
                    if info.filename.startswith(prefix)
                    and (info.filename.endswith(f"{COLONY}_lin.mat")
                         or info.filename.endswith(f"{COLONY}_Tdata.mat")
                         or "/cellSegmentationData/" in info.filename
                         and info.filename.endswith(".mat"))]
        for info in selected:
            blob = archive.read(info)
            relative = info.filename.removeprefix(prefix)
            path = DEST / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
            print(relative, len(blob), hashlib.sha256(blob).hexdigest(), flush=True)
    print("extracted", len(selected), "files from", COLONY, flush=True)


if __name__ == "__main__":
    main()
