"""Range-fetch only rpsM lineage and centroid features from the Zenodo archive.

Use --dev for the parser-development colony. Use --holdout only after the
evaluation protocol and code have been frozen. Raw images are not fetched.
"""

import argparse
import hashlib
import io
import sys
import time
import zipfile
from collections import OrderedDict
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.family_set import ROOT  # noqa: E402

URL = "https://zenodo.org/api/records/268921/files/rpsM.zip/content"
DEV = "151029_E1-1"
HELD_OUT = ("151029_E1-5", "151029_E1-6", "151101_E2-1",
            "151101_E2-2", "151101_E3-11", "151101_E3-12",
            "151101_E4-12", "151101_E4-19", "151101_E4-20")


class CachedHTTPRangeReader(io.RawIOBase):
    """Seekable remote ZIP reader with bounded 4 MiB HTTP range cache."""

    BLOCK = 4 * 1024 * 1024
    MAX_BLOCKS = 8

    def __init__(self, url):
        self.url = url
        self.session = requests.Session()
        response = self.session.head(url, timeout=30)
        response.raise_for_status()
        self.size = int(response.headers["Content-Length"])
        self.position = 0
        self.cache = OrderedDict()

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

    def _block(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        start = index * self.BLOCK
        end = min(start + self.BLOCK, self.size) - 1
        for attempt in range(8):
            response = self.session.get(self.url, headers={"Range":
                                        f"bytes={start}-{end}"}, timeout=90)
            if response.status_code != 429:
                break
            delay = min(5 * (2 ** attempt), 60)
            print(f"Zenodo rate limit; retry after {delay}s", flush=True)
            time.sleep(delay)
        if response.status_code != 206:
            raise RuntimeError(f"Expected HTTP 206, got {response.status_code}")
        result = response.content
        if len(result) != end - start + 1:
            raise IOError(f"Incomplete HTTP range {start}-{end}")
        self.cache[index] = result
        if len(self.cache) > self.MAX_BLOCKS:
            self.cache.popitem(last=False)
        return result

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.position
        end = min(self.position + n, self.size)
        chunks = []
        while self.position < end:
            block = self._block(self.position // self.BLOCK)
            offset = self.position % self.BLOCK
            take = min(len(block) - offset, end - self.position)
            chunks.append(block[offset:offset + take])
            self.position += take
        return b"".join(chunks)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dev", action="store_true")
    group.add_argument("--holdout", action="store_true")
    args = parser.parse_args()
    names = (DEV,) if args.dev else HELD_OUT
    base = ROOT / "data/raw" / ("rpsm_parser_dev" if args.dev else "rpsm_holdout")
    with zipfile.ZipFile(CachedHTTPRangeReader(URL)) as archive:
        selected = {}
        for name in names:
            prefix = f"rpsM/{name}/"
            files = [info for info in archive.infolist()
                     if info.filename.startswith(prefix)
                     and (info.filename == f"{prefix}data/{name}_lin.mat"
                          or (info.filename.startswith(prefix + "cellSegmentationData/")
                              and info.filename.endswith(".mat")))]
            if not any(info.filename.endswith(f"{name}_lin.mat") for info in files):
                raise FileNotFoundError(f"Missing lineage for {name}")
            if not any("/cellSegmentationData/" in info.filename for info in files):
                raise FileNotFoundError(f"Missing centroid features for {name}")
            selected[name] = files
        for name, files in selected.items():
            for info in files:
                relative = info.filename.removeprefix(f"rpsM/{name}/")
                path = base / name / relative
                if path.exists() and path.stat().st_size == info.file_size:
                    continue
                blob = archive.read(info)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(blob)
                print(name, relative, len(blob), hashlib.sha256(blob).hexdigest(),
                      flush=True)
            print(name, len(files), "files complete", flush=True)


if __name__ == "__main__":
    main()
