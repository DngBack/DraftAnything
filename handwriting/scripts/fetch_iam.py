"""Parallel ranged download of the IAM Zenodo record (19003721) with md5 verification.

Zenodo throttles each connection to ~20 KB/s here, so files are fetched as many
concurrent byte ranges. Usage: python scripts/fetch_iam.py OUT_DIR file [file ...]
"""
import hashlib
import json
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REC = "https://zenodo.org/api/records/19003721"
CHUNK = 4 * 1024 * 1024


def fetch_range(url, start, end, path):
    want = end - start + 1
    for _ in range(30):
        have = path.stat().st_size if path.exists() else 0
        if have >= want:
            return
        with open(path, "ab") as fo:
            subprocess.run(["curl", "-sfL", "--max-time", "900", "-r", f"{start + have}-{end}", url],
                           stdout=fo, check=False)
    raise RuntimeError(f"range {start}-{end} incomplete")


def main(out, names):
    out.mkdir(parents=True, exist_ok=True)
    files = {f["key"]: f for f in json.load(urllib.request.urlopen(REC))["files"]}
    for name in names:
        f = files[name]
        size, md5 = f["size"], f["checksum"].split(":")[1]
        dst = out / name
        if dst.exists() and dst.stat().st_size == size:
            print(name, "exists", flush=True)
            continue
        parts = out / f".{name}.parts"
        parts.mkdir(exist_ok=True)
        url = f["links"]["self"]
        jobs = [(s, min(s + CHUNK, size) - 1, parts / f"{i:05d}")
                for i, s in enumerate(range(0, size, CHUNK))]
        with ThreadPoolExecutor(48) as ex:
            list(ex.map(lambda j: fetch_range(url, *j), jobs))
        h = hashlib.md5()
        with open(dst, "wb") as fo:
            for *_, p in jobs:
                b = p.read_bytes()
                h.update(b)
                fo.write(b)
        ok = h.hexdigest() == md5
        print(name, size, "md5", "OK" if ok else f"MISMATCH {h.hexdigest()} != {md5}", flush=True)
        if ok:
            for *_, p in jobs:
                p.unlink()
            parts.rmdir()


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2:])
