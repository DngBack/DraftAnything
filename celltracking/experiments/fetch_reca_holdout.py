"""Fetch the nine predeclared recA test colonies after freezing the protocol.

The parser-development colony 151027-05 is excluded. The archive is accessed
by HTTP ranges, so no microscope images or unrelated colonies are downloaded.
Run from celltracking/: python experiments/fetch_reca_holdout.py
"""

import hashlib
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.family_set import ROOT  # noqa: E402
from experiments.fetch_reca_parser_dev import HTTPRangeReader, URL  # noqa: E402
from experiments.reca_microcolony_external_test import HELD_OUT  # noqa: E402


def main():
    destination = ROOT / "data/raw/reca_holdout"
    selected = {}
    with zipfile.ZipFile(HTTPRangeReader(URL)) as archive:
        for colony in HELD_OUT:
            prefix = f"recA/{colony}/"
            files = [info for info in archive.infolist()
                     if info.filename.startswith(prefix)
                     and (info.filename.endswith(f"{colony}_lin.mat")
                          or ("/cellSegmentationData/" in info.filename
                              and info.filename.endswith(".mat")))]
            if not any(info.filename.endswith(f"{colony}_lin.mat") for info in files):
                raise FileNotFoundError(f"Missing lineage file for {colony}")
            if not any("/cellSegmentationData/" in info.filename for info in files):
                raise FileNotFoundError(f"Missing segmentation data for {colony}")
            selected[colony] = files
        for colony, files in selected.items():
            for info in files:
                relative = info.filename.removeprefix(f"recA/{colony}/")
                path = destination / colony / relative
                if path.exists() and path.stat().st_size == info.file_size:
                    continue
                blob = archive.read(info)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(blob)
                print(colony, relative, len(blob), hashlib.sha256(blob).hexdigest(),
                      flush=True)
            print(colony, len(files), "files complete", flush=True)


if __name__ == "__main__":
    main()
