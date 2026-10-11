"""Blind test 4: range-fetch lineage and centroid features of the unused van Vliet 2018
subsets (trpL, metA, pheA; Zenodo 268921). Raw images are not fetched.

  python experiments/fetch_vv_blind.py list    # colony names only (zip directory)
  python experiments/fetch_vv_blind.py fetch   # after docs/blind_test4_protocol.md is fixed
"""

import hashlib
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.family_set import ROOT  # noqa: E402
from experiments.fetch_rpsm import CachedHTTPRangeReader  # noqa: E402

SUBSETS = ("trpL", "metA", "pheA")
MAX_COLONIES = None  # all colonies, as fixed in docs/blind_test4_protocol.md


def url(subset):
    return f"https://zenodo.org/api/records/268921/files/{subset}.zip/content"


def colonies(archive, subset):
    """Colonies with both a Schnitzcells lineage file and segmentation features, sorted."""
    names = {}
    for info in archive.infolist():
        parts = info.filename.split("/")
        if len(parts) < 3 or not info.filename.endswith(".mat"):
            continue
        name = parts[-3] if parts[-2] in ("data", "cellSegmentationData") else None
        if name is None:
            continue
        root = info.filename[: info.filename.index(f"/{name}/") + len(name) + 2]
        d = names.setdefault(name, {"root": root, "lin": None, "seg": []})
        if parts[-2] == "data" and parts[-1] == f"{name}_lin.mat":
            d["lin"] = info
        elif parts[-2] == "cellSegmentationData":
            d["seg"].append(info)
    return {n: d for n, d in sorted(names.items()) if d["lin"] is not None and d["seg"]}


def fix_offset(reader, info):
    """Some entries of these archives carry zip64 offsets that are 2**32 too large; locate
    the local header whose stored filename matches."""
    name = info.filename.encode()
    for off in (info.header_offset, info.header_offset - 2 ** 32, info.header_offset + 2 ** 32):
        if off < 0:
            continue
        reader.seek(off)
        head = reader.read(30 + len(name))
        if head[:4] == b"PK\x03\x04" and head[30:30 + len(name)] == name:
            info.header_offset = off
            return
    raise IOError(f"local header not found for {info.filename}")


def read_entry(reader, info):
    """Read one member from its local header, bypassing zipfile's overlap check (which the
    wrong central-directory offsets trigger)."""
    import struct
    import zlib
    reader.seek(info.header_offset)
    head = reader.read(30)
    n, m = struct.unpack("<HH", head[26:30])
    reader.seek(info.header_offset + 30 + n + m)
    data = reader.read(info.compress_size)
    if info.compress_type == zipfile.ZIP_STORED:
        out = data
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        out = zlib.decompress(data, -15)
    else:
        raise ValueError(f"compression {info.compress_type}")
    if len(out) != info.file_size or zlib.crc32(out) != info.CRC:
        raise IOError(f"CRC/size mismatch for {info.filename}")
    return out


def main():
    mode = sys.argv[1]
    for subset in SUBSETS:
        reader = CachedHTTPRangeReader(url(subset))
        with zipfile.ZipFile(reader) as archive:
            cols = colonies(archive, subset)
            print(subset, len(cols), "colonies:", " ".join(cols), flush=True)
            if mode != "fetch":
                continue
            base = ROOT / "data/raw" / f"vv_{subset}"
            for name in list(cols)[:MAX_COLONIES]:
                d = cols[name]
                for info in [d["lin"]] + d["seg"]:
                    rel = info.filename.removeprefix(d["root"])
                    path = base / name / rel
                    if path.exists() and path.stat().st_size == info.file_size:
                        continue
                    fix_offset(reader, info)
                    blob = read_entry(reader, info)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(blob)
                print(subset, name, 1 + len(d["seg"]), "files", flush=True)


if __name__ == "__main__":
    main()
