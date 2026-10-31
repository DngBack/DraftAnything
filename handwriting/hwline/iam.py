"""IAM offline (lines + XML) parsing and the Large Writer Independent split."""
from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

SPLIT_FILES = {
    "train": "trainset.txt",
    "val1": "validationset1.txt",
    "val2": "validationset2.txt",
    "test": "testset.txt",
}


def read_split(task_dir: Path) -> dict[str, str]:
    """line_id -> split name, from the official task files."""
    out: dict[str, str] = {}
    for split, fname in SPLIT_FILES.items():
        for ln in (task_dir / fname).read_text().split():
            ln = ln.strip()
            if not ln:
                continue
            if ln in out:
                raise ValueError(f"{ln} appears in {out[ln]} and {split}")
            out[ln] = split
    return out


def parse_form(xml_path: Path) -> list[dict]:
    """One record per handwritten line; word boxes are in form coordinates."""
    root = ET.parse(xml_path).getroot()
    form_id, writer_id = root.attrib["id"], root.attrib["writer-id"]
    recs = []
    for line in root.iter("line"):
        words = []
        for w in line.iter("word"):
            cmps = [(int(c.attrib["x"]), int(c.attrib["y"]),
                     int(c.attrib["width"]), int(c.attrib["height"])) for c in w.iter("cmp")]
            if cmps:
                x0 = min(c[0] for c in cmps); y0 = min(c[1] for c in cmps)
                x1 = max(c[0] + c[2] for c in cmps); y1 = max(c[1] + c[3] for c in cmps)
                box = (x0, y0, x1, y1)
            else:
                box = None
            words.append({"word_id": w.attrib["id"], "text": w.attrib["text"], "box": box})
        recs.append({
            "line_id": line.attrib["id"], "form_id": form_id, "writer_id": writer_id,
            "transcript": line.attrib["text"],
            "segmentation": line.attrib.get("segmentation", ""),
            "threshold": int(line.attrib.get("threshold", 0)),
            "words": words,
        })
    return recs


def line_image_path(lines_dir: Path, line_id: str) -> Path:
    a, b, _ = line_id.split("-")
    return lines_dir / a / f"{a}-{b}" / f"{line_id}.png"


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()
