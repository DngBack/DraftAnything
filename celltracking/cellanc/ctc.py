"""Read CTC-format sequences: images, TRA markers, man_track.txt (doc §3)."""

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

_FRAME_RE = re.compile(r"(\d+)\.tif$")


def frame_index(path: Path) -> int:
    """Numeric frame index from t000.tif / t0000.tif / man_track0000.tif (doc §6.1)."""
    m = _FRAME_RE.search(path.name)
    if m is None:
        raise ValueError(f"no frame index in {path}")
    return int(m.group(1))


def list_frames(folder: Path, prefix: str) -> dict[int, Path]:
    """{frame_index: path}, sorted numerically; rejects duplicate indices."""
    out: dict[int, Path] = {}
    for p in folder.glob(f"{prefix}*.tif"):
        t = frame_index(p)
        if t in out:
            raise ValueError(f"duplicate frame {t}: {out[t]} vs {p}")
        out[t] = p
    return dict(sorted(out.items()))


@dataclass
class Sequence:
    dataset: str
    root: Path  # dataset folder containing 0N, 0N_GT
    seq: str  # "01", "02", ...

    @property
    def image_dir(self) -> Path:
        return self.root / self.seq

    @property
    def tra_dir(self) -> Path:
        return self.root / f"{self.seq}_GT" / "TRA"

    @property
    def seg_dir(self) -> Path:
        return self.root / f"{self.seq}_GT" / "SEG"

    def images(self) -> dict[int, Path]:
        return list_frames(self.image_dir, "t")

    def markers(self) -> dict[int, Path]:
        return list_frames(self.tra_dir, "man_track")

    def segs(self) -> dict[int, Path]:
        return list_frames(self.seg_dir, "man_seg") if self.seg_dir.exists() else {}

    def tracks(self) -> dict[int, tuple[int, int, int]]:
        return read_man_track(self.tra_dir / "man_track.txt")


def find_sequences(dataset: str, root: Path) -> list[Sequence]:
    """Every folder <root>/**/NN that has a sibling NN_GT/TRA/man_track.txt."""
    seqs = []
    for txt in sorted(root.rglob("man_track.txt")):
        gt = txt.parent.parent  # .../NN_GT
        if txt.parent.name != "TRA" or not gt.name.endswith("_GT"):
            continue
        seqs.append(Sequence(dataset, gt.parent, gt.name[: -len("_GT")]))
    return seqs


def read_man_track(path: Path) -> dict[int, tuple[int, int, int]]:
    """{L: (B, E, P)}; B and E are inclusive frame indices, P=0 means no parent."""
    tracks: dict[int, tuple[int, int, int]] = {}
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        L, B, E, P = (int(x) for x in line.split())
        if L in tracks:
            raise ValueError(f"{path}:{n}: duplicate track {L}")
        tracks[L] = (B, E, P)
    return tracks


def seg_tra_match(seg_path: Path, tra_path: Path, min_overlap: float = 0.5) -> dict:
    """Match SEG instances to TRA IDs by overlap, checking one-to-one (doc §4).

    A SEG instance matches a TRA ID when that ID covers > min_overlap of the TRA marker.
    """
    seg, tra = tifffile.imread(seg_path), tifffile.imread(tra_path)
    fg = (seg > 0) & (tra > 0)
    pairs, n = np.unique(np.stack([seg[fg], tra[fg]]), axis=1, return_counts=True)
    tra_ids, tra_px = np.unique(tra[tra > 0], return_counts=True)
    seg_ids = np.unique(seg[seg > 0])
    tra_size = dict(zip(tra_ids.tolist(), tra_px.tolist()))
    good = [(s, t) for (s, t), c in zip(pairs.T.tolist(), n.tolist()) if c > min_overlap * tra_size[t]]
    s_count, t_count = Counter(s for s, _ in good), Counter(t for _, t in good)
    one_to_one = sum(s_count[s] == 1 and t_count[t] == 1 for s, t in good)
    return {"frame": frame_index(tra_path), "n_seg": len(seg_ids), "n_tra": len(tra_ids),
            "one_to_one": one_to_one, "tra_unmatched": len(tra_ids) - len(t_count),
            "seg_unmatched": len(seg_ids) - len(s_count),
            "ambiguous": sum(v > 1 for v in s_count.values()) + sum(v > 1 for v in t_count.values())}


def marker_centroids(path: Path) -> dict[int, tuple[float, float, int]]:
    """{track_id: (center_y, center_x, marker_pixels)} for one TRA frame.

    Marker pixel count is only for audit; it is NOT cell area (doc §4).
    """
    lab = tifffile.imread(path)
    ids = np.unique(lab)
    ids = ids[ids > 0]
    if ids.size == 0:
        return {}
    cy_cx = ndimage.center_of_mass(np.ones_like(lab, dtype=np.uint8), lab, ids)
    sizes = ndimage.sum_labels(np.ones_like(lab, dtype=np.uint32), lab, ids)
    return {int(i): (float(c[0]), float(c[1]), int(s)) for i, c, s in zip(ids, cy_cx, sizes)}
