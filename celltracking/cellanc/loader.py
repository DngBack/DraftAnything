"""Model-side loader for benchmark v0. Reads only inputs/; never labels/ (doc §10, §15)."""

import json
from pathlib import Path

import pandas as pd
import tifffile


class Windows:
    def __init__(self, root: Path, data_dir: Path, split: str | None = None):
        self.root, self.data_dir = Path(root), Path(data_dir)
        self.windows = [w for f in sorted((self.root / "inputs/windows").glob("*.jsonl"))
                        for w in map(json.loads, f.read_text().splitlines())
                        if split is None or w["split"] == split]
        self._det, self._frames = {}, {}

    def __len__(self):
        return len(self.windows)

    def _seq(self, alias):
        if alias not in self._det:
            self._det[alias] = pd.read_parquet(self.root / "inputs/detections" / f"{alias}.parquet")
            self._frames[alias] = pd.read_parquet(
                self.root / "inputs/frames" / f"{alias}.parquet").set_index("frame_index")
        return self._det[alias], self._frames[alias]

    def load(self, w: dict, images: bool = False) -> dict:
        """Detections (and optionally raw images) for the observed frames of window `w` only."""
        det, frames = self._seq(w["sequence_id"])
        obs = w["observed_frames"]
        d = det[det.frame_index.isin(obs)]
        out = {
            "window": w,
            "detections": {t: g[["local_detection_id", "center_y", "center_x"]].to_numpy()
                           for t, g in d.groupby("frame_index")},
        }
        assert set(out["detections"]) <= set(obs)
        if images:  # read per frame, keep original dtype (doc §14)
            out["images"] = {t: tifffile.imread(self.data_dir / frames.at[t, "image_path"])
                             for t in obs}
        return out

    def __iter__(self):
        for w in self.windows:
            yield self.load(w)
