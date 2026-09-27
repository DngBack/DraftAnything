"""Per-cell major axis from the TRA masks (oracle segmentation, like marker_pixels):
unit vector (axis_y, axis_x) and major/minor std along/across it, from second moments.

  uv run python experiments/shape.py  -> outputs/index/one_in_a_million/<seq>/shape.parquet
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage

IDX = Path(__file__).parents[1] / "outputs/index/one_in_a_million"


def frame_shape(job):
    t, path = job
    lab = tifffile.imread(path)
    ids = np.unique(lab)
    ids = ids[ids > 0]
    yy, xx = np.indices(lab.shape, dtype=np.float64)
    n = ndimage.sum_labels(np.ones_like(yy), lab, ids)
    m = {k: ndimage.sum_labels(v, lab, ids) / n
         for k, v in {"y": yy, "x": xx, "yy": yy * yy, "xx": xx * xx, "xy": xx * yy}.items()}
    cov = np.stack([np.stack([m["yy"] - m["y"] ** 2, m["xy"] - m["x"] * m["y"]], -1),
                    np.stack([m["xy"] - m["x"] * m["y"], m["xx"] - m["x"] ** 2], -1)], -2)
    w, v = np.linalg.eigh(cov)  # ascending
    return pd.DataFrame({"frame_index": t, "gt_track_id": ids.astype(int),
                         "axis_y": v[:, 0, 1], "axis_x": v[:, 1, 1],
                         "major_sd": np.sqrt(np.maximum(w[:, 1], 0)),
                         "minor_sd": np.sqrt(np.maximum(w[:, 0], 0))})


if __name__ == "__main__":
    for seq in sorted(p.name for p in IDX.iterdir()):
        fr = pd.read_parquet(IDX / seq / "frames.parquet").dropna(subset=["marker_path"])
        with ProcessPoolExecutor(96) as ex:
            out = pd.concat(ex.map(frame_shape, zip(fr.frame_index, fr.marker_path), chunksize=8))
        out.to_parquet(IDX / seq / "shape.parquet", index=False)
        print(seq, len(out), out.major_sd.median().round(2), out.minor_sd.median().round(2), flush=True)
