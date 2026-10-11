"""Segment raw frames with a generalist segmenter (Cellpose-SAM) for end-to-end evaluation.

Runs in the separate segmentation environment (.venv-seg). Reads a frame list
(alias, frame_index, image_path), writes one label image and one detection table
per frame; detections use the same second-moment shape features as the GT index
(experiments/shape.py), so they plug into the ancestry models unchanged.

  .venv-seg/bin/python experiments/segment_frames.py --frames outputs/seg/frames.csv --out outputs/seg/cpsam
  .venv-omni/bin/python experiments/segment_frames.py --frames outputs/seg/frames.csv --out outputs/seg/omni --model bact_phase_omni
(Omnipose settings: docs/second_segmenter_protocol.md)
"""

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

spec = importlib.util.spec_from_file_location("shape", Path(__file__).parent / "shape.py")
shape = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--model", default="cpsam", help="cpsam (Cellpose-SAM, .venv-seg) or bact_phase_omni (Omnipose, .venv-omni)")
    a = ap.parse_args()
    if a.model == "cpsam":
        from cellpose import models
        model = models.CellposeModel(gpu=True)
        run = lambda im: model.eval(im, batch_size=a.batch)[0]  # noqa: E731
    else:
        from cellpose_omni import models
        model = models.CellposeModel(gpu=True, model_type=a.model)
        omni = dict(channels=[0, 0], rescale=None, mask_threshold=-2, flow_threshold=0, transparency=True, omni=True,
                    cluster=True, resample=True, verbose=False, tile=False, niter=None, augment=False, affinity_seg=False)
        run = lambda im: model.eval(im, batch_size=a.batch, **omni)[0]  # noqa: E731
    jobs = pd.read_csv(a.frames)
    out = Path(a.out)
    t0 = time.time()
    for k, r in enumerate(jobs.itertuples()):
        dst = out / r.alias / f"{r.frame_index:05d}"
        if Path(f"{dst}.parquet").exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        im = tifffile.imread(r.image_path)
        masks = run(im).astype(np.int32)
        tifffile.imwrite(f"{dst}.tif", masks, compression="zlib")
        det = shape.frame_shape((int(r.frame_index), f"{dst}.tif")).rename(columns={"gt_track_id": "label"})
        ids = det.label.to_numpy()
        from scipy import ndimage
        c = np.array(ndimage.center_of_mass(np.ones_like(masks), masks, ids)).reshape(-1, 2)
        det["center_y"], det["center_x"] = c[:, 0], c[:, 1]
        det.to_parquet(f"{dst}.parquet", index=False)
        print(k, r.alias, r.frame_index, len(det), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
