"""Object-centric instances for phase-contrast images: detect every compact object (cells, debris), crop a
window around each, so that one instance is one object instead of a 224 px tile that mixes many cells
with shared nuisance (out-of-focus particles, a large bright artifact, background).

Detection is unsupervised and identical for every image: PHOEBI's illumination correction, local
contrast |g - G_sigma * g|, a global threshold, connected components, and a size filter that drops
specks and the large optical artifact. Objects are ranked by integrated contrast; the top `max_objects`
are kept. No shape filter is applied on purpose: debris occurs in every class, so exact negatives (no
object of a bag lacking k is k) teach the decoder to ignore it.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy import ndimage as ndi

SIGMA_BG = 3.0
THRESH_SD = 3.0
MIN_AREA, MAX_AREA = 25, 4000


def detect(gray: np.ndarray, max_objects: int = 48) -> List[Dict]:
    """gray: illumination-corrected uint8/float image -> list of {cy, cx, area, contrast, elong}."""
    g = gray.astype(np.float32)
    c = np.abs(g - ndi.gaussian_filter(g, SIGMA_BG))
    thr = np.median(c) + THRESH_SD * 1.4826 * np.median(np.abs(c - np.median(c)))
    mask = ndi.binary_closing(c > thr, iterations=2)
    lab, n = ndi.label(mask)
    if n == 0:
        return []
    idx = np.arange(1, n + 1)
    area = ndi.sum(mask, lab, idx)
    strength = ndi.sum(c, lab, idx)
    keep = np.nonzero((area >= MIN_AREA) & (area <= MAX_AREA))[0]
    keep = keep[np.argsort(-strength[keep])][:max_objects]
    slices = ndi.find_objects(lab)
    objs = []
    for i in keep:
        sl = slices[i]
        ys, xs = np.nonzero(lab[sl] == i + 1)
        ys = ys + sl[0].start; xs = xs + sl[1].start
        if len(ys) > 2:
            ev = np.linalg.eigvalsh(np.cov(np.vstack([ys, xs]).astype(np.float64)))
            elong = float(np.sqrt(max(ev[1], 1e-6) / max(ev[0], 1e-6)))
        else:
            elong = 1.0
        objs.append({"cy": float(ys.mean()), "cx": float(xs.mean()), "area": float(area[i]),
                     "contrast": float(strength[i]), "elong": elong})
    objs.sort(key=lambda o: -o["contrast"])
    return objs[:max_objects]


def crops(gray_rgb: np.ndarray, objs: List[Dict], size: int = 64) -> np.ndarray:
    """(H, W, 3) image -> (len(objs), size, size, 3) crops centred on objects (edge-padded)."""
    h = size // 2
    pad = np.pad(gray_rgb, ((h, h), (h, h), (0, 0)), mode="edge")
    out = np.empty((len(objs), size, size, 3), dtype=gray_rgb.dtype)
    for i, o in enumerate(objs):
        y, x = int(round(o["cy"])) + h, int(round(o["cx"])) + h
        out[i] = pad[y - h:y + h, x - h:x + h]
    return out
