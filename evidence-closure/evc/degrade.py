"""Image-quality levels for the attribution curves. Deterministic per (uid, level)."""
import io
import random

from PIL import Image, ImageFilter

SCALES = {"s100": 1.0, "s60": 0.6, "s50": 0.5, "s40": 0.4, "s30": 0.3, "s20": 0.2}
LEVELS = (*SCALES, "photo")


def degrade(img, level, uid=""):
    if level in SCALES:
        s = SCALES[level]
        if s == 1.0:
            return img
        return img.resize((max(28, round(img.width * s)), max(28, round(img.height * s))), Image.BICUBIC)
    if level == "photo":  # scan/photo-like, at the s60 resolution: small rotation, blur, heavy JPEG
        rng = random.Random(f"{uid}/{level}")
        x = img.resize((round(img.width * 0.6), round(img.height * 0.6)), Image.BICUBIC)
        x = x.rotate(rng.uniform(-2, 2), resample=Image.BICUBIC, expand=True, fillcolor=(235, 235, 235))
        x = x.filter(ImageFilter.GaussianBlur(0.8))
        buf = io.BytesIO()
        x.save(buf, "JPEG", quality=25)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    raise ValueError(level)
