"""Content check for any line-image list: CER of an off-the-shelf HTR against the transcript.

Used to drop generator outputs whose content is wrong (degenerate or repeated words), which
would otherwise corrupt word segmentation and the dynamics profile. TrOCR-base-handwritten
was fine-tuned on IAM lines, so CER on real IAM lines is optimistic; use it as a filter
reference (same HTR for every source), not as a content-fidelity result.
Writes the list with a `cer` column.
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
from PIL import Image

MODEL = "microsoft/trocr-base-handwritten"


def cer(ref, hyp):
    ref, hyp = ref.strip(), hyp.strip()
    d = np.arange(len(hyp) + 1)
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[-1] / max(len(ref), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    proc = TrOCRProcessor.from_pretrained(MODEL)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL, use_safetensors=True).cuda().eval().half()
    lst = pd.read_csv(a.list, dtype={"writer_id": str})
    hyps = []
    for k in range(0, len(lst), a.batch):
        ims = [Image.open(p).convert("RGB") for p in lst.image_path.iloc[k:k + a.batch]]
        px = proc(images=ims, return_tensors="pt").pixel_values.cuda().half()
        with torch.inference_mode():
            ids = model.generate(px, max_new_tokens=128, num_beams=1)
        hyps += proc.batch_decode(ids, skip_special_tokens=True)
    lst["hyp"] = hyps
    lst["cer"] = [cer(r, h) for r, h in zip(lst.transcript, hyps)]
    lst.to_csv(a.out, index=False)
    q = lst.cer.quantile([0.25, 0.5, 0.75, 0.9]).round(3).to_dict()
    print(json.dumps({"list": a.list, "n": len(lst), "cer_quantiles": q,
                      "share_cer_le_0.2": round(float((lst.cer <= 0.2).mean()), 3)}))


if __name__ == "__main__":
    main()
