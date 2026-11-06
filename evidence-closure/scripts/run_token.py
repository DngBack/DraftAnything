"""Protocol T outputs (docs/protocol_token.md): same information, different input size or tile count. Resumable."""
import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from evc.data import DATA  # noqa: E402
from evc.degrade import degrade  # noqa: E402
from evc.vlm import load_vlm  # noqa: E402
from run_decide import PROMPTS, gen  # noqa: E402


def scaled(x, k):
    return x.resize((round(x.width * k), round(x.height * k)), Image.BICUBIC)


def plan_qwen(img, uid):
    """(level, cond, max_patches, image)"""
    yield "s100", "k1", None, img
    for lv in ("s40", "s30"):
        x = degrade(img, lv, uid)
        yield lv, "k1", None, x
        yield lv, "k1.5", None, scaled(x, 1.5)
        yield lv, "k2", None, scaled(x, 2)
        yield lv, "full", None, x.resize(img.size, Image.BICUBIC)
        yield lv, "full_nearest", None, x.resize(img.size, Image.NEAREST)


def plan_intern(img, uid):
    for lv in ("s100", "s40", "s30"):
        x = img if lv == "s100" else degrade(img, lv, uid)
        for mp in (1, 4, 12):
            yield lv, f"tile{mp}", mp, x
        if lv != "s100":
            yield lv, "full", 12, x.resize(img.size, Image.BICUBIC)


def size_of(vlm, img):
    """Image tokens (Qwen) or tiles (InternVL) that this image becomes under the current processor settings."""
    p = vlm.proc.image_processor(images=[img], return_tensors="pt")
    if "image_grid_thw" in p:
        return int(p["image_grid_thw"].prod()) // vlm.proc.image_processor.merge_size ** 2
    return int(p["pixel_values"].shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--set", default="token_set.json")
    ap.add_argument("--name", default="token")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    items = json.load(open(DATA / a.set))[: a.limit or None]
    out = ROOT / "results" / f"{a.name}_{a.tag}_0.jsonl"
    done = set()
    if out.exists():
        done = {tuple(json.loads(l)[x] for x in ("qid", "level", "cond")) for l in open(out)}
    vlm = load_vlm(a.model)
    plan = plan_intern if "intern" in a.model.lower() else plan_qwen
    f = open(out, "a")
    t0 = time.time()
    for n, it in enumerate(items):
        img = Image.open(DATA / "pages" / f"{it['uid']}.png").convert("RGB")
        for lv, cond, mp, x in plan(img, it["uid"]):
            if (it["qid"], lv, cond) in done:
                continue
            if mp is not None:
                vlm.proc.image_processor.max_patches = mp
            r = gen(vlm, x, PROMPTS["copy"].format(row=it["row"], col=it["col"], q=it["question"]))
            r.update(qid=it["qid"], uid=it["uid"], level=lv, cond=cond, size=size_of(vlm, x), img_wh=list(x.size))
            f.write(json.dumps(r) + "\n")
            f.flush()
        print(f"[{n + 1}/{len(items)}] {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
