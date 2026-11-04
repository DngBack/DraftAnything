"""Build the IAM line manifest + split audit (plan §3.1-3.2).

Expects the Zenodo tars extracted under IAM_ROOT: xml/, lines/, ascii/lines.txt, task/.
Writes data/manifest.csv and reports/audit/split_audit.{json,md}.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image

from hwline.iam import line_image_path, md5, parse_form, read_split


def read_lines_txt(path: Path) -> dict:
    """ascii/lines.txt: id seg threshold n_cmp x y w h transcript."""
    out = {}
    for ln in path.read_text().splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        p = ln.split(" ")
        out[p[0]] = {"seg": p[1], "x": int(p[4]), "y": int(p[5]), "w": int(p[6]), "h": int(p[7])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iam-root", type=Path, default=Path("/home/jovyan/bachdx2/datasets/iam"))
    ap.add_argument("--out", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--report", type=Path, default=Path("reports/audit"))
    a = ap.parse_args()
    split = read_split(a.iam_root / "task")
    ltxt = read_lines_txt(a.iam_root / "ascii" / "lines.txt")

    rows = []
    for xp in sorted((a.iam_root / "xml").glob("*.xml")):
        for r in parse_form(xp):
            lid = r["line_id"]
            img = line_image_path(a.iam_root / "lines", lid)
            if not img.exists():
                continue
            W, H = Image.open(img).size
            lt = ltxt[lid]
            flags = []
            if r["segmentation"] != "ok":
                flags.append("seg_err")
            if (lt["w"], lt["h"]) != (W, H):
                flags.append("bbox_size_mismatch")
            boxes, oob = [], False
            for w in r["words"]:
                if w["box"] is None:
                    boxes.append(None)
                    continue
                x0, y0, x1, y1 = w["box"]
                b = [x0 - lt["x"], y0 - lt["y"], x1 - lt["x"], y1 - lt["y"]]
                oob |= b[0] < -2 or b[1] < -2 or b[2] > W + 2 or b[3] > H + 2
                boxes.append([max(b[0], 0), max(b[1], 0), min(b[2], W), min(b[3], H)])
            if oob:
                flags.append("word_box_oob")
            if any(b is None for b in boxes):
                flags.append("word_without_box")
            rows.append({
                "image_path": str(img), "line_id": lid, "form_id": r["form_id"],
                "writer_id": r["writer_id"], "transcript": r["transcript"],
                "split": split.get(lid, "unassigned"), "orig_height": H, "orig_width": W,
                "word_texts": json.dumps([w["text"] for w in r["words"]]),
                "word_boxes": json.dumps(boxes), "threshold": r["threshold"],
                "quality_flag": "|".join(flags) or "ok", "parent_source_id": r["form_id"],
                "md5": md5(img),
            })
    df = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)

    # ---- audit
    a.report.mkdir(parents=True, exist_ok=True)
    au = {"n_lines_total": len(df), "split_ids_listed": len(split),
          "split_ids_missing_image": len(set(split) - set(df.line_id))}
    for s, g in df.groupby("split"):
        au[f"{s}"] = {"lines": len(g), "writers": g.writer_id.nunique(), "forms": g.form_id.nunique(),
                      "quality_ok": int((g.quality_flag == "ok").sum())}
    sp = [s for s in ["train", "val1", "val2", "test"] if s in set(df.split)]
    for key in ["writer_id", "form_id"]:
        sets = {s: set(df.loc[df.split == s, key]) for s in sp}
        au[f"{key}_overlap"] = {f"{x}&{y}": len(sets[x] & sets[y]) for i, x in enumerate(sp) for y in sp[i + 1:]}
    dup = df[df.duplicated("md5", keep=False)]
    au["duplicate_md5_lines"] = len(dup)
    au["quality_flag_counts"] = df.quality_flag.value_counts().to_dict()
    # writers per form (IAM forms are single-writer) and forms with >1 split
    fs = defaultdict(set)
    for f, s in zip(df.form_id, df.split):
        fs[f].add(s)
    au["forms_spanning_splits"] = sum(len(v) > 1 for v in fs.values())
    (a.report / "split_audit.json").write_text(json.dumps(au, indent=2))
    print(json.dumps(au, indent=2))


if __name__ == "__main__":
    main()
