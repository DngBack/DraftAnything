"""E9 report: per-action recovery / breakage by E8 failure type, scale repair, and the oracle ceiling of a policy."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_e8 import prog_ok  # noqa: E402
from run_e8 import read_recall, safe_eval  # noqa: E402


def has_expr(t):
    return t is not None and safe_eval(t) is not None and any(o in t for o in "+-*/")


def value_of(text, r):
    """Realistic routing used everywhere: executable expression -> Python, else the direct answer."""
    return prog_ok(text, r, False) if has_expr(text) else r["direct_v"]


def failure_type(r):
    rd = max(r["read_recall"], read_recall(r["program"], r["facts"])) >= 1
    if not rd:
        return "read"
    return "reason" if r["answer_type"] == "arithmetic" else "span"


def main(tag, prefix):
    B = {json.loads(l)["qid"]: json.loads(l) for l in open(ROOT / "results" / f"e8_{tag}.jsonl")}
    A = [json.loads(l) for l in open(ROOT / "results" / f"{prefix}_{tag}.jsonl")]
    R = []
    for a in A:
        r = dict(B[a["qid"]])
        r.update(a)
        r["v_base"] = value_of(r["program"], r)
        r["v_zoom"] = value_of(r["zoom"], r) if r["zoom"] else r["v_base"]
        r["v_verify"] = value_of(r["verify"], r)
        r["ftype"] = failure_type(r)
        R.append(r)
    n = len(R)
    L = [f"# E9 — hành động sửa lỗi có mục tiêu, {tag}", "", f"n = {n} câu dev. Base = route thực tế của E8 "
         "(biểu thức tính được thì dùng Python, không thì dùng `direct`).", ""]
    # scale
    S = [r for r in R if r["v_base"]]
    L += ["## Scale (trên các câu mà giá trị đã đúng)", "", "| nguồn scale | đúng scale gold (kể cả rỗng) | trên câu có scale |", "|---|---|---|"]
    for k, nm in (("scale_direct", "từ câu trả lời `direct`"), ("scale_vlm", "hỏi lại VLM (1 lượt)"),
                  ("scale_ocr", "OCR + closure quan hệ (0 lượt)")):
        acc = np.mean([r[k] == r["scale"] for r in S])
        acc_s = np.mean([r[k] == r["scale"] for r in S if r["scale"]])
        L.append(f"| {nm} | {100 * acc:.1f} | {100 * acc_s:.1f} |")
    # official accuracy pipelines
    L += ["", "## Chấm chính thức (giá trị + scale)", "", "| pipeline | lượt gọi VLM | accuracy |", "|---|---|---|"]
    off = lambda v, s, g: bool(v) and s == g  # noqa: E731
    rows = [("direct", 1, [off(r["direct_v"], r["scale_direct"], r["scale"]) for r in R]),
            ("program→Py + scale từ direct", 2, [off(r["v_base"], r["scale_direct"], r["scale"]) for r in R]),
            ("program→Py + scale VLM", 3, [off(r["v_base"], r["scale_vlm"], r["scale"]) for r in R]),
            ("**program→Py + scale OCR-closure**", 2, [off(r["v_base"], r["scale_ocr"], r["scale"]) for r in R]),
            ("program→Py + scale OCR-closure + verify", 3, [off(r["v_verify"], r["scale_ocr"], r["scale"]) for r in R]),
            ("program→Py + scale OCR-closure + zoom", 3, [off(r["v_zoom"], r["scale_ocr"], r["scale"]) for r in R]),
            ("oracle: tốt nhất trong {base, zoom, verify} + scale OCR", 4,
             [off(r["v_base"] or r["v_zoom"] or r["v_verify"], r["scale_ocr"], r["scale"]) for r in R]),
            ("oracle: giá trị đúng + scale gold", 2, [bool(r["v_base"]) for r in R])]
    for nm, c, v in rows:
        L.append(f"| {nm} | {c} | {100 * np.mean(v):.1f} |")
    # recovery / breakage of value actions
    L += ["", "## Hành động trên giá trị: cứu / làm hỏng theo loại lỗi của base", "",
          "| hành động | cứu: read | cứu: reason | cứu: span | làm hỏng câu đúng | thay đổi ròng |", "|---|---|---|---|---|---|"]
    for k in ("v_zoom", "v_verify"):
        rec = defaultdict(list)
        for r in R:
            if not r["v_base"]:
                rec[r["ftype"]].append(r[k])
        brk = [not r[k] for r in R if r["v_base"]]
        net = np.mean([r[k] for r in R]) - np.mean([r["v_base"] for r in R])
        cells = [f"{sum(rec[t])}/{len(rec[t])}" for t in ("read", "reason", "span")]
        L.append(f"| {k[2:]} | " + " | ".join(cells) + f" | {sum(brk)}/{len(brk)} | {100 * net:+.1f} |")
    (ROOT / "results" / f"{prefix}_{tag}.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="qwen25vl7b")
    ap.add_argument("--prefix", default="e9")
    a = ap.parse_args()
    main(a.tag, a.prefix)
