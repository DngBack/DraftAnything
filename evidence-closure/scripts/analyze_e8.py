"""E8 error taxonomy: reading vs calculation vs reasoning, and the ceiling of each fix."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from run_e8 import read_recall, safe_eval  # noqa: E402
from evc.metrics import value_correct  # noqa: E402


def prog_ok(text, r, pct_norm):
    """Score an expression by executing it; pct_norm also accepts x100 when the gold scale is percent."""
    v = safe_eval(text)
    if v is None:
        return value_correct(text, r["answer"])
    cands = [v] + ([v * 100] if pct_norm and r["scale"] == "percent" else [])
    return any(value_correct(f"{c}", r["answer"]) for c in cands)


def boot_diff(a, b, docs, n=2000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ud = sorted(set(docs))
    idx = {d: [i for i, x in enumerate(docs) if x == d] for d in ud}
    rng = np.random.default_rng(seed)
    est = (a - b).mean()
    bs = []
    for _ in range(n):
        s = rng.choice(len(ud), len(ud))
        ii = [i for k in s for i in idx[ud[k]]]
        bs.append((a[ii] - b[ii]).mean())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return f"{100 * est:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"


def main(tag, pct_norm):
    R = [json.loads(l) for l in open(ROOT / "results" / f"e8_{tag}.jsonl")]
    for r in R:
        r["program_v"] = prog_ok(r["program"], r, pct_norm)
        r["hintprog_v"] = prog_ok(r["hintprog"], r, pct_norm)
        r["gold_deriv_v"] = prog_ok(r["derivation"], r, pct_norm) if r["derivation"] else None
        r["read_ok"] = max(r["read_recall"], read_recall(r["program"], r["facts"])) >= 1.0
    mode = "percent-norm (×100 khi scale gold = percent)" if pct_norm else "strict"
    L = [f"# E8 — phân loại lỗi, {tag} — chấm program: {mode}", "", f"n = {len(R)} câu dev, {len({r['uid'] for r in R})} tài liệu.", ""]
    ar = [r for r in R if r["answer_type"] == "arithmetic"]
    ok = [r["gold_deriv_v"] for r in ar if r["gold_deriv_v"] is not None]
    L += [f"Kiểm tra hàm chấm: thực thi `derivation` gold cho đúng đáp án ở {100 * np.mean(ok):.1f} % câu arithmetic "
          f"(n = {len(ok)}).", ""]
    L += ["## Accuracy theo loại câu", "",
          "| loại | n | direct | hint (cho sẵn fact gold) | program→Python | hint+program→Python | đọc đủ fact |",
          "|---|---|---|---|---|---|---|"]
    for t in ("span", "multi-span", "arithmetic", "count", "all"):
        S = R if t == "all" else [r for r in R if r["answer_type"] == t]
        if not S:
            continue
        pr = f"{100 * np.mean([r['program_v'] for r in S]):.1f}" if t == "arithmetic" else "–"
        hp = f"{100 * np.mean([r['hintprog_v'] for r in S]):.1f}" if t == "arithmetic" else "–"
        L.append(f"| {t} | {len(S)} | {100 * np.mean([r['direct_v'] for r in S]):.1f} | "
                 f"{100 * np.mean([r['hint_v'] for r in S]):.1f} | {pr} | {hp} | "
                 f"{100 * np.mean([r['read_ok'] for r in S]):.1f} |")
    # taxonomy of direct errors
    L += ["", "## Phân loại câu `direct` sai", ""]
    cat = Counter()
    for r in R:
        if r["direct_v"]:
            continue
        if not r["read_ok"]:
            cat["đọc sai / chọn nhầm ô (perception, localization)"] += 1
        elif r["answer_type"] == "arithmetic":
            cat["đọc đủ, biểu thức đúng → chỉ tính nhầm (calculation)" if r["program_v"]
                else "đọc đủ, biểu thức sai (reasoning: phép toán / thứ tự / điều kiện)"] += 1
        else:
            cat["đọc đủ, trả lời sai (chọn / định dạng span)"] += 1
    nerr = sum(cat.values())
    L += [f"Tổng câu sai: {nerr} / {len(R)}.", "", "| loại lỗi | số câu | % lỗi |", "|---|---|---|"]
    for k, v in cat.most_common():
        L.append(f"| {k} | {v} | {100 * v / nerr:.1f} |")
    # fixes
    docs = [r["uid"] for r in R]
    ad = [r["uid"] for r in ar]
    best_tool = [r["direct_v"] or r["program_v"] for r in ar]
    L += ["", "## Mức cải thiện tối đa của từng cách sửa (CI 95 % bootstrap theo tài liệu)", "",
          f"- **Bằng chứng đầy đủ** (hint − direct, mọi câu): {boot_diff([r['hint_v'] for r in R], [r['direct_v'] for r in R], docs)}",
          f"- **Thực thi bằng Python** (program − direct, arithmetic): {boot_diff([r['program_v'] for r in ar], [r['direct_v'] for r in ar], ad)}",
          f"- **Oracle chọn đúng giữa direct và program** (arithmetic): {boot_diff(best_tool, [r['direct_v'] for r in ar], ad)}",
          f"- **Bằng chứng đầy đủ + Python** (hintprog − direct, arithmetic): {boot_diff([r['hintprog_v'] for r in ar], [r['direct_v'] for r in ar], ad)}",
          f"- Scale sai dù giá trị đúng: {sum(1 for r in R if r['direct_v'] and r['direct_s'] is False)} câu "
          f"/ {sum(1 for r in R if r['direct_v'] and r['scale'])} câu có scale."]
    # by number of facts
    L += ["", "## Theo số fact", "", "| n_facts | n | direct | hint | đọc đủ |", "|---|---|---|---|---|"]
    for k in sorted({min(r["n_facts"], 4) for r in R}):
        S = [r for r in R if min(r["n_facts"], 4) == k]
        L.append(f"| {k}{'+' if k == 4 else ''} | {len(S)} | {100 * np.mean([r['direct_v'] for r in S]):.1f} | "
                 f"{100 * np.mean([r['hint_v'] for r in S]):.1f} | {100 * np.mean([r['read_ok'] for r in S]):.1f} |")
    (ROOT / "results" / f"e8_{tag}{'_pct' if pct_norm else ''}.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="qwen25vl7b")
    ap.add_argument("--pct", action="store_true")
    a = ap.parse_args()
    main(a.tag, a.pct)
