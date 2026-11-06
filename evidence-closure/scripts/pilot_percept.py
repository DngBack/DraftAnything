"""Pilot A (docs/protocol_percept.md): is reading stable across prompts when pixels are fixed? No GPU."""
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evc.data import is_num, load_doc, page_words  # noqa: E402

NUM = re.compile(r"\d[\d,]*\.?\d*")
CONST = {"1", "2", "3", "4", "5", "10", "12", "100", "1000"}
REFUSE = re.compile(r"cannot|can't|unable|illegible|not (legible|clear|visible|readable)|too (blurry|small)|blurry|sorry|not possible", re.I)
LEVELS = ("s100", "s60", "s50", "s40", "s30", "photo")


def dig(s):
    return re.sub(r"\D", "", s).lstrip("0") or "0"


def nums(t):
    return [d for d in (dig(m) for m in NUM.findall(t or "")) if d not in CONST]


def ed(a, b):
    p = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        c = [i]
        for j, y in enumerate(b, 1):
            c.append(min(p[j] + 1, c[-1] + 1, p[j - 1] + (x != y)))
        p = c
    return p[-1]


def read_of(f, got):
    """The emitted number nearest to fact f, if within half its length."""
    if not got:
        return None
    r = min(got, key=lambda g: ed(f, g))
    return r if ed(f, r) <= max(1, len(f) // 2) else None


def boot(values, docs, n=1000):
    values, docs = np.asarray(values, float), np.asarray(docs)
    ud = sorted(set(docs))
    idx = {d: np.where(docs == d)[0] for d in ud}
    rng = np.random.default_rng(0)
    out = [values[np.concatenate([idx[ud[k]] for k in rng.choice(len(ud), len(ud))])].mean() for _ in range(n)]
    return values.mean(), *np.percentile(out, [2.5, 97.5])


def pct(t):
    return f"{100 * t[0]:.1f} [{100 * t[1]:.1f}, {100 * t[2]:.1f}]"


def pair_stats(R, a, b):
    dis, docs, ok = [], [], []
    for r in R:
        if r.get(a) is None or r.get(b) is None:
            continue
        ga, gb = nums(r[a]), nums(r[b])
        for f in r["facts"]:
            f = dig(str(f))
            if not re.search(r"\d", str(f)) or f in CONST:
                continue
            x, y = read_of(f, ga), read_of(f, gb)
            if x and y:
                dis.append(float(x != y))
                docs.append(r["uid"])
                ok.append(float(x == f))
    return np.array(dis), docs, np.array(ok)


def page_nums(uid, split, cache={}):
    if uid not in cache:
        words, _ = page_words(load_doc(uid, split))
        cache[uid] = {dig(w["text"]) for w in words if is_num(w["text"])} - CONST
    return cache[uid]


def origin(R, split):
    c = dict(gold=0, page=0, near=0, fab=0)
    rnd = dict(page=0, near=0, n=0)
    rng = random.Random(0)
    for r in R:
        pg = page_nums(r["uid"], split)
        gold = {dig(str(f)) for f in r["facts"]}
        for g in nums(r.get("program")):
            near = any(abs(len(g) - len(p)) <= 1 and ed(g, p) == 1 for p in pg) and len(g) >= 3
            c["gold" if g in gold else "page" if g in pg else "near" if near else "fab"] += 1
            z = str(rng.randint(1, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(len(g) - 1))
            rnd["n"] += 1
            if z in pg:
                rnd["page"] += 1
            elif len(z) >= 3 and any(abs(len(z) - len(p)) <= 1 and ed(z, p) == 1 for p in pg):
                rnd["near"] += 1
    n = max(1, sum(c.values()))
    return {k: 100 * v / n for k, v in c.items()}, 100 * rnd["page"] / max(1, rnd["n"]), 100 * rnd["near"] / max(1, rnd["n"]), n


def main():
    L = ["# Pilot A: bản đọc theo prompt khi pixel giữ nguyên", ""]
    for tag in ("qwen25vl7b", "qwen3vl8b"):
        L += [f"## {tag}", "", "| mức | n fact | M1 bất ổn định | P(đúng \\| đồng ý) | P(đúng \\| không đồng ý) | n đồng ý / không |", "|---|---|---|---|---|---|"]
        for lv in ("s100", "s30"):
            R = [json.loads(l) for l in open(ROOT / "results" / f"attr_{tag}_{lv}.jsonl")]
            R = [r for r in R if r["answer_type"] == "arithmetic"]
            d, docs, ok = pair_stats(R, "program", "opprog")
            agree = ok[d == 0]
            dis = ok[d == 1]
            L.append(f"| {lv} | {len(d)} | {pct(boot(d, docs))} | {100 * agree.mean():.1f} | {100 * dis.mean():.1f} | {len(agree)} / {len(dis)} |")
        e8 = ROOT / "results" / f"e8_{tag}.jsonl"
        R = [json.loads(l) for l in open(e8)]
        R = [r for r in R if r["answer_type"] == "arithmetic"]
        d, docs, ok = pair_stats(R, "program", "extract")
        L += ["", f"E8 dev s100, `extract` so với `program`: n fact = {len(d)}, bất ổn định = {pct(boot(d, docs))}, "
              f"P(đúng | đồng ý) = {100 * ok[d == 0].mean():.1f}, P(đúng | không đồng ý) = {100 * ok[d == 1].mean():.1f}", ""]
        L += ["M3, nguồn gốc số trong `program` (%):", "", "| mức | n số | fact gold | số khác trên trang | cách 1 chữ số | bịa | ngẫu nhiên: trên trang / cách 1 |", "|---|---|---|---|---|---|---|"]
        M4 = []
        for lv in LEVELS:
            R = [json.loads(l) for l in open(ROOT / "results" / f"attr_{tag}_{lv}.jsonl")]
            A = [r for r in R if r["answer_type"] == "arithmetic"]
            o, rp, rn, n = origin(A, "train")
            L.append(f"| {lv} | {n} | {o['gold']:.1f} | {o['page']:.1f} | {o['near']:.1f} | {o['fab']:.1f} | {rp:.1f} / {rn:.1f} |")
            M4.append(f"{lv} {100 * np.mean([bool(REFUSE.search(r['direct'])) for r in R]):.1f}")
        L += ["", "M4, tỷ lệ `direct` từ chối (%): " + ", ".join(M4), ""]
    out = ROOT / "results" / "pilot_percept.md"
    out.write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
