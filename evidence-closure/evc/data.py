"""TAT-DQA loading: questions, OCR words, gold fact words, page rendering."""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
NUM_RE = re.compile(r"^[\(\-–$€£]*\d[\d,]*\.?\d*[\)%]*$")


def is_num(t: str) -> bool:
    return bool(NUM_RE.match(t.strip()))


def load_doc(uid: str, split: str = "dev") -> dict:
    return json.load(open(DATA / f"docs_{split}" / split / f"{uid}.json"))


def page_words(doc: dict, page: int = 0) -> tuple[list[dict], tuple[int, int]]:
    """Flatten blocks of one page into words with char offsets inside their block."""
    p = doc["pages"][page]
    words = []
    for bi, b in enumerate(sorted(p["blocks"], key=lambda b: b["order"])):
        text = b["text"]
        cur = 0
        for w, bb in zip(b["words"]["word_list"], b["words"]["bbox_list"]):
            s = text.find(w, cur)
            if s < 0:
                s = cur
            e = s + len(w)
            cur = e
            words.append(dict(text=w, bbox=[float(v) for v in bb], block=bi,
                              uuid=b["uuid"], s=s, e=e))
    W, H = p["bbox"][2], p["bbox"][3]
    return words, (W, H)


def fact_word_ids(q: dict, words: list[dict]) -> list[list[int]]:
    """For each block_mapping entry, ids of words overlapping its char span."""
    by_uuid = defaultdict(list)
    for i, w in enumerate(words):
        by_uuid[w["uuid"]].append(i)
    out = []
    for m in q["block_mapping"]:
        for uuid, (s, e) in m.items():
            ids = [i for i in by_uuid.get(uuid, []) if words[i]["s"] < e and words[i]["e"] > s]
            out.append(ids)
    return out


def render_page(uid: str, size: tuple[int, int], split: str = "dev") -> Path:
    """Render PDF page 1 so that pixel coords equal OCR coords."""
    import fitz  # pymupdf

    out = DATA / "pages" / f"{uid}.png"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    pdf = fitz.open(DATA / f"docs_{split}" / split / f"{uid}.pdf")
    pg = pdf[0]
    W, H = size
    mat = fitz.Matrix(W / pg.rect.width, H / pg.rect.height)
    pix = pg.get_pixmap(matrix=mat, alpha=False)
    pix.save(out)
    return out


def build_set(n: int = 600, seed: int = 0, split: str = "dev", qfile: str | None = None) -> list[dict]:
    d = json.load(open(DATA / (qfile or f"tatdqa_dataset_{split}.json")))
    items = []
    for e in d:
        if e["doc"]["page"] != 1:
            continue
        uid = e["doc"]["uid"]
        doc = load_doc(uid, split)
        words, size = page_words(doc)
        for q in e["questions"]:
            if not q["block_mapping"]:
                continue
            groups = fact_word_ids(q, words)
            if not groups or any(len(g) == 0 for g in groups):
                continue
            fids = sorted({i for g in groups for i in g})
            if not any(is_num(words[i]["text"]) for i in fids):
                continue
            # fact lives in a table-like line: its block has >= 2 numeric words
            blocks = {words[i]["block"] for i in fids}
            if not any(sum(is_num(w["text"]) for w in words if w["block"] == b) >= 2 for b in blocks):
                continue
            items.append(dict(qid=q["uid"], uid=uid, question=q["question"], answer=q["answer"],
                              answer_type=q["answer_type"], scale=q["scale"], fact_ids=fids,
                              n_facts=len(groups), size=list(size), split=split))
    rng = random.Random(seed)
    by_type = defaultdict(list)
    for it in items:
        by_type[it["answer_type"]].append(it)
    total = len(items)
    chosen = []
    for t, lst in sorted(by_type.items()):
        rng.shuffle(lst)
        chosen += lst[: max(1, round(n * len(lst) / total))]
    rng.shuffle(chosen)
    return chosen[:n], total
