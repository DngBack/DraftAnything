"""Pilot C (docs/protocol_percept_nc.md): how often does a target cell sit in a verifiable column-sum relation on the OCR page?"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evc.data import DATA, is_num, load_doc, page_words  # noqa: E402


def val(t):
    x = re.sub(r"[^\d.]", "", t)
    try:
        v = float(x)
    except ValueError:
        return None
    return -v if t.strip().startswith("(") or t.strip().startswith("-") else v


def columns(words):
    cols = []
    for i, w in enumerate(words):
        if not is_num(w["text"]) or val(w["text"]) is None:
            continue
        for c in cols:
            if abs(words[c[0]]["bbox"][2] - w["bbox"][2]) <= 12:
                c.append(i)
                break
        else:
            cols.append([i])
    return [sorted(c, key=lambda i: words[i]["bbox"][1]) for c in cols]


def covered(words):
    """Word ids taking part in a run a[k..m-1] summing to a[m], tolerance max(1, 0.1 %), total >= 10."""
    out = set()
    for c in columns(words):
        v = [val(words[i]["text"]) for i in c]
        for m in range(2, len(c)):
            for k in range(m - 2, -1, -1):
                s = sum(v[k:m])
                if abs(v[m]) >= 10 and abs(s - v[m]) <= max(1, 0.001 * abs(v[m])):
                    out.update(c[k:m + 1])
                    break
    return out


def main():
    S = json.load(open(DATA / "percept_set.json"))
    hit = 0
    tot = 0
    for it in S:
        words, _ = page_words(load_doc(it["uid"], "train"))
        ids = [i for i, w in enumerate(words) if w["text"] == it["word"] and [round(x) for x in w["bbox"]] == [round(x) for x in it["bbox"]]]
        tot += 1
        hit += bool(ids) and ids[0] in covered(words)
    print(f"target cells in a verifiable column-sum relation: {hit}/{tot} = {100 * hit / tot:.1f} %")


if __name__ == "__main__":
    main()
