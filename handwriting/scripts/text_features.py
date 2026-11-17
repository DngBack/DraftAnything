"""Linguistic features per IAM word token, from text only (no image, no gap).

For every form: rebuild the handwritten word sequence from the XML, recover sentence
boundaries from ascii/sentences.txt, compute GPT-2 surprisal given all preceding form text,
and a constituency parse (Stanza) for boundary strength.

Writes data/ling/tokens.csv.gz. Safe to run on sealed splits: it never reads an image.
"""
import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

NO_SPACE_BEFORE = {".", ",", ";", ":", "?", "!", ")", "]", "'s", "n't", "'t", "'re", "'ve", "'ll", "'d", "'m"}
NO_SPACE_AFTER = {"(", "["}
CLAUSE_LABELS = {"S", "SBAR", "SBARQ", "SINV", "SQ"}


def read_sentences(path: Path) -> dict[str, list[tuple[int, str]]]:
    """form_id -> [(sentence_idx, token)], in reading order."""
    out = defaultdict(list)
    for ln in path.read_text().splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        p = ln.split(" ", 9)
        form = "-".join(p[0].split("-")[:2])
        sidx = int(p[0].split("-")[2][1:])
        for t in p[9].split("|"):
            out[form].append((sidx, t.replace(" ", "")))
    return out


def read_form(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    rows = []
    for line in root.iter("line"):
        for pos, w in enumerate(line.iter("word")):
            rows.append({"form_id": root.attrib["id"], "writer_id": root.attrib["writer-id"],
                         "line_id": line.attrib["id"], "pos": pos, "word_id": w.attrib["id"],
                         "text": w.attrib["text"], "lob_tag": w.attrib.get("tag", "")})
    return rows


def assign_sentences(rows, sent_tokens):
    """Align on the character stream, so a difference in tokenisation cannot shift labels."""
    owner, n_skip = [], 0
    for sidx, tok in sent_tokens:
        owner.extend([sidx] * len(tok))
    stream = "".join(t for _, t in sent_tokens)
    i = 0
    for r in rows:
        t = r["text"].replace(" ", "")
        j = stream.find(t, i, i + len(t) + 40) if t else -1
        if j < 0:
            r["sent"] = -1
            n_skip += 1
        else:
            r["sent"] = owner[j]
            i = j + len(t)
    return n_skip


def detok(tokens: list[str]):
    """Join tokens into natural text; returns text and each token's (start, end) span."""
    s, spans, quote_open, no_space_next = "", [], False, True
    for t in tokens:
        is_quote = t == '"'
        glue = no_space_next or t in NO_SPACE_BEFORE or t.startswith("'") and len(t) <= 3
        if is_quote and quote_open:
            glue = True
        if not glue:
            s += " "
        spans.append((len(s), len(s) + len(t)))
        s += t
        no_space_next = t in NO_SPACE_AFTER or (is_quote and not quote_open)
        if is_quote:
            quote_open = not quote_open
    return s, spans


@torch.no_grad()
def surprisals(tok, model, device, tokens: list[str], max_len: int = 1000) -> np.ndarray:
    """Bits for each word token given all preceding text; summed over its subword pieces."""
    text, spans = detok(tokens)
    enc = tok(text, return_offsets_mapping=True, truncation=True, max_length=max_len)
    ids = [tok.bos_token_id] + enc["input_ids"]
    offs = enc["offset_mapping"]
    logits = model(torch.tensor([ids], device=device)).logits[0, :-1]
    lp = torch.log_softmax(logits, -1).gather(1, torch.tensor(ids[1:], device=device)[:, None])[:, 0]
    bits = (-lp / np.log(2)).cpu().numpy()
    out = np.zeros(len(tokens))
    k = 0
    for b, (a, e) in zip(bits, offs):
        mid = (a + e) / 2
        while k < len(spans) - 1 and mid >= spans[k][1]:
            k += 1
        out[k] += b
    return out


def tree_features(tree, n: int):
    """Per word: tree path to next word, and whether the word starts a clause."""
    leaves, parents = [], {}

    def walk(node, parent):
        parents[id(node)] = parent
        if node.is_leaf():
            leaves.append(node)
        for c in node.children:
            walk(c, node)

    walk(tree, None)
    if len(leaves) != n:
        return None

    def path(node):
        p = []
        while node is not None:
            p.append(node)
            node = parents[id(node)]
        return p[::-1]

    paths = [path(l) for l in leaves]
    dist, start = [np.nan] * n, [0.0] * n
    for i in range(n - 1):
        a, b = paths[i], paths[i + 1]
        d = 0
        while d < min(len(a), len(b)) and a[d] is b[d]:
            d += 1
        dist[i] = (len(a) - d) + (len(b) - d)
    first_of = {}
    for i, p in enumerate(paths):
        for node in p:
            if id(node) not in first_of:
                first_of[id(node)] = i
    for i, p in enumerate(paths):
        for depth, node in enumerate(p):
            # depth 0-1 are ROOT and the sentence's own top node: not an embedded clause start.
            if depth >= 2 and first_of[id(node)] == i and node.label in CLAUSE_LABELS:
                start[i] = 1.0
    return dist, start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iam-root", type=Path, default=Path("data/iam"))
    ap.add_argument("--out", type=Path, default=Path("data/ling/tokens.csv.gz"))
    ap.add_argument("--model", default="gpt2-large")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)

    import stanza
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model).to(device).eval()
    nlp = stanza.Pipeline("en", processors="tokenize,pos,constituency", tokenize_pretokenized=True,
                          use_gpu=device == "cuda", verbose=False)

    sents = read_sentences(a.iam_root / "ascii" / "sentences.txt")
    xmls = sorted((a.iam_root / "xml").glob("*.xml"))
    if a.limit:
        xmls = xmls[:a.limit]
    all_rows, skipped, total, parse_fail, n_sent = [], 0, 0, 0, 0
    for fi, xp in enumerate(xmls):
        rows = read_form(xp)
        form = rows[0]["form_id"]
        skipped += assign_sentences(rows, sents.get(form, []))
        total += len(rows)
        bits = surprisals(tok, model, device, [r["text"] for r in rows])
        for r, b in zip(rows, bits):
            r["surprisal"], r["tree_dist_next"], r["clause_start"] = float(b), np.nan, 0.0
        # Consecutive tokens of one sentence, in reading order, are parsed together.
        groups, cur = [], []
        for r in rows:
            if cur and r["sent"] != cur[-1]["sent"]:
                groups.append(cur)
                cur = []
            cur.append(r)
        if cur:
            groups.append(cur)
        groups = [g for g in groups if g[0]["sent"] >= 0]
        doc = nlp([[r["text"].replace(" ", "_") for r in g] for g in groups])
        for g, s in zip(groups, doc.sentences):
            n_sent += 1
            res = tree_features(s.constituency, len(g))
            if res is None:
                parse_fail += 1
                continue
            for r, d, c in zip(g, *res):
                r["tree_dist_next"], r["clause_start"] = d, c
        all_rows += rows
        if fi % 100 == 0:
            print(fi, len(xmls), flush=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(a.out, index=False)
    print(f"tokens {total}; unaligned to a sentence {skipped}; sentences {n_sent}; parse failures {parse_fail}")
    print(df[["surprisal", "tree_dist_next", "clause_start"]].describe().T)


if __name__ == "__main__":
    main()
