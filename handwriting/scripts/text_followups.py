"""Text-only features for the follow-up analyses in docs/acl_prereg_gaps.md.

surprisal_nc: bits of each word given only the start-of-text token.
clause_strict: clause onset using the labels the plan names (S, SBAR, SBARQ).
clause_check: the original five-label definition, recomputed to verify the parse matches
tokens.csv.gz. Reads no image and no gap.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import text_features as tf

STRICT = {"S", "SBAR", "SBARQ"}
ORIGINAL = set(tf.CLAUSE_LABELS)


@torch.no_grad()
def nocontext_bits(tok, model, words):
    out = {}
    for w in words:
        ids = [tok.bos_token_id] + tok(" " + w)["input_ids"]
        logits = model(torch.tensor([ids])).logits[0, :-1]
        lp = torch.log_softmax(logits, -1).gather(1, torch.tensor(ids[1:])[:, None])[:, 0]
        out[w] = float((-lp / np.log(2)).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=Path, default=Path("data/ling/tokens.csv.gz"))
    ap.add_argument("--out", type=Path, default=Path("data/ling/followups.csv.gz"))
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    import stanza
    from transformers import AutoModelForCausalLM, AutoTokenizer
    df = pd.read_csv(a.tokens, dtype={"writer_id": str}, keep_default_na=False, na_values=[""])
    if a.limit:
        df = df[df.form_id.isin(df.form_id.unique()[:a.limit])].copy()
    tok = AutoTokenizer.from_pretrained("gpt2-large")
    model = AutoModelForCausalLM.from_pretrained("gpt2-large").eval()
    words = sorted(df.text.astype(str).unique())
    nc = nocontext_bits(tok, model, words)
    print("unique words", len(words), flush=True)
    df["surprisal_nc"] = df.text.astype(str).map(nc)

    nlp = stanza.Pipeline("en", processors="tokenize,pos,constituency", tokenize_pretokenized=True,
                          use_gpu=False, verbose=False)
    df["clause_strict"], df["clause_check"] = 0.0, 0.0
    strict, check = df.columns.get_loc("clause_strict"), df.columns.get_loc("clause_check")
    df = df.reset_index(drop=True)
    idx_by_form = df.groupby("form_id", sort=False).indices
    for fi, (form, idx) in enumerate(idx_by_form.items()):
        idx = np.sort(idx)
        groups, cur = [], []
        sent = df.sent.to_numpy()
        for i in idx:   # same grouping as text_features.main: consecutive tokens of one sentence
            if cur and sent[i] != sent[cur[-1]]:
                groups.append(cur)
                cur = []
            cur.append(i)
        if cur:
            groups.append(cur)
        groups = [g for g in groups if sent[g[0]] >= 0]
        text = df.text.astype(str).to_numpy()
        doc = nlp([[text[i].replace(" ", "_") for i in g] for g in groups])
        for g, s in zip(groups, doc.sentences):
            for col, labels in [(strict, STRICT), (check, ORIGINAL)]:
                tf.CLAUSE_LABELS = labels
                res = tf.tree_features(s.constituency, len(g))
                if res is not None:
                    df.iloc[g, col] = res[1]
        if fi % 100 == 0:
            print(fi, len(idx_by_form), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df[["line_id", "pos", "surprisal_nc", "clause_strict", "clause_check"]].to_csv(a.out, index=False)
    same = (df.clause_check == df.clause_start).mean()
    print(f"clause_check equals stored clause_start on {same:.5f} of tokens")
    print(f"tokens flagged: stored {df.clause_start.mean():.4f}, strict {df.clause_strict.mean():.4f}")
    print("spearman surprisal vs nocontext", df[["surprisal", "surprisal_nc"]].corr("spearman").iloc[0, 1])


if __name__ == "__main__":
    main()
