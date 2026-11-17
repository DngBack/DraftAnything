"""Surprisal from an alternative language model, for the pre-stated robustness check."""
import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from text_features import surprisals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=Path, default=Path("data/ling/tokens.csv.gz"))
    ap.add_argument("--model", default="EleutherAI/pythia-1b-deduped")
    ap.add_argument("--out", type=Path, default=Path("data/ling/surprisal_pythia.csv.gz"))
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16).to(device).eval()
    if tok.bos_token_id is None:
        tok.bos_token_id = tok.eos_token_id
    df = pd.read_csv(a.tokens, dtype={"writer_id": str}, keep_default_na=False, na_values=[""])
    out = []
    for i, (_, g) in enumerate(df.groupby("form_id", sort=False)):
        g = g.copy()
        g["surprisal_alt"] = surprisals(tok, model, device, g.text.astype(str).tolist())
        out.append(g[["line_id", "pos", "surprisal_alt"]])
        if i % 300 == 0:
            print(i, flush=True)
    r = pd.concat(out)
    r.to_csv(a.out, index=False)
    m = df.merge(r, on=["line_id", "pos"])
    print("spearman with gpt2-large", m[["surprisal", "surprisal_alt"]].corr("spearman").iloc[0, 1])


if __name__ == "__main__":
    main()
