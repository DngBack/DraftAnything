"""Exploratory, CPU-only audit of cached illusion answer states and native heads.

No generation, downloads, training of the VLM, or changes to existing results.
The classifiers here are new fixed-C sklearn baselines, not reproductions of
the paper's Adam probes. Run with mllm_readout/.venv/bin/python.
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from transformers import AutoConfig, AutoTokenizer


ROOT = Path(__file__).resolve().parent
MODELS = {
    "qwen2vl_2b": ("Qwen/Qwen2-VL-2B-Instruct", "g3_heldout"),
    "qwen3vl4b": ("Qwen/Qwen3-VL-4B-Instruct", "g3"),
    "gemma3_4b": ("google/gemma-3-4b-it", "g3"),
}


def native_rows(repo, ids):
    cache = Path(os.environ.get("HF_HUB_CACHE", str(Path.home() / ".cache/huggingface/hub")))
    snapshots = sorted((cache / ("models--" + repo.replace("/", "--")) / "snapshots").glob("*"))
    snapshot = next(p for p in snapshots if (p / "model.safetensors.index.json").exists())
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    config = AutoConfig.from_pretrained(str(snapshot), local_files_only=True)
    keys = [k for k in index if k.endswith("lm_head.weight")]
    tied = not keys
    if tied:
        text_config = config.get_text_config()
        if not (getattr(config, "tie_word_embeddings", False) or getattr(text_config, "tie_word_embeddings", False)):
            raise ValueError("No lm_head found and tied embeddings not declared")
        keys = [k for k in index if k.endswith("embed_tokens.weight")]
    if len(keys) != 1:
        raise ValueError(keys)
    key = keys[0]
    with safe_open(str(snapshot / index[key]), framework="pt", device="cpu") as f:
        tensor = f.get_slice(key)
        rows = torch.cat([tensor[i:i + 1] for i in ids]).float().numpy()
    cap = getattr(config.get_text_config(), "final_logit_softcapping", None)
    return rows, cap, tied


def predict(X, y, fold, subsets=None):
    subsets = np.zeros(len(y), int) if subsets is None else subsets
    predictions = np.empty_like(y)
    for f in np.unique(fold):
        for g in np.unique(subsets):
            tr, te = (fold != f) & (subsets == g), (fold == f) & (subsets == g)
            if not te.any():
                continue
            if len(np.unique(y[tr])) < 2:
                raise ValueError("Single-class training split")
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
            clf.fit(X[tr], y[tr])
            predictions[te] = clf.predict(X[te])
    return predictions


def audit(name, seed):
    repo, directory = MODELS[name]
    with np.load(ROOT / "results" / directory / f"{name}_illusions_states.npz") as d:
        y, raw, scores, key, subset = [d[k] for k in ("y", "ans", "scores", "key", "group")]
        H = d["last"][:, -1].astype(np.float32)
    tok = AutoTokenizer.from_pretrained(repo, local_files_only=True)
    cids = [sorted({i for prefix in ("", " ") for v in (c, c.lower(), c.capitalize())
                    for i in tok.encode(prefix + v, add_special_tokens=False)
                    if tok.decode([i]).strip().lower() == c.lower()}) for c in ("Yes", "No")]
    ids = sorted(set(sum(cids, [])))
    W, cap, tied = native_rows(repo, ids)
    Z = H @ W.T
    if cap:
        Z = cap * np.tanh(Z / cap)
    recomputed = np.stack([Z[:, [ids.index(i) for i in cls]].max(1) for cls in cids], axis=1)
    # Include ALL native token rows, not one argmax variant per class.
    _, sv, vh = np.linalg.svd(W, full_matrices=False)
    rank = int((sv > sv[0] * 1e-6).sum())
    basis = vh[:rank]
    row_coordinates = H @ basis.T
    Hnull = H - row_coordinates @ basis
    u, inv = np.unique(key, return_inverse=True)
    fold = (torch.randperm(len(u), generator=torch.Generator().manual_seed(seed)) % 5).numpy()[inv]
    features = {"pooled_global": scores, "unpooled_global": Z,
                "rowspace_global": row_coordinates, "nullspace_global": Hnull,
                "hidden_global": H}
    preds = {k: predict(X, y, fold) for k, X in features.items()}
    preds["pooled_per_subset"] = predict(scores, y, fold, subset)
    # A purely algebraic check: erasing the null component leaves candidate
    # token logits unchanged up to fp32 rounding, before optional soft-capping.
    null_token_max = float(np.max(np.abs(Hnull @ W.T)))
    row = dict(model=name, seed=seed, n=len(y), n_images=len(u), n_variant_tokens=len(ids),
               native_token_row_rank=rank, tied_head=tied, softcap=cap,
               raw=float(np.mean(raw == y)), score_mae=float(np.mean(np.abs(recomputed - scores))),
               score_argmax_agreement=float(np.mean(recomputed.argmax(1) == raw)),
               null_token_max=null_token_max)
    for k, p in preds.items():
        row[k] = float(np.mean(p == y))
    rng = np.random.default_rng(0)
    difference = (preds["hidden_global"] == y).astype(float) - (preds["pooled_global"] == y)
    delta = []
    for _ in range(2000):
        weights = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))[inv]
        delta.append(float(np.sum(weights * difference) / weights.sum()))
    row["hidden_minus_pooled_lo"], row["hidden_minus_pooled_hi"] = np.percentile(delta, [2.5, 97.5])
    print(json.dumps(row), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--output", type=Path, default=ROOT / "results" / "exploratory_geometry_20261016.csv")
    args = ap.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        rows = []
        for name in args.models:
            for seed in args.seeds:
                rows.append(audit(name, seed))
                args.output.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
