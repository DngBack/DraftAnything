"""Exploratory CPU pilot: matched-dimension readouts and stricter transfer.

Uses existing final answer states and cached native token rows; no VLM inference.
This is not a reproduction of the paper's Adam probes or an online repair.
Five folds stratify base images by (illusion subset, label), keeping all six
views together. Cross-prompt fits use only one wording on training images and
the other wording on held-out images. Family transfer leaves out an entire
illusion subset. Hyperparameters are fixed, not selected on these test sets.

Run with mllm_readout/.venv/bin/python. Existing experiment files are untouched.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from transformers import AutoTokenizer

from readout_geometry_audit import MODELS, ROOT, native_rows


def image_folds(y, key, subset, seed):
    unique, first, inv = np.unique(key, return_index=True, return_inverse=True)
    assert np.all(y == y[first][inv]), "An image has inconsistent labels"
    assert np.all(subset == subset[first][inv])
    strata = np.array([f"{g}:{c}" for g, c in zip(subset[first], y[first])])
    fold = np.empty(len(unique), int)
    for f, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(unique, strata)):
        fold[te] = f
    return fold[inv]


def fit_predict(X, y, tr, te):
    assert tr.any() and te.any()
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1500))
    clf.fit(X[tr], y[tr])
    return clf.predict(X[te])


def summarize(y, raw, pred, key, seed):
    unique, inv = np.unique(key, return_inverse=True)
    correct, base = pred == y, raw == y
    delta = correct.astype(float) - base.astype(float)
    per_image = np.bincount(inv, weights=delta) / np.bincount(inv)
    rng = np.random.default_rng(seed)
    boot = np.mean(per_image[rng.integers(len(unique), size=(2000, len(unique)))], axis=1)
    return dict(acc=float(correct.mean()), raw=float(base.mean()),
                gain_lo=float(np.percentile(boot, 2.5)), gain_hi=float(np.percentile(boot, 97.5)),
                fixed=float(correct[~base].mean()), broken=float((~correct[base]).mean()))


def cell(name, seed, protocols):
    repo, directory = MODELS[name]
    with np.load(ROOT / "results" / directory / f"{name}_illusions_states.npz") as d:
        y, raw, scores, key, subset, q = [d[k] for k in ("y", "ans", "scores", "key", "group", "q")]
        H = d["last"][:, -1].astype(np.float32)
    tok = AutoTokenizer.from_pretrained(repo, local_files_only=True)
    cids = [sorted({i for p in ("", " ") for v in (c, c.lower(), c.capitalize())
                    for i in tok.encode(p + v, add_special_tokens=False)
                    if tok.decode([i]).strip().lower() == c.lower()}) for c in ("Yes", "No")]
    ids = sorted(set(sum(cids, [])))
    W, cap, tied = native_rows(repo, ids)
    _, sv, vh = np.linalg.svd(W, full_matrices=False)
    rank = int((sv > sv[0] * 1e-6).sum())
    basis = vh[:rank]
    coords = H @ basis.T
    Hnull = H - coords @ basis
    rng = np.random.default_rng(seed + 1701)
    random_basis = np.linalg.qr(rng.standard_normal((H.shape[1], 32)))[0].astype(np.float32)
    Z = H @ W.T
    if cap:
        Z = cap * np.tanh(Z / cap)
    reconstructed = np.stack([Z[:, [ids.index(i) for i in cls]].max(1) for cls in cids], axis=1)
    features = {"pooled": scores, "margin": (scores[:, 0] - scores[:, 1])[:, None],
                "candidate_rowspace": coords, "random_same_rank": H @ random_basis[:, :rank],
                "random_32": H @ random_basis, "candidate_nullspace": Hnull, "hidden": H}
    wording = np.empty(len(y), int)
    for g in np.unique(subset):
        qs = sorted(np.unique(q[subset == g]))
        assert len(qs) == 2
        for i, question in enumerate(qs):
            wording[(subset == g) & (q == question)] = i
    fold = image_folds(y, key, subset, seed)
    splits = {
        "image_cv": [(fold != f, fold == f) for f in range(5)],
        "prompt_0_to_1": [((fold != f) & (wording == 0), (fold == f) & (wording == 1)) for f in range(5)],
        "prompt_1_to_0": [((fold != f) & (wording == 1), (fold == f) & (wording == 0)) for f in range(5)],
        "prompt_0_to_0": [((fold != f) & (wording == 0), (fold == f) & (wording == 0)) for f in range(5)],
        "prompt_1_to_1": [((fold != f) & (wording == 1), (fold == f) & (wording == 1)) for f in range(5)],
        "family_holdout": [(subset != g, subset == g) for g in np.unique(subset)],
    }
    rows, predictions = [], {}
    for protocol in protocols:
        for feature, X in features.items():
            pred = np.full(len(y), -1, int)
            for tr, te in splits[protocol]:
                assert not set(key[tr]) & set(key[te]), "Image leakage"
                if protocol in ("prompt_0_to_1", "prompt_1_to_0"):
                    assert not set(q[tr]) & set(q[te]), "Wording leakage"
                pred[te] = fit_predict(X, y, tr, te)
            ok = pred >= 0
            row = dict(model=name, seed=seed, protocol=protocol, feature=feature,
                       n_items=int(ok.sum()), n_images=len(np.unique(key[ok])), d=X.shape[1],
                       **summarize(y[ok], raw[ok], pred[ok], key[ok], seed))
            rows.append(row)
            predictions[f"{protocol}__{feature}"] = pred
        print(json.dumps(dict(model=name, seed=seed, protocol=protocol,
                              accuracies={r["feature"]: round(r["acc"], 4) for r in rows if r["protocol"] == protocol})), flush=True)
    metadata = dict(model=name, seed=seed, native_rank=rank, n_candidate_tokens=len(ids),
                    score_mae=float(np.mean(np.abs(reconstructed - scores))),
                    argmax_reconstruction=float(np.mean(reconstructed.argmax(1) == raw)),
                    null_token_max=float(np.max(np.abs(Hnull @ W.T))),
                    n_images=len(np.unique(key)), n_items=len(y), tied_head=tied,
                    softcap=cap, classifier="StandardScaler + sklearn LogisticRegression C=1 max_iter=1500",
                    final_state_dtype="cached float16 converted to float32",
                    warning="Exploratory cached-state classification; not online generation or causal identification")
    return rows, dict(y=y, raw=raw, key=key, fold=fold, **predictions), metadata


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--protocols", nargs="+", choices=["image_cv", "prompt_0_to_1", "prompt_1_to_0",
                                                     "prompt_0_to_0", "prompt_1_to_1", "family_holdout"],
                    default=["image_cv", "prompt_0_to_1", "prompt_1_to_0", "prompt_0_to_0", "prompt_1_to_1", "family_holdout"])
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "ver29_cached_pilot")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    rows, metadata = [], []
    with threadpool_limits(limits=2):
        for model in a.models:
            for seed in a.seeds:
                r, predictions, meta = cell(model, seed, a.protocols)
                rows.extend(r)
                metadata.append(meta)
                np.savez_compressed(a.out / f"{model}_seed{seed}_predictions.npz", **predictions)
                pd.DataFrame(rows).to_csv(a.out / "metrics.csv", index=False)
                (a.out / "metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
