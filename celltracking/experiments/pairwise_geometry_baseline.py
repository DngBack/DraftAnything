"""Strong direct pairwise association control for family-center regression.

Train on OIAM M0-M2 at H=33/66/132; evaluate M3/M4. The pair classifier uses
the same endpoint point clouds and horizon, plus all 23 anchor context features
and explicit anchor-target displacement. It sees all true links, 16 nearby
negatives and 8 random negatives per labeled target in training. At inference
it scores *all* anchor-target pairs, so candidate recall is 100%. No lineage
or temporal track ID is an inference feature. M3/M4 were viewed previously.

Run from celltracking/: python experiments/pairwise_geometry_baseline.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.evaluate import score_window  # noqa: E402
from cellanc.loader import Windows  # noqa: E402
from experiments.family_set import BENCH, RESULTS, ROOT, examples  # noqa: E402
from experiments.learned_family_center_probe import geometry  # noqa: E402
from experiments.multihorizon_center_context import fit_multi, multi_cost  # noqa: E402


HORIZONS = (33, 66, 132)
K_NEAR = 16
K_RANDOM = 8


def pair_features(e, target_rows, anchor_rows, anchor_context=None, spacing=None):
    src = e["anchor"][["center_y", "center_x"]].to_numpy(dtype=np.float32)
    dst = e["target"][["center_y", "center_x"]].to_numpy(dtype=np.float32)
    if anchor_context is None or spacing is None:
        anchor_context, _, spacing, _ = geometry(e)
    ca = np.median(src, axis=0)
    cb = np.median(dst, axis=0)
    scale_b = max(float(np.percentile(np.linalg.norm(dst - cb, axis=1), 80)), 1.0)
    delta = (dst[target_rows] - src[anchor_rows]) / spacing
    target_rel = (dst[target_rows] - cb) / scale_b
    horizon = np.full((len(target_rows), 1),
                      np.log2(e["window"]["horizon_frames"] / 132), dtype=np.float32)
    return np.column_stack([anchor_context[anchor_rows], delta,
                            np.linalg.norm(delta, axis=1), target_rel,
                            horizon]).astype(np.float32)


def sampled_pairs(e, rng):
    src_id = {int(v): i for i, v in enumerate(e["anchor"].local_detection_id)}
    dst_id = {int(v): i for i, v in enumerate(e["target"].local_detection_id)}
    src = e["anchor"][["center_y", "center_x"]].to_numpy()
    dst = e["target"][["center_y", "center_x"]].to_numpy()
    near = cKDTree(src).query(dst, k=min(K_NEAR, len(src)))[1]
    near = np.asarray(near).reshape(len(dst), -1)
    valid = e["targets"]
    valid = valid[valid.in_foi & (valid.target_status == "valid_anchor")]
    target_rows, anchor_rows, labels = [], [], []
    for target_id, ancestor_id in zip(valid.target_detection_id, valid.ancestor_detection_id):
        if int(target_id) not in dst_id or int(ancestor_id) not in src_id:
            continue
        j = dst_id[int(target_id)]
        truth = src_id[int(ancestor_id)]
        candidates = set(near[j].tolist())
        candidates.update(rng.choice(len(src), size=min(K_RANDOM, len(src)), replace=False).tolist())
        candidates.add(truth)
        for i in candidates:
            target_rows.append(j)
            anchor_rows.append(i)
            labels.append(i == truth)
    return (np.asarray(target_rows, dtype=int), np.asarray(anchor_rows, dtype=int),
            np.asarray(labels, dtype=np.uint8))


def train_classifier(items):
    rng = np.random.default_rng(20261003)
    features, labels = [], []
    for index, e in enumerate(items):
        target_rows, anchor_rows, y = sampled_pairs(e, rng)
        features.append(pair_features(e, target_rows, anchor_rows))
        labels.append(y)
        if (index + 1) % 50 == 0:
            print("prepared", index + 1, "windows", flush=True)
    x, y = np.concatenate(features), np.concatenate(labels)
    weight = np.where(y == 1, 12.0, 1.0)
    print("pair training", x.shape, "positive", int(y.sum()), "rate", round(y.mean(), 4), flush=True)
    model = HistGradientBoostingClassifier(max_iter=180, learning_rate=0.05,
            max_leaf_nodes=63, min_samples_leaf=30, l2_regularization=1,
            random_state=0).fit(x, y, sample_weight=weight)
    return model


def pair_assignment(e, model):
    return pair_probabilities(e, model).argmax(axis=1)


def pair_probabilities(e, model):
    n_anchor = len(e["anchor"])
    n_target = len(e["target"])
    context, _, spacing, _ = geometry(e)
    result = np.empty((n_target, n_anchor), dtype=np.float32)
    for start in range(0, n_target, 64):
        stop = min(start + 64, n_target)
        target_rows = np.repeat(np.arange(start, stop), n_anchor)
        anchor_rows = np.tile(np.arange(n_anchor), stop - start)
        x = pair_features(e, target_rows, anchor_rows, context, spacing)
        p = model.predict_proba(x)[:, 1].reshape(stop - start, n_anchor)
        result[start:stop] = p
    return result


def score(e, assignment):
    aid = e["anchor"].local_detection_id.to_numpy(dtype=int)
    tid = e["target"].local_detection_id.to_numpy(dtype=int)
    pred = dict(zip(tid, aid[assignment]))
    empty = pd.DataFrame(columns=["det_1", "det_2"])
    return score_window(pred, set(), e["targets"], empty)


def main():
    windows = Windows(BENCH, ROOT / "data").windows
    train = [e for alias in ("M0", "M1", "M2") for h in HORIZONS
             for e in examples(alias, windows, compensate=False, horizon=h)]
    pair_model = train_classifier(train)
    center_model = fit_multi(train)
    rows = []
    for alias in ("M3", "M4"):
        for horizon in HORIZONS:
            items = examples(alias, windows, compensate=False, horizon=horizon)
            for e in items:
                assignments = {
                    "nearest_anchor": e["cost"].argmin(axis=1),
                    "family_center": multi_cost(e, center_model).argmin(axis=1),
                    "pairwise_geometry": pair_assignment(e, pair_model),
                }
                for name, assignment in assignments.items():
                    s = score(e, assignment)
                    rows.append({"sequence_id": alias, "window_id": e["window"]["window_id"],
                                 "horizon_frames": horizon, "n_anchors": len(e["anchor"]),
                                 "method": name, "n_scored": s["n_scored"],
                                 "n_correct": s["n_correct"], "set_f1_sum": s["set_f1_sum"],
                                 "count_n_anchors": s["count_n_anchors"]})
            print(alias, horizon, "complete", flush=True)
    out = pd.DataFrame(rows)
    path = RESULTS / "pairwise_geometry_baseline.parquet"
    out.to_parquet(path, index=False)
    for label, data in (("all", out), ("dense", out[out.n_anchors > 200])):
        acc = (data.n_correct / data.n_scored).groupby([data.horizon_frames, data.method]).mean()
        f1 = (data.set_f1_sum / data.count_n_anchors).groupby([data.horizon_frames, data.method]).mean()
        print(label, pd.DataFrame({"acc": acc, "f1": f1}).round(4).to_string(), flush=True)
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
