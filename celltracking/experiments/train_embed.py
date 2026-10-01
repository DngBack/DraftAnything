"""Train the learned-cost embedding (cellanc/embed.py) on train movies only (M0-M2 = 00/01/02).

Supervision: for a curriculum of gaps Delta in GAPS, and reference frames t on a stride grid,
every cell present at t+Delta with a resolved ancestor at t (lineage.trace_to_anchor) is one
training example: candidates = every cell present at t, label = index of the true ancestor.
Cross-entropy per example, batched as one frame-pair per step (frame sizes vary 1..~1500).

Using dense real frames (not benchmark schedules) and every gap up to the largest test horizon
means the model sees both single-step transitions and the same sparsity it will be evaluated at,
which is what experiments/pyuat_run.py's collapse at 2 frames shows a fixed-adjacency tracker
cannot do.

  uv run python experiments/train_embed.py [--epochs 8] [--t-stride 10] [--tag v0]
    -> outputs/results/embed_<tag>.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.embed import Embed  # noqa: E402
from cellanc.lineage import trace_to_anchor  # noqa: E402
from main import DATASETS  # noqa: E402

ROOT = Path(__file__).parents[1]
OUT = ROOT / "outputs"
GAPS = [1, 2, 4, 8, 16, 32, 64, 132]  # frames; OIAM default (1 min/frame, so also minutes)
DROP_VEL_PROB = 0.3  # used only by the legacy oracle-velocity diagnostic


def load_seq(dataset: str, alias: str):
    idx_dir = {v: k for k, v in DATASETS[dataset]["aliases"].items()}[alias]
    idx = OUT / "index" / dataset / idx_dir
    obs = pd.read_parquet(idx / "observations.parquet",
                          columns=["frame_index", "gt_track_id", "center_y", "center_x"])
    shape_path = idx / "shape.parquet"
    if shape_path.exists():
        m = obs.merge(pd.read_parquet(shape_path), on=["frame_index", "gt_track_id"], how="left")
    else:  # no full mask for this dataset (CTC gold SEG is sparse) -> oracle-location only
        m = obs.assign(axis_y=np.nan, axis_x=np.nan, major_sd=np.nan, minor_sd=np.nan)
    trk = pd.read_parquet(idx / "tracks.parquet",
                          columns=["gt_track_id", "start_frame", "end_frame", "parent_id"])
    tracks = {int(r.gt_track_id): (int(r.start_frame), int(r.end_frame), int(r.parent_id))
             for r in trk.itertuples()}
    return dict(list(m.groupby("frame_index"))), tracks


def frame_features(g: pd.DataFrame, g_prev: pd.DataFrame | None = None,
                   dt_prev: float | None = None, *,
                   oracle_velocity: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(gt_track_id per row, feature matrix) -- see cellanc.embed.N_FEATURES for the layout.

    Velocity obtained by matching gt_track_id across frames reveals temporal identity. It is
    zero by default. oracle_velocity=True exists solely to reproduce earlier diagnostic runs;
    it must never be used to support a benchmark or paper claim.
    """
    pos = g[["center_y", "center_x"]].to_numpy()
    c = pos.mean(0)
    r = np.sqrt(((pos - c) ** 2).sum(1).mean()) or 1.0
    yx = (pos - c) / r
    if len(pos) > 1:
        dist = cKDTree(pos).query(pos, k=min(6, len(pos)))[0][:, 1:]
        spacing, nn_dist = dist.mean(1), dist[:, 0]
    else:
        spacing = nn_dist = np.ones(len(pos))
    axis = g[["axis_y", "axis_x"]].fillna(0.0).to_numpy()
    logsz = np.log(g[["major_sd", "minor_sd"]].fillna(1.0).to_numpy() + 1e-3)
    vel = np.zeros((len(g), 2), dtype=np.float32)
    has_v = np.zeros(len(g), dtype=np.float32)
    if oracle_velocity and g_prev is not None and dt_prev:
        prev_pos = dict(zip(g_prev.gt_track_id, g_prev[["center_y", "center_x"]].to_numpy()))
        for i, tid in enumerate(g.gt_track_id.to_numpy()):
            if tid in prev_pos:
                vel[i] = (pos[i] - prev_pos[tid]) / r / dt_prev
                has_v[i] = 1.0
    feat = np.concatenate([yx, axis, logsz, np.log(spacing + 1e-3)[:, None], vel, has_v[:, None],
                          np.log(nn_dist + 1e-3)[:, None]], axis=1)
    return g.gt_track_id.to_numpy(), feat.astype(np.float32)


def build_example(t: int, b: int, by_t: dict, tracks: dict, rng: np.random.Generator | None = None,
                  oracle_velocity: bool = False):
    H = b - t
    g_a_prev = by_t.get(t - H) if oracle_velocity else None
    if g_a_prev is not None and rng is not None and rng.random() < DROP_VEL_PROB:
        g_a_prev = None
    ids_a, feat_a = frame_features(by_t[t], g_a_prev, H,
                                   oracle_velocity=oracle_velocity)
    ids_b, feat_b = frame_features(by_t[b], by_t[t] if oracle_velocity else None, H,
                                   oracle_velocity=oracle_velocity)
    present_a = set(int(i) for i in ids_a)
    idx_a = {int(i): k for k, i in enumerate(ids_a)}
    labels, keep = [], []
    for k, u in enumerate(ids_b):
        anc, status, _ = trace_to_anchor(int(u), t, tracks, present_a)
        if status == "valid_anchor":
            keep.append(k)
            labels.append(idx_a[anc])
    if not labels:
        return None
    return feat_a, feat_b[keep], np.array(labels, dtype=np.int64)


def sample_pairs(by_t: dict, t_stride: int, gaps: list[int], seed: int) -> list[tuple[int, int]]:
    ts = sorted(by_t)
    pairs = [(t, t + H) for t in ts[::t_stride] for H in gaps if t + H in by_t]
    np.random.default_rng(seed).shuffle(pairs)
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--t-stride", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tag", default="v0")
    ap.add_argument("--oracle-velocity", action="store_true",
                    help="reproduce legacy GT-ID velocity experiments; diagnostic only")
    ap.add_argument("--dataset", default="one_in_a_million", choices=DATASETS)
    ap.add_argument("--split", default=None, help="which DATASETS[dataset]['splits'] value to "
                    "train on; defaults to 'train' for one_in_a_million, else 'domain_adapt'")
    args = ap.parse_args()
    split = args.split or ("train" if args.dataset == "one_in_a_million" else "domain_adapt")
    time_step_min = DATASETS[args.dataset]["time_step_min"]
    cfg_path = OUT / "benchmarks/v1/config.json"
    gaps = json.loads(cfg_path.read_text())[args.dataset]["deltas_frames"] if cfg_path.exists() \
        else GAPS
    train_aliases = [a for a, s in DATASETS[args.dataset]["splits"].items() if s == split]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Embed().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    seqs = {a: load_seq(args.dataset, a) for a in train_aliases}
    all_pairs = [(a, *p) for a in seqs for p in sample_pairs(seqs[a][0], args.t_stride, gaps, seed=0)]
    print(f"{len(all_pairs)} training frame-pairs across {train_aliases}", flush=True)

    for epoch in range(args.epochs):
        rng = np.random.default_rng(epoch)
        order = rng.permutation(len(all_pairs))
        tot_loss, n_ex, n_correct, n_seen = 0.0, 0, 0, 0
        for k in order:
            a, t, b = all_pairs[k]
            by_t, tracks = seqs[a]
            ex = build_example(t, b, by_t, tracks, rng, args.oracle_velocity)
            if ex is None:
                continue
            feat_a, feat_b, labels = ex
            fa = torch.as_tensor(feat_a, device=dev)
            fb = torch.as_tensor(feat_b, device=dev)
            lab = torch.as_tensor(labels, device=dev)
            ea, eb = model.embed(fa), model.embed(fb)
            cost = model.cost(ea, eb, float((b - t) * time_step_min))  # (n_a, n_b), dt in minutes
            logits = -cost.T  # one row per target, one column per candidate ancestor
            loss = F.cross_entropy(logits, lab)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(labels)
            n_correct += (logits.argmax(1) == lab).sum().item()
            n_seen += len(labels)
            n_ex += 1
        print(f"epoch {epoch}: {n_ex} pairs, {n_seen} targets, "
             f"loss {tot_loss / max(n_seen, 1):.4f}, train_acc {n_correct / max(n_seen, 1):.4f}",
             flush=True)

    out = OUT / "results" / f"embed_{args.tag}.pt"
    torch.save(model.state_dict(), out)
    out.with_suffix(".json").write_text(json.dumps({
        "dataset": args.dataset, "train_aliases": train_aliases,
        "oracle_velocity": args.oracle_velocity, "epochs": args.epochs,
        "t_stride": args.t_stride, "gaps_frames": gaps,
    }, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
