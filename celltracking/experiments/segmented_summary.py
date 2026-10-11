"""Summary table: true positions vs off-the-shelf segmenters, end-to-end (docs/second_segmenter_protocol.md).

Reads outputs/results/segmented_eval_{gt,cpsam_all,omni}{,_detection}.parquet (segmented_eval.py) and writes
outputs/results/segmented_summary.csv: per detector and model family (argmax; 3-seed mean, min, max), window-mean
end-to-end accuracy at 33/66/132 min, all windows and dense (n_anchors > 200), plus frame-mean detection recall/precision.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.family_set import RESULTS  # noqa: E402

SOURCES = {"gt": "True positions", "cpsam_all": "Cellpose-SAM", "omni": "Omnipose bact_phase_omni"}
FAMILIES = {"raft_v3": ("raft_v3", "raft_v3_s1", "raft_v3_s2"),
            "raft_v3p": ("raft_v3p", "raft_v3p_s1", "raft_v3p_s2"),
            "raft_sim_loc": ("raft_sim_loc", "raft_sim_loc_s1", "raft_sim_loc_s2"),
            "nearest": ("nearest",)}
H = (33, 66, 132)


def main():
    rows = []
    for src, name in SOURCES.items():
        p = RESULTS / f"segmented_eval_{src}.parquet"
        if not p.exists():
            continue
        df, det = pd.read_parquet(p), pd.read_parquet(RESULTS / f"segmented_eval_{src}_detection.parquet")
        for fam, tags in FAMILIES.items():
            methods = [t if t == "nearest" else f"{t}_argmax" for t in tags]
            d = df[df.method.isin(methods)]
            if d.method.nunique() != len(methods):
                continue
            r = {"detector": name, "model": fam, "n_seeds": len(methods),
                 "det_recall": det.recall.mean(), "det_precision": det.precision.mean()}
            for sub, dd in (("all", d), ("dense", d[d.n_anchors > 200])):
                for h in H:
                    per_seed = dd[dd.H == h].groupby("method").end_to_end.mean()
                    r[f"{sub}_{h}"] = per_seed.mean()
                    r[f"{sub}_{h}_min"], r[f"{sub}_{h}_max"] = per_seed.min(), per_seed.max()
                    if fam == "raft_v3":  # paper's end-to-end model is seed 0
                        r[f"{sub}_{h}_seed0"] = per_seed["raft_v3_argmax"]
            rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "segmented_summary.csv", index=False)
    cols = ["detector", "model", "det_recall", "det_precision"] + [f"{s}_{h}" for s in ("all", "dense") for h in H]
    print(out[cols].round(3).to_string(index=False))
    print("\nseed range (min-max):")
    print(out[["detector", "model"] + [f"{s}_{h}_{m}" for s in ("all", "dense") for h in H for m in ("min", "max")]]
          .round(3).to_string(index=False))
    if "all_33_seed0" in out:
        print("\nraft_v3 seed 0 (paper):")
        print(out[out.model == "raft_v3"][["detector"] + [f"{s}_{h}_seed0" for s in ("all", "dense") for h in H]]
              .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
