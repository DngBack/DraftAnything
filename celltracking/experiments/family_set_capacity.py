"""Check whether family affinity adds value to a capacity-constrained decoder.

Uses the already trained M0-M2 family affinity. Selects all decoder choices on
M3, then evaluates M4 once. Reports both capacity alone and affinity + capacity.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.baselines import BASELINES  # noqa: E402
from cellanc.evaluate import score_window  # noqa: E402
from cellanc.loader import Windows  # noqa: E402
from experiments.family_set import (Affinity, BENCH, RESULTS, ROOT, examples, metrics,
                                    probabilities, summarize)  # noqa: E402


def established_baselines(alias: str, all_windows: list[dict]) -> pd.DataFrame:
    loader = Windows(BENCH, ROOT / "data")
    selected = [w for w in all_windows if w["sequence_id"] == alias and
                w["horizon_frames"] == 132 and w["schedule_id"] == "ends" and
                "A_h132" in w["experiments"]]
    targets = pd.read_parquet(BENCH / "labels/targets" / f"{alias}.parquet")
    by_id = dict(list(targets.groupby("window_id")))
    rows = []
    for w in selected:
        win = loader.load(w)
        for name in ("nearest_anchor_rc", "chained_lap_c4", "chained_lap_c4_rc"):
            pred = BASELINES[name](win)
            s = score_window(pred, set(), by_id[w["window_id"]],
                             pd.DataFrame(columns=["det_1", "det_2"]))
            rows.append({"method": name, "window_id": w["window_id"],
                         "sequence_id": alias, "n_scored": s["n_scored"],
                         "n_correct": s["n_correct"], "set_f1_sum": s["set_f1_sum"],
                         "count_n_anchors": s["count_n_anchors"]})
    return pd.DataFrame(rows)


def main():
    saved = torch.load(RESULTS / "family_set_h132.pt", map_location="cpu", weights_only=False)
    model = Affinity()
    model.load_state_dict(saved["model"])
    model.eval()
    all_windows = Windows(BENCH, ROOT / "data").windows
    val_ex = examples("M3", all_windows)
    probabilities(model, saved["mean"], saved["std"], val_ex)
    prior = json.loads((RESULTS / "family_set_h132_config.json").read_text())
    trials = []
    for beta in (0, prior["beta"]):
        for capacity in (4, 8):
            for penalty in (0, 0.5, 1, 2):
                config = {"beta": beta, "threshold": prior["threshold"],
                          "temperature": prior["temperature"], "capacity": capacity,
                          "extra_child_penalty": penalty}
                df = summarize(val_ex, **config)
                acc, f1 = metrics(df)
                trials.append({**config, "acc": acc, "set_f1": f1})
    rank = pd.DataFrame(trials).sort_values(["set_f1", "acc"], ascending=False)
    best = rank.iloc[0]
    config = {k: int(best[k]) if k == "capacity" else float(best[k]) for k in
              ("beta", "threshold", "temperature", "capacity", "extra_child_penalty")}
    print("M3 selected", config, "metrics", tuple(round(x, 3) for x in metrics(
          summarize(val_ex, **config))), flush=True)
    for beta in (0, prior["beta"]):
        subset = rank[rank.beta == beta].iloc[0]
        print("M3 beta", beta, "best acc/F1", round(subset.acc, 3),
              round(subset.set_f1, 3), flush=True)

    test_ex = examples("M4", all_windows)
    probabilities(model, saved["mean"], saved["std"], test_ex)
    rows = []
    for alias, ex in (("M3", val_ex), ("M4", test_ex)):
        base = established_baselines(alias, all_windows)
        chosen = summarize(ex, **config).assign(method="family_set_capacity")
        rows.extend((base, chosen))
        for name, group in base.groupby("method"):
            print(alias, name, tuple(round(x, 3) for x in metrics(group)), flush=True)
        print(alias, "family_set_capacity", tuple(round(x, 3) for x in metrics(chosen)), flush=True)
    pd.concat(rows).to_parquet(RESULTS / "family_set_h132_capacity_windows.parquet", index=False)
    rank.to_csv(RESULTS / "family_set_h132_capacity_val_grid.csv", index=False)
    (RESULTS / "family_set_h132_capacity_config.json").write_text(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
