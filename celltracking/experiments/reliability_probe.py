"""Does decode disagreement (OT vs LAP, same learned cost, no labels needed) predict where the
cost is untrustworthy? Tests a label-free reliability signal against the true accuracy we
happen to have for these benchmark datasets -- validating whether it could be trusted on a
genuinely unlabeled target dataset.

For every window: run the SAME checkpoint through both decode strategies, measure the fraction
of final-frame cells where they pick a different ancestor (label-free), and separately score
each against ground truth (label-needed, only possible here because these are benchmarks).
Correlating disagreement against true accuracy across settings (in-domain, zero-shot transfer,
domain-adapted, one dataset where each fails) is the actual test.

  uv run python experiments/reliability_probe.py -> outputs/results/reliability_probe.parquet
"""

import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from cellanc.embed import Embed  # noqa: E402
from cellanc.evaluate import score_window  # noqa: E402
from cellanc.loader import Windows  # noqa: E402
from experiments.eval_embed import _load, embed_chain  # noqa: E402
from main import DATASETS  # noqa: E402

ROOT = Path(__file__).parents[1]
OUT = ROOT / "outputs"
BENCH = OUT / "benchmarks" / "v1"

# (setting label, checkpoint tag, dataset, seqs, H) -- covers in-domain, zero-shot transfer
# (1 collapse + 2 ok-ish), and domain-adapted (1 win + 1 loss), i.e. every regime already
# characterized in the synthesis report.
RUNS = [
    ("in_domain_oiam", "v3", "one_in_a_million", ["M3", "M4"], 132),
    ("zero_shot_sim_plus", "v3", "sim_plus", ["SIM-01", "SIM-02"], 57),
    ("zero_shot_musc", "v3", "musc", ["MuSC-01", "MuSC-02"], 110),
    ("zero_shot_hsc", "v3", "hsc", ["HSC-01", "HSC-02"], 239),
    ("domain_adapt_musc_win", "musc_da", "musc", ["MuSC-02"], 110),
    ("domain_adapt_hsc_loss", "hsc_da", "hsc", ["HSC-02"], 239),
]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    windows = Windows(BENCH, ROOT / "data")
    rows = []
    for setting, tag, dataset, seqs, H in RUNS:
        state = torch.load(OUT / "results" / f"embed_{tag}.pt", map_location=dev)
        model = Embed(n_in=state["phi.0.weight"].shape[1]).to(dev)
        model.load_state_dict(state)
        model.eval()

        ws = [w for w in windows.windows if w["sequence_id"] in seqs
              and w["horizon_frames"] == H and w["schedule_id"] in ("ends", "s16", "s32")
              and f"A_h{H}" in w["experiments"]]
        for w in ws:
            m, tg_by = _load(dataset, w["sequence_id"])
            m_by_frame = dict(list(m[m.frame_index.isin(w["observed_frames"])].groupby("frame_index")))
            time_step_min = w["horizon_min"] / w["horizon_frames"]
            pred_ot = embed_chain(w, model, dev, m_by_frame, time_step_min, "ot", eps=0.5, tau=None)
            pred_lap = embed_chain(w, model, dev, m_by_frame, time_step_min, "lap", cap=2)
            common = set(pred_ot) & set(pred_lap)
            disagreement = sum(pred_ot[c] != pred_lap[c] for c in common) / len(common)
            tg = tg_by[w["window_id"]]
            sib = pd.DataFrame(columns=["det_1", "det_2"])
            s_ot = score_window(pred_ot, set(), tg, sib)
            s_lap = score_window(pred_lap, set(), tg, sib)
            rows.append({"setting": setting, "dataset": dataset, "sequence_id": w["sequence_id"],
                        "schedule_id": w["schedule_id"], "window_id": w["window_id"],
                        "disagreement": disagreement,
                        "ot_acc": s_ot["n_correct"] / s_ot["n_scored"] if s_ot["n_scored"] else None,
                        "lap_acc": s_lap["n_correct"] / s_lap["n_scored"] if s_lap["n_scored"] else None})
        print(setting, "done,", len([r for r in rows if r["setting"] == setting]), "windows", flush=True)

    R = pd.DataFrame(rows).dropna(subset=["ot_acc"])
    out = OUT / "results" / "reliability_probe.parquet"
    R.to_parquet(out, index=False)

    print("\nper-setting means:")
    print(R.groupby("setting")[["disagreement", "ot_acc"]].mean().round(3))
    print("\noverall Pearson corr(disagreement, ot_acc):", R.disagreement.corr(R.ot_acc).round(3))
    print("overall Spearman corr:", R.disagreement.corr(R.ot_acc, method="spearman").round(3))
    print("saved", out)


if __name__ == "__main__":
    main()
