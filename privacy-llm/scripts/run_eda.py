"""Run the full EDA and write figures/tables to reports/eda/.   Run:  python scripts/run_eda.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from privacy_llm import eda, plots  # noqa: E402
from privacy_llm.config import REPORT_DIR  # noqa: E402
from privacy_llm.data import load_records, profiles_frame, to_frame  # noqa: E402

OUT = REPORT_DIR / "eda"


def save_fig(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / "figures" / f"{name}.png")
    plt.close(fig)


def save_table(df: pd.DataFrame, name: str, index: bool = True) -> None:
    df.to_csv(OUT / "tables" / f"{name}.csv", index=index)


def main() -> None:
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)

    splits = {s: load_records(s) for s in ("full", "small", "swap_original", "swap_flipped")}
    full = splits["full"]
    df, profiles = to_frame(full), profiles_frame(full)

    norm = eda.norm_matrix(df)
    domains = df.drop_duplicates("scenario").set_index("scenario")["domain"]
    sim = eda.scenario_similarity(norm)
    surface = eda.leak_surface(df)
    collisions = eda.value_collisions(profiles)
    baselines = eda.trivial_baselines(df)

    summary = {
        "overview": {s: eda.overview(to_frame(r)) for s, r in splits.items()},
        "mixed_label_cells": len(eda.mixed_label_cells(df)),
        "answer_checks": eda.answer_checks(df),
        "small_vs_full": eda.subset_report(splits["small"], full),
        "swap": eda.swap_report(splits["swap_original"], splits["swap_flipped"], full),
        "baselines": baselines.set_index("predictor")["accuracy"].to_dict(),
        "never_shared_fields": norm.columns[norm.sum() == 0].tolist(),
        "value_collisions": len(collisions),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    save_table(norm, "norm_matrix")
    save_table(eda.share_rate(df, ["domain", "scenario"]), "share_rate_by_scenario")
    save_table(eda.share_rate(df, "field"), "share_rate_by_field")
    save_table(eda.share_rate(df, "field_group"), "share_rate_by_field_group")
    save_table(sim.round(3), "scenario_similarity")
    save_table(eda.field_value_stats(profiles), "field_value_stats", index=False)
    save_table(surface, "leak_surface")
    save_table(collisions, "value_collisions", index=False)
    save_table(baselines, "baselines", index=False)
    save_table(profiles, "profiles", index=False)

    save_fig(plots.norm_heatmap(norm), "01_norm_matrix")
    save_fig(plots.share_count_by_scenario(norm, domains), "02_share_by_scenario")
    save_fig(plots.share_count_by_field(norm), "03_share_by_field")
    save_fig(plots.similarity_heatmap(sim), "04_scenario_similarity")
    save_fig(plots.profile_demographics(profiles), "05_profiles")
    save_fig(plots.leak_surface_bars(surface), "06_leak_surface")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
