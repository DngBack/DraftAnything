"""Protocol audit of Table 2 (OIAM M3/M4; review item 3.3).

Every row is re-derived from its per-window results through make_tables.oiam_rows (the same code that writes the
table) and checked against one reference window set (raft_v3 argmax windows). Per row and gap: number of windows,
windows missing from / extra to the reference, window-mean accuracy (the table value), 95% CIs from a window
bootstrap and from a time-block bootstrap (blocks of 4 consecutive windows within a movie, which respects the overlap
of windows), the per-movie means (M3, M4) and the cell-weighted accuracy. Seeded rows: per-window seed means.

  python experiments/table2_audit.py      # writes outputs/results/table2_audit.csv and paper/tables/oiam_ci.tex
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "paper"))
import make_tables as mt  # noqa: E402
from experiments.block_bootstrap import ci  # noqa: E402
from experiments.family_set import RESULTS  # noqa: E402

HS = (33, 66, 132)


def per_window(frames):
    d = pd.concat(frames, ignore_index=True)
    agg = {"value": "mean", "n_anchors": "first"}
    if "n_scored" in d:
        agg["n_scored"] = "first"
    return d.groupby(["H", "window_id"], as_index=False).agg(agg)


def by_movie(x):
    x = x.assign(seq=x.window_id.str[:2], anchor=x.window_id.str.extract(r"_a(\d+)_", expand=False).astype(int))
    return [g.sort_values("anchor").value.to_numpy() for _, g in x.groupby("seq")], x


def main():
    mt.FRAMES.clear()
    rows = mt.oiam_rows()
    ref = per_window(mt.FRAMES["\\ours, full"])
    out = []
    for name, frames in mt.FRAMES.items():
        if rows.get(name) is None:
            continue
        d = per_window(frames)
        if "n_scored" not in d or d.n_scored.isna().any():  # decoder files lack it; same windows as the reference
            d = d.drop(columns="n_scored", errors="ignore").merge(ref[["H", "window_id", "n_scored"]], how="left")
        for h in HS:
            for dense in (False, True):
                x = d[(d.H == h) & ((d.n_anchors > 200) if dense else True)]
                r = ref[(ref.H == h) & ((ref.n_anchors > 200) if dense else True)]
                if not len(x):
                    continue
                vals, xs = by_movie(x)
                m, wlo, whi = ci(vals, 0)
                _, blo, bhi = ci(vals, 4)
                cw = (float((x.value * x.n_scored).sum() / x.n_scored.sum())
                      if "n_scored" in x and x.n_scored.notna().all() else np.nan)
                out.append({"row": name, "H": h, "dense": dense, "n": len(x),
                            "missing": len(set(r.window_id) - set(x.window_id)),
                            "extra": len(set(x.window_id) - set(r.window_id)),
                            "seeds": len(frames), "mean": m, "win_lo": wlo, "win_hi": whi,
                            "block_lo": blo, "block_hi": bhi,
                            "M3": xs[xs.seq == "M3"].value.mean(), "M4": xs[xs.seq == "M4"].value.mean(),
                            "cell_weighted": cw,
                            "table": rows[name][1 if dense else 0][HS.index(h)]})
    df = pd.DataFrame(out)
    bad = df[(df["mean"] - df.table).abs() > 1e-9]
    print("rows whose audit mean differs from the table value:", len(bad))
    if len(bad):
        print(bad[["row", "H", "dense", "mean", "table"]].to_string())
    pd.set_option("display.width", 250)
    print(df[["row", "H", "dense", "n", "missing", "extra", "seeds", "mean", "block_lo", "block_hi", "M3", "M4",
              "cell_weighted"]].round(3).to_string(index=False))
    df.to_csv(RESULTS / "table2_audit.csv", index=False)
    write_tex(df)


def write_tex(df):
    keep = [k for k in df.row.unique() if k not in mt.OIAM_COMPACT_DROP]
    lines = [r"\begin{table*}[t]", r"\centering\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{adjustbox}{max width=\textwidth}", r"\begin{tabular}{l" + "c" * 6 + "}", r"\toprule",
             r"& \multicolumn{3}{c}{All windows: mean [block-bootstrap 95\% CI] (M3 / M4)} & "
             r"\multicolumn{3}{c}{Dense ($N_a{>}200$): mean [CI]} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}", r"Method & 33 & 66 & 132 & 33 & 66 & 132 \\", r"\midrule"]
    for k in keep:
        s = df[df.row == k].set_index(["dense", "H"])
        cells = []
        for dense in (False, True):
            for h in HS:
                if (dense, h) not in s.index:
                    cells.append("--")
                    continue
                r = s.loc[(dense, h)]
                c = f"{r['mean']:.3f} [{r.block_lo:.3f}, {r.block_hi:.3f}]".replace("0.", ".")
                if not dense:
                    c += f" ({r.M3:.2f}/{r.M4:.2f})".replace("0.", ".")
                cells.append(c)
        lines.append(k + " & " + " & ".join(cells) + r" \\")
    n = df[(df.row == "\\ours, full") & ~df.dense].set_index("H").n
    nd = df[(df.row == "\\ours, full") & df.dense].set_index("H").n
    miss = df.groupby("row")[["missing", "extra"]].max().max()
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}",
              r"\caption{\textbf{Table~2 under one protocol.} Every row recomputed from its per-window results by the "
              r"code that writes Tab.~2, on the same windows (" + "/".join(str(int(v)) for v in n) + " windows at "
              r"33/66/132 minutes, " + "/".join(str(int(v)) for v in nd) + r" dense; at most "
              + f"{int(miss.missing)} missing and {int(miss.extra)} extra windows in any row"
              r"). CIs: time-block bootstrap, blocks of four consecutive windows within each movie (5000 resamples), "
              r"which respects the overlap of windows; in parentheses the means over M3 and M4 separately. Seeded "
              r"rows: per-window means over three seeds. Windows, not movies, are the resampling unit because only two "
              r"test movies exist; the per-movie means show the between-movie spread.}",
              r"\label{tab:oiamci}", r"\end{table*}"]
    (ROOT / "paper/tables/oiam_ci.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
