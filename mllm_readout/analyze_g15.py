"""G1.5 decision (docs/g15_protocol.md) from results/*_states.npz.

Per cell: model acc, calibration acc (affine refit of own answer logits), best probe acc over
layers x {last, img}, and the bias/loss/perc decomposition of model errors (all 5-fold CV).
"""
import glob, os
import numpy as np, pandas as pd
from pilot import probe_cv, OUT


def cell(path):
    d = np.load(path)
    y, ans, scores = d["y"], d["ans"], d["scores"]
    err = ans != y
    cal = probe_cv(scores, y)[0]
    best = None
    for pos in ("last", "img"):
        pr, _ = probe_cv(d[pos if pos == "last" else "imgm"], y)  # (layers, n)
        acc = (pr == y).mean(1)
        li = int(acc.argmax())
        if best is None or acc[li] > best[0]:
            best = (acc[li], li, pos, pr[li])
    pacc, li, pos, prb = best
    ne = max(int(err.sum()), 1)
    bias = err & (cal == y)
    loss = err & (cal != y) & (prb == y)
    m, c = 1 - err.mean(), (cal == y).mean()
    return dict(cell=os.path.basename(path).replace("_states.npz", ""), n=len(y), n_err=int(err.sum()),
                model=m, calib=c, probe=pacc, probe_at=f"L{li}/{pos}",
                bias=bias.sum() / ne, loss=loss.sum() / ne, perc=1 - (bias.sum() + loss.sum()) / ne,
                cal_recovers=(c - m) / (pacc - m) if pacc > m else np.nan)


def main():
    df = pd.DataFrame([cell(p) for p in sorted(glob.glob(f"{OUT}/*_states.npz"))])
    df.to_csv(f"{OUT}/g15_table.csv", index=False)
    print(df.round(3).to_string(index=False))
    ok = df[df.n_err >= 30]
    pcal = (ok.cal_recovers >= 0.5).mean()
    pread = ((ok.loss >= 0.3) & (ok.loss > ok.bias)).mean()
    verdict = "P-cal" if pcal >= 2 / 3 else "P-read" if pread >= 2 / 3 else "mixed -> 4-way diagnostic"
    print(f"\ncells with >=30 errors: {len(ok)} | calib recovers >=50% of gap: {pcal:.2f} | "
          f"loss-dominant: {pread:.2f} | verdict: {verdict}")


if __name__ == "__main__":
    main()
