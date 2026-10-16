"""R1 B + C1 (docs/r1_protocol.md): counterfactual partners and a paraphrased prompt for the synthetic closed-form tasks.

For item s (same seeds as pilot.py): the partner (stimuli.py: one object removed / objects swapped / target recoloured;
same prompt) -> candidate scores + last-position states (all layers); the original image with a paraphrased prompt ->
candidate scores. Saved to results/r1/<model>_<family>_cf.npz.

usage: python cf_pairs.py --model qwen3vl4b [--n 600]
"""
import argparse, os, re
import numpy as np, torch
import stimuli as S
from pilot import CANDS, MODELS, OUT, Runner

PARA = {"count": lambda q: "Count the {}s in this picture. Reply with one digit only.".format(
            re.match(r"How many (\w+)s are", q).group(1)),
        "spatial": lambda q: "Relative to the blue {1}, where is the red {0}? Reply with exactly one of: left, right, above, below.".format(
            *re.match(r"Where is the red (\w+) relative to the blue (\w+)\?", q).groups()),
        "attr": lambda q: "Which colour is the {}? Reply with a single word.".format(re.match(r"What color is the (\w+)\?", q).group(1))}
GEN = {"count": S.count_item, "spatial": S.spatial_item, "attr": S.attr_item}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--n", type=int, default=600)
    a = ap.parse_args()
    os.makedirs(f"{OUT}/r1", exist_ok=True)
    R = None
    for fam in GEN:
        path = f"{OUT}/r1/{a.model}_{fam}_cf.npz"
        if os.path.exists(path) or not os.path.exists(f"{OUT}/{a.model}_{fam}_states.npz"):
            continue
        cands = CANDS[fam]
        if R is None:
            R = Runner(a.model, cands)
        else:  # same model, new candidate set (as in Runner.__init__)
            tok = R.p.tokenizer
            R.cand = [sorted({tok.encode(v, add_special_tokens=False)[0] for v in (c, c.capitalize(), " " + c, " " + c.capitalize())})
                      for c in cands]
        pidx, py, pans, psc, plast, qy, qans, qsc = [], [], [], [], [], [], [], []
        for s in range(a.n):
            it, pt = GEN[fam](s)
            para = dict(it, q=PARA[fam](it["q"]))
            sc, _ = R.run(R.inputs(para))
            qy.append(cands.index(it["label"])), qans.append(int(sc.argmax())), qsc.append(sc.cpu().numpy())
            if pt is None:
                continue
            sc, hs = R.run(R.inputs(pt), hidden=True)
            pidx.append(s), py.append(cands.index(pt["label"])), pans.append(int(sc.argmax())), psc.append(sc.cpu().numpy())
            plast.append(torch.stack([h[0, -1] for h in hs]).float().cpu().half())
        np.savez(path, pidx=np.array(pidx), p_y=np.array(py), p_ans=np.array(pans), p_scores=np.stack(psc),
                 p_last=torch.stack(plast).numpy(), q_y=np.array(qy), q_ans=np.array(qans), q_scores=np.stack(qsc))
        print(f"{a.model} {fam}: partners={len(pidx)} partner_acc={np.mean(np.array(pans) == np.array(py)):.3f} "
              f"para_acc={np.mean(np.array(qans) == np.array(qy)):.3f}", flush=True)


if __name__ == "__main__":
    main()
