"""Reproduce 2607.09544's counting setting (CountBench, counts 1-9, 50/50 stratified split) and compare
logit calibration with their re-prompting ("Always Reprompt" is their strongest baseline; their detector-guided method
is within 0.5 pt of it on CountBench for Qwen3-VL-8B).

Their prompts: system "You are a helpful assistant. Output only the final count in the format <count>."
first pass "How many object(s) are in this image? Answer:"; correction = first pass + previous answer +
"That's definitely wrong. The correct count should be:". We score digits 1-9 by first-token logits.

usage: python count_repro.py --model qwen3vl8b   (caches results/countrepro_<model>.npz)
       python count_repro.py --report
"""
import argparse, glob, os, warnings
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from pilot import MODELS, OUT

warnings.filterwarnings("ignore")
SYS = "You are a helpful assistant. Output only the final count in the format <count>."
Q = "How many object(s) are in this image? Answer:"
DIGITS = [str(i) for i in range(1, 10)]


def items():
    from datasets import load_dataset
    out = []
    for r in load_dataset("vikhyatk/CountBenchQA", split="test"):
        if 1 <= int(r["number"]) <= 9:
            im = r["image"].convert("RGB")
            im.thumbnail((768, 768))
            out.append(dict(image=im, y=int(r["number"]) - 1))
    return out


def score(model, mt=False):
    from pilot import Runner
    R = Runner(model, DIGITS)

    def inp(im, text, prev=None):
        msgs = [{"role": "system", "content": [{"type": "text", "text": SYS}]},
                {"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": text}]}]
        if prev is not None:  # multi-turn correction: previous answer as the assistant turn
            msgs += [{"role": "assistant", "content": [{"type": "text", "text": prev}]},
                     {"role": "user", "content": [{"type": "text", "text": "That's definitely wrong. The correct count should be:"}]}]
        return R.p.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors="pt").to("cuda")
    its = items()
    S1, S2 = [], []
    for it in its:
        s1 = R.run(inp(it["image"], Q))[0].cpu().numpy()
        prev = DIGITS[int(s1.argmax())]
        s2 = R.run(inp(it["image"], Q, prev) if mt else
                   inp(it["image"], f"{Q} {prev}. That's definitely wrong. The correct count should be:"))[0].cpu().numpy()
        S1.append(s1); S2.append(s2)
    np.savez(f"{OUT}/countrepro_{model}{'_mt' if mt else ''}.npz", S1=np.stack(S1), S2=np.stack(S2), y=np.array([it["y"] for it in its]))


WORDS = "one two three four five six seven eight nine ten".split()


def parse(txt):
    """first integer (or number word) in the generated text; -1 if none."""
    import re
    m = re.search(r"\d+|\b(?:" + "|".join(WORDS) + r")\b", txt.lower())
    return -1 if m is None else (int(m.group()) if m.group().isdigit() else WORDS.index(m.group()) + 1)


@torch.no_grad()
def generate(model):
    """Their protocol with open generation (greedy, parsed): first pass, then multi-turn 'definitely wrong' re-prompt."""
    from pilot import Runner
    R = Runner(model, DIGITS)

    def gen(msgs):
        x = R.p.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
        o = R.m.generate(**x, max_new_tokens=16, do_sample=False)
        return R.p.decode(o[0, x["input_ids"].shape[1]:], skip_special_tokens=True)
    its = items()
    A1, A2 = [], []
    for it in its:
        msgs = [{"role": "system", "content": [{"type": "text", "text": SYS}]},
                {"role": "user", "content": [{"type": "image", "image": it["image"]}, {"type": "text", "text": Q}]}]
        t1 = gen(msgs)
        t2 = gen(msgs + [{"role": "assistant", "content": [{"type": "text", "text": t1}]},
                         {"role": "user", "content": [{"type": "text", "text": "That's definitely wrong. The correct count should be:"}]}])
        A1.append(parse(t1)); A2.append(parse(t2))
    y = np.array([it["y"] + 1 for it in its])
    A1, A2 = np.array(A1), np.array(A2)
    np.savez(f"{OUT}/countgen_{model}.npz", A1=A1, A2=A2, y=y)
    print(f"{model}: generation raw {100 * (A1 == y).mean():.1f}  always-reprompt {100 * (A2 == y).mean():.1f}  "
          f"(unparsed {(A1 < 0).mean():.3f}/{(A2 < 0).mean():.3f}; e.g. {t1!r} -> {t2!r})", flush=True)


def report(draws=20):
    print(f"{'model':12s} n    raw    always-reprompt  calib(50%)   calib(k=5/class)  calib(1st+2nd pass)  [test half, mean over {draws} stratified splits]")
    for p in sorted(glob.glob(f"{OUT}/countrepro_*.npz")):
        d, rng = np.load(p), np.random.default_rng(0)  # per-model seed: adding a model must not change the others' splits
        S1, S2, y = d["S1"].astype(np.float64), d["S2"].astype(np.float64), d["y"]
        S1, S2 = S1 - S1.mean(1, keepdims=True), S2 - S2.mean(1, keepdims=True)
        S12 = np.concatenate([S1, S2], 1)
        r = {k: [] for k in ("raw", "rep", "cal", "k5", "cal_rep")}
        for _ in range(draws):
            tr = np.concatenate([rng.permutation(np.where(y == c)[0])[: (y == c).sum() // 2] for c in np.unique(y)])
            te = np.setdiff1d(np.arange(len(y)), tr)
            r["raw"].append((S1[te].argmax(1) == y[te]).mean())
            r["rep"].append((S2[te].argmax(1) == y[te]).mean())
            r["cal"].append((LogisticRegression(max_iter=5000).fit(S1[tr], y[tr]).predict(S1[te]) == y[te]).mean())
            r["cal_rep"].append((LogisticRegression(max_iter=5000).fit(S12[tr], y[tr]).predict(S12[te]) == y[te]).mean())
            k5 = np.concatenate([rng.choice(np.where(y[tr] == c)[0], min(5, (y[tr] == c).sum()), replace=False) for c in np.unique(y[tr])])
            r["k5"].append((LogisticRegression(max_iter=5000).fit(S1[tr][k5], y[tr][k5]).predict(S1[te]) == y[te]).mean())
        f = lambda x: f"{100 * np.mean(x):.1f}±{100 * np.std(x):.1f}"
        print(f"{os.path.basename(p)[11:-4]:12s} {len(y):4d} {f(r['raw'])}  {f(r['rep'])}       {f(r['cal'])}   {f(r['k5'])}       {f(r['cal_rep'])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--mt", action="store_true", help="multi-turn correction prompt")
    ap.add_argument("--gen", action="store_true", help="open generation + parsing (their scoring)")
    a = ap.parse_args()
    if a.model and a.gen:
        generate(a.model)
    elif a.model:
        score(a.model, a.mt)
    if a.report:
        report()
