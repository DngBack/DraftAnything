"""Logit calibration applied to free-form generation.

Greedy generation is left untouched until the first step whose (raw) argmax is an answer token; at that step the
candidate scores (max over surface variants per class) are recorded, and in calibrated mode the class is re-chosen by an
affine calibrator (logistic regression on centred candidate scores, fitted on the train half) and that class's best
variant token is forced. The rest of the answer is generated freely and the text is parsed as usual.

Tasks
- countbench: 2607.09544 setting (count_repro.SYS / Q), classes 1-9.
- vlmbias:    VLMBias "main" (counterfactual images), prompts as given ("... in curly brackets"), subsets whose answers are
              single tokens. Calibration is only meaningful where an answer value is the gold in some items and the biased
              answer in others (VALID); in the other subsets gold and bias values are disjoint, so any label-fitted
              calibrator can learn "never say the typical value" -> reported, but flagged.

usage: python gen_calib.py --task vlmbias --model qwen3vl8b     (results/gencal_<task>_<model>.parquet)
       python gen_calib.py --report
"""
import argparse, glob, os, re, warnings
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from transformers import LogitsProcessor, LogitsProcessorList
from pilot import MODELS, OUT

warnings.filterwarnings("ignore")
SUBSETS = {  # sub_topic -> candidate answers (single-token)
    "Ebbinghaus illusion": ["Yes", "No"], "Müller-Lyer illusion": ["Yes", "No"], "Poggendorff illusion": ["Yes", "No"],
    "Ponzo illusion": ["Yes", "No"], "Vertical-Horizontal illusion": ["Yes", "No"], "Zöllner illusion": ["Yes", "No"],
    "Dice Pattern": ["2", "3", "4", "5"], "Tally Pattern": ["2", "3", "4", "5"], "Car Logos": ["3", "4", "5"],
    "Shoe Logos": ["1", "2", "3", "4"], "Animal Add Legs": ["2", "3", "4", "5"], "Chess Grid": ["7", "8", "9"]}
ILLUSIONS = {"Ebbinghaus illusion", "Müller-Lyer illusion", "Poggendorff illusion", "Ponzo illusion",
             "Vertical-Horizontal illusion", "Zöllner illusion"}  # the only subsets with both values in both roles, balanced
CANDS_OF = lambda g: SUBSETS.get(g) or (["a", "b"] if g == "MMVP" else ["Yes", "No"] if g in ("adversarial", "popular", "random")
                                      else [str(i) for i in range(1, 10)])
VALID = {"Ebbinghaus illusion", "Müller-Lyer illusion", "Poggendorff illusion", "Ponzo illusion",
         "Vertical-Horizontal illusion", "Zöllner illusion", "Dice Pattern", "Tally Pattern", "Car Logos"}


def items(task):
    from datasets import load_dataset
    if task == "countbench":
        from count_repro import items as cb, SYS, Q
        return [dict(image=it["image"], sys=SYS, q=Q, cands=[str(i) for i in range(1, 10)], y=str(it["y"] + 1),
                     group="countbench") for it in cb()]
    if task == "mmvp":  # 150 pairs; both images of a pair share the question and have opposite answers
        from huggingface_hub import snapshot_download
        from PIL import Image
        root = snapshot_download("MMVP/MMVP", repo_type="dataset")
        q = pd.read_csv(os.path.join(root, "Questions.csv"))
        return [dict(image=Image.open(os.path.join(root, "MMVP Images", f"{r['Index']}.jpg")).convert("RGB"), sys=None, gate="letter",
                     q=f"{r['Question']}\n{r['Options']}\nAnswer with the option's letter from the given choices directly.",
                     cands=["a", "b"], y=r["Correct Answer"].strip("()"), group="MMVP", pair=(r["Index"] - 1) // 2)
                for r in q.to_dict("records")]
    if task == "pope":  # balanced Yes/No object probing; 3 negative-sampling settings; split by image (18 questions each)
        ims, out = {}, []
        for r in load_dataset("lmms-lab/POPE", split="test"):
            k = r["image_source"]
            if k not in ims:
                ims[k] = r["image"].convert("RGB")
            out.append(dict(image=ims[k], sys=None, q=f"{r['question']} Please answer yes or no.", cands=["Yes", "No"],
                            y=r["answer"].strip().capitalize(), group=r["category"], pair=list(ims).index(k)))
        return out
    out = []
    for r in load_dataset("anvo25/vlms-are-biased", split="main"):
        if r["sub_topic"] in SUBSETS:
            im = r["image"].convert("RGB")
            im.thumbnail((768, 768))
            out.append(dict(image=im, sys=None, q=r["prompt"], cands=SUBSETS[r["sub_topic"]], y=r["ground_truth"],
                            bias=r["expected_bias"], group=r["sub_topic"], gate="{",
                            key=re.sub(r"_px\d+", "", r["image_path"])))  # same image at 3 resolutions -> one key
    return out


def parse(txt, cands):
    """answer inside {...} if present, else whole text; first candidate-like token (int or word)."""
    m = re.search(r"\{([^}]*)\}", txt)
    s = (m.group(1) if m else txt).strip()
    if cands == ["a", "b"]:
        w = re.search(r"\(([ab])\)", txt.lower()) or re.match(r"\W*([ab])\b", txt.lower())
        return w.group(1) if w else ""
    if cands[0] in ("Yes", "No"):
        w = re.search(r"\b(yes|no)\b", s.lower())
        return w.group(1).capitalize() if w else ""
    n = re.search(r"\d+", s)
    return n.group() if n else ""


class Decide(LogitsProcessor):
    """Record candidate scores at the first step whose argmax is an answer-like token (`trig_ids`: any digit for numeric
    tasks, so a multi-digit answer such as "12" is decided at its first digit); optionally re-choose the class with `pick`
    and, after forcing, ban `stop_ids` (digits) for one step so the forced answer stays a single number."""
    def __init__(self, cand_ids, pick=None, trig_ids=None, stop_ids=()):
        self.cand_ids, self.pick, self.done, self.scores, self.ban = cand_ids, pick, False, None, False
        self.trig = set(trig_ids) if trig_ids else {i for ids in cand_ids for i in ids}
        self.stop = list(stop_ids)
        self.ready = lambda ids: True  # extra gate on the generated prefix

    def __call__(self, input_ids, scores):
        if self.ban:
            self.ban = False
            scores[0, self.stop] = float("-inf")
            return scores
        if self.done:
            return scores
        if int(scores[0].argmax()) not in self.trig or not self.ready(input_ids):
            return scores
        self.done = True
        s = scores[0].float()
        self.scores = np.array([float(s[ids].max()) for ids in self.cand_ids])
        if self.pick is not None:
            c = self.pick(self.scores)
            ids = self.cand_ids[c]
            best = ids[int(s[ids].argmax())]
            scores = torch.full_like(scores, float("-inf"))
            scores[0, best] = 0
            self.ban = bool(self.stop)
        return scores


def run(task, model):
    from pilot import Runner
    its = items(task)
    R = Runner(model, ["x"])  # candidate ids are built per item below
    tok = R.p.tokenizer
    ids_cache = {}

    def cand_ids(cands, paren=False):
        """token ids that themselves decode to a candidate; with paren, also "(a"-style merged tokens (option letters)"""
        k = (tuple(cands), paren)
        if k not in ids_cache:
            pre = ("", " ", "(", " (") if paren else ("", " ")
            norm = (lambda t: t.strip().lstrip("(").lower()) if paren else (lambda t: t.strip().lower())
            ids_cache[k] = [sorted({i for p in pre for v in (c, c.lower(), c.capitalize())
                                    for i in tok.encode(p + v, add_special_tokens=False) if norm(tok.decode([i])) == c.lower()})
                            for c in cands]
        return ids_cache[k]

    # all pieces, not just the first: Llama-style tokenizers encode "3" as ["▁", "3"]
    digit_ids = sorted({i for d in range(10) for p in ("", " ") for i in tok.encode(p + str(d), add_special_tokens=False)
                        if tok.decode([i]).strip() == str(d)})

    @torch.no_grad()
    def gen(it, pick=None):
        msgs = ([{"role": "system", "content": [{"type": "text", "text": it["sys"]}]}] if it["sys"] else []) + \
               [{"role": "user", "content": [{"type": "image", "image": it["image"]}, {"type": "text", "text": it["q"]}]}]
        x = R.p.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
        numeric = it["cands"][0].isdigit()
        d = Decide(cand_ids(it["cands"], it.get("gate") == "letter"), pick, digit_ids if numeric else None, digit_ids if numeric else ())
        n0 = x["input_ids"].shape[1]
        if it.get("gate") == "{":  # vlmbias: answer requested in braces; older models (Qwen2-VL, LLaVA-1.5) answer directly
            d.ready = lambda ids, n0=n0: tok.decode(ids[0, n0:]).strip() == "" or tok.decode(ids[0, n0:][-4:]).rstrip().endswith("{")
        elif it.get("gate") == "letter":  # mmvp: the option letter must open the answer (not the article "a")
            d.ready = lambda ids, n0=n0: tok.decode(ids[0, n0:]).strip().strip("*").strip() in ("", "(", "Answer:", "Answer: (")
        o = R.m.generate(**x, max_new_tokens=384 if it.get("gate") == "{" else 24, do_sample=False, logits_processor=LogitsProcessorList([d]))
        return R.p.decode(o[0, x["input_ids"].shape[1]:], skip_special_tokens=True), d.scores

    rows = []
    for i, it in enumerate(its):
        txt, sc = gen(it)
        rows.append(dict(i=i, group=it["group"], y=it["y"], bias=it.get("bias"), raw_txt=txt, raw=parse(txt, it["cands"]),
                         scores=None if sc is None else sc.tolist()))
    df = pd.DataFrame(rows)
    # one stratified 50/50 split; calibrator per group, fitted on train-half decision-step scores
    rng = np.random.default_rng(0)
    df["train"] = False
    if "pair" in its[0]:  # keep both images of a pair on the same side
        pairs = np.array([it["pair"] for it in its])
        up = rng.permutation(np.unique(pairs))
        df["train"] = np.isin(pairs, up[: len(up) // 2])
    else:
        for _, g in df.groupby(["group", "y"]):
            df.loc[rng.permutation(g.index)[: len(g) // 2], "train"] = True
    df["pair"] = [it.get("pair", -1) for it in its]
    df["cal"], df["cal_txt"], df["cal_offline"] = df.raw, df.raw_txt, df.raw
    for grp, g in df.groupby("group"):
        cands = its[g.index[0]]["cands"]
        tr = g[g.train & g.scores.notna() & g.y.isin(cands)]
        if tr.y.nunique() < 2:
            continue
        X = lambda s: np.asarray(s) - np.mean(s)
        clf = LogisticRegression(max_iter=5000).fit(np.stack([X(s) for s in tr.scores]), tr.y.map(cands.index))
        pick = lambda s, clf=clf: int(clf.predict(X(s)[None])[0])
        for j in g[~g.train].index:
            if df.at[j, "scores"] is not None:
                df.at[j, "cal_offline"] = cands[pick(df.at[j, "scores"])]
                txt, _ = gen(its[j], pick)
                df.at[j, "cal_txt"], df.at[j, "cal"] = txt, parse(txt, cands)
    # restricted control: argmax over the candidate scores at the decision step (answer space constrained, no calibration)
    df["restricted"] = [r.raw if r.scores is None else its[r.i]["cands"][int(np.argmax(r.scores))] for r in df.itertuples()]
    df.to_parquet(f"{OUT}/gencal_{task}_{model}.parquet")
    report(f"{OUT}/gencal_{task}_{model}.parquet")


def report(path=None):
    for p in [path] if path else sorted(glob.glob(f"{OUT}/gencal_*.parquet")):
        df = pd.read_parquet(p)
        te = df[~df.train]
        print(f"== {os.path.basename(p)[7:-8]}  (test half, n={len(te)})")
        rows = []
        tr_all = df[df.train]
        remap = {}  # output-remap control: train-half lookup "model said X -> most frequent gold", per group
        for grp, t in tr_all.groupby("group"):
            mp = t.groupby("raw").y.agg(lambda v: v.mode()[0])
            remap[grp] = lambda r, mp=mp, maj=t.y.mode()[0]: mp.get(r, maj)
        te = te.assign(remap=[remap[r.group](r.raw) for r in te.itertuples()])
        aggs = [("ALL", te)] if ("countbench" in p or "mmvp" in p or "pope" in p) else [("ILLUSIONS", te[te.group.isin(ILLUSIONS)]), ("ALL valid", te[te.group.isin(VALID)])]
        for grp, g in list(te.groupby("group")) + aggs:
            trg = df[df.train & (df.group.isin(g.group.unique()))]
            const = trg.groupby("group").y.agg(lambda v: v.mode()[0])  # label-prior-only control (majority train label per group)
            rows.append(dict(group=grp, n=len(g), valid=grp in VALID or grp in ("countbench", "ALL valid", "ALL", "ILLUSIONS"),
                             const=(g.y == g.group.map(const)).mean(),
                             raw=(g.raw == g.y).mean(), remap=(g.remap == g.y).mean(),
                             restricted=(g.restricted == g.y).mean() if "restricted" in g else np.nan, calibrated=(g.cal == g.y).mean(), offline=(g.cal_offline == g.y).mean(),
                             says_bias=(g.raw == g.bias).mean() if "bias" in g and g.bias.notna().any() else np.nan))
        print(pd.DataFrame(rows).round(3).to_string(index=False), flush=True)
        if "pair" in te and (te.pair >= 0).any():  # MMVP's own metric: both images of a pair correct
            pa = lambda col: te.assign(ok=te[col] == te.y).groupby("pair").ok.all().mean()
            print(f"   pair accuracy: raw {pa('raw'):.3f}  remap {pa('remap'):.3f}  calibrated {pa('cal'):.3f}", flush=True)


def stats(draws=20):
    """Per model x answer set: online (the single generated split) accuracy with exact McNemar p-values of calibrated vs
    raw and vs the output-remap control; offline mean+-sd over `draws` stratified 50/50 splits (decision-step scores)."""
    from scipy.stats import binomtest
    sets = {"countbench": ["countbench"], "MMVP": ["MMVP"], "illusions": sorted(ILLUSIONS), "Car Logos": ["Car Logos"],
            "Dice Pattern": ["Dice Pattern"], "Tally Pattern": ["Tally Pattern"], "POPE": ["adversarial", "popular", "random"]}

    def mcnemar(a, b):
        n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
        return 1.0 if n01 + n10 == 0 else binomtest(n01, n01 + n10).pvalue

    def remap_pred(tr, te):
        mp, maj = tr.groupby("raw").y.agg(lambda v: v.mode()[0]), tr.y.mode()[0]
        return te.raw.map(lambda r: mp.get(r, maj))

    rows = []
    for p in sorted(glob.glob(f"{OUT}/gencal_*.parquet")):
        df, model = pd.read_parquet(p), os.path.basename(p)[7:-8].split("_", 1)[1]
        for name, groups in sets.items():
            d = df[df.group.isin(groups)]
            if d.empty:
                continue
            te = d[~d.train]
            rm = pd.concat([remap_pred(d[d.train & (d.group == g)], te[te.group == g]) for g in groups])
            ok = lambda col: (col == te.y).to_numpy()
            raw, cal, rem = ok(te.raw), ok(te.cal), ok(rm.loc[te.index])
            r = dict(model=model, set=name, n=len(te), const=np.nan, raw=raw.mean(), remap=rem.mean(), cal=cal.mean(),
                     p_raw=mcnemar(cal, raw), p_remap=mcnemar(cal, rem), nodec=round(float(d.scores.isna().mean()), 3))
            if "pair" in te and (te.pair >= 0).all():  # MMVP: both images of a pair right
                pr = lambda v: pd.Series(v, index=te.index).groupby(te.pair).all().mean()
                r |= dict(pair_raw=pr(raw), pair_remap=pr(rem), pair_cal=pr(cal))
            if name == "POPE":  # yes-ratio (POPE's bias statistic) and the hardest setting
                adv = (te.group == "adversarial").to_numpy()
                r |= dict(yes_raw=(te.raw == "Yes").mean(), yes_cal=(te.cal == "Yes").mean(),
                          adv_raw=raw[adv].mean(), adv_cal=cal[adv].mean())
            rng, acc = np.random.default_rng(0), {k: [] for k in ("raw", "remap", "cal", "const")}
            for _ in range(draws):  # offline re-splits
                tr = np.zeros(len(d), bool)
                pos = {ix: k for k, ix in enumerate(d.index)}
                if "pair" in d and (d.pair >= 0).all():  # MMVP: split by pair
                    up = rng.permutation(d.pair.unique())
                    tr = d.pair.isin(up[: len(up) // 2]).to_numpy()
                else:
                    for _, g in d.groupby(["group", "y"]):
                        tr[[pos[i] for i in rng.permutation(g.index)[: len(g) // 2]]] = True
                for k in acc:
                    acc[k].append(0)
                for g in groups:
                    gm = (d.group == g).to_numpy()
                    a, b = d[gm & tr], d[gm & ~tr]
                    cands = CANDS_OF(g)
                    pred = b.raw.to_numpy(dtype=object).copy()
                    fit = a[a.scores.notna() & a.y.isin(cands)]
                    if fit.y.nunique() >= 2:
                        X = lambda S: np.stack([np.asarray(s) - np.mean(s) for s in S])
                        clf = LogisticRegression(max_iter=5000).fit(X(fit.scores), fit.y.map(cands.index))
                        has = b.scores.notna().to_numpy()
                        if has.any():
                            pred[has] = [cands[c] for c in clf.predict(X(b.scores[has]))]
                    acc["cal"][-1] += (pred == b.y.to_numpy(dtype=object)).sum()
                    acc["raw"][-1] += (b.raw == b.y).sum()
                    acc["remap"][-1] += (remap_pred(a, b) == b.y).sum()
                    acc["const"][-1] += (b.y == a.y.mode()[0]).sum()
                for k in acc:
                    acc[k][-1] /= (~tr).sum()
            r["const"] = np.mean(acc["const"])
            r |= {f"off_{k}": f"{100 * np.mean(v):.1f}±{100 * np.std(v):.1f}" for k, v in acc.items() if k != "const"}
            rows.append(r)
    out = pd.DataFrame(rows)
    for c in ("const", "raw", "remap", "cal", "pair_raw", "pair_remap", "pair_cal", "yes_raw", "yes_cal", "adv_raw", "adv_cal"):
        if c in out:
            out[c] = (100 * out[c]).round(1)
    out[["p_raw", "p_remap"]] = out[["p_raw", "p_remap"]].map(lambda v: f"{v:.1e}")
    print(out.sort_values(["set", "model"]).to_string(index=False))
    out.to_csv(f"{OUT}/gencal_stats.csv", index=False)


def _check():
    assert parse("{5}", ["2", "5"]) == "5" and parse("The answer is {Yes}.", ["Yes", "No"]) == "Yes"
    assert parse("(b) Closed", ["a", "b"]) == "b" and parse("A", ["a", "b"]) == "a" and parse("a dog", ["a", "b"]) == "a"
    assert parse("<10>", [str(i) for i in range(1, 10)]) == "10" and parse("none", ["Yes", "No"]) == ""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["countbench", "vlmbias", "mmvp", "pope"])
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--stats", action="store_true", help="McNemar tests + offline re-splits")
    a = ap.parse_args()
    _check()
    if a.model:
        run(a.task, a.model)
    elif a.report:
        report()
    elif a.stats:
        stats()
