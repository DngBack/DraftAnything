"""Protocol D analysis (docs/protocol_decide.md). Written before any main-run output existed."""
import argparse
import json
import re
import sys
from glob import glob
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_percept import auroc, threshold  # noqa: E402
from evc.data import DATA  # noqa: E402
from evc.degrade import SCALES  # noqa: E402
from evc.stimuli import swap_digit  # noqa: E402
from pilot_percept import dig, nums  # noqa: E402

CONF = re.compile(r"\d+(?:\.\d+)?")


def split_conf(text):
    """Number part and stated confidence of a `conf` output; (text, nan) when no '|' is present."""
    if "|" not in text:
        return text, float("nan")
    a, b = text.split("|", 1)
    m = CONF.search(b)  # first number after the bar: some models keep writing prose afterwards
    return a, float(m.group(0)) if m else float("nan")


def load(name, tag, setname):
    S = {i["qid"]: i for i in json.load(open(DATA / setname))}
    rows = [json.loads(l) for p in sorted(glob(str(ROOT / "results" / f"{name}_{tag}_*.jsonl"))) for l in open(p)]
    return S, rows


class Table:
    def __init__(self, S, rows):
        self.qs = sorted({r["qid"] for r in rows})
        self.ix = {q: k for k, q in enumerate(self.qs)}
        self.S = S
        self.D = {}
        n = len(self.qs)
        for r in rows:
            it = S[r["qid"]]
            text, conf = split_conf(r["text"]) if r["prompt"] == "conf" else (r["text"], float("nan"))
            got = set(nums(text))
            orig, shown, plc = dig(it["word"]), dig(swap_digit(it["word"], it["pos"], it["d_s"])), dig(swap_digit(it["word"], it["pos"], it["d_a"]))
            target = shown if r["cond"] == "cf" else dig(it["decoy_word"]) if r["cond"] == "decoy" else orig
            dl = [x for x, w in zip(r["lps"], r["toks"]) if any(c.isdigit() for c in w)] or [0.0]
            v = dict(T=target in got, O=orig in got, A=plc in got, lp=r["lp"], U="unreadable" in r["text"].lower(), C=conf,
                     V=orig in got, lp1=r["lps"][0] if r["lps"] else 0.0, lpd=float(np.mean(dl)), lpm=min(dl))
            for k, x in v.items():
                self.D.setdefault((k, r["block"], r["level"], r["cond"], r["prompt"]), np.full(n, np.nan))[self.ix[r["qid"]]] = float(x)
        rng = np.random.default_rng(0)
        self.boots = [rng.integers(0, n, n) for _ in range(2000)]
        self.h = np.array([S[q]["bbox"][3] - S[q]["bbox"][1] for q in self.qs], float)

    def g(self, k, b, lv, c, p):
        return self.D[(k, b, lv, c, p)]

    def ci(self, fn):
        return np.nanpercentile([fn(i) for i in self.boots], [2.5, 97.5])


def f(e, ci, k=100):
    star = " *" if ci[0] > 0 or ci[1] < 0 else ""
    return f"{k * e:+.1f} [{k * ci[0]:+.1f}, {k * ci[1]:+.1f}]{star}"


def m(x):
    return f"{100 * np.nanmean(x):.1f}"


def au(score, y, i=None):
    i = np.arange(len(y)) if i is None else i
    s, t = score[i], y[i]
    ok = ~np.isnan(s + t)
    return auroc(s[ok], t[ok] > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--name", default="decide")
    ap.add_argument("--set", default="decide_set.json")
    a = ap.parse_args()
    S, rows = load(a.name, a.tag, a.set)
    t = Table(S, rows)
    g = t.g
    L = [f"# Protocol D, {a.tag}, n = {len(t.qs)} ô", ""]
    # validity
    to, tn, tc = g("T", 1, "s100", "orig", "copy"), g("T", 1, "s100", "null", "copy"), g("T", 1, "s100", "cf", "copy")
    dn = tn - to
    L += ["## Hợp lệ", "", f"- T(null) − T(orig), s100: {f(np.nanmean(dn), t.ci(lambda i: np.nanmean(dn[i])))} (cần trị tuyệt đối ≤ 5)",
          f"- T(cf), s100: {m(tc)} (cần ≥ 80); T(orig) {m(to)}",
          f"- `boxonly` chép ô mồi: s100 {m(g('T', 3, 's100', 'decoy', 'boxonly'))} (cần ≥ 70), khung s40 {m(g('T', 3, 's40', 'decoy', 'boxonly'))}",
          f"- `L` trên ảnh s100: {m(g('T', 3, 's100', 'L', 'boxcopy'))}", ""]
    # T table
    L += ["## Khối 1: T theo mức", "", "| mức | orig copy | null copy | cf copy | orig addr | cf addr |", "|---|---|---|---|---|---|"]
    for lv in ("s100", "s50", "s40", "s30"):
        L.append(f"| {lv} | {m(g('T', 1, lv, 'orig', 'copy'))} | {m(g('T', 1, lv, 'null', 'copy'))} | {m(g('T', 1, lv, 'cf', 'copy'))} | {m(g('T', 1, lv, 'orig', 'addr'))} | {m(g('T', 1, lv, 'cf', 'addr'))} |")
    # D1
    L += ["", "## D1: định vị × đọc nét", "", "| mức | base | L | G | LG | φ_L | φ_G | tương tác | φ_G − φ_L | base − orig hạ thật |", "|---|---|---|---|---|---|---|---|---|---|"]
    for lv in ("s50", "s40", "s30"):
        b, l, gg, lg = g("T", 2, lv, "base", "copy"), g("T", 2, lv, "L", "boxcopy"), g("T", 2, lv, "G", "copy"), g("T", 2, lv, "LG", "boxcopy")
        pl = ((l - b) + (lg - gg)) / 2
        pg = ((gg - b) + (lg - l)) / 2
        it = lg - l - gg + b
        ci = lambda x: f(np.nanmean(x), t.ci(lambda i: np.nanmean(x[i])))  # noqa: E731,B023
        dd = b - g("T", 1, lv, "orig", "copy")
        L.append(f"| {lv} | {m(b)} | {m(l)} | {m(gg)} | {m(lg)} | {ci(pl)} | {ci(pg)} | {ci(it)} | {ci(pg - pl)} | {ci(dd)} |")
    # D2
    L += ["", "## D2: O − A trên ảnh cf", "", "| mức | prompt | O | A | O − A |", "|---|---|---|---|---|"]
    for lv in ("s50", "s40", "s30"):
        for p in ("copy", "addr"):
            o, x = g("O", 1, lv, "cf", p), g("A", 1, lv, "cf", p)
            d = o - x
            L.append(f"| {lv} | {p} | {m(o)} | {m(x)} | {f(np.nanmean(d), t.ci(lambda i: np.nanmean(d[i])))} |")  # noqa: B023
    # D3
    L += ["", "## D3: T(copy) − T(addr), ảnh orig", ""]
    for lv in ("s100", "s50", "s40", "s30"):
        d = g("T", 1, lv, "orig", "copy") - g("T", 1, lv, "orig", "addr")
        L.append(f"- {lv}: {f(np.nanmean(d), t.ci(lambda i: np.nanmean(d[i])))}")  # noqa: B023
    # D4a
    L += ["", "## D4a: tín hiệu ẩn (copy, orig)", "", "| mức | đúng % | AUROC log-prob | AUROC cỡ chữ | hiệu | cặp bất đồng đúng > sai |", "|---|---|---|---|---|---|"]
    for lv in ("s100", "s50", "s40", "s30"):
        y, lp = g("T", 1, lv, "orig", "copy"), g("lp", 1, lv, "orig", "copy")
        e = au(lp, y)
        c = t.ci(lambda i: au(lp, y, i))  # noqa: B023
        hu = au(t.h, y)
        dc = t.ci(lambda i: au(lp, y, i) - au(t.h, y, i))  # noqa: B023
        conds = [("orig", "copy"), ("null", "copy"), ("cf", "copy")]
        Y = np.stack([g("T", 1, lv, *x) for x in conds], 1)
        P = np.stack([g("lp", 1, lv, *x) for x in conds], 1)
        per = np.full(len(t.qs), np.nan)
        for q in range(len(t.qs)):
            w = [P[q, u] - P[q, v] for u in range(3) for v in range(3) if Y[q, u] == 1 and Y[q, v] == 0]
            if w:
                per[q] = np.mean(np.array(w) > 0)
        pc = t.ci(lambda i: np.nanmean(per[i]))  # noqa: B023
        L.append(f"| {lv} | {m(y)} | {e:.3f} [{c[0]:.3f}, {c[1]:.3f}] | {hu:.3f} | {f(e - hu, dc, 1)} | {100 * np.nanmean(per):.1f} [{100 * pc[0]:.1f}, {100 * pc[1]:.1f}], {int((~np.isnan(per)).sum())} ô |")
    # D4b
    L += ["", "## D4b: từ chối bằng lời", "", "| mức | prompt | tỷ lệ từ chối | P(từ chối \\| copy sai) | P(từ chối \\| copy đúng) | hiệu |", "|---|---|---|---|---|---|"]
    for lv in ("s100", "s50", "s40", "s30"):
        y = g("T", 1, lv, "orig", "copy")
        for p in ("copyu", "copyu2"):
            u = g("U", 1, lv, "orig", p)

            def dif(i):  # noqa: B023
                yi, ui = y[i], u[i]  # noqa: B023
                return np.nanmean(ui[yi == 0]) - np.nanmean(ui[yi == 1])
            L.append(f"| {lv} | {p} | {m(u)} | {m(u[y == 0])} | {m(u[y == 1])} | {f(dif(np.arange(len(y))), t.ci(dif))} |")
    # D4c
    L += ["", "## D4c: log-prob so với độ tự tin ghi ra (prompt conf)", "", "| mức | đúng % | có độ tự tin % | TB tự tin (đúng / sai) | AUROC log-prob | AUROC tự tin | hiệu |", "|---|---|---|---|---|---|---|"]
    for lv in ("s100", "s50", "s40", "s30"):
        y, lp, c = g("T", 1, lv, "orig", "conf"), g("lp", 1, lv, "orig", "conf"), g("C", 1, lv, "orig", "conf")
        have = ~np.isnan(c)
        e1, e2 = au(lp, y), au(c, y)
        d = t.ci(lambda i: (lambda j: au(lp, y, j) - au(c, y, j))(i[have[i]]))  # noqa: B023
        L.append(f"| {lv} | {m(y)} | {100 * have.mean():.1f} | {np.nanmean(c[y == 1]):.1f} / {np.nanmean(c[y == 0]):.1f} | {au(lp[have], y[have]):.3f} | {e2:.3f} | {f(au(lp[have], y[have]) - e2, d, 1)} |")
    # descriptive
    L += ["", "## Mô tả: ngưỡng chiều cao chữ (px, mức đúng 50 %)", ""]
    for p in ("copy", "addr", "program"):
        hs = np.concatenate([t.h * SCALES[lv] for lv in ("s100", "s50", "s40", "s30")])
        ys = np.concatenate([g("T", 1, lv, "orig", p) for lv in ("s100", "s50", "s40", "s30")])
        ok = ~np.isnan(ys)
        n = len(t.qs)
        ci = np.nanpercentile([threshold(hs[np.r_[i, i + n, i + 2 * n, i + 3 * n]][ok[np.r_[i, i + n, i + 2 * n, i + 3 * n]]],
                                         ys[np.r_[i, i + n, i + 2 * n, i + 3 * n]][ok[np.r_[i, i + n, i + 2 * n, i + 3 * n]]]) for i in t.boots[:300]], [2.5, 97.5])
        L.append(f"- {p}: {threshold(hs[ok], ys[ok]):.1f} [{ci[0]:.1f}, {ci[1]:.1f}]")
    L += ["", "## Mô tả: từ chối bằng log-prob (ngưỡng chọn trên nửa tài liệu, đánh giá trên nửa kia, độ phủ mục tiêu 50 %)", "", "| mức | đúng khi trả lời hết | phủ thực | đúng khi trả lời | `copyu2`: phủ / đúng khi trả lời |", "|---|---|---|---|---|"]
    half = np.arange(len(t.qs)) % 2 == 0
    for lv in ("s50", "s40", "s30"):
        y, lp = g("T", 1, lv, "orig", "copy"), g("lp", 1, lv, "orig", "copy")
        tau = np.nanmedian(lp[half])
        keep = (~half) & (lp >= tau)
        u = g("U", 1, lv, "orig", "copyu2")
        y2 = g("T", 1, lv, "orig", "copyu2")
        kv = (~half) & (u == 0)
        L.append(f"| {lv} | {m(y[~half])} | {100 * keep.sum() / (~half).sum():.1f} | {m(y[keep])} | {100 * kv.sum() / (~half).sum():.1f} / {m(y2[kv])} |")
    # additions from the related-work review, still written before main-run output
    L += ["", "## Mô tả: các cách gộp log-prob (AUROC, copy, orig)", "", "| mức | TB mọi token | token đầu | TB chữ số | min chữ số |", "|---|---|---|---|---|"]
    for lv in ("s100", "s50", "s40", "s30"):
        y = g("T", 1, lv, "orig", "copy")
        L.append(f"| {lv} | " + " | ".join(f"{au(g(k, 1, lv, 'orig', 'copy'), y):.3f}" for k in ("lp", "lp1", "lpd", "lpm")) + " |")
    hp = np.concatenate([t.h * SCALES[lv] for lv in ("s100", "s50", "s40", "s30")])
    yy = np.concatenate([g("T", 1, lv, "orig", "conf") for lv in ("s100", "s50", "s40", "s30")])
    lpp = np.concatenate([g("lp", 1, lv, "orig", "conf") for lv in ("s100", "s50", "s40", "s30")])
    cc = np.concatenate([g("C", 1, lv, "orig", "conf") for lv in ("s100", "s50", "s40", "s30")])
    L += ["", "## Mô tả: khoảng cách log-prob và lời nói theo chiều cao chữ (prompt conf, gộp các mức)", "", "| px | n | đúng % | TB tự tin | AUROC log-prob | AUROC tự tin | ECE tự tin |", "|---|---|---|---|---|---|---|"]
    for lo, hi in ((0, 6), (6, 8), (8, 10), (10, 14), (14, 99)):
        k = (hp >= lo) & (hp < hi) & ~np.isnan(yy) & ~np.isnan(cc)
        if k.sum() < 20:
            L.append(f"| {lo}–{hi} | {k.sum()} | – | – | – | – | – |")
            continue
        pc, yk = cc[k] / 100, yy[k]
        ece = sum(((pc >= a0) & (pc < a0 + 0.1 + (a0 > 0.85))).mean() * abs(pc[(pc >= a0) & (pc < a0 + 0.1 + (a0 > 0.85))].mean() - yk[(pc >= a0) & (pc < a0 + 0.1 + (a0 > 0.85))].mean())
                  for a0 in np.arange(0, 1, 0.1) if ((pc >= a0) & (pc < a0 + 0.1 + (a0 > 0.85))).any())
        L.append(f"| {lo}–{hi} | {k.sum()} | {100 * yk.mean():.1f} | {cc[k].mean():.1f} | {auroc(lpp[k], yk > 0):.3f} | {auroc(cc[k], yk > 0):.3f} | {100 * ece:.1f} |")
    L += ["", "## Mô tả: đối chứng", ""]
    for lv in ("s100", "s40"):
        L.append(f"- hộp đỏ trên ô mồi, prompt vẫn nêu nhãn ô đích, {lv}: chép ô mồi {m(g('T', 3, lv, 'decoy', 'boxcopy'))}, chép ô đích {m(g('V', 3, lv, 'decoy', 'boxcopy'))}")
    L.append(f"- ảnh trắng: `copyu2` từ chối {m(g('U', 3, 's100', 'blank', 'copyu2'))}; `copy` có ra số {100 * np.mean([bool(nums(r['text'])) for r in rows if r['cond'] == 'blank' and r['prompt'] == 'copy']):.1f}; `conf` TB tự tin {np.nanmean(g('C', 3, 's100', 'blank', 'conf')):.1f}")
    L += ["", "## Mô tả: đọc khi chép so với khi dùng (`program`, T, orig)", ""]
    for lv in ("s100", "s50", "s40", "s30"):
        d = g("T", 1, lv, "orig", "copy") - g("T", 1, lv, "orig", "program")
        L.append(f"- {lv}: copy {m(g('T', 1, lv, 'orig', 'copy'))}, program {m(g('T', 1, lv, 'orig', 'program'))}, hiệu {f(np.nanmean(d), t.ci(lambda i: np.nanmean(d[i])))}")  # noqa: B023
    out = ROOT / "results" / f"{a.name}_{a.tag}.md"
    out.write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
