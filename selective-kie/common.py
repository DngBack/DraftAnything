"""Shared data loading, normalization, error taxonomy and risk-coverage metrics for the CORD pilot."""
import json
import re

import numpy as np

# schema field -> (CORD gt_parse path, description used in prompts)
FIELDS = {
    "subtotal": (("sub_total", "subtotal_price"), "subtotal amount (before tax, service charge and discount)"),
    "tax": (("sub_total", "tax_price"), "tax amount"),
    "service": (("sub_total", "service_price"), "service charge amount"),
    "discount": (("sub_total", "discount_price"), "receipt-level discount amount"),
    "total": (("total", "total_price"), "total amount to pay"),
    "cash": (("total", "cashprice"), "cash amount tendered by the customer"),
    "change": (("total", "changeprice"), "change returned to the customer"),
    "card": (("total", "creditcardprice"), "amount paid by credit/debit card"),
    "qty": (("total", "menuqty_cnt"), "total quantity of items"),
}
CAT2FIELD = {".".join(p): f for f, (p, _) in FIELDS.items()}
EXCLUDED = "__EXCLUDED__"  # multi-valued gold: out of scope (single-span fields only)


def parse_num(s):
    """Parse a printed amount/count to float, or None. '60.000'->60000, '2.00'->2, '60,000.50'->60000.5.

    Sign is dropped: a discount printed '-5.000' and answered '5.000' is the same value.
    ponytail: separator heuristic (3 trailing digits = thousands, else decimal) - fails on '1.500' meaning 1.5.
    """
    if s is None:
        return None
    t = re.sub(r"[^\d.,]", "", str(s))
    if not re.search(r"\d", t):
        return None
    t = t.strip(".,")
    seps = [i for i, c in enumerate(t) if c in ".,"]
    if seps:
        last = seps[-1]
        if len(t) - last - 1 == 3:
            t = re.sub(r"[.,]", "", t)
        else:
            t = re.sub(r"[.,]", "", t[:last]) + "." + t[last + 1:]
    return float(t)


NUM = re.compile(r"\d[\d.,]*\d|\d")


def numbers_in(text):
    """Numeric substrings of an OCR line, e.g. 'P. Resto 10% 2,700' -> ['10', '2,700']."""
    return NUM.findall(text)


def load_cord(split):
    """Return list of docs: id, image, gold{field: str|None|EXCLUDED}, lines[{text, cat, box}]."""
    import datasets

    ds = datasets.load_dataset("naver-clova-ix/cord-v2", split=split)
    docs = []
    for i, x in enumerate(ds):
        g = json.loads(x["ground_truth"])
        gp = g["gt_parse"]
        gold = {}
        for f, ((a, b), _) in FIELDS.items():
            sect = gp.get(a, {})
            if not isinstance(sect, dict):
                gold[f] = EXCLUDED
                continue
            v = sect.get(b)
            gold[f] = EXCLUDED if isinstance(v, list) else v
        lines = []
        for l in g["valid_line"]:
            q = [w["quad"] for w in l["words"]]
            box = [min(w["x1"] for w in q), min(w["y1"] for w in q), max(w["x3"] for w in q), max(w["y3"] for w in q)]
            lines.append({
                "text": " ".join(w["text"] for w in l["words"]),
                "value": " ".join(w["text"] for w in l["words"] if not w["is_key"]),  # annotation only, for labels
                "cat": l["category"], "box": box,
            })
        docs.append({"id": f"{split}/{i}", "image": x["image"].convert("RGB"), "gold": gold, "lines": lines})
    return docs


def candidates(lines):
    """Inference-time candidate values: unique numbers in OCR text (no categories or key flags used)."""
    seen, out = set(), []
    for l in lines:
        for t in numbers_in(l["text"]):
            v = parse_num(t)
            if v not in seen:
                seen.add(v)
                out.append({"surface": t, "value": v})
    return out


def norm_value(s):
    """Operational value: a printed zero amount is equivalent to ABSENT."""
    v = parse_num(s)
    return None if v == 0 else v


def label(pred, gold, lines, field):
    """Return (correct, error_type), error_type in {None, binding, binding_absent, absent_halluc, miss, transcription}.

    binding(_absent) = predicted value is printed on the page but annotated as another category
    (gold present / gold ABSENT respectively).
    """
    pv, gv = norm_value(pred), norm_value(gold)
    if pv == gv:
        return True, None
    if pv is None:
        return False, "miss"
    own = CAT2FIELD_INV[field]
    if pv in {parse_num(l["value"]) for l in lines if l["cat"] != own}:
        return False, "binding" if gv is not None else "binding_absent"
    return False, "absent_halluc" if gv is None else "transcription"


CAT2FIELD_INV = {f: c for c, f in CAT2FIELD.items()}


# ---------------- risk-coverage ----------------
def rc_curve(score, err):
    """Sort by score desc; return coverage and risk arrays for accepting the top-k, k=1..N (ties grouped)."""
    score, err = np.asarray(score, float), np.asarray(err, float)
    o = np.argsort(-score, kind="stable")
    s, e = score[o], err[o]
    last = np.r_[s[1:] != s[:-1], True]  # only thresholds at tie-group ends are realisable
    k = np.arange(1, len(s) + 1)[last]
    risk = np.cumsum(e)[last] / k
    return k / len(s), risk, s[last]


def coverage_at_risk(score, err, target):
    """Max coverage with empirical risk <= target (in-sample, for curves/oracle only)."""
    cov, risk, _ = rc_curve(score, err)
    ok = risk <= target + 1e-12
    return float(cov[ok].max()) if ok.any() else 0.0


def threshold_at_risk(score, err, target):
    """Threshold chosen on calibration data: lowest score whose accepted-set risk <= target."""
    cov, risk, thr = rc_curve(score, err)
    ok = risk <= target + 1e-12
    return float(thr[ok][np.argmax(cov[ok])]) if ok.any() else np.inf


def apply_threshold(score, err, thr):
    score, err = np.asarray(score), np.asarray(err, float)
    acc = score >= thr
    return acc.mean(), (err[acc].mean() if acc.any() else float("nan"))


def aurc(score, err):
    cov, risk, _ = rc_curve(score, err)
    n = len(err)
    # area under the step curve at tie-group ends, weighted by group size
    widths = np.diff(np.r_[0, cov])
    return float((risk * widths).sum()) if n else float("nan")


if __name__ == "__main__":
    assert parse_num("60.000") == 60000 and parse_num("2.00") == 2 and parse_num("Rp 60,000.50") == 60000.5
    assert parse_num("-60.000") == 60000 and parse_num(None) is None and parse_num("null") is None
    assert parse_num("1.5") == 1.5 and parse_num("12") == 12 and parse_num("60 .000") == 60000
    assert numbers_in("P. Resto 10% 2,700") == ["10", "2,700"] and numbers_in("Rp. 50.000") == ["50.000"]
    lines = [{"value": "118.000", "cat": "total.total_price"}, {"value": "18.000", "cat": "total.changeprice"}]
    assert label("18.000", "118.000", lines, "total") == (False, "binding")
    assert label("118,000", "118.000", lines, "total") == (True, None)
    assert label(None, "118.000", lines, "total") == (False, "miss")
    assert label("5.000", None, lines, "tax") == (False, "absent_halluc")
    assert label("18.000", None, lines, "tax") == (False, "binding_absent")
    assert label("0", None, lines, "tax") == (True, None) and label(None, "0", lines, "tax") == (True, None)
    assert label("119.000", "118.000", lines, "total") == (False, "transcription")
    # risk-coverage: perfect ranking accepts all correct ones
    s, e = [0.9, 0.8, 0.7, 0.1], [0, 0, 0, 1]
    assert coverage_at_risk(s, e, 0.0) == 0.75 and coverage_at_risk(s, e, 0.25) == 1.0
    # ties cannot be split
    assert coverage_at_risk([1, 1, 0], [0, 1, 0], 0.0) == 0.0
    t = threshold_at_risk(s, e, 0.0)
    assert t == 0.7 and apply_threshold(s, e, t) == (0.75, 0.0)
    assert abs(aurc([3, 2, 1], [0, 0, 1]) - (0 + 0 + 1 / 3) / 3) < 1e-9
    print("common.py self-check ok")
