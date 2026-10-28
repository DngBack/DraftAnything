"""Kleister-Charity loading and typed candidate generation.

A *mention* is a text span (start, end) of one type; a *candidate* is (field, mention, value).
Money mentions yield two candidates (value as written, and x1000 for accounts in GBP'000).
Values are normalised to the expected.tsv format so a candidate matches gold by string equality.
"""
import lzma
import re
from pathlib import Path

import numpy as np

ROOT = Path('/home/jovyan/bachdx2/datasets/kleister/kleister-charity')

# Fields with a closed candidate space. Names, towns and street lines are span extraction, out of scope.
FIELDS = ['income_annually_in_british_pounds', 'spending_annually_in_british_pounds', 'report_date',
          'charity_number', 'address__postcode']
FTYPE = {'income_annually_in_british_pounds': 'money', 'spending_annually_in_british_pounds': 'money',
         'report_date': 'date', 'charity_number': 'number', 'address__postcode': 'postcode'}
SHORT = dict(zip(FIELDS, ['income', 'spending', 'date', 'number', 'postcode']))

MONTHS = {m: i + 1 for i, m in enumerate(
    ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
     'november', 'december'])}
MONTHS.update({m[:3]: v for m, v in list(MONTHS.items())})
MONTHS['sept'] = 9
MON = '|'.join(sorted(MONTHS, key=len, reverse=True))

RE_MONEY = re.compile(r'(?<![\d.,/])(\(?)(£\s?)?(\d{1,3}(?:,\s?\d{3})+|\d{3,9})(\.\d{2})?(?![\d/]|,\d)')
RE_NUMBER = re.compile(r'(?<![\d.,/£])(\d{6,7})(?!\d|[.,/]\d)')
RE_POSTCODE = re.compile(r'(?<![A-Z0-9])([A-Z]{1,2}\d[A-Z\d]?) ?(\d[A-Z]{2})(?![A-Z0-9])')
RE_DATES = [
    (re.compile(rf'(?<!\d)(\d{{1,2}})(?:st|nd|rd|th)?\s+({MON})\.?,?\s+((?:19|20)\d\d)(?!\d)', re.I), 'dmy'),
    (re.compile(rf'(?<![a-z])({MON})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+((?:19|20)\d\d)(?!\d)', re.I), 'mdy'),
    (re.compile(r'(?<![\d/.])(\d{1,2})[/.\-](\d{1,2})[/.\-]((?:19|20)?\d\d)(?![\d/.])'), 'num'),
]


def load_split(split):
    """Return list of dicts: id, text, gold {field: value} restricted to FIELDS."""
    rows = [l.rstrip('\n').split('\t') for l in lzma.open(ROOT / split / 'in.tsv.xz', 'rt')]
    exp = (ROOT / split / 'expected.tsv').read_text().splitlines()
    docs = []
    for r, e in zip(rows, exp):
        gold = dict(kv.split('=', 1) for kv in e.split())
        text = r[5].replace('\\n', '\n').replace('\\t', ' ')
        docs.append(dict(id=r[0], text=text, gold={f: gold[f] for f in FIELDS if f in gold}))
    return docs


def _iso(d, m, y):
    y = int(y); y += 2000 if y < 100 else 0
    d, m = int(d), int(m)
    if not (1 <= m <= 12 and 1 <= d <= 31 and 1990 <= y <= 2030):
        return None
    return f'{y:04d}-{m:02d}-{d:02d}'


def mentions(text):
    """List of (type, start, end, values, flags). flags: paren, pound sign."""
    out = []
    for m in RE_MONEY.finditer(text):
        n = float(re.sub(r'[,\s]', '', m.group(3)) + (m.group(4) or ''))
        if 1990 <= n <= 2030 and m.group(4) is None and ',' not in m.group(3):
            continue   # bare years
        out.append(('money', m.start(3), m.end(), [f'{n:.2f}', f'{n * 1000:.2f}'],
                    (bool(m.group(1)), bool(m.group(2)))))
    for m in RE_NUMBER.finditer(text):
        out.append(('number', m.start(1), m.end(1), [m.group(1).lstrip('0') or '0'], (False, False)))
    for m in RE_POSTCODE.finditer(text.upper()):
        out.append(('postcode', m.start(), m.end(), [f'{m.group(1)}_{m.group(2)}'], (False, False)))
    for rx, kind in RE_DATES:
        for m in rx.finditer(text):
            a, b, y = m.groups()
            if kind == 'dmy':
                v = _iso(a, MONTHS[b.lower()], y)
            elif kind == 'mdy':
                v = _iso(b, MONTHS[a.lower()], y)
            else:
                v = _iso(a, b, y)
            if v:
                out.append(('date', m.start(), m.end(), [v], (False, False)))
    out.sort(key=lambda x: x[1])
    return out


def candidates(doc):
    """Mentions + per-field candidate table.

    Returns dict with
      spans [M, 2], mtype [M]  (mention-level, embedded once)
      per field f: cand_m [C] mention index, cand_v [C] value str, cand_scale [C], label [C] bool
    """
    text = doc['text']
    ms = mentions(text)
    spans = np.array([(s, e) for _, s, e, _, _ in ms], np.int64).reshape(-1, 2)
    out = dict(spans=spans, mtype=[t for t, *_ in ms], flags=np.array([f for *_, f in ms], bool).reshape(-1, 2),
               fields={})
    for f in FIELDS:
        cm, cv, cs = [], [], []
        for i, (t, _, _, vals, _) in enumerate(ms):
            if t != FTYPE[f]:
                continue
            for k, v in enumerate(vals):
                cm.append(i); cv.append(v); cs.append(k)
        g = doc['gold'].get(f)
        out['fields'][f] = dict(cand_m=np.array(cm, np.int64), cand_v=cv, cand_scale=np.array(cs, np.int64),
                                label=np.array([v == g for v in cv], bool))
    return out


def context(text, s, e, left=240, right=80):
    """Candidate context with the span marked, newlines kept as ' | ' (table rows survive OCR as lines)."""
    a = text[max(0, s - left):s].replace('\n', ' | ')
    b = text[e:e + right].replace('\n', ' | ')
    return f'{a} [[ {text[s:e]} ]] {b}'
