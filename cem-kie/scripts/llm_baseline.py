"""Reference: zero-shot full-text LLM reader (Qwen3-VL-4B-Instruct, text only) on dev-0, same 5 fields and metric.

Documents longer than MAX_TOK tokens keep head and tail (accounts usually sit at the end).
results/llm/metrics.json and preds.json, same row format as train.py.
"""
import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cemkie.data import FIELDS, SHORT, _iso, load_split, MONTHS
from cemkie.metrics import counts, f1

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'Qwen/Qwen3-VL-4B-Instruct'
MAX_TOK = 24000

PROMPT = """Below is the OCR text of an annual report of a British charity.
Extract these fields and answer with one JSON object only:
- "income": total annual income of the charity in GBP for the reporting year (a number, in pounds, not thousands)
- "spending": total annual expenditure/spending in GBP for the reporting year (a number, in pounds)
- "report_date": end date of the reporting period, YYYY-MM-DD
- "charity_number": registered charity number (digits only)
- "postcode": postcode of the charity's own address
Use null for a field that is not in the document.

DOCUMENT:
"""


def norm(f, v):
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in ('null', 'none', 'n/a'):
        return None
    if f in ('income', 'spending'):
        s = re.sub(r'[£,\s]', '', s)
        try:
            return f'{float(s):.2f}'
        except ValueError:
            return None
    if f == 'report_date':
        m = re.match(r'(\d{4})-(\d{1,2})-(\d{1,2})', s)
        if m:
            return _iso(m.group(3), m.group(2), m.group(1))
        m = re.match(r'(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})', s)
        return _iso(m.group(1), MONTHS.get(m.group(2).lower()[:3], 0), m.group(3)) if m else None
    if f == 'charity_number':
        d = re.sub(r'\D', '', s)
        return d.lstrip('0') or None
    m = re.match(r'([A-Z]{1,2}\d[A-Z\d]?)\s*(\d[A-Z]{2})', s.upper())
    return f'{m.group(1)}_{m.group(2)}' if m else None


def main():
    out_dir = ROOT / 'results' / 'llm'
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
    tok = proc.tokenizer
    docs = load_split('dev-0')
    keymap = dict(zip(FIELDS, ['income', 'spending', 'report_date', 'charity_number', 'postcode']))
    rows, t0, raw = [], time.time(), []
    for i, d in enumerate(docs):
        ids = tok(d['text'], add_special_tokens=False)['input_ids']
        if len(ids) > MAX_TOK:
            ids = ids[:MAX_TOK * 2 // 3] + ids[-MAX_TOK // 3:]
        text = tok.decode(ids)
        msgs = [{'role': 'user', 'content': [{'type': 'text', 'text': PROMPT + text}]}]
        x = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                     return_tensors='pt').to('cuda')
        with torch.no_grad():
            y = model.generate(**x, max_new_tokens=160, do_sample=False)
        ans = tok.decode(y[0, x['input_ids'].shape[1]:], skip_special_tokens=True)
        m = re.search(r'\{.*\}', ans, re.S)
        try:
            js = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            js = {}
        raw.append(dict(id=d['id'], answer=ans))
        for f in FIELDS:
            pred, gold = norm(keymap[f], js.get(keymap[f])), d['gold'].get(f)
            rows.append(dict(id=d['id'], field=SHORT[f], pred=pred, gold=gold, c=counts(pred, gold)))
        if i % 20 == 0:
            print(i, f'{time.time() - t0:.0f}s', f'micro so far {f1(r["c"] for r in rows):.3f}', flush=True)
    res = {fs: f1(r['c'] for r in rows if r['field'] == fs) for fs in SHORT.values()}
    res['micro'] = f1(r['c'] for r in rows)
    json.dump(dict(model=MODEL, max_tok=MAX_TOK, test=res, seconds=time.time() - t0),
              open(out_dir / 'metrics.json', 'w'), indent=1)
    json.dump(rows, open(out_dir / 'preds.json', 'w'))
    json.dump(raw, open(out_dir / 'raw.json', 'w'))
    print(res)


if __name__ == '__main__':
    main()
