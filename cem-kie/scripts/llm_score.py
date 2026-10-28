"""Step 5 (CoSolve oracle): LLM log-probs for money candidates, as unaries for the joint solver.

For each val/test document: candidates = union of set4 top-M over seeds (results/topm) + the LLM greedy
answer + NULL. Each candidate is scored as log p('{\\n  "<field>": <value>,' | document) with the document
prefilled once and the KV cache cropped back after every branch. Val docs also get a greedy answer
(test reuses results/llm/raw.json).

results/llm_score/{val,test}.json: per doc {id, gold, greedy, cands: {field: {value: logp}}}.
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cemkie.data import load_split
from llm_baseline import MAX_TOK, MODEL, PROMPT, norm
from train import SPLIT_SEED, VAL_FRAC

ROOT = Path(__file__).resolve().parents[1]
MONEY = {'income': 'income_annually_in_british_pounds', 'spending': 'spending_annually_in_british_pounds'}


def as_text(v):
    """Candidate value as the LLM writes it (integer pounds when whole)."""
    if v is None:
        return 'null'
    x = float(v)
    return str(int(x)) if x == int(x) else f'{x:.2f}'


def candidate_sets(split):
    out = {}
    files = sorted((ROOT / 'results' / 'topm').glob('set4-s*/' + ('val_preds.json' if split == 'val' else 'preds.json')))
    for f in files:
        for r in json.load(open(f)):
            if r['field'] in MONEY:
                out.setdefault((r['id'], r['field']), set()).update(v for v, _ in r['top'])
    return out


@torch.no_grad()
def branch_logp(model, cache, L0, ids):
    """Sum log-prob of `ids` continuing the cached prefix of length L0; restores the cache."""
    x = torch.tensor([ids], device='cuda')
    logits = model(input_ids=x, past_key_values=cache, use_cache=True).logits[0].float()
    cache.crop(-len(ids))   # this transformers version takes a negative count to remove
    return logits, x[0]


def main():
    out_dir = ROOT / 'results' / 'llm_score'
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
    tok = proc.tokenizer

    tr = load_split('train')
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(tr))
    splits = {'val': [tr[i] for i in perm[:int(VAL_FRAC * len(perm))]], 'test': load_split('dev-0')}
    greedy_test = {x['id']: x['answer'] for x in json.load(open(ROOT / 'results' / 'llm' / 'raw.json'))}

    for split, docs in splits.items():
        if len(sys.argv) > 1 and split not in sys.argv[1:]:
            continue
        cands = candidate_sets(split)
        res, t0 = [], time.time()
        for i, d in enumerate(docs):
            ids = tok(d['text'], add_special_tokens=False)['input_ids']
            if len(ids) > MAX_TOK:
                ids = ids[:MAX_TOK * 2 // 3] + ids[-MAX_TOK // 3:]
            msgs = [{'role': 'user', 'content': [{'type': 'text', 'text': PROMPT + tok.decode(ids)}]}]
            x = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                         return_tensors='pt').to('cuda')
            if split == 'test':
                ans = greedy_test[d['id']]
            else:
                y = model.generate(**x, max_new_tokens=160, do_sample=False)
                ans = tok.decode(y[0, x['input_ids'].shape[1]:], skip_special_tokens=True)
            m = re.search(r'\{.*\}', ans, re.S)
            try:
                js = json.loads(m.group(0)) if m else {}
            except json.JSONDecodeError:
                js = {}
            greedy = {f: norm(f, js.get(f)) for f in MONEY}

            with torch.no_grad():
                out = model(input_ids=x['input_ids'], use_cache=True, logits_to_keep=1)
            cache, L0 = out.past_key_values, x['input_ids'].shape[1]
            scores = {}
            for f, full in MONEY.items():
                vals = set(cands.get((d['id'], f), set())) | {greedy[f]}
                head = tok(f'{{\n  "{f}": ', add_special_tokens=False)['input_ids']
                sc = {}
                for v in vals:
                    tail = tok(as_text(v) + ',', add_special_tokens=False)['input_ids']
                    logits, seq = branch_logp(model, cache, L0, head + tail)
                    # logits[t] predicts seq[t+1]; the first head token is predicted by the prefix's last logit
                    lp = torch.log_softmax(logits[:-1], -1).gather(1, seq[1:, None]).squeeze(1)
                    sc['null' if v is None else v] = float(lp[len(head) - 1:].sum())
                scores[f] = sc
            del cache, out
            res.append(dict(id=d['id'], gold={f: d['gold'].get(full) for f, full in MONEY.items()},
                            greedy=greedy, cands=scores))
            if i % 20 == 0:
                print(split, i, f'{time.time() - t0:.0f}s', flush=True)
        json.dump(res, open(out_dir / f'{split}.json', 'w'))


if __name__ == '__main__':
    main()
