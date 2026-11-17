"""(1) Is the HWD feature space sensitive to *content-specific* shape on real data?
    Compare, for the same writer: two real instances of the SAME word vs two DIFFERENT words of equal length.
    If the gap is tiny relative to the writer gap, HWD cannot see join-level differences.
(2) Render a visual panel REAL | COV | UNC | OTHER | SELF from the logged supports."""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from h1_lib import load_words, load_generator, generate, HWDFeatures  # noqa: E402

data, out = sys.argv[1], Path(sys.argv[2])
dev = 'cuda'
rows = [r for r in load_words(data) if r['text'].isalpha() and len(r['text']) >= 3]
hwd = HWDFeatures(dev)
byw = defaultdict(list)
for r in rows:
    byw[r['writer']].append(r)
rng = random.Random(0)
same, diff, other = [], [], []
writers = sorted(byw)
for w in writers:
    bytext = defaultdict(list)
    for r in byw[w]:
        bytext[r['text']].append(r)
    for t, inst in bytext.items():
        if len(inst) < 2:
            continue
        a, b = inst[:2]
        cands = [r for r in byw[w] if r['text'] != t and len(r['text']) == len(t)]
        ow = rng.choice([x for x in writers if x != w])
        ocands = [r for r in byw[ow] if r['text'] == t] or [r for r in byw[ow] if len(r['text']) == len(t)]
        if not cands or not ocands:
            continue
        c, o = rng.choice(cands), rng.choice(ocands)
        f = hwd([np.array(Image.open(x['path']).convert('L')) for x in (a, b, c, o)])
        same.append(float((f[0] - f[1]).norm()))
        diff.append(float((f[0] - f[2]).norm()))
        other.append(float((f[0] - f[3]).norm()))
same, diff, other = map(np.array, (same, diff, other))
print(f'real pairs n={len(same)}: same writer+same word {same.mean():.3f} | same writer, different word (equal length) '
      f'{diff.mean():.3f} | other writer {other.mean():.3f}')
print(f'content-specific gap (diff-same) = {diff.mean() - same.mean():.3f}; writer gap (other-same) = {other.mean() - same.mean():.3f}; '
      f'ratio {100 * (diff.mean() - same.mean()) / (other.mean() - same.mean()):.1f}%')
json.dump(dict(n=len(same), same=same.mean(), diff=diff.mean(), other=other.mean()),
          open(out / 'hwd_sensitivity.json', 'w'), indent=1)

# (2) panel
byid = {r['word_id']: r for r in load_words(data)}
sups = [json.loads(l) for l in open(out / 'supports.jsonl')]
sups = [s for s in sups if s['draw'] == 0]
rng.shuffle(sups)
sups = sups[:14]
g = load_generator(dev)
conds = ['COV', 'UNC', 'OTHER', 'SELF']
cols = {'REAL': [np.array(Image.open(byid[s['word_id']]['path']).convert('L')) for s in sups]}
for c in conds:
    cols[c] = generate(g, [[byid[i]['path'] for i in s[c]] for s in sups], [byid[s['word_id']]['text'] for s in sups],
                       dev, seed=1)
names = list(cols)
colw = [max(a.shape[1] for a in cols[n]) + 12 for n in names]
canvas = Image.new('L', (sum(colw), 36 * len(sups)), 255)
for i in range(len(sups)):
    x = 0
    for n, cw in zip(names, colw):
        canvas.paste(Image.fromarray(cols[n][i]), (x, i * 36))
        x += cw
canvas.save(out / 'panel.png')
print('panel columns', names)
