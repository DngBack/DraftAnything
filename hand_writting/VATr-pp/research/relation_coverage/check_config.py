"""Pick the VATr++ English inference flags empirically: paired HWD distance to the real word, random own supports."""
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from h1_lib import load_words, load_generator, generate, HWDFeatures  # noqa: E402

data, out = sys.argv[1], Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
dev = 'cuda'
rows = load_words(data)
byw = defaultdict(list)
for r in rows:
    if r['text'].isalpha() and len(r['text']) >= 2:
        byw[r['writer']].append(r)
rng = random.Random(0)
writers = sorted(w for w in byw if len(byw[w]) >= 60)[:12]
items = []
for w in writers:
    pool = [r for r in byw[w] if r['width32'] <= 192]
    for _ in range(8):
        tgt = rng.choice([r for r in byw[w] if 3 <= len(r['text']) <= 9])
        sup = rng.sample([r for r in pool if r['word_id'] != tgt['word_id']], 15)
        items.append((w, tgt, [s['path'] for s in sup]))
hwd = HWDFeatures(dev)
real = hwd([np.array(Image.open(t['path']).convert('L')) for _, t, _ in items])
panels = {}
for layout in (False, True):
    for qpos in (False, True):
        g = load_generator(dev, correct_layout=layout, query_position=qpos)
        fakes = generate(g, [s for *_, s in items], [t['text'] for _, t, _ in items], dev, seed=0)
        d = (hwd(fakes) - real).norm(dim=1)
        # shuffled-writer control: same fake vs a real word of another writer
        perm = torch.roll(real, 8, 0)
        dshuf = (hwd(fakes) - perm).norm(dim=1)
        print(f'layout_fix={layout} query_pos={qpos}: paired d={d.mean():.3f}  vs other-writer real {dshuf.mean():.3f}')
        panels[(layout, qpos)] = fakes[:10]
# visual panel: real row, then each config
rows_img = [[np.array(Image.open(t['path']).convert('L')) for _, t, _ in items[:10]]] + list(panels.values())
H = 36 * len(rows_img)
W = max(sum(a.shape[1] + 6 for a in r) for r in rows_img)
canvas = Image.new('L', (W, H), 255)
for i, r in enumerate(rows_img):
    x = 0
    for a in r:
        canvas.paste(Image.fromarray(a), (x, i * 36))
        x += a.shape[1] + 6
canvas.save(out / 'config_panel.png')
print('rows: real, (F,F), (F,T), (T,F), (T,T)', [t['text'] for _, t, _ in items[:10]])
