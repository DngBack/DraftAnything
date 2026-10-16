"""Seeded parametric stimuli with counterfactual partners.

COUNT:   n shapes (2-9); partner = same layout minus one shape (label n-1, n>=3).
SPATIAL: red target vs blue target among gray distractors; partner = targets swapped.
ATTR:    colour of the single shape of the queried kind; partner = target recoloured.
"""
import random
from PIL import Image, ImageDraw

SIZE = 448
COLORS = ["red", "blue", "green", "orange", "purple", "black", "brown", "magenta"]
REL = ["left", "right", "above", "below"]
# attr uses prototypical RGBs (PIL's "brown"/"magenta" read as red/purple, which makes errors a naming artefact)
ATTR_RGB = {"red": "#dc1414", "blue": "#1e3cdc", "green": "#14a028", "yellow": "#f0d200",
            "orange": "#ff8c00", "purple": "#8228aa", "black": "#000000", "pink": "#ff78be"}
ATTR_COLORS = list(ATTR_RGB)


def _shape(d, kind, x, y, r, color):
    if kind == "circle":
        d.ellipse([x - r, y - r, x + r, y + r], fill=color)
    elif kind == "square":
        d.rectangle([x - r, y - r, x + r, y + r], fill=color)
    else:
        d.polygon([(x, y - r), (x - r, y + r), (x + r, y + r)], fill=color)


def _place(rng, k, rmin, rmax, taken=()):
    out = list(taken)
    while len(out) < len(taken) + k:
        r = rng.randint(rmin, rmax)
        x, y = rng.randint(r + 4, SIZE - r - 4), rng.randint(r + 4, SIZE - r - 4)
        if all((x - a) ** 2 + (y - b) ** 2 > (r + c + 6) ** 2 for a, b, c in out):
            out.append((x, y, r))
    return out[len(taken):]


def render(objs):
    im = Image.new("RGB", (SIZE, SIZE), "white")
    d = ImageDraw.Draw(im)
    for kind, x, y, r, color in objs:
        _shape(d, kind, x, y, r, ATTR_RGB.get(color, color))
    return im


def count_item(seed):
    rng = random.Random(seed)
    n = rng.randint(2, 9)
    kind = rng.choice(["circle", "square", "triangle"])
    objs = [(kind, x, y, r, rng.choice(COLORS)) for x, y, r in _place(rng, n, 9, 26)]
    q = f"How many {kind}s are in the image? Answer with a single digit."
    item = dict(family="count", label=str(n), q=q, objs=objs)
    partner = dict(item, label=str(n - 1), objs=objs[:-1]) if n >= 3 else None
    return item, partner


def spatial_item(seed):
    rng = random.Random(seed)
    rel = rng.choice(REL)
    kinds = rng.sample(["circle", "square", "triangle"], 2)
    r = rng.randint(10, 20)
    # red at a, blue at b with a clearly dominant axis offset
    cx, cy = rng.randint(120, 328), rng.randint(120, 328)
    off, side = rng.randint(70, 110), rng.randint(-25, 25)
    dx, dy = {"left": (-off, side), "right": (off, side), "above": (side, -off), "below": (side, off)}[rel]
    a, b = (cx + dx // 2, cy + dy // 2), (cx - dx // 2, cy - dy // 2)
    dist = _place(rng, rng.randint(4, 10), 8, 18, taken=[(*a, r + 4), (*b, r + 4)])
    gray = [(rng.choice(["circle", "square", "triangle"]), x, y, rr, "gray") for x, y, rr in dist]
    q = (f"Where is the red {kinds[0]} relative to the blue {kinds[1]}? "
         "Answer with one word: left, right, above, or below.")
    flip = {"left": "right", "right": "left", "above": "below", "below": "above"}
    mk = lambda pa, pb: [(kinds[0], *pa, r, "red"), (kinds[1], *pb, r, "blue")] + gray
    item = dict(family="spatial", label=rel, q=q, objs=mk(a, b))
    partner = dict(item, label=flip[rel], objs=mk(b, a))
    return item, partner


def attr_item(seed):
    rng = random.Random(seed)
    kinds = ["circle", "square", "triangle"]
    kind = rng.choice(kinds)
    others = [k for k in kinds if k != kind]
    pos = _place(rng, rng.randint(6, 10), 8, 14)
    cols = rng.sample(ATTR_COLORS, 3)  # target colour + 2 distractor colours, so the colour alone does not give the answer
    objs = [(kind, *pos[0], cols[0])] + [(rng.choice(others), x, y, r, rng.choice(cols)) for x, y, r in pos[1:]]
    q = f"What color is the {kind}? Answer with one word."
    item = dict(family="attr", label=cols[0], q=q, objs=objs)
    new = rng.choice([c for c in ATTR_COLORS if c != cols[0]])
    partner = dict(item, label=new, objs=[(kind, *pos[0], new)] + objs[1:])
    return item, partner


def _check():
    for s in range(200):
        it, p = spatial_item(s)
        (_, ax, ay, *_), (_, bx, by, *_) = it["objs"][:2]
        dx, dy = ax - bx, ay - by
        want = ("left" if dx < 0 else "right") if abs(dx) > abs(dy) else ("above" if dy < 0 else "below")
        assert want == it["label"], (s, want, it["label"])
        c, cp = count_item(s)
        assert int(c["label"]) == len(c["objs"]) and (cp is None or int(cp["label"]) == len(cp["objs"]))
        a, ap = attr_item(s)
        assert sum(o[0] == a["objs"][0][0] for o in a["objs"]) == 1 and a["objs"][0][4] == a["label"]
        assert ap["objs"][0][4] == ap["label"] != a["label"]
    print("stimuli ok")


if __name__ == "__main__":
    _check()
