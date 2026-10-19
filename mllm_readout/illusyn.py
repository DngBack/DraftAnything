"""G5 S (docs/g5_protocol.md): procedurally rendered, shortcut-controlled illusions ("illusyn").

Three families: Mueller-Lyer (two stacked shafts with fins), Ponzo (two bars between converging rails), vertical-horizontal
(inverted T). Label: equal (diff 0) or unequal (|L1 - L2| / mean in {0.10, 0.15, 0.20, 0.30}, longer line random).
The two compared lengths always sum to 2 * base (ink matched; base jittered per image), and the illusion context (fin
direction, which bar sits near the apex) is drawn independently of which line is longer. Regime A: black on white,
width 2-4 px. Regime B: coloured lines on grey or noise, width 5-8 px, rotation +-8 deg. 200 equal + 200 unequal per
family, half per regime, seeded. Also: pixel_stats(), the 64-statistic pixel baseline.

usage: python illusyn.py        (writes a contact sheet to results/g5/illusyn_sheet.png)
"""
import colorsys, numpy as np
from PIL import Image, ImageDraw

S = 512
DIFFS = (0.10, 0.15, 0.20, 0.30)
Q = {"mullerlyer": "Are the two horizontal lines equal in length? Answer in curly brackets, e.g., {Yes} or {No}.",
     "ponzo": "Are the two horizontal bars equal in length? Answer in curly brackets, e.g., {Yes} or {No}.",
     "verticalhorizontal": "Are the horizontal and vertical lines equal in length? Answer in curly brackets, e.g., {Yes} or {No}."}


def lengths(rng, diff, lo=150, hi=210):
    base = rng.uniform(lo, hi)
    a, b = base * (1 + diff / 2), base * (1 - diff / 2)  # (a - b) / mean = diff, a + b = 2 base
    return (a, b) if rng.random() < 0.5 else (b, a)


def segments(fam, rng, diff):
    """line segments [(x0, y0, x1, y1)] in a 512 canvas centred near (256, 256); the first two are the compared lines"""
    l1, l2 = lengths(rng, diff, *((110, 150) if fam == "ponzo" else (150, 210)))
    cx, cy = 256 + rng.uniform(-30, 30), 256 + rng.uniform(-30, 30)
    seg = []
    if fam == "mullerlyer":
        gap = rng.uniform(90, 130)
        fin, ang = rng.uniform(22, 32), np.deg2rad(rng.uniform(30, 45))
        out_top = rng.random() < 0.5  # which shaft gets outward fins: independent of which is longer
        fins = []
        for k, (L, y) in enumerate(((l1, cy - gap / 2), (l2, cy + gap / 2))):
            x0, x1 = cx - L / 2, cx + L / 2
            seg.append((x0, y, x1, y))
            out = out_top if k == 0 else not out_top
            s = -1 if out else 1  # fins point away from (out) or toward (in) the shaft centre
            for xe, d in ((x0, 1), (x1, -1)):
                dx, dy = s * d * fin * np.cos(ang), fin * np.sin(ang)
                fins += [(xe, y, xe + dx, y - dy), (xe, y, xe + dx, y + dy)]
        seg += fins
    elif fam == "ponzo":
        top, bot = cy - 170, cy + 170
        spread_top, spread_bot = rng.uniform(60, 80), rng.uniform(230, 250)
        rails = [(cx - spread_top, top, cx - spread_bot, bot), (cx + spread_top, top, cx + spread_bot, bot)]
        ya, yb = cy - rng.uniform(60, 90), cy + rng.uniform(60, 90)
        upper_first = rng.random() < 0.5  # which compared line is drawn near the apex: independent of length
        ys = (ya, yb) if upper_first else (yb, ya)
        seg = [(cx - l1 / 2, ys[0], cx + l1 / 2, ys[0]), (cx - l2 / 2, ys[1], cx + l2 / 2, ys[1])] + rails
    else:  # inverted T: horizontal l1, vertical l2 from a point on it
        hx0 = cx - l1 / 2
        at = hx0 + l1 * rng.choice([0.5, rng.uniform(0.3, 0.7)])
        yb = cy + l2 / 2
        seg = [(hx0, yb, hx0 + l1, yb), (at, yb, at, yb - l2)]
    return seg


def first_longer(fam, seg):
    """signed label: Mueller-Lyer top shaft / Ponzo upper bar / vertical-horizontal horizontal line is the longer one"""
    l = np.hypot(seg[:2, 2] - seg[:2, 0], seg[:2, 3] - seg[:2, 1])
    if fam == "verticalhorizontal":
        return int(l[0] > l[1])
    top = int(np.argmin(seg[:2, 1]))
    return int(l[top] > l[1 - top])


def render(fam, regime, diff, rng, info=None):
    seg = np.array(segments(fam, rng, diff), float)
    if info is not None:
        info["longer"] = first_longer(fam, seg) if diff else -1
    w = int(rng.integers(2, 5)) if regime == "A" else int(rng.integers(5, 9))
    if regime == "B":
        th = np.deg2rad(rng.uniform(-8, 8))
        c, s = np.cos(th), np.sin(th)
        for k in (0, 2):
            x, y = seg[:, k] - 256, seg[:, k + 1] - 256
            seg[:, k], seg[:, k + 1] = 256 + c * x - s * y, 256 + s * x + c * y
        if rng.random() < 0.5:
            g = int(rng.integers(150, 220))
            bg = np.full((S, S, 3), g, np.uint8)
        else:
            bg = np.clip(rng.normal(rng.uniform(150, 210), 25, (S, S, 1)).repeat(3, 2), 0, 255).astype(np.uint8)
        col = tuple(int(255 * v) for v in colorsys.hsv_to_rgb(rng.random(), rng.uniform(0.6, 1), rng.uniform(0.3, 0.6)))
    else:
        bg, col = np.full((S, S, 3), 255, np.uint8), (0, 0, 0)
    im = Image.fromarray(bg).resize((2 * S, 2 * S))
    dr = ImageDraw.Draw(im)
    for x0, y0, x1, y1 in seg:
        dr.line([(2 * x0, 2 * y0), (2 * x1, 2 * y1)], fill=col, width=2 * w)
    return im.resize((S, S), Image.LANCZOS)


def items(n_per=100, diffs=DIFFS, seed=2026, tag=""):
    rng = np.random.default_rng(seed)
    out = []
    for fam in Q:
        for regime in "AB":
            for k in range(2 * n_per):
                eq = k < n_per
                diff = 0.0 if eq else diffs[k % len(diffs)]
                info = {}
                out.append(dict(image=render(fam, regime, diff, rng, info), longer=info["longer"], sys=None, q=Q[fam], cands=["Yes", "No"],
                                y="Yes" if eq else "No", group=fam, gate="{", regime=regime, diff=diff,
                                key=f"{tag}{fam}_{regime}_{k:03d}"))
    return out


def pixel_stats(im):
    """64 hand-made statistics: ink count, bbox (4), centre of mass (2), row / column ink projections (16 + 16),
    4x4 grid ink fractions (16), grey moments (4), mean RGB of ink (3), edge density (1), top-half ink share (1)"""
    a = np.asarray(im, np.float32)
    g = a.mean(2)
    bgc = np.median(a.reshape(-1, 3), 0)
    ink = np.linalg.norm(a - bgc, axis=2) > 60
    n = ink.sum() + 1e-6
    ys, xs = np.nonzero(ink) if ink.any() else (np.array([0]), np.array([0]))
    bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()]) / S
    com = np.array([xs.mean(), ys.mean()]) / S
    rows = ink.reshape(16, -1).sum(1) / n
    cols = ink.T.reshape(16, -1).sum(1) / n
    grid = ink.reshape(4, S // 4, 4, S // 4).mean((1, 3)).ravel()
    z = (g - g.mean()) / (g.std() + 1e-6)
    mom = np.array([g.mean() / 255, g.std() / 255, (z ** 3).mean(), (z ** 4).mean()])
    rgb = (a[ink].mean(0) if ink.any() else bgc) / 255
    edge = (np.abs(np.diff(ink.astype(np.int8), axis=0)).sum() + np.abs(np.diff(ink.astype(np.int8), axis=1)).sum()) / n
    top = ink[: S // 2].sum() / n
    return np.concatenate([[n / S ** 2], bbox, com, rows, cols, grid, mom, rgb, [edge, top]]).astype(np.float32)


if __name__ == "__main__":
    import os
    from pilot import OUT
    its = items()
    print(len(its), len(pixel_stats(its[0]["image"])))
    pick = [its[i] for i in (0, 150, 200, 350, 400, 550, 600, 750, 800, 950, 1000, 1150)]
    sheet = Image.new("RGB", (6 * 256, 2 * 256), "white")
    for k, it in enumerate(pick):
        sheet.paste(it["image"].resize((256, 256)), ((k % 6) * 256, (k // 6) * 256))
    os.makedirs(f"{OUT}/g5", exist_ok=True)
    sheet.save(f"{OUT}/g5/illusyn_sheet.png")
