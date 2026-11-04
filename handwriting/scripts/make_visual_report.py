"""Visual summary of weeks 1-3: sample lines from every source with measured word baselines,
perception stimuli, dynamics profile with CIs, and per-line distributions.

Sources: real IAM val1 lines, Emuru (CVPR'25, zero-shot), TS-GAN (BMVC'20, IAM lines),
B0 = SpiS-GAN words stitched into lines (leaky public checkpoint: reference only).
Profile numbers come from reports/h2/profile_3way.csv (589 lines with CER <= 0.2 in real,
Emuru and TS-GAN) and reports/h2/profile_b0.csv (unfiltered). Writes one self-contained HTML.
"""
import base64
import html
import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from hwline.profile import add_features, per_line_stats

OUT = Path("reports/visual/report.html")
SRC = [  # key, label, colour var, words table, image path pattern
    ("real", "Ảnh thật (IAM)", "--s-real", "data/dyn/words_real32_val1.csv.gz", None),
    ("emuru", "Emuru (CVPR'25)", "--s-1", "data/dyn/words_emuru32_val1.csv.gz", "data/gen/emuru_val1/{}.png"),
    ("tsgan", "TS-GAN (BMVC'20)", "--s-2", "data/dyn/words_tsgan32_val1.csv.gz", "data/gen/tsgan_val1/{}.png"),
    ("b0", "B0: ghép từ SpiS", "--s-3", "data/dyn/words_b0_line.csv.gz", "data/gen/b0_line/{}.png"),
]
HEX = {"real": "#52514e", "emuru": "#2a78d6", "tsgan": "#eb6834", "b0": "#1baf7a"}


def png_uri(im, fmt="PNG"):
    buf = io.BytesIO(); im.save(buf, fmt, optimize=True)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def common_ids():
    ok = lambda s: set(pd.read_csv(f"data/dyn/cer_{s}_val1.csv").query("cer <= 0.2").line_id)
    return ok("real") & ok("emuru") & ok("tsgan")


def line_image(key, line_id, man):
    p = man.loc[line_id, "image_path"] if key == "real" else dict((s[0], s[4]) for s in SRC)[key].format(line_id)
    im = Image.open(p).convert("L")
    return im.resize((max(int(round(im.width * 32 / im.height)), 1), 32), Image.BILINEAR)


def annotate(im32, words, colour, scale=2):
    """Upscale a 32 px line and mark each word's measured baseline plus the line's linear fit."""
    im = im32.resize((im32.width * scale, 32 * scale), Image.LANCZOS).convert("RGB")
    d = ImageDraw.Draw(im)
    w = words[words.v_baseline.astype(bool)]
    xs = ((w.box_x0 + w.box_x1) / 2).to_numpy() * scale
    ys = w.baseline_y.to_numpy() * scale
    for x0, x1, y in zip(w.box_x0 * scale, w.box_x1 * scale, ys):
        d.line([(x0, y), (x1, y)], fill=colour, width=2)
    if len(xs) >= 3:
        k, b = np.polyfit(xs, ys, 1)
        X = np.array([0, im.width])
        for x in range(0, im.width, 10):
            d.line([(x, k * x + b), (x + 5, k * (x + 5) + b)], fill=(120, 120, 120), width=1)
    return im


# ---------------------------------------------------------------- SVG charts
def dot_ci_panel(title, unit, rows, ticks, w=330, h=170):
    """Horizontal dot + CI per source on one shared axis; zero line marked."""
    pl, pr, pt, pb = 118, 14, 30, 30
    xmin, xmax = ticks[0], ticks[-1]
    X = lambda v: pl + (v - xmin) / (xmax - xmin) * (w - pl - pr)
    step = (h - pt - pb) / len(rows)
    out = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="{html.escape(title)}">',
           f'<text x="0" y="16" class="pt">{html.escape(title)}</text>']
    for t in ticks:
        out.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{pt}" y2="{h-pb}" class="grid"/>'
                   f'<text x="{X(t):.1f}" y="{h-pb+16}" class="tick" text-anchor="middle">{t:g}</text>')
    if xmin < 0 < xmax:
        out.append(f'<line x1="{X(0):.1f}" x2="{X(0):.1f}" y1="{pt}" y2="{h-pb}" class="zero"/>')
    for i, r in enumerate(rows):
        y = pt + step * (i + 0.5)
        tip = f'{r["label"]}: {r["v"]:+.3f}' + (f' [{r["lo"]:+.3f}, {r["hi"]:+.3f}]' if r.get("lo") is not None else "")
        out.append(f'<g class="mk"><title>{html.escape(tip)}</title>'
                   f'<rect x="{pl}" y="{y-step/2:.1f}" width="{w-pl-pr}" height="{step:.1f}" fill="transparent"/>')
        if r.get("lo") is not None:
            out.append(f'<line x1="{X(r["lo"]):.1f}" x2="{X(r["hi"]):.1f}" y1="{y:.1f}" y2="{y:.1f}" '
                       f'stroke="var({r["var"]})" stroke-width="2" stroke-linecap="round"/>')
        out.append(f'<circle cx="{X(r["v"]):.1f}" cy="{y:.1f}" r="5" fill="var({r["var"]})" stroke="var(--surface)" stroke-width="2"/></g>'
                   f'<text x="{pl-8}" y="{y+4:.1f}" class="lab" text-anchor="end">{html.escape(r["label"])}</text>')
    out.append(f'<text x="{w-pr}" y="{h-2}" class="tick" text-anchor="end">{html.escape(unit)}</text></svg>')
    return "".join(out)


def density_panel(title, xlabel, series, xmin, xmax, w=500, h=220, bw=None):
    """Overlaid kernel density lines, one per source (legend sits above the panels)."""
    pl, pr, pt, pb = 14, 14, 28, 34
    xs = np.linspace(xmin, xmax, 160)
    dens = []
    for s in series:
        v = s["values"][np.isfinite(s["values"])]
        b = bw or 1.06 * v.std() * len(v) ** -0.2
        dens.append(np.exp(-0.5 * ((xs[:, None] - v[None]) / b) ** 2).sum(1) / (len(v) * b * np.sqrt(2 * np.pi)))
    ymax = max(d.max() for d in dens) * 1.05
    X = lambda v: pl + (v - xmin) / (xmax - xmin) * (w - pl - pr)
    Y = lambda v: h - pb - v / ymax * (h - pt - pb)
    out = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="{html.escape(title)}">',
           f'<text x="0" y="16" class="pt">{html.escape(title)}</text>',
           f'<line x1="{pl}" x2="{w-pr}" y1="{h-pb}" y2="{h-pb}" class="axis"/>']
    for t in np.linspace(xmin, xmax, 6):
        out.append(f'<text x="{X(t):.1f}" y="{h-pb+15}" class="tick" text-anchor="middle">{t:.2g}</text>')
    if xmin < 0 < xmax:
        out.append(f'<line x1="{X(0):.1f}" x2="{X(0):.1f}" y1="{pt}" y2="{h-pb}" class="zero"/>')
    for s, d in zip(series, dens):
        pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, d))
        v = s["values"][np.isfinite(s["values"])]
        tip = f'{s["label"]}: n={len(v)}, trung vị {np.median(v):.3f}, IQR [{np.quantile(v,.25):.3f}, {np.quantile(v,.75):.3f}]'
        out.append(f'<g class="mk"><title>{html.escape(tip)}</title>'
                   f'<polyline points="{pts}" fill="none" stroke="var(--surface)" stroke-width="6"/>'
                   f'<polyline points="{pts}" fill="none" stroke="var({s["var"]})" stroke-width="2" stroke-linejoin="round"/></g>')
    out.append(f'<text x="{w-pr}" y="{h-2}" class="tick" text-anchor="end">{html.escape(xlabel)}</text></svg>')
    return "".join(out)


def legend(keys):
    lab = {s[0]: (s[1], s[2]) for s in SRC}
    return '<div class="legend">' + "".join(
        f'<span><i style="background:var({lab[k][1]})"></i>{html.escape(lab[k][0])}</span>' for k in keys) + "</div>"


def main():
    man = pd.read_csv("data/manifest.csv", dtype={"writer_id": str}).set_index("line_id")
    ids = common_ids()
    words = {k: pd.read_csv(f, dtype={"writer_id": str}) for k, _, _, f, _ in SRC}
    have = set.intersection(*[set(w.line_id) for w in words.values()]) & ids
    stats = {k: per_line_stats(w[w.line_id.isin(ids if k != "b0" else set(w.line_id))]) for k, w in words.items()}

    # Gallery: 3 lines with the clearest real drift + 3 random, among lines all sources measured.
    rs = stats["real"].set_index("line_id").reindex(sorted(have)).dropna(subset=["slope"])
    rs = rs[man.loc[rs.index, "transcript"].str.len().between(35, 60).to_numpy()]
    pick = list(rs.slope.abs().sort_values(ascending=False).index[:3])
    pick += list(rs.drop(pick).sample(3, random_state=7).index)
    rgb = {"real": (82, 81, 78), "emuru": (42, 120, 214), "tsgan": (235, 104, 52), "b0": (27, 175, 122)}
    gal = []
    for n, lid in enumerate(pick):
        tag = "drift rõ ở ảnh thật" if n < 3 else "chọn ngẫu nhiên"
        rows = [f'<div class="gl-h">{html.escape(man.loc[lid, "transcript"])} '
                f'<span class="muted">· {lid} · writer {man.loc[lid, "writer_id"]} · {tag}</span></div>']
        for k, label, var, _, _ in SRC:
            w = words[k][words[k].line_id == lid]
            im = annotate(line_image(k, lid, man), w, rgb[k])
            st = stats[k].set_index("line_id")
            s = st.loc[lid] if lid in st.index else None
            meta = (f'độ dốc {s.slope:+.2f} · SD baseline {s.base_sd:.2f} · SD gap {s.gap_sd:.2f}'
                    if s is not None and np.isfinite(s.slope) else "")
            rows.append(f'<div class="gl-r"><div class="gl-l"><i style="background:var({var})"></i>{label}'
                        f'<div class="muted small">{meta}</div></div><img src="{png_uri(im)}" alt="{label} {lid}"></div>')
        gal.append('<div class="gl">' + "".join(rows) + "</div>")

    # Perception stimuli (two lines, all conditions).
    st = pd.read_csv("data/perception/stimuli.csv")
    stim_html = []
    names = {"orig": "Gốc (ghép lại y nguyên)", "shuffle": "Xáo: giữ biến thiên, phá thứ tự",
             "detrend": "Bỏ drift tuyến tính", "flat": "Phẳng: như ghép từ", "catch": "Catch: jitter mạnh"}
    for lid in st.sort_values("slope_pct", key=abs, ascending=False).line_id.iloc[[0, 10]]:
        rows = [f'<div class="gl-h">{html.escape(st.set_index("line_id").loc[lid, "transcript"])} <span class="muted">· {lid}</span></div>']
        for c, nm in names.items():
            im = Image.open(f"data/perception/{lid}__{c}.png")
            im = im.resize((int(im.width * 64 / im.height), 64), Image.LANCZOS)
            rows.append(f'<div class="gl-r"><div class="gl-l">{nm}</div><img src="{png_uri(im)}" alt="{c}"></div>')
        stim_html.append('<div class="gl">' + "".join(rows) + "</div>")

    # Profile with CIs.
    p3 = pd.read_csv("reports/h2/profile_3way.csv")
    p3 = p3[p3.source.isin(["real_3w", "emuru_3w", "tsgan_3w"])].assign(key=lambda d: d.source.str[:-3])
    pb = pd.read_csv("reports/h2/profile_b0.csv").query("source == 'b0_line'").assign(key="b0")
    P = pd.concat([p3, pb]).set_index(["key", "feature"])
    lab = {s[0]: (s[1], s[2]) for s in SRC}

    def rows_for(feat, v, lo=None, hi=None):
        return [{"label": lab[k][0], "var": lab[k][1], "v": P.loc[(k, feat), v],
                 "lo": P.loc[(k, feat), lo] if lo else None, "hi": P.loc[(k, feat), hi] if hi else None}
                for k in ["real", "emuru", "tsgan", "b0"]]
    panels = [
        dot_ci_panel("Drift baseline trong dòng", "trend R² vượt null", rows_for("baseline", "trend_r2_excess", "trend_lo", "trend_hi"), [-0.05, 0, 0.05, 0.1, 0.15, 0.2]),
        dot_ci_panel("Phụ thuộc baseline giữa từ kề nhau", "tương quan lag-1 vượt null", rows_for("baseline", "lag1_excess", "lag1_lo", "lag1_hi"), [-0.1, 0, 0.1, 0.2, 0.3, 0.4]),
        dot_ci_panel("Phụ thuộc gap giữa từ kề nhau", "tương quan lag-1 vượt null", rows_for("gap", "lag1_excess", "lag1_lo", "lag1_hi"), [-0.1, 0, 0.1, 0.2]),
        dot_ci_panel("Biến thiên baseline trong dòng", "SD residual (core height)", rows_for("baseline", "within_sd"), [0, 0.1, 0.2, 0.3]),
        dot_ci_panel("Biến thiên khoảng cách giữa từ", "SD residual (core height)", rows_for("gap", "within_sd"), [0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        dot_ci_panel("Slant (đối chứng: không khác biệt)", "tương quan lag-1 vượt null", rows_for("slant_grad", "lag1_excess", "lag1_lo", "lag1_hi"), [-0.1, 0, 0.1, 0.2]),
    ]

    ser = lambda col: [{"label": lab[k][0], "var": lab[k][1], "values": stats[k][col].to_numpy(float)}
                       for k in ["real", "emuru", "tsgan", "b0"]]
    dens = [
        density_panel("Độ lệch chuẩn baseline giữa các từ, theo dòng", "core height", ser("base_sd"), 0, 0.8),
        density_panel("Độ lệch chuẩn khoảng cách giữa từ, theo dòng", "core height", ser("gap_sd"), 0, 1.6),
    ]

    # Table view of the profile.
    trs = []
    for k in ["real", "emuru", "tsgan", "b0"]:
        for f in ["baseline", "gap", "slant_grad"]:
            r = P.loc[(k, f)]
            trs.append(f"<tr><td>{lab[k][0]}</td><td>{f}</td><td>{int(r.n_lines)}</td>"
                       f"<td>{r.lag1_excess:+.3f} [{r.lag1_lo:+.3f}, {r.lag1_hi:+.3f}]</td>"
                       f"<td>{r.lag1_detrended:+.3f} [{r.lag1d_lo:+.3f}, {r.lag1d_hi:+.3f}]</td>"
                       f"<td>{r.trend_r2_excess:+.3f} [{r.trend_lo:+.3f}, {r.trend_hi:+.3f}]</td><td>{r.within_sd:.3f}</td></tr>")

    page = PAGE.format(
        legend=legend(["real", "emuru", "tsgan", "b0"]), gallery="".join(gal), stim="".join(stim_html),
        panels="".join(f'<div class="panel">{p}</div>' for p in panels),
        dens="".join(f'<div class="panel">{p}</div>' for p in dens), rows="".join(trs),
        n3=len(ids))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page)
    print(OUT, round(OUT.stat().st_size / 1e6, 2), "MB")


PAGE = """<!DOCTYPE html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dynamics trong dòng chữ viết tay: kết quả tuần 1–3</title>
<style>
.viz-root{{color-scheme:light;--surface:#fcfcfb;--text:#0b0b0b;--text2:#52514e;--muted:#7a7974;--grid:#e6e5e1;--axis:#b9b8b2;
--s-real:#52514e;--s-1:#2a78d6;--s-2:#eb6834;--s-3:#1baf7a;--card:#ffffff}}
@media (prefers-color-scheme:dark){{:root:where(:not([data-theme="light"])) .viz-root{{color-scheme:dark;--surface:#1a1a19;--text:#f0efec;
--text2:#c3c2b7;--muted:#9a998f;--grid:#2e2e2b;--axis:#55554f;--s-real:#c3c2b7;--s-1:#3987e5;--s-2:#d95926;--s-3:#199e70;--card:#232321}}}}
:root[data-theme="dark"] .viz-root{{color-scheme:dark;--surface:#1a1a19;--text:#f0efec;--text2:#c3c2b7;--muted:#9a998f;--grid:#2e2e2b;
--axis:#55554f;--s-real:#c3c2b7;--s-1:#3987e5;--s-2:#d95926;--s-3:#199e70;--card:#232321}}
body{{margin:0}}.viz-root{{background:var(--surface);color:var(--text);font:15px/1.5 system-ui,sans-serif;padding:28px;max-width:1180px;margin:auto}}
h1{{font-size:24px;margin:0 0 6px}}h2{{font-size:19px;margin:34px 0 6px}}p{{color:var(--text2);max-width:880px}}
.muted{{color:var(--muted)}}.small{{font-size:12px}}
.legend{{display:flex;gap:18px;flex-wrap:wrap;margin:10px 0 14px;color:var(--text2);font-size:14px}}
.legend i,.gl-l i{{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px}}
.gl{{background:var(--card);border:1px solid var(--grid);border-radius:10px;padding:12px 14px;margin:12px 0;overflow-x:auto}}
.gl-h{{font-weight:600;margin-bottom:6px}}.gl-r{{display:flex;align-items:center;gap:12px;padding:3px 0;border-top:1px solid var(--grid)}}
.gl-l{{flex:0 0 210px;font-size:13px;color:var(--text2)}}.gl-r img{{height:64px;background:#fff;border-radius:4px}}
.panels{{display:flex;flex-wrap:wrap;gap:14px}}.panel{{background:var(--card);border:1px solid var(--grid);border-radius:10px;padding:10px 12px}}
svg text{{fill:var(--text2);font:12px system-ui,sans-serif}}svg .pt{{fill:var(--text);font-weight:600;font-size:13px}}
svg .lab{{font-size:12px}}svg .tick{{fill:var(--muted);font-size:11px}}
svg .grid{{stroke:var(--grid);stroke-width:1}}svg .axis{{stroke:var(--axis)}}svg .zero{{stroke:var(--axis);stroke-dasharray:3 3}}
svg .mk:hover circle{{r:6}}svg .mk:hover polyline:last-child{{stroke-width:3}}
.key{{background:var(--card);border-left:4px solid var(--s-2);padding:10px 14px;border-radius:6px;margin:12px 0}}
table{{border-collapse:collapse;font-size:13px;margin-top:8px}}td,th{{border-bottom:1px solid var(--grid);padding:4px 10px;text-align:left}}
details{{margin-top:10px}}summary{{cursor:pointer;color:var(--text2)}}
</style></head><body><div class="viz-root">
<h1>Dynamics trong dòng chữ viết tay: kết quả tuần 1–3</h1>
<p>IAM val1 (Large Writer Independent split). Ba câu hỏi: dòng chữ thật có cấu trúc hình học giữa các từ không (H1); người đọc có nhận ra nó không (khảo sát, chưa có dữ liệu); các generator hiện có có tái tạo được nó không (H2).</p>
<div class="key"><b>Kết luận chính.</b> Ở chữ thật, baseline trôi theo dòng (drift) và khoảng cách giữa từ biến thiên mạnh. Cả hai generator cấp dòng đã thử đều thiếu điều này: Emuru có drift nhỏ hơn khoảng 7 lần; TS-GAN gần như thẳng tuyệt đối (SD baseline ≈ 1/5 ảnh thật). Spacing của cả hai chỉ biến thiên khoảng 40% so với chữ thật. Slant thì không khác biệt.</div>

<h2>1. Mẫu ảnh: cùng một dòng, bốn nguồn</h2>
<p>Đoạn ngang màu là baseline đo được của từng từ (ở 32 px, cùng một bộ đo cho mọi nguồn); nét đứt xám là đường thẳng khớp qua các baseline đó, tức drift của dòng. Độ dốc tính bằng core height trên mỗi 10 core height chiều ngang. Ba dòng đầu được chọn vì drift rõ ở ảnh thật, nên chúng minh họa hiện tượng chứ không phải mẫu đại diện; ba dòng sau chọn ngẫu nhiên. B0 dùng checkpoint SpiS công khai bị rò rỉ writer, nên chỉ để tham khảo.</p>
{legend}{gallery}

<h2>2. Hồ sơ dynamics: thật vs generator</h2>
<p>{n3} dòng val1 có CER ≤ 0.2 (TrOCR) ở cả ảnh thật, Emuru và TS-GAN; B0 chưa lọc CER. Thanh ngang là CI 95% bootstrap theo writer. Residual được tính sau khi đã loại hiệu ứng nội dung và hằng số theo dòng. Rê chuột lên điểm để xem giá trị.</p>
{legend}<div class="panels">{panels}</div>

<h2>3. Phân phối theo từng dòng</h2>
<p>Mỗi đường là mật độ của một thống kê tính riêng cho từng dòng. <b>Trái:</b> các từ trong một dòng lệch nhau theo chiều dọc bao nhiêu. Chữ thật có trung vị 0.20 core height. Emuru thấp hơn (0.12), TS-GAN gần như thẳng tuyệt đối (0.05). B0 cao hơn chữ thật (0.37), nhưng đó là các từ nhảy ngẫu nhiên, không có drift hay phụ thuộc (xem phần 2). <b>Phải:</b> độ đều của khoảng cách giữa từ. Chữ thật trải rộng (trung vị 0.43). Emuru và TS-GAN dồn quanh 0.18. B0 dùng khoảng cách cố định (≈0).</p>
{legend}<div class="panels">{dens}</div>
<details><summary>Bảng số liệu (dạng bảng của các biểu đồ trên)</summary>
<table><tr><th>Nguồn</th><th>Đặc trưng</th><th>Số dòng</th><th>lag-1 vượt null</th><th>lag-1 sau detrend</th><th>trend R² vượt null</th><th>SD trong dòng</th></tr>{rows}</table></details>

<h2>4. Stimuli của thí nghiệm cảm nhận</h2>
<p>Hình dạng từng chữ được giữ nguyên, chỉ vị trí các từ thay đổi. Mọi điều kiện, kể cả bản gốc, đều đi qua cùng pipeline cắt–ghép. Người tham gia so dòng gốc với một trong các phiên bản còn lại và chọn dòng nào là bản gốc.</p>
{stim}

<h2>Lưu ý khi đọc</h2>
<p>Lọc nội dung là bắt buộc: trước khi lọc CER, 16% dòng Emuru hỏng nội dung tạo ra phụ thuộc giả (baseline lag-1 +0.17). Emuru sinh ở 64 px, được thu về 32 px; slant đo kém ở 32 px. Mới có 2 generator cấp dòng; DiffBrush chưa chạy. Khảo sát cảm nhận chưa có dữ liệu, nên chưa biết người đọc có nhận ra khác biệt này hay không.</p>
</div></body></html>"""

if __name__ == "__main__":
    main()
