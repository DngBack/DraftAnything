"""Plots for the PartScribe technical report, read from the run logs and result files in runs/.

    python paper/make_figures.py        (from hand_writting/PartScribe; writes paper/figures/*.pdf)
"""
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
R = ROOT / 'runs'
OUT = ROOT / 'paper' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02})
C_NFD, C_CHAR, C_VATR, C_REAL, C_GREY = '#1f6fb2', '#d9822b', '#7a7a7a', '#2e8b57', '#bbbbbb'


def jl(p):
    return [json.loads(x) for x in open(p)]


def smooth(steps, vals, win=5000):
    steps, vals = np.asarray(steps), np.asarray(vals)
    out_s, out_v = [], []
    for s in range(int(steps.min()) + win, int(steps.max()) + 1, win // 2):
        m = (steps > s - win) & (steps <= s)
        if m.any():
            out_s.append(s); out_v.append(vals[m].mean())
    return np.array(out_s), np.array(out_v)


# --------------------------------------------------------------- training curves
def fig_training():
    runs = [('gen_vn_nfd', 'tần suất tự nhiên (ablation)', C_GREY, ':'),
            ('gen_vn_nfd_lt', '_lt: long-tail sampler', '#9ecae1', '-'),
            ('gen_vn_nfd_v2a', 'v2a: ghép từ lệch dòng', '#fdae6b', '--'),
            ('gen_vn_nfd_v2', 'v2 (multiword, mark weight, HAT)', C_NFD, '-'),
            ('gen_vn_nfd_v3', 'v3 (+ chữ hoa PNL)', '#08306b', '-')]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for name, lab, col, ls in runs:
        L = jl(R / name / 'log.jsonl')
        s, v = smooth([r['step'] for r in L], [r['flow'] for r in L])
        ax[0].plot(s / 1000, v, ls, color=col, label=lab, lw=1.4)
    ax[0].set_xlabel('bước train (nghìn)'); ax[0].set_ylabel('flow loss (trung bình 5k bước)')
    ax[0].axvline(100, color='k', lw=0.5, alpha=0.3); ax[0].axvline(300, color='k', lw=0.5, alpha=0.3)
    ax[0].legend(fontsize=6.5, frameon=False)
    for name, lab, col in [('gen_vn_nfd_v2', 'v2', C_NFD), ('gen_vn_nfd_v3', 'v3', '#08306b')]:
        L = jl(R / name / 'log.jsonl')
        for key, ls in [('anti', '-'), ('supcon', '--')]:
            s, v = smooth([r['step'] for r in L], [r[key] for r in L])
            ax[1].plot(s / 1000, v, ls, color=col, lw=1.2, label=f'{lab}: {key}')
    ax[1].set_xlabel('bước train (nghìn)'); ax[1].set_ylabel('loss phụ của style g')
    ax[1].legend(fontsize=6.5, frameon=False)
    fig.savefig(OUT / 'training_curves.pdf'); plt.close(fig)


def fig_vae():
    L = jl(R / 'vae_vn32' / 'log.jsonl')
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    s = np.array([r['step'] for r in L]) / 1000
    ax.plot(s, [r['l1'] for r in L], color=C_NFD, label='L1 tái tạo')
    ax.plot(s, [r['edge'] for r in L], color=C_CHAR, label='L1 biên (Sobel)')
    ax.set_yscale('log'); ax.set_xlabel('bước (nghìn)'); ax.legend(frameon=False, fontsize=7)
    fig.savefig(OUT / 'vae_curve.pdf'); plt.close(fig)


# --------------------------------------------------------------- held-out results vs step
def held_rows():
    rows = []
    spec = [('vn_25000', 25, 'ok', 'tự nhiên'), ('vn_lt_50000', 50, 'ok', '_lt'), ('vn_lt_75000', 75, 'ok', '_lt'),
            ('vn_lt_local_100000', 100, 'pair', '_lt')]
    spec += [(f'vn_v2_{k}000', k, 'pair', 'v2') for k in (125, 150, 175, 200, 225, 250, 275, 300)]
    spec += [('vn_v2_300000_edgefix', 300, 'pair', 'v2 (sửa mép)')]
    spec += [(f'vn_v3_{k}000', k, 'pair', 'v3') for k in (325, 350, 375, 400)]
    for d, k, metric, run in spec:
        f = R / 'eval' / d / ('compare_pairs.json' if metric == 'pair' else 'compare.json')
        x = json.loads(f.read_text())
        rows.append({'dir': d, 'k': k, 'metric': metric, 'run': run, 'nfd': x['C1+C2']['B'], 'char': x['C1+C2']['A'],
                     'ci': x['C1+C2']['ci95'], 'C1': x['C1_tone_on_new_base']['B'],
                     'C2': x['C2_stacked_pair_seen_on_other_base']['B'], 'C3': x['C3_stacked_pair_never_seen']['B']})
    return rows


def fig_heldout():
    rows = held_rows()
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    for run, col, mk in [('tự nhiên', C_GREY, 's'), ('_lt', '#9ecae1', 'o'), ('v2', C_NFD, 'o'),
                         ('v2 (sửa mép)', '#e31a1c', '*'), ('v3', '#08306b', 'D')]:
        rr = [r for r in rows if r['run'] == run]
        if not rr:
            continue
        x = [r['k'] for r in rr]; y = [r['nfd'] for r in rr]
        # CI is for the nfd-char difference; char is ~0, so it is shown around nfd
        lo = [r['ci'][0] + r['char'] for r in rr]; hi = [r['ci'][1] + r['char'] for r in rr]
        ax.errorbar(x, y, yerr=[np.array(y) - lo, np.array(hi) - y], fmt=mk + '-', color=col, ms=4, lw=1,
                    capsize=2, label=f'nfd {run}')
    ax.plot([r['k'] for r in rows], [r['char'] for r in rows], 'x', color=C_CHAR, label='char (mọi mốc)')
    ax.axvspan(0, 100, color='#f0f0f0', zorder=0); ax.text(3, 0.95, 'OCR CRNN', fontsize=6.5, color='#666')
    ax.text(108, 0.95, 'LocalOCR, pair_ok', fontsize=6.5, color='#666')
    ax.axvline(350, color='#08306b', lw=0.6, ls=':'); ax.text(352, 0.2, 'bản chốt\nv3 350k', fontsize=6.5, color='#08306b')
    ax.set_xlabel('bước train (nghìn)'); ax.set_ylabel('độ đúng ký tự giữ lại C1+C2')
    ax.set_ylim(-0.03, 1.0); ax.legend(fontsize=6.5, frameon=False, ncol=6, loc='upper center', bbox_to_anchor=(0.5, -0.22))
    fig.savefig(OUT / 'heldout_vs_steps.pdf'); plt.close(fig)
    return rows


def fig_heldout_groups():
    rows = {r['dir']: r for r in held_rows()}
    picks = [('vn_lt_local_100000', '_lt 100k'), ('vn_v2_300000_edgefix', 'v2 300k'), ('vn_v3_350000', 'v3 350k')]
    groups = ['C1', 'C2', 'C3']
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    w = 0.25
    for i, (d, lab) in enumerate(picks):
        ax.bar(np.arange(3) + (i - 1) * w, [rows[d][g] for g in groups], w, label=lab,
               color=['#9ecae1', C_NFD, '#08306b'][i])
    ax.bar(np.arange(3) + 1.6 * w, [0.0, 0.0, 0.0], w * 0.6, color=C_CHAR, label='char (≈0)')
    ax.set_xticks(range(3)); ax.set_xticklabels(['C1\nè é ẻ ẹ ỷ', 'C2\nễ ỡ ử', 'C3\nẳ'])
    ax.set_ylabel('pair_ok'); ax.set_ylim(0, 1); ax.legend(fontsize=6.5, frameon=False, ncol=2)
    fig.savefig(OUT / 'heldout_groups.pdf'); plt.close(fig)


# --------------------------------------------------------------- standard metrics
def fig_std():
    vn = json.loads((R / 'std_vn' / 'scores.json').read_text())
    pn = json.loads((R / 'std_pnl' / 'scores.json').read_text())
    models = [('vatrpp_175k', 'VATr++\n175,5k', C_VATR), ('vatrpp_comp_250k', 'VATr++\n250k', '#555555'),
              ('partscribe_char_300k_edgefix_s0', 'char\nv2 300k', C_CHAR), ('partscribe_nfd_300k_edgefix_s0', 'nfd\nv2 300k', C_NFD),
              ('partscribe_nfd_v3_350k_s0', 'nfd v3\n350k', '#08306b')]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for a, (key, lab, scale) in zip(ax, [('FID', 'FID ↓', 1), ('HWD', 'HWD ↓', 1), ('CER', 'CER (%) ↓', 100)]):
        vals = [vn[m][key] * scale for m, _, _ in models]
        a.bar(range(len(models)), vals, color=[c for _, _, c in models])
        if key == 'CER':
            a.axhline(vn['real']['CER'] * 100, color=C_REAL, ls='--', lw=1); a.text(0, vn['real']['CER'] * 100 + 0.2, 'ảnh thật', color=C_REAL, fontsize=7)
        if key == 'FID':
            a.axhline(vn['real_half_vs_half']['FID'], color=C_REAL, ls='--', lw=1, label='thật vs thật'); a.legend(frameon=False, fontsize=6.5, loc='upper right')
        a.set_xticks(range(len(models))); a.set_xticklabels([m[1].replace('\n', ' ') for m in models], fontsize=6.3, rotation=40, ha='right'); a.set_title(lab, fontsize=9)
        for i, v in enumerate(vals):
            a.text(i, v, f'{v:.2f}' if key != 'FID' else f'{v:.1f}', ha='center', va='bottom', fontsize=6)
    fig.savefig(OUT / 'std_metrics.pdf'); plt.close(fig)

    models = [('vatrpp_comp_250k', 'VATr++\n250k', '#555555'), ('char_v2_300k', 'char\nv2', C_CHAR), ('nfd_v2_300k', 'nfd\nv2', C_NFD),
              ('char_v3_325k', 'char\nv3 325k', '#f4b183'), ('nfd_v3_350k', 'nfd v3\n350k', '#08306b'), ('nfd_v3_400k', 'nfd v3\n400k', '#6baed6')]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for a, (key, lab, scale) in zip(ax, [('FID', 'FID ↓', 1), ('HWD', 'HWD ↓', 1), ('CER', 'CER (%) ↓', 100)]):
        vals = [pn[m][key] * scale for m, _, _ in models]
        a.bar(range(len(models)), vals, color=[c for _, _, c in models])
        a.set_xticks(range(len(models))); a.set_xticklabels([m[1].replace('\n', ' ') for m in models], fontsize=6.3, rotation=40, ha='right'); a.set_title(lab, fontsize=9)
        for i, v in enumerate(vals):
            a.text(i, v, f'{v:.2f}' if key != 'FID' else f'{v:.1f}', ha='center', va='bottom', fontsize=6)
    fig.savefig(OUT / 'c4_metrics.pdf'); plt.close(fig)


def fig_sweep():
    lab = {'c3_s1.5_m0_n50': 'mặc định', 'c2_s1.5_m0_n50': 'content 2', 'c4.5_s1.5_m0_n50': 'content 4,5',
           'c3_s1.0_m0_n50': 'style 1,0', 'c3_s2.5_m0_n50': 'style 2,5', 'c3_s1.5_m1_n50': 'mark 1',
           'c3_s1.5_m2_n50': 'mark 2', 'c3_s1.5_m0_n100': '100 bước'}
    r = json.loads((R / 'std_vn_sweep' / 'scores.json').read_text()) if (R / 'std_vn_sweep' / 'scores.json').exists() else None
    pts = {}
    if r:
        for k in lab:
            if k in r:
                pts[k] = (r[k]['CER'] * 100, r[k]['HWD'])
    if len(pts) < len(lab):   # values logged during the sweep (scores.json of that run was overwritten)
        pts = {'c3_s1.5_m0_n50': (3.37, 1.173), 'c2_s1.5_m0_n50': (4.82, 1.161), 'c4.5_s1.5_m0_n50': (2.38, 1.220),
               'c3_s1.0_m0_n50': (3.77, 1.204), 'c3_s2.5_m0_n50': (4.36, 1.164), 'c3_s1.5_m1_n50': (4.03, 1.175),
               'c3_s1.5_m2_n50': (4.36, 1.166), 'c3_s1.5_m0_n100': (3.30, 1.173)}
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for k, (x, y) in pts.items():
        ax.scatter(x, y, color='#e31a1c' if k == 'c3_s1.5_m0_n50' else C_NFD, s=18, zorder=3)
        off = {'c3_s1.5_m0_n50': (-38, -10), 'c3_s1.5_m0_n100': (4, 4), 'c3_s1.5_m1_n50': (4, 4), 'c3_s1.5_m2_n50': (4, -9),
               'c3_s2.5_m0_n50': (-36, 2), 'c2_s1.5_m0_n50': (-40, -3)}.get(k, (4, 3))
        ax.annotate(lab[k], (x, y), fontsize=6.5, xytext=off, textcoords='offset points')
    ax.set_xlabel('CER (%) ↓  (độ đọc đúng)'); ax.set_ylabel('HWD ↓  (độ giống writer)')
    fig.savefig(OUT / 'sampling_sweep.pdf'); plt.close(fig)


# --------------------------------------------------------------- edge line
def fig_edge():
    jobs = json.loads((R / 'std_vn' / 'jobs.json').read_text())
    def prof(name):
        out = []
        for j in jobs[::7][:400]:
            a = 255 - np.asarray(Image.open(R / 'std_vn' / name / j['file']).convert('L')).astype(float)
            out.append(np.median(a, 0)[-8:])
        return np.mean(out, 0)
    before, after = prof('partscribe_nfd_300k_s0'), prof('partscribe_nfd_300k_edgefix_s0')
    fig, ax = plt.subplots(figsize=(3.6, 2.3))
    x = np.arange(-8, 0) + 1
    ax.plot(x, before, 'o-', color='#e31a1c', label='trước khi sửa'); ax.plot(x, after, 's-', color=C_NFD, label='sau decode_generated')
    ax.set_xlabel('cột tính từ mép phải (0 = cột cuối)'); ax.set_ylabel('độ đậm trung vị theo cột (0–255)')
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(OUT / 'edge_profile.pdf'); plt.close(fig)


# --------------------------------------------------------------- data statistics
def fig_data():
    from partscribe.datasets import load_vn, load_pnl
    from partscribe.holdout import held_chars
    from partscribe.runtime import VATR
    man = str(VATR / 'files/vietnamese/manifest.json')
    tr = load_vn(man, str(ROOT / 'data/vn'), 'train', apply_holdout=True)
    full = load_vn(man, str(ROOT / 'data/vn'), 'train', apply_holdout=False)
    cnt = Counter(c for t in tr.texts for c in t if c.isalpha())
    cnt_full = Counter(c for t in full.texts for c in t if c.isalpha())
    letters = sorted(set(cnt_full) | set(cnt), key=lambda c: -cnt_full.get(c, cnt.get(c, 0)))
    held = held_chars('vn')
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    vals = [max(cnt.get(c, 0), 0.8) for c in letters]
    cols = ['#e31a1c' if c in held else ('#08306b' if c.isupper() else C_NFD) for c in letters]
    ax.bar(range(len(letters)), vals, color=cols, width=0.85)
    ax.set_yscale('log'); ax.set_ylabel('số lần xuất hiện trong train\n(sau khi giữ lại)')
    ax.set_xticks([]); ax.set_xlim(-1, len(letters)); ax.set_xlabel(f'{len(letters)} chữ cái có trong train, xếp theo tần suất giảm dần')
    for i, c in enumerate(letters):
        if c in held or i < 3 or c in 'ẫỵẸỄỶ' or (c.isupper() and cnt.get(c, 0) and i % 9 == 0):
            ax.text(i, vals[i] * 1.3, c, ha='center', fontsize=6.5, color=cols[i])
    ax.text(0.99, 0.62, '+ 42 chữ hoa có dấu khác: 0 ảnh trong train VN\n(không có trên trục; ví dụ Ễ, Ê, Ộ)', transform=ax.transAxes,
            ha='right', fontsize=7, color='#08306b')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_NFD, label='chữ thường'), Patch(color='#08306b', label='chữ hoa'),
                       Patch(color='#e31a1c', label='bị giữ lại (0 ảnh train)')], frameon=False, fontsize=7)
    fig.savefig(OUT / 'char_longtail.pdf'); plt.close(fig)
    upper_acc = [c for c in letters if c.isupper() and c not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
    zero_up = [c for c in upper_acc if cnt.get(c, 0) == 0]
    pn = load_pnl(ROOT / 'data/pnl32.npz', 'train')
    pc = Counter(c for t in pn.texts for c in t if c.isupper())
    stats = {'n_train_images': len(tr), 'n_train_writers': len(set(map(int, tr.writers))),
             'n_letters': len(letters), 'upper_accented_seen': len(upper_acc), 'upper_accented_zero_in_vn_train': zero_up,
             'pnl_train_capitals': len(pc), 'median_letter_count': int(np.median([cnt.get(c, 0) for c in letters]))}
    (OUT / 'data_stats.json').write_text(json.dumps(stats, ensure_ascii=False, indent=1))


def fig_ocr_gate():
    g = json.loads((R / 'eval_ocr_vn_local' / 'gate.json').read_text())
    fig, ax = plt.subplots(figsize=(3.6, 2.3))
    labels = ['chấm dưới\n(n = 32)', 'dấu ở trên\n(n = 151)']
    crnn = [19 / 32, 59 / 151]; loc = [g['dot_below']['read_after_erase'] / 32, g['above']['read_after_erase'] / 151]
    ax.bar(np.arange(2) - 0.18, crnn, 0.36, color=C_VATR, label='CRNN + BiLSTM')
    ax.bar(np.arange(2) + 0.18, loc, 0.36, color=C_NFD, label='LocalOCR (29 px)')
    ax.set_xticks(range(2)); ax.set_xticklabels(labels)
    ax.set_ylabel('tỉ lệ vẫn "đọc ra dấu"\nsau khi xóa dấu khỏi ảnh thật'); ax.legend(frameon=False, fontsize=7)
    fig.savefig(OUT / 'ocr_gate.pdf'); plt.close(fig)


def fig_align_bias():
    """Alignment bias B(x, k) of the DiT content cross-attention, for a toy word."""
    word, widths = 'lịch sử', np.array([2.2, 1.6, 3.4, 4.2, 2.6, 3.4, 3.6])
    centers = np.cumsum(widths) - widths / 2
    cols = np.arange(0, widths.sum(), 0.25)
    B = np.stack([np.exp(-0.5 * ((cols - c) / (w / 2 + 0.5)) ** 2) for c, w in zip(centers, widths)])
    fig, ax = plt.subplots(figsize=(5.0, 1.9))
    ax.imshow(B, aspect='auto', cmap='Blues', extent=[0, widths.sum(), len(word) - 0.5, -0.5])
    ax.set_yticks(range(len(word))); ax.set_yticklabels([c if c != ' ' else '␣' for c in word])
    ax.set_xlabel('cột latent (4 px mỗi cột)'); ax.set_ylabel('ký tự (key)')
    for c, w in zip(np.cumsum(widths), widths):
        ax.axvline(c, color='w', lw=0.6)
    fig.savefig(OUT / 'align_bias.pdf'); plt.close(fig)


def fig_flow_schematic():
    rng = np.random.default_rng(0)
    data = np.c_[rng.normal(-1.5, 0.25, 40), rng.normal(1.0, 0.25, 40)]
    noise = rng.normal(0, 1, (40, 2)) + [1.5, -0.8]
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    for a, b in zip(data, noise):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=C_GREY, lw=0.5)
    ax.scatter(*data.T, color=C_NFD, s=8, label='dữ liệu (t = 0)', zorder=3)
    ax.scatter(*noise.T, color='#e31a1c', s=8, label='nhiễu (t = 1)', zorder=3)
    ax.annotate('', xy=data[0], xytext=noise[0], arrowprops=dict(arrowstyle='->', color='k', lw=1))
    ax.text(*(0.5 * (data[0] + noise[0]) + [0.1, 0.15]), 'v = ε − x₀', fontsize=8)
    ax.set_xticks([]); ax.set_yticks([]); ax.legend(frameon=False, fontsize=7, loc='upper right')
    fig.savefig(OUT / 'flow_schematic.pdf'); plt.close(fig)


if __name__ == '__main__':
    for f in [fig_training, fig_vae, fig_heldout, fig_heldout_groups, fig_std, fig_sweep, fig_edge, fig_ocr_gate,
              fig_align_bias, fig_flow_schematic, fig_data]:
        f(); print('ok', f.__name__, flush=True)
