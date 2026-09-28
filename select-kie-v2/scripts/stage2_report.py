"""Measured stage-2 tables and risk-coverage plots."""
import csv,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];out=ROOT/'results/stage2'
sys.path.insert(0,str(ROOT.parent/'selective-kie'))
from common import rc_curve
cord=json.loads((out/'cord-metrics.json').read_text())
lines=['# Stage 2 — kết quả thực nghiệm', '',
       'Ngày 28/09/2026. Tham khảo [protocol](stage2-protocol.md). CORD vẫn là exploratory; VRDU dùng split chính thức unseen-template, extractor mới trên OCR thật. Tất cả confidence heads HGB so sánh có hyperparameters giống nhau.','']
fig,axes=plt.subplots(2,2,figsize=(12,9))
for ax,(model,r) in zip(axes[0],cord.items()):
    lines += ['## CORD: '+model,'','| HGB features | AURC ↓ | Mean AURC 5-fold document CV ↓ | Coverage @ target 5% | Risk test |', '|---|---:|---:|---:|---:|']
    for name,m in r['measures'].items():
        t=m['targets']['0.05'];lines.append(f"| {name} | {m['aurc']:.5f} | {m['cv_mean_aurc']:.5f} | {t['coverage']:.2%} | {t['risk']:.2%} |")
    lines += ['','| Contrast | 95% CI ΔAURC | 95% CI Δcoverage (điểm %) |','|---|---:|---:|']
    for name,c in r['contrasts'].items():
        al,ah=c['aurc_diff_ci95'];cl,ch=c['coverage_diff_ci95'];lines.append(f'| {name} | [{al:+.5f}, {ah:+.5f}] | [{100*cl:+.2f}, {100*ch:+.2f}] |')
    lines += ['']
    rows=list(csv.DictReader((out/f'cord-scores-{model}.csv').open()));e=[int(x['error']) for x in rows]
    for n in ['fusion','fusion+binding','fusion+evidence','all']:
        cov,risk,_=rc_curve([float(x[n]) for x in rows],e);ax.plot(cov,risk,label=n)
    ax.set(title='CORD '+model,ylim=(0,.15))
for ax,stem in zip(axes[1],['vrdu-predictions','vrdu-qwen35-predictions']):
    r=json.loads((out/f'{stem}-metrics.json').read_text());model='Qwen2.5-1.5B' if stem=='vrdu-predictions' else 'Qwen3.5-2B'
    lines += ['## VRDU: '+model,'',f"Documents: {r['counts_docs']}. Decisions: {r['counts_decisions']}. Excluded multi-value: {r['excluded']}.",
              f"Accept-all error test: {r['accept_all_error']['test']:.2%}. Input truncated: {r['truncated_inputs']}; output capped: {r['capped_outputs']}.",'',
              '| Head | AURC ↓ | Brier ↓ | Coverage @ target 5% | Risk test | Errors / accepted |','|---|---:|---:|---:|---:|---:|']
    for name,m in r['measures'].items():
        t=m['targets']['0.05'];risk=f"{t['risk']:.2%}" if t['risk'] is not None else '—'
        lines.append(f"| {name} | {m['aurc']:.5f} | {m['brier']:.5f} | {t['coverage']:.2%} | {risk} | {t['errors']} / {t['accepted']} |")
    lines += ['','| Contrast | 95% CI ΔAURC | 95% CI Δcoverage (điểm %) |','|---|---:|---:|']
    for name,c in r['contrasts'].items():
        al,ah=c['aurc_diff_ci95'];cl,ch=c['coverage_diff_ci95'];lines.append(f'| {name} | [{al:+.5f}, {ah:+.5f}] | [{100*cl:+.2f}, {100*ch:+.2f}] |')
    lines += ['','### Calibration domain: cùng head fit150 docs','',
              '| Head | Source-template calibration: coverage / risk | Target-template calibration: coverage / risk |','|---|---:|---:|']
    for name,c in r['calibration_shift_secondary'].items():
        s,t=c['source_calibration'],c['target_calibration'];sr=f"{s['risk']:.2%}" if s['risk'] is not None else '—';tr=f"{t['risk']:.2%}" if t['risk'] is not None else '—'
        lines.append(f"| {name} | {s['coverage']:.2%} / {sr} | {t['coverage']:.2%} / {tr} |")
    lines += ['','Primary thresholds dùng 50 nhãn calibration từ template Short-Form mới; không phải zero-shot threshold transfer. Secondary source/target calibration giữ cùng score/head, dùng 50 docs mỗi bên.','']
    rows=list(csv.DictReader((out/f'{stem}-test.csv').open()));e=[int(x['err']) for x in rows]
    for n in ['HGB-fusion','HGB-evidence','HGB-no-geometry']:
        cov,risk,_=rc_curve([float(x[n]) for x in rows],e);ax.plot(cov,risk,label=n)
    ax.set(title='VRDU OCR '+model,ylim=(0,.9))
for ax in axes.ravel():
    ax.axhline(.05,color='gray',linestyle='--');ax.set(xlabel='Coverage',ylabel='Accepted error');ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(out/'risk-coverage.png',dpi=160);plt.close(fig)
lines += ['## Diễn giải và giới hạn','',
          'Evidence features thắng HGB mạnh về ranking trên Qwen2 CORD và cả hai model VRDU. Trên Qwen3 CORD gain nhỏ, CI chứa 0. Tại endpoint 5% VRDU, ranking gain không chuyển thành coverage gain ổn định; risk thực tế và coverage phải được đọc cùng nhau.',
          '', 'NameMatch gốc giữ punctuation/case. [Sensitivity](../results/stage2/label-sensitivity.json) không refit head hoặc threshold: Qwen3.5 HGB-evidence có 2/52 lỗi official, cả hai là dấu chấm cuối tên; dưới quy tắc normalized secondary còn 0/52. Không coi 0/52 là risk guarantee và không thay primary result.',
          '', 'Corpus CORD dùng annotation OCR; VRDU dùng OCR cung cấp trong benchmark. Các features là rules schema-specific từ text/boxes, không phải architecture mới hay visual encoder đã học. Classifier có field identity; chưa chứng minh schema-agnostic generalization.',
          '', 'Bootstrap 1000 lần theo document, resample calibration/test và chọn lại threshold. Head cố định, không tính fit variance hoặc dependence giữa cùng registrant. Development VRDU50 không dùng fit/tune confidence head; 2 mẫu được dùng phát triển schema của stage 3.',
          '', '![Risk–coverage](../results/stage2/risk-coverage.png)', '',
          'Prior: [ExtractConf](https://arxiv.org/abs/2606.24420), [PLV](https://arxiv.org/abs/2609.20110), [Valid selective risk control](https://arxiv.org/abs/2608.14639). Kết quả hiện chưa vượt các prior này trong matched reproduction; không claim novelty chỉ từ feature fusion.']
(ROOT/'docs/stage2-results.md').write_text('\n'.join(lines)+'\n')
print('Saved stage2-results.md')
