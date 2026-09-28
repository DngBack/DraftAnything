"""Generate measured confirmation tables, figure, and a local audit viewer."""
import csv,hashlib,html,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];out=ROOT/'results/stage3'
r=json.loads((out/'metrics.json').read_text())
lines=['# Stage 3 — xác nhận trên 100 tài liệu mới','',
       'Qwen3.5-2B, bốn điều kiện ảnh/OCR × schema ngắn/làm rõ vai trò, 600 quyết định mỗi điều kiện. '
       '100 Short Form không nằm trong 600 tài liệu stage 2; chọn bằng hash filename trước inference. '
       'Không fit/tune trên holdout mới. [Protocol](stage3-protocol.md).','',
       '| Đầu vào / schema | Accuracy official | Accuracy normalized (secondary) | Field present accuracy | Field ABSENT accuracy |',
       '|---|---:|---:|---:|---:|']
for name,m in r['measures'].items():
    lines.append(f"| {name} | {m['accuracy']:.2%} | {m['normalized_accuracy']:.2%} | {m['present_accuracy']:.2%} | {m['absent_accuracy']:.2%} |")
lines += ['','## Paired document bootstrap, 2000 lần','',
          '| Contrast | Δ accuracy official (điểm %) | 95% CI |','|---|---:|---:|']
for name,c in r['contrasts'].items():
    v=c['correct'];lo,hi=v['ci95'];lines.append(f"| {name} | {100*v['accuracy_difference']:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] |")
lines += ['','## Confidence cao và transfer head đã học','',
          'Confidence cao = mean token probability >99%; không coi logprob là xác suất correctness. '
          'Head và ngưỡng bên dưới giữ nguyên từ stage 2: không calibration lại trên holdout.','',
          '| Điều kiện | Lỗi / quyết định confidence cao | HGB-fusion coverage / official risk | HGB-evidence coverage / official risk |',
          '|---|---:|---:|---:|']
for name,m in r['measures'].items():
    def cell(head):
        t=m['frozen_selector_transfer'][head];risk=f"{t['official_risk']:.2%}" if t['official_risk'] is not None else '—'
        return f"{t['coverage']:.2%} / {risk} ({t['official_errors']}/{t['accepted']})"
    stats=m['stats'];lines.append(f"| {name} | {stats.get('high_confidence_errors',0)} / {stats.get('high_confidence_decisions',0)} | {cell('HGB-fusion')} | {cell('HGB-evidence')} |")
lines += ['','HGB-evidence reject toàn bộ hai điều kiện ảnh ở threshold cũ: an toàn bằng zero coverage chưa giải quyết automation. '
          'HGB-fusion trên image-baseline chấp nhận 23 quyết định và sai18, còn image-clarified chấp nhận115 và sai0. '
          'Không diễn giải 0/115 thành risk guarantee; thay schema đã thay accuracy gốc và phân phối score.', '',
          'Trong image-clarified, 3 high-confidence errors đều là foreign_principle_name: 2 prediction bỏ qualifier trong ngoặc, 1 prediction điền vào field gold ABSENT. '
          'Không đủ cơ sở gọi cả3 là lỗi binding thật; cần audit về completeness và missing annotations.', '',
          '## Per-field official accuracy','',
          '| Field | OCR baseline | OCR clarified | Image baseline | Image clarified |','|---|---:|---:|---:|---:|']
names=list(r['measures'])
for field in r['measures'][names[0]]['per_field']:
    vals=[]
    for n in names:
        v=r['measures'][n]['per_field'][field];vals.append(f"{v['correct']/v['decisions']:.1%}")
    lines.append('| '+field+' | '+' | '.join(vals)+' |')
lines += ['','## Chi phí đo trên A30','',
          '| Điều kiện | Generate seconds / doc (batch amortized) | CPU evidence seconds / doc | Peak allocated GiB |', '|---|---:|---:|---:|']
for n,m in r['measures'].items():
    lines.append(f"| {n} | {m['mean_amortized_generate_seconds']:.3f} | {m['mean_evidence_feature_seconds_per_doc']:.4f} | {m['max_peak_memory_gib']:.2f} |")
lines += ['','Generation time không gồm loading model, tải/render PDF, processor và ghi output. '
          'Feature time chỉ gồm xử lý OCR/boxes, không gồm chạy OCR. Không có input truncation hay output cap ở bốn điều kiện.', '',
          '## Giới hạn diễn giải','',
          '- Schema làm rõ chứa thêm thông tin và nhiều token hơn. Gain là extraction gain do thay cách mô tả target, chưa phải novelty hay architecture superiority. Cần length-matched control và paraphrase controls.',
          '- Image model 2B, ảnh mọi page cạnh dài1280px; OCR cung cấp sẵn có thể được tạo ở độ phân giải khác. Kết quả không chứng minh OCR luôn hơn VLM hoặc cô lập riêng ảnh hưởng layout.',
          '- Giá trị nằm trên trang nhưng sai official matcher còn có thể do partial value, punctuation hoặc schema ambiguity. Nhãn tự động không thay audit người.',
          '- Document holdout chưa loại dependence theo registrant/person. Source vẫn là VRDU, không phải dataset thứ ba.',
          '- Hai head transfer đã fit trên OCR baseline, vì vậy thất bại trên ảnh/schema mới là domain shift diagnostic, không phải so sánh optimal heads huấn luyện cùng modality.', '',
          '## Artifact audit','',
          '[Viewer 100 tài liệu và bốn prediction](../results/stage3/audit-viewer.html), '
          '[2400 quyết định audit](../results/stage3/audit.csv), '
          '[metrics.json](../results/stage3/metrics.json), '
          '[manifest holdout](../results/stage3/holdout-manifest.json). Các cột human còn trống.', '',
          '![Accuracy trên holdout mới](../results/stage3/accuracy.png)']
(ROOT/'docs/stage3-results.md').write_text('\n'.join(lines)+'\n')
fig,ax=plt.subplots(figsize=(8,4.5));pos=np.arange(4)
values=[r['measures'][n]['accuracy'] for n in names]
ax.bar(pos,values,color=['#78909c','#1976d2','#90a4ae','#00897b'])
ax.set(xticks=pos,xticklabels=['OCR\nbaseline','OCR\nclarified','Image\nbaseline','Image\nclarified'],ylabel='Official value accuracy',ylim=(0,1))
for p,v in zip(pos,values):ax.text(p,v+.025,f'{v:.1%}',ha='center')
ax.set_title('100 fresh VRDU documents · Qwen3.5-2B');fig.tight_layout();fig.savefig(out/'accuracy.png',dpi=170);plt.close(fig)
rows=list(csv.DictReader((out/'audit.csv').open()));manifest=json.loads((out/'holdout-manifest.json').read_text());viewer=['<!doctype html><meta charset="utf-8"><title>VRDU audit</title><style>body{font:15px system-ui;margin:24px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:8px;text-align:left;vertical-align:top}summary{cursor:pointer;margin:16px 0;font-weight:bold}iframe{width:100%;height:700px}small{color:#555}</style><h1>VRDU holdout audit</h1><p>Automatic matcher results; no human audit has been completed. Compare the source PDF, gold, and four outputs. Record reviews in audit.csv.</p>']
for doc in manifest['filenames']:
    selected=[v for v in rows if v['doc']==doc];fields=list(dict.fromkeys(v['field'] for v in selected));slug=hashlib.sha256(doc.encode()).hexdigest()
    viewer.append('<details><summary>'+html.escape(doc)+'</summary><p><a href="../../data/vrdu/pdfs/'+slug+'.pdf">Open source PDF</a></p><table><tr><th>Field / gold</th>'+''.join('<th>'+html.escape(n)+'</th>' for n in names)+'</tr>')
    for field in fields:
        subset={v['condition']:v for v in selected if v['field']==field};gold=subset[names[0]]['gold']
        viewer.append('<tr><th>'+html.escape(field)+'<br><small>'+html.escape(gold or 'ABSENT')+'</small></th>')
        for n in names:
            v=subset[n];viewer.append('<td>'+html.escape(v['pred'] or 'null')+'<br><small>official='+v['correct']+'; normalized='+v['normalized_correct']+'</small></td>')
        viewer.append('</tr>')
    viewer.append('</table></details>')
(out/'audit-viewer.html').write_text('\n'.join(viewer))
print('Saved stage3 report, figure and audit viewer')
