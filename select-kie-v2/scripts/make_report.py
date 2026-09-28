"""Render measured tables without manually copying numbers."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / 'results/metrics.json').read_text())
lines = ['# Kết quả pilot v2 — 28/09/2026', '',
         'Đã huấn luyện correctness heads mới trên đầu ra extractor có sẵn. '
         'Nguồn: [CORD-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2), '
         '800 train / 100 validation / 100 test, 8 field tiền. '
         'Đây là lượt phân tích thăm dò; test đã được dùng trong pilot cũ.', '',
         'Coverage là tỷ lệ quyết định được tự động chấp nhận. Risk là tỷ lệ sai trong phần được chấp nhận. '
         'Threshold được chọn trên validation với target 5%; risk test có thể vượt 5%. '
         'AURC thấp hơn là tốt hơn. Không có risk certification.', '']
for model, result in data.items():
    lines += ['## ' + model, '',
              f"Số quyết định train/validation/test: {result['counts']}. Tỷ lệ sai accept-all trên test: {result['test_error']:.2%}.", '',
              '| Head | AURC ↓ | Brier ↓ | Coverage @ target 5% | Risk test | Lỗi / accepted |',
              '|---|---:|---:|---:|---:|---:|']
    for name, m in result['measures'].items():
        t = m['targets']['0.05']; risk = f"{t['risk']:.2%}" if t['risk'] is not None else '—'
        lines.append(f"| {name} | {m['aurc']:.4f} | {m['brier']:.4f} | {t['coverage']:.2%} | {risk} | {t['errors']} / {t['accepted']} |")
    lines += ['', '### Bất định của so sánh', '',
              'Bootstrap ghép cặp 1000 lần theo tài liệu, resample cả validation/test rồi chọn lại threshold. '
              'Head giữ nguyên; CI chưa tính bất định do fitting train hay template clustering.', '',
              '| So sánh (A − B) | Chênh coverage quan sát (điểm %) | 95% CI coverage | 95% CI AURC |',
              '|---|---:|---:|---:|']
    for name, c in result['contrasts'].items():
        a,b = name.split(' minus ')
        diff = result['measures'][a]['targets']['0.05']['coverage'] - result['measures'][b]['targets']['0.05']['coverage']
        lo,hi = c['coverage_diff_ci95']; al,ah = c['aurc_diff_ci95']
        lines.append(f'| {name} | {100*diff:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] | [{al:+.4f}, {ah:+.4f}] |')
    c = result['contrasts']['fusion+binding minus fusion']['risk_exceedance_given_nonempty']
    lines += ['', f"Trong bootstrap có tập accept không rỗng, risk vượt target 5%: fusion+binding {c['fusion+binding']:.1%}, fusion {c['fusion']:.1%}. Đây là thống kê mô tả.", '',
              '### Taxonomy tự động', '',
              '| Loại lỗi | Tất cả split | Test |', '|---|---:|---:|']
    for key,val in result['taxonomy_all'].items():
        lines.append(f"| {key} | {val} | {result['taxonomy_test'].get(key,0)} |")
    lines += ['', '### Sensitivity: bỏ subtotal ABSENT nhưng prediction = gold total', '',
              f"Bỏ {result['sensitivity_dropped_val_test'][0]} quyết định validation và {result['sensitivity_dropped_val_test'][1]} test. Không fit lại head; chọn lại threshold trên phần validation còn lại. Đây là diagnostic dùng gold, không triển khai được như quy tắc selection.", '',
              '| Head | Coverage | Risk |', '|---|---:|---:|']
    for name in ['fusion','fusion+binding','fusion-HGB','same-info-HGB']:
        m = result['sensitivity'][name]; risk = f"{m['risk']:.2%}" if m['risk'] is not None else '—'
        lines.append(f"| {name} | {m['coverage']:.2%} | {risk} |")
    lines += ['']
lines += ['## Giới hạn', '',
          '- LR fusion+binding và generic HGB có cùng feature matrix, nhãn train và candidate information. '
          'Thí nghiệm này đo ích lợi tín hiệu binding và baseline correctness, chưa kiểm chứng architecture binding mới.',
          '- OCR/candidate dùng annotation CORD. Chưa chạy OCR thật hoặc unseen-template split; chưa có nguồn thứ hai.',
          '- Không fine-tune VLM; binding scores cũ cần nhiều teacher-forcing pass. Chưa đáp ứng head một lượt.',
          '- Normalization kế thừa coi 0 như ABSENT và bỏ dấu âm. Taxonomy tự động chưa được hai người audit.',
          '- Sample audit ngẫu nhiên và sample lỗi bổ sung được đánh dấu riêng trong CSV; không trộn để ước lượng prevalence.', '',
          '![Risk–coverage](../results/risk-coverage.png)', '',
          'Số liệu đầy đủ và endpoint secondary 1%, 2%, 10%: [metrics.json](../results/metrics.json). '
          'Mẫu audit: [audit.csv](../results/audit.csv). '
          'Protocol: [pilot-protocol.md](pilot-protocol.md).']
(ROOT / 'docs/pilot-results.md').write_text('\n'.join(lines)+'\n')
print(ROOT / 'docs/pilot-results.md')
