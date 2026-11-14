"""Summarize actual component-continuation benchmarks, including partial runs."""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=ROOT/'runs/component_continuation_250k_v1')
    p.add_argument('--output', type=Path, default=ROOT/'docs/acl_longtail/261114_report_05_Continuation_component_den_250k.md')
    args = p.parse_args()
    status = json.loads((args.run/'status.json').read_text())
    files = [path for path in sorted((args.run/'expanded').glob('benchmark_*.json')) if path.stem.split('_')[-1].isdigit()]
    if not files:
        raise ValueError('No completed expanded benchmark yet')
    rows = [json.loads(path.read_text()) for path in files]
    assets = args.output.parent/'component_250k_assets'
    assets.mkdir(parents=True, exist_ok=True)
    with (assets/'benchmarks.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['step', 'cer', 'cer_without_diacritics', 'exact'])
        for row in rows:
            writer.writerow([row['step'], row['guidance_ocr_cer'], row['guidance_ocr_cer_without_diacritics'], row['exact_transcript_rate']])
    first, last = rows[0], rows[-1]
    best_path = args.run/'best_selection.json'
    best = json.loads(best_path.read_text()) if best_path.exists() else None
    text = ['# Tiếp tục component sampling đến 250.000 steps', '',
            f"Trạng thái: **{status['state']}**. Nguồn **{status['source_step']:,}**, mục tiêu **{status['target_step']:,}** tổng outer steps. Benchmark mới nhất đã hoàn thành: **{last['step']:,}**.", '',
            'Fine-tune native VATr++ với component-aware transcript sampling. Giữ nguyên kiến trúc, G/D/writer/OCR, Adam và LR từ nguồn; batch/support 8/15, OCR frozen, không thêm auxiliary loss.', '',
            '| Mốc | CER ↓ | CER bỏ dấu ↓ | Exact ↑ |', '|---|---:|---:|---:|',
            f"| Nguồn {first['step']} | {100*first['guidance_ocr_cer']:.2f}% | {100*first['guidance_ocr_cer_without_diacritics']:.2f}% | {100*first['exact_transcript_rate']:.2f}% |",
            f"| Mới nhất {last['step']} | {100*last['guidance_ocr_cer']:.2f}% | {100*last['guidance_ocr_cer_without_diacritics']:.2f}% | {100*last['exact_transcript_rate']:.2f}% |", '',
            '| Prompt | Exact nguồn | Exact mới nhất | CER mới nhất |', '|---|---:|---:|---:|']
    for prompt in ['Tiếng', 'Việt', 'mã', 'mạ', 'NGUYỄN', 'Ắ', 'Ễ', 'Ỡ', 'Ự']:
        a, b = first['per_prompt'][prompt], last['per_prompt'][prompt]
        text.append(f"| {prompt} | {round(a['exact_transcript_rate']*30)}/30 | {round(b['exact_transcript_rate']*30)}/30 | {100*b['cer']:.2f}% |")
    text += ['', '![Diễn biến benchmark](component_250k_assets/progress.png)', '',
             'Panel cố định 600 mẫu / 30 writer. Đây là OCR supervisor proxy, không phải điểm đẹp hay đánh giá style độc lập. Không suy ra lợi ích bền vững từ một mốc; rolling mean dùng để chọn một checkpoint hiện tại, không phải trung bình trọng số.', '']
    if best:
        text += [f"Best đang giữ: bước {best['step']}, CER riêng {100*best['current_cer']:.2f}%, rolling CER {100*best['rolling_cer']:.2f}% trên cửa sổ {best['window']} mốc.", '']
    text += [f"[Panel nguồn]({args.run/'expanded'/('benchmark_%06d.png'%first['step'])}) · [Panel mới nhất]({args.run/'expanded'/('benchmark_%06d.png'%last['step'])})", '',
             f"- Trạng thái: `{args.run/'status.json'}`.",
             f"- Log: `{args.run/'train.log'}`.",
             f"- Checkpoint: `{args.run/'last.pth'}`, `best.pth`, `best_focus.pth` và 3 checkpoint định kỳ gần nhất.",
             '- Checkpoint/evaluation mỗi 1.000 steps; log mỗi 50 steps. Không tự thay thế checkpoint nguồn.', '']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(text), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    x = [row['step'] for row in rows]
    axes[0].plot(x, [100*row['guidance_ocr_cer'] for row in rows], marker='.', label='CER')
    axes[1].plot(x, [100*row['exact_transcript_rate'] for row in rows], marker='.', label='Exact')
    for axis, title in zip(axes, ['Supervisor CER (%)', 'Exact transcript (%)']):
        axis.set_title(title); axis.set_xlabel('Total outer steps'); axis.grid(alpha=.25)
    fig.suptitle('Pretrained VATr++ component-sampling continuation')
    fig.tight_layout(); fig.savefig(assets/'progress.png', dpi=160); plt.close(fig)
    print(json.dumps({'latest_benchmark_step': last['step'], 'state': status['state'], 'report': str(args.output)}))


if __name__ == '__main__':
    main()
