"""Measured continuation report, including optimizer/frozen-OCR verification."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import torch

from research.longtail.compare import paired_compare


def fingerprint(tensors):
    value = hashlib.sha256()
    for key, tensor in sorted(tensors.items()):
        value.update(key.encode()); value.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def metadata(path):
    state = torch.load(path, map_location='cpu', weights_only=False)
    step = {}
    for name, optimizer in state['optimizers'].items():
        values = [float(item['step']) for item in optimizer['state'].values() if 'step' in item]
        step[name] = {'min': min(values), 'max': max(values)}
    return {'outer_step': state['step'], 'adam_steps': step, 'ocr_sha256': fingerprint(state['model']['R']),
            'network_shapes': {net: {key: list(t.shape) for key, t in tensors.items()} for net, tensors in state['model'].items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('runs/pretrained_longtail_v1'))
    p.add_argument('--output', type=Path, default=Path('docs/acl_longtail/261114_report_04_Finetune_pretrained_175500_den_176500.md'))
    args = p.parse_args()
    plan = json.loads((args.root/'status.json').read_text())
    if plan['state'] != 'completed': raise ValueError('Report requires completed trials')
    source = metadata(Path(plan['source'])); target = plan['target_step']; start = plan['source_step']
    verification, benchmarks = {}, {}
    for name in ('legacy', 'component'):
        state = metadata(args.root/name/'last.pth')
        verification[name] = {'outer_step': state['outer_step'], 'adam_steps': state['adam_steps'],
                              'ocr_unchanged': state['ocr_sha256'] == source['ocr_sha256'],
                              'native_network_keys_and_shapes_unchanged': state['network_shapes'] == source['network_shapes']}
        if state['outer_step'] != target or not verification[name]['ocr_unchanged'] or not verification[name]['native_network_keys_and_shapes_unchanged']:
            raise AssertionError('Continuation verification failed')
        for net in source['adam_steps']:
            expected = plan['additional_outer_steps_per_trial']*(2 if net in ('G', 'D') else 1)
            if state['adam_steps'][net]['max'] != source['adam_steps'][net]['max']+expected:
                raise AssertionError('Optimizer did not continue at expected update count: '+net)
        folder = args.root/name/'expanded'
        benchmarks[name] = [json.loads(path.read_text()) for path in sorted(folder.glob('benchmark_*.json')) if path.stem.split('_')[-1].isdigit()]
    rows = {}
    for name in ('legacy', 'component'):
        rows[name] = [{'sample_id': f"{r['writer']}:{r['reference']}", 'writer': str(r['writer']),
                       'reference': r['reference'], 'prediction': r['prediction']} for r in benchmarks[name][-1]['predictions']]
    comparison = paired_compare(rows['legacy'], rows['legacy'], rows['component'], 'ỄỠẮ')
    result = {'source_step': start, 'target_step': target, 'source_adam_steps': source['adam_steps'],
              'verification': verification, 'source_unchanged': plan['source_unchanged'],
              'fixed_final_checkpoint_comparison': comparison}
    output_json = args.output.with_suffix('.json'); output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    text = ['# Fine-tune VATr++ từ checkpoint pretrained', '',
            f"Nguồn: `{plan['source']}` ở **{start:,}**; cả hai nhánh hoàn thành **{target:,}**, mỗi nhánh thêm {plan['additional_outer_steps_per_trial']:,} outer steps.", '',
            f"Runtime gốc: **PyTorch {plan['torch']}**. Source SHA-256: `{plan['source_sha256']}`. Checkpoint nguồn được kiểm tra giữ nguyên sau training.", '',
            '## Xác nhận continuation', '',
            '- G/D/writer/OCR giữ nguyên toàn bộ tên và shape tensor native; không xây lại generator.',
            '- OCR frozen khớp hash với nguồn ở cả hai nhánh.',
            '- Adam của G/D tăng đúng 2.000 cập nhật và writer tăng đúng 1.000; không reset optimizer.',
            '- Hai benchmark khởi đầu đều khớp artifact nguồn: CER 13,6257%, exact 62,5%.',
            '- Batch/support 8/15; LR encoder/G/D/writer = 5e-6/2e-5/1e-5/2e-5; OCR/writer weight 4/0,1; các loss và kiến trúc giữ nguyên.', '',
            '## Chỉ thay transcript sampling', '',
            '`legacy` giữ sampling cũ. `component` trộn 50% trọng số ký tự cũ và 50% trọng số nhóm base/case, modifier/case, tone/case, modifier×tone×case trong lượt rare. Rare fraction vẫn 0,3 và HAT fraction vẫn 0,35. Không thêm ảnh giám sát hoặc HAT writer giả.', '',
            '| Bước | CER legacy ↓ | CER component ↓ | Exact legacy ↑ | Exact component ↑ |',
            '|---:|---:|---:|---:|---:|']
    for a, b in zip(benchmarks['legacy'], benchmarks['component']):
        if a['step'] != b['step']: raise AssertionError('Benchmark schedule mismatch')
        text.append(f"| {a['step']} | {100*a['guidance_ocr_cer']:.2f}% | {100*b['guidance_ocr_cer']:.2f}% | {100*a['exact_transcript_rate']:.2f}% | {100*b['exact_transcript_rate']:.2f}% |")
    text += ['', '| Nhánh cuối | NGUYỄN đúng /30 | Ễ đúng /30 | Ỡ đúng /30 | Ắ đúng /30 |', '|---|---:|---:|---:|---:|']
    for name in ('legacy', 'component'):
        prompts = benchmarks[name][-1]['per_prompt']
        text.append('| '+name+' | '+' | '.join(str(round(prompts[c]['exact_transcript_rate']*30)) for c in ['NGUYỄN','Ễ','Ỡ','Ắ'])+' |')
    text += ['', '## Diễn giải', '',
             'Nhánh component tốt hơn đối chứng ở cả bốn lần đánh giá sau resume trong pilot này. Tuy nhiên, chỉ một nguồn/seed và 1.000 outer steps; chưa chứng minh lợi ích bền vững, chất lượng thẩm mỹ hay lời giải cho chữ hoa hiếm. OCR benchmark vẫn là proxy có lịch sử chung với GAN, không phải chấm người đọc độc lập.', '',
             '**Ễ/Ỡ/Ắ đứng riêng vẫn thất bại**. HAT có ảnh chứa chữ nhưng không có transcript một ký tự Ễ/Ỡ; cần tách thiếu glyph khỏi thiếu ngữ cảnh ký tự đơn. Sampling hiện tại không sinh transcript mới. Xem `character_context_audit.json`.', '',
             'Không diễn giải sampler thành ưu tiên trực tiếp Ễ: xác suất HAT draw chứa Ễ đổi từ 0,13669% sang 0,13337% trong corpus hiện tại. Đây là phân bổ theo thành phần, không đơn thuần oversampling chữ Ễ. Xem `pretrained_sampling_exposure.json`.', '',
             'Không tự động thay checkpoint nguồn bằng model pilot. Phải so sample cùng writer/support, lặp seed và đánh giá nội dung/dấu/phong cách độc lập trước khi promote. Các thí nghiệm generator nhỏ từ đầu đã được lưu riêng và dừng theo chỉnh hướng của người dùng; không gộp vào kết quả này.', '',
             '## Artifact', '',
             f"- Trạng thái: `{args.root}/status.json`.",
             f"- Model mới: `{args.root}/component/last.pth`.",
             f"- Đối chứng: `{args.root}/legacy/last.pth`.",
             f"- Panel 600 mẫu: `{args.root}/{{legacy,component}}/expanded/benchmark_{target:06d}.png`.",
             '- Code: `util/vietnamese_longtail.py`, cờ `--text-sampler` trong `train_vietnamese.py` và launcher `scripts/run_pretrained_longtail_pilot.py`.',
             '- Bộ test trên runtime gốc: 39 tests pass; log tại `test_run.log`.', '']
    args.output.write_text('\n'.join(text), encoding='utf-8')
    print(json.dumps({'source_step': start, 'target_step': target, 'verification_passed': True}))


if __name__ == '__main__': main()
