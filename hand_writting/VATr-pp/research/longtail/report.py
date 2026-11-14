"""Build measured pilot tables/curves; never fill missing runs with estimates."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

from research.longtail.prepare import write_json


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def percent(value):
    return f'{100*value:.2f}%' if value is not None else 'not measured'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('runs/acl_longtail_v1'))
    p.add_argument('--output', type=Path, default=Path('docs/acl_longtail'))
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    rec_rows, curves = [], []
    for mode in ('whole', 'factorized', 'glyph'):
        for seed in (742, 743, 744):
            name = ({'whole': 'clean_recognizer_whole', 'factorized': 'clean_recognizer'}.get(mode)
                    if seed == 742 and mode != 'glyph' else f'recognizer_{mode}_seed{seed}')
            folder = args.root/name
            metric = read(folder/'validation_005000.json')
            if metric:
                rec_rows.append({'mode': mode, 'seed': seed, 'path': str(folder), **metric})
            points = [read(path) for path in sorted(folder.glob('validation_*.json'))]
            if points: curves.append((mode, seed, points))
    generation = []
    for phase, step in [('guided', 1000), ('gan', 2000)]:
        for mode in ('glyph', 'factorized'):
            folder = args.root/f'{phase}_{mode}_seed742'
            metric = read(folder/f'ocr_eval_{step}/metrics.json')
            prompts = read(folder/'prompts_ocr_eval/metrics.json')
            complete = read(folder/'completion.json')
            config = read(folder/'config.json')
            generation.append({'phase': phase, 'mode': mode, 'step': step,
                               'status': 'complete' if complete else 'pending', 'path': str(folder),
                               'external_ocr': metric, 'prompt_ocr': prompts,
                               'parameters': config.get('trainable_parameters') if config else None})
    summary = {'updated_utc': datetime.now(timezone.utc).isoformat(),
               'recognition': rec_rows, 'generation': generation,
               'note': 'Exploratory development only; no final-test or blinded human results.'}
    write_json(args.output/'measured_results.json', summary)
    text = ['# Measured long-tail research pilot', '', f"Updated UTC: {summary['updated_utc']}", '',
            'These are actual completed measurements. Missing entries remain pending. No final-test scoring or blinded human quality evaluation was performed.', '',
            '## Clean recognition — fixed 5,000 updates', '',
            'Same clean data and CNN backbone; heads and parameter counts differ. Evaluation is a fixed 187-image, 30-writer development panel, not full development. Three supported holdout glyphs are ễ/ỡ/ẵ; uppercase targets have no paired real images.', '',
            '| Head | Seed | CER ↓ | Word exact ↑ | Holdout glyph macro-F1 ↑ |',
            '|---|---:|---:|---:|---:|']
    for row in rec_rows:
        text.append(f"| {row['mode']} | {row['seed']} | {percent(row['cer'])} | {percent(row['word_exact'])} | {percent(row['tail_macro_f1'])} |")
    text += ['', 'Across completed seeds (sample SD; do not treat a one-seed row as replicated):', '',
             '| Head | Seeds | CER mean ± SD | Tail macro-F1 mean ± SD |', '|---|---:|---:|---:|']
    for mode in ('whole', 'factorized', 'glyph'):
        rows = [r for r in rec_rows if r['mode'] == mode]
        if rows:
            def aggregate(key):
                mean = statistics.mean(r[key] for r in rows)
                sd = statistics.stdev(r[key] for r in rows) if len(rows) > 1 else None
                return percent(mean)+' ± '+(percent(sd) if sd is not None else 'not replicated')
            text.append(f"| {mode} | {len(rows)} | {aggregate('cer')} | {aggregate('tail_macro_f1')} |")
    text += ['', 'A conventional unseen output row has no learned semantic route to the heldout class. Its zero F1 is an expected structural limitation, not by itself a novel result. The canonical-glyph prototype head is a stronger semantic control, with full target printed glyph access. These recognition results do not prove generation/style quality.', '',
             '## Generation — separate legacy OCR evaluation', '',
             '97 fixed paired development targets. Same immutable clean CTC guidance checkpoint (selected at step 4,500 by seen CER), reconstruction recipe and seed. GAN stage continues each 1,000-update run for another 1,000 updates with newly initialized D; it is a changed objective and larger compute budget.', '',
             '| Stage | Renderer | Total G updates | Status | External CER ↓ | Word exact ↑ | Tail macro-F1 ↑ |',
             '|---|---|---:|---|---:|---:|---:|']
    for row in generation:
        m = row['external_ocr'] or {}
        text.append(f"| {row['phase']} | {row['mode']} | {row['step']} | {row['status']} | {percent(m.get('cer'))} | {percent(m.get('word_exact'))} | {percent(m.get('tail_macro_f1'))} |")
    text += ['', 'The external OCR is separate from new G training but was selected using legacy development. It is a screening proxy. Architecture capacity is not matched. Single-seed generator differences are exploratory; inspect real/generated/support images before any quality conclusion.', '',
             '## Uppercase prompt-only diagnostics', '',
             '| Stage | Renderer | Prompt | OCR exact / 30 writers |', '|---|---|---|---:|']
    for row in generation:
        m = row['prompt_ocr']
        if not m: continue
        for prompt in ('Ễ', 'Ỡ', 'Ẵ', 'NGUYỄN'):
            score = m.get('per_prompt', {}).get(prompt)
            if score: text.append(f"| {row['phase']} | {row['mode']} | {prompt} | {round(score['word_exact']*30)}/30 |")
    text += ['', 'These 300-prompt panels use known input transcripts and clean writer supports, without paired real target images. They test readability proxies; they do not establish exact writer-specific target-glyph fidelity. Standalone OCR itself may be unreliable.', '',
             '## Failed pilot and remaining gaps', '',
             '- Plain L1 collapsed toward white at 100 real-data updates. It was retained as a failure, not promoted for low loss.',
             '- Guided generation produced visible gray/noisy backgrounds at 250 and 1,000 updates. Better OCR scores alone do not fix that.',
             '- No component-local writer memory or annotated mark geometry is trained. No advantage over native VATr++, WriteViT or CONSTANT is established.',
             '- Uppercase genuine-writer data collection, a stronger baseline suite, independent mark/style/human evaluation, and three-seed generation confirmation remain required.',
             '- The current study is not an ACL-ready paper. See STUDY_PROTOCOL.md, PAPER_DRAFT.md and STATUS.md.', '']
    (args.output/'261114_report_03_Ket_qua_pilot_long_tail.md').write_text('\n'.join(text), encoding='utf-8')
    if curves:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        colors = {'whole': '#8a8a8a', 'factorized': '#1878b5', 'glyph': '#c36a17'}
        labeled = set()
        for mode, seed, points in curves:
            label = mode if mode not in labeled else None; labeled.add(mode)
            for axis, key in zip(axes, ('cer', 'tail_macro_f1')):
                axis.plot([r['step'] for r in points], [100*r[key] for r in points], color=colors[mode], alpha=.7,
                          linestyle='--' if mode == 'glyph' else '-', label=label)
        for axis, title in zip(axes, ('Development CER (%) ↓', 'Holdout glyph macro-F1 (%) ↑')):
            axis.set_title(title); axis.set_xlabel('Recognizer optimizer updates'); axis.grid(alpha=.2)
        axes[0].legend(); fig.suptitle('Exploratory clean recognition; fixed development panel')
        fig.tight_layout(); fig.savefig(args.output/'recognition_curves.png', dpi=180); plt.close(fig)
    print(json.dumps({'completed_recognition_runs': len(rec_rows), 'generation_metrics': sum(r['external_ocr'] is not None for r in generation)}))


if __name__ == '__main__': main()
