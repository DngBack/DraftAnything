"""Export orthographic prompts and blank annotation templates, not new data."""
import argparse
import csv
from dataclasses import replace
from pathlib import Path
from research.longtail.prepare import write_json
from research.vietcompose.orthography import decompose_character, compose_character, TONES


def inventory():
    rows = []
    for case in ('lower', 'upper'):
        for vowel in 'aăâeêioôơuưy':
            factors = decompose_character(vowel)
            for tone in TONES:
                f = replace(factors, case=case, tone=tone)
                char = compose_character(f)
                rows.append({'prompt_id': f'glyph-{ord(char):04X}', 'text': char, 'unit': 'glyph', **f.as_dict()})
    rows.append({'prompt_id': 'context-nguyen', 'text': 'NGUYỄN', 'unit': 'context',
                 'base': '', 'shape': '', 'tone': '', 'case': 'upper'})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    prompts = inventory()
    with (args.output/'prompts.csv').open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prompts[0])); writer.writeheader(); writer.writerows(prompts)
    with (args.output/'annotations.csv').open('x', encoding='utf-8', newline='') as handle:
        fields = ['writer_pseudonym', 'split', 'prompt_id', 'text', 'image_path', 'license_or_permission',
                  'consent_record_id', 'image_quality_checked', 'transcription_checked',
                  'body_box', 'modifier_box', 'tone_box', 'annotator_id']
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerows({'prompt_id': r['prompt_id'], 'text': r['text']} for r in prompts)
    write_json(args.output/'status.json', {'status': 'templates-only-no-data-collected', 'prompts': len(prompts),
               'instructions': 'Assign train/development/test by writer BEFORE acquisition. Do not interpret blank fields as consent, valid labels, or completed samples. Boxes use original-image x0,y0,x1,y1; fused marks may need uncertainty flags. Glyph prompts are valid letters, not claims of valid syllables under arbitrary tone substitution.'})


if __name__ == '__main__': main()
