"""Audit composition coverage and materialize a leakage-aware future protocol.

This never relabels old checkpoints as zero-shot. Target handwriting, transcripts
and style supports must all be removed during a fresh controlled training run.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import unicodedata

from data.vietnamese import PackedH5,load_manifest
from research.vietcompose.orthography import decompose_character,SHAPES,TONES


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,default=Path('files/vietnamese/manifest.json'));p.add_argument('--output',type=Path,default=Path('docs/vietcompose/protocol'));p.add_argument('--holdout',default='ỄễỠỡẴẵ');args=p.parse_args()
    m=load_manifest(args.manifest);data=PackedH5(m,include_hat=True);args.output.mkdir(parents=True,exist_ok=True)
    holdout=set(unicodedata.normalize('NFC',args.holdout));splits={};coverage={}
    for split in ['train','hat_ocr','val']:
        ids=m['indices'][split];clean=[i for i in ids if not holdout.intersection(data.labels[i])];held=[i for i in ids if holdout.intersection(data.labels[i])]
        splits[split]={'clean_indices':clean,'heldout_combination_indices':held}
        chars=Counter(c for i in ids for c in data.labels[i]);clean_chars=Counter(c for i in clean for c in data.labels[i]);roles={role:Counter() for role in ['base','shape','tone','case']}
        for c,count in clean_chars.items():
            factors=decompose_character(c)
            for role in roles:roles[role][getattr(factors,role)]+=count
        coverage[split]={'all_images':len(ids),'clean_images':len(clean),'heldout_images':len(held),'characters':dict(chars),'clean_component_occurrences':{k:dict(v) for k,v in roles.items()}}
    pooled=Counter(c for split in ['train','hat_ocr'] for i in splits[split]['clean_indices'] for c in data.labels[i]);primitive={role:Counter() for role in ['base','shape','tone','case']}
    for c,count in pooled.items():
        factors=decompose_character(c)
        for role in primitive:primitive[role][getattr(factors,role)]+=count
    target_audit={}
    for char in sorted(holdout):
        f=decompose_character(char);counts={role:primitive[role][getattr(f,role)] for role in primitive};target_audit[char]={'factors':f.as_dict(),'clean_primitive_occurrences':counts,'all_primitives_covered':all(counts.values()),'train_images':sum(char in data.labels[i] for i in m['indices']['train']),'hat_images':sum(char in data.labels[i] for i in m['indices']['hat_ocr']),'val_images':sum(char in data.labels[i] for i in m['indices']['val'])}
    # Candidate quadruple benchmark; actual support sampling must use clean pools.
    protocol={'version':1,'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),'holdout_characters':sorted(holdout),'splits':splits,'requirements':['Filter target images, generated text corpora, OCR supervision, style supports and retrieval memories using clean_indices.','Do not warm-start the strict experiment from our 100k/150k/200k checkpoints: they have target-character exposure.','Keep writer train/development disjoint; report seen/unseen combination crossed with seen/unseen writer.','Pre-register whether canonical glyphs of held-out characters are allowed; match that access for all baselines.','Independent evaluation OCR may know held-out characters but must not supply training gradients.'],'execution_status':'Protocol materialized, not enforced by existing legacy GAN trainer; no strict zero-shot experiment has run.'}
    (args.output/'split_indices.json').write_text(json.dumps(protocol,ensure_ascii=False))
    (args.output/'coverage.json').write_text(json.dumps({'coverage':coverage,'heldout_targets':target_audit,'primitive_counts':{k:dict(v) for k,v in primitive.items()}},ensure_ascii=False,indent=2))
    print(json.dumps({'split_sizes':{k:{x:len(y) for x,y in v.items()} for k,v in splits.items()},'targets':target_audit},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
