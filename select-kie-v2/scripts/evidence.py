"""Inference-only text/box support. This module never accepts gold/category labels."""
import re
import numpy as np

CORD_ALIASES = {
 'subtotal':r'\b(sub\s*total|subtotal|jumlah harga)\b',
 'tax':r'\b(tax|pajak|ppn|pb1)\b',
 'service':r'\b(service|servis|layanan|svc)\b',
 'discount':r'\b(discount|diskon|disc|potongan)\b',
 'total':r'(?<!sub)(?<!sub )\b(total|grand total|amount due|jumlah bayar)\b',
 'cash':r'\b(cash|tunai|paid|tendered|bayar)\b',
 'change':r'\b(change|kembali|kembalian)\b',
 'card':r'\b(card|credit|kredit|debit|visa|mastercard)\b',
}
VRDU_ALIASES = {
 'registration_num':r'\b(registration (no|number)|registrant.*(no|number))\b',
 'registrant_name':r'\b(name of registrant|registrant|full name)\b',
 'foreign_principle_name':r'\b(foreign principal|principal.*name|name.*principal)\b',
 'file_date':r'\b(date|dated)\b',
 'signer_name':r'\b(signature|signatures|print name|printed name|signed)\b',
 'signer_title':r'\b(title|capacity|position)\b',
}
EVIDENCE = ['key_present','key_count','support_dist','support_same_row','support_right',
            'rival_key_dist','support_key_margin','rival_value_dist','support_value_margin',
            'matched_line_count','key_value_candidates','shared_prediction']
GEOMETRY = ['support_dist','support_same_row','support_right','rival_key_dist','support_key_margin',
            'rival_value_dist','support_value_margin']

def support(lines, field, matches, candidate_lines, aliases, shared=0):
    """lines contain only text+normalized boxes; matches/candidates are OCR-derived indices."""
    keys={f:[i for i,l in enumerate(lines) if re.search(rx,l['text'].lower())] for f,rx in aliases.items()}
    own=keys[field]
    def distance(a,b):
        la,lb=lines[a]['box'],lines[b]['box']
        xa,ya=(la[0]+la[2])/2,(la[1]+la[3])/2
        xb,yb=(lb[0]+lb[2])/2,(lb[1]+lb[3])/2
        return min(2.,4*abs(yb-ya)+abs(xb-xa))
    pairs=[(distance(k,v),k,v) for k in own for v in matches]
    best=min(pairs) if pairs else (2.,None,None)
    dist,k,v=best
    others=[i for f,ks in keys.items() if f!=field for i in ks]
    od=min([distance(k,v) for k in others for v in matches],default=2.)
    rival=min([distance(k,v) for k in own for v in candidate_lines if v not in matches],default=2.)
    same=right=0
    if k is not None:
        bk,bv=lines[k]['box'],lines[v]['box']
        same=int(abs((bk[1]+bk[3]-bv[1]-bv[3])/2)<.012)
        right=int((bv[0]+bv[2])/2>=(bk[0]+bk[2])/2)
    return dict(zip(EVIDENCE,[int(bool(own)),len(own),dist,same,right,od,od-dist,rival,rival-dist,
                              len(matches),len(candidate_lines),shared]))

def clean_text(s):
    return re.sub(r'[^a-z0-9]','',str(s).lower()) if s is not None else ''

def vrdu_features(ocr,record,field):
    """Reconstruct the stage-2 inference vector; accepts OCR and predictions only."""
    raw=record['fields'][field]; pred=raw['pred'];pred=None if pred is None or not str(pred).strip() else str(pred)
    pv=clean_text(pred);text=clean_text(ocr['text']);lp=raw['lps'];v=np.array(lp or [0.])
    lines=[{'text':l['text'],'box':[l['bbox'][1],l['bbox'][2]+page['page_id'],l['bbox'][3],l['bbox'][4]+page['page_id']]}
           for page in ocr['pages'] for l in page['lines']]
    matches=[i for i,l in enumerate(lines) if pv and pv in clean_text(l['text'])]
    candidates=[i for i,l in enumerate(lines) if re.search(r'\d',l['text'])] if field=='registration_num' else list(range(len(lines)))
    shared=sum(pv==clean_text(record['fields'][f]['pred']) for f in VRDU_ALIASES if f!=field) if pv else 0
    return {'field':field,'lp_mean':float(v.mean()),'lp_min':float(v.min()),'lp_first':float(v[0]),'n_tok':len(lp),'is_null':int(pred is None),
            'missing_key':int(raw['missing_key']),'on_page':int(bool(pv and pv in text)),
            'n_occ':text.count(pv) if pv else 0,'pred_len':len(pv),'numeric':int(bool(pv and pv.isdigit())),'n_lines':len(lines),
            **support(lines,field,matches,candidates,VRDU_ALIASES,shared)}
