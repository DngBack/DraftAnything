"""Deterministic Vietnamese base/shape/tone/case factorization.

Canonical mark ordering is not treated as semantic ordering. Đ/đ is an explicit
crossbar exception because Unicode NFD does not decompose it to D/d.
"""
from dataclasses import dataclass, asdict
import unicodedata as ud

SHAPES = ('none','circumflex','breve','horn','crossbar')
TONES = ('none','acute','grave','hook','tilde','dot_below')
SHAPE_MARKS = {'\u0302':'circumflex','\u0306':'breve','\u031b':'horn'}
TONE_MARKS = {'\u0301':'acute','\u0300':'grave','\u0309':'hook','\u0303':'tilde','\u0323':'dot_below'}
CASE = ('lower','upper','literal')


@dataclass(frozen=True)
class Factors:
    base: str
    shape: str
    tone: str
    case: str

    def as_dict(self):return asdict(self)


def decompose_character(character):
    character = ud.normalize('NFC', character)
    if len(character)!=1:raise ValueError('Expected one NFC character/grapheme')
    if character in 'Đđ':return Factors('d','crossbar','none','upper' if character=='Đ' else 'lower')
    nfd = ud.normalize('NFD',character)
    base = nfd[0];shape=tone='none'
    if not base.isascii():raise ValueError(f'Unsupported base: {character!r}')
    for mark in nfd[1:]:
        if mark in SHAPE_MARKS and shape=='none':shape=SHAPE_MARKS[mark]
        elif mark in TONE_MARKS and tone=='none':tone=TONE_MARKS[mark]
        else:raise ValueError(f'Unsupported or repeated mark: {character!r}')
    case='upper' if base.isupper() else 'lower' if base.islower() else 'literal'
    return Factors(base.lower() if base.isalpha() else base,shape,tone,case)


def compose_character(factors):
    base=factors.base.upper() if factors.case=='upper' else factors.base
    if factors.shape=='crossbar':
        if factors.base!='d' or factors.tone!='none':raise ValueError('Invalid crossbar combination')
        return 'Đ' if factors.case=='upper' else 'đ'
    shape={v:k for k,v in SHAPE_MARKS.items()}.get(factors.shape,'')
    tone={v:k for k,v in TONE_MARKS.items()}.get(factors.tone,'')
    return ud.normalize('NFC',base+shape+tone)


def strip_tone(text):
    return ''.join(compose_character(Factors(f.base,f.shape,'none',f.case)) for f in map(decompose_character,ud.normalize('NFC',text)))


def component_errors(reference,prediction):
    """Levenshtein alignment of graphemes, then role mismatch accounting.

Insertions/deletions count once for each role; denominators are reference length.
This is a transparent diagnostic, not an independent visual-quality metric.
"""
    reference,prediction=ud.normalize('NFC',reference),ud.normalize('NFC',prediction)
    n,m=len(reference),len(prediction)
    dp=[[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1):dp[i][0]=i
    for j in range(m+1):dp[0][j]=j
    for i in range(1,n+1):
        for j in range(1,m+1):dp[i][j]=min(dp[i-1][j]+1,dp[i][j-1]+1,dp[i-1][j-1]+(reference[i-1]!=prediction[j-1]))
    counts={role:0 for role in ('base','shape','tone','case')};i,j=n,m
    while i or j:
        if i and j and dp[i][j]==dp[i-1][j-1]+(reference[i-1]!=prediction[j-1]):
            a,b=decompose_character(reference[i-1]),decompose_character(prediction[j-1])
            for role in counts:counts[role]+=getattr(a,role)!=getattr(b,role)
            i-=1;j-=1
        else:
            for role in counts:counts[role]+=1
            if i and dp[i][j]==dp[i-1][j]+1:i-=1
            else:j-=1
    return {'reference_characters':n,'character_errors':dp[n][m],**{role+'_errors':int(value) for role,value in counts.items()}}
