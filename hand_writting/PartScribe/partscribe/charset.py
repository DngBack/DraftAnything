"""Character set and Unicode-part utilities.

The model never sees NFD parts. They are used only to (1) define and audit the
held-out protocol and (2) build probe labels and mark masks for evaluation.
"""
import unicodedata as ud

# Identical to VATr-pp/files/vietnamese/manifest.json["alphabet"] (checked in tests).
ALPHABET = (" %()+,-./0123456789:=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            "ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯư"
            "ẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ")

# Combining marks used in Vietnamese, by role.
MODIFIERS = {'̂': 'circumflex', '̆': 'breve', '̛': 'horn'}
TONES = {'̀': 'grave', '́': 'acute', '̉': 'hook', '̃': 'tilde', '̣': 'dot_below'}
MARKS = {**MODIFIERS, **TONES}
MARK_NAMES = sorted(set(MARKS.values()))


def glyph_chars():
    """Visible characters (space excluded)."""
    return [c for c in ALPHABET if not c.isspace()]


def parts(ch):
    """(base, [mark names]) from NFD; Đ/đ have no combining mark and stay atomic."""
    d = ud.normalize('NFD', ch)
    names = [MARKS[m] for m in d[1:] if m in MARKS]
    if len(names) != len(d) - 1:
        raise ValueError(f'unexpected combining mark in {ch!r}')
    # Canonical order: modifier before tone, as in Vietnamese writing.
    names.sort(key=lambda n: (n not in MODIFIERS.values(), n))
    return d[0], names


def composition_chain(ch):
    """Precomposed prefixes base -> base+modifier -> full, e.g. ễ: [e, ê, ễ].

    Used to isolate the pixels of each mark by rendering differences.
    """
    d = ud.normalize('NFD', ch)
    base, marks = d[0], list(d[1:])
    marks.sort(key=lambda m: (m not in MODIFIERS, m))
    chain = [base]
    for k in range(1, len(marks) + 1):
        chain.append(ud.normalize('NFC', base + ''.join(marks[:k])))
    for c in chain:
        assert len(c) == 1, f'no precomposed form for prefix of {ch!r}: {c!r}'
    return chain, [MARKS[m] for m in marks]
