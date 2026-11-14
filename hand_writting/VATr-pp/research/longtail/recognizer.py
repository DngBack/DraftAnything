"""Joint-alignment compositional CTC: factor scores recombine BEFORE CTC.

A single blank is separate from semantic 'none'. Unlike separate role-CTC
streams, this has one shared alignment over complete legal NFC characters.
It is an experimental recognition model, not an independent visual-style judge.
"""
from collections import OrderedDict
import torch
from torch import nn
from research.vietcompose.orthography import decompose_character


class FactorizedCTC(nn.Module):
    def __init__(self, alphabet, hidden=128):
        super().__init__()
        if len(set(alphabet)) != len(alphabet) or not alphabet:
            raise ValueError('Unique nonempty alphabet required')
        self.alphabet = alphabet
        factors = [decompose_character(c) for c in alphabet]
        self.roles = ('base', 'shape', 'tone', 'case')
        self.vocabularies = {role: sorted({getattr(f, role) for f in factors}) for role in self.roles}
        for role in self.roles:
            vocab = self.vocabularies[role]
            self.register_buffer('map_'+role, torch.tensor([vocab.index(getattr(f, role)) for f in factors]))
        self.cnn = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 32, 3, padding=1)), ('relu1', nn.GELU()), ('pool1', nn.MaxPool2d(2)),
            ('conv2', nn.Conv2d(32, 64, 3, padding=1)), ('relu2', nn.GELU()), ('pool2', nn.MaxPool2d(2)),
            ('conv3', nn.Conv2d(64, hidden, 3, padding=1)), ('relu3', nn.GELU()),
            ('conv4', nn.Conv2d(hidden, hidden, 3, padding=1)), ('relu4', nn.GELU())]))
        self.context = nn.Conv1d(hidden, hidden, 5, padding=2)
        self.heads = nn.ModuleDict({role: nn.Linear(hidden, len(self.vocabularies[role])) for role in self.roles})
        self.blank = nn.Linear(hidden, 1)

    def forward(self, images):
        value = self.cnn(images).mean(2)
        value = value+torch.nn.functional.gelu(self.context(value))
        value = value.permute(2, 0, 1)
        scores = sum(head(value).index_select(-1, getattr(self, 'map_'+role)) for role, head in self.heads.items())
        return torch.cat((self.blank(value), scores), dim=-1).log_softmax(-1)

    @staticmethod
    def input_lengths(widths):
        return torch.as_tensor(widths, dtype=torch.long)//4


def targets(texts, alphabet):
    lookup = {c: i+1 for i, c in enumerate(alphabet)}
    lengths = torch.tensor([len(t) for t in texts], dtype=torch.long)
    encoded = torch.tensor([lookup[c] for t in texts for c in t], dtype=torch.long)
    return encoded, lengths


def ctc_loss(network, images, texts, widths):
    encoded, lengths = targets(texts, network.alphabet)
    available = network.input_lengths(widths)
    minimum = torch.tensor([len(t)+sum(a == b for a, b in zip(t, t[1:])) for t in texts])
    if torch.any(available < minimum):
        raise ValueError('Insufficient timesteps for CTC alignment')
    probabilities = network(images)
    if int(available.max()) > probabilities.shape[0]:
        raise ValueError('CTC width exceeds actual network timesteps')
    return nn.functional.ctc_loss(probabilities, encoded, available, lengths, blank=0, zero_infinity=False)


def decode(probabilities, widths, alphabet):
    paths = probabilities.argmax(-1).detach().cpu()
    result = []
    for b, length in enumerate(FactorizedCTC.input_lengths(widths)):
        prior, text = None, []
        for index in paths[:int(length), b].tolist():
            if index and index != prior: text.append(alphabet[index-1])
            prior = index
        result.append(''.join(text))
    return result


class WholeCharacterCTC(FactorizedCTC):
    """Same visual backbone with a conventional complete-character output head."""
    def __init__(self, alphabet, hidden=128):
        super().__init__(alphabet, hidden)
        del self.heads
        del self.blank
        self.classifier = nn.Linear(hidden, len(alphabet)+1)

    def forward(self, images):
        value = self.cnn(images).mean(2)
        value = value+torch.nn.functional.gelu(self.context(value))
        return self.classifier(value.permute(2, 0, 1)).log_softmax(-1)


class GlyphPrototypeCTC(FactorizedCTC):
    """Canonical-glyph semantic baseline able to represent unseen whole labels.

    A conventional untrained output row has no route to identify a heldout label.
    This control instead learns handwritten-feature/printed-glyph compatibility.
    Full target canonical glyph access matches the declared exposure assumption.
    """
    def __init__(self, alphabet, glyphs, hidden=128):
        super().__init__(alphabet, hidden)
        del self.heads
        import numpy as np
        if any(c not in glyphs for c in alphabet): raise ValueError('Missing canonical glyph')
        self.register_buffer('canonical_glyphs', torch.tensor(np.stack([glyphs[c] for c in alphabet]), dtype=torch.float32).flatten(1))
        self.glyph_projection = nn.Sequential(nn.Linear(256, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.log_scale = nn.Parameter(torch.tensor(2.3025851))

    def forward(self, images):
        value = self.cnn(images).mean(2)
        value = (value+torch.nn.functional.gelu(self.context(value))).permute(2, 0, 1)
        glyphs = torch.nn.functional.normalize(self.glyph_projection(self.canonical_glyphs), dim=-1)
        scores = self.log_scale.exp().clamp(max=100)*(torch.nn.functional.normalize(value, dim=-1) @ glyphs.T)
        return torch.cat((self.blank(value), scores), dim=-1).log_softmax(-1)
