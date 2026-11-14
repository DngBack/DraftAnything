"""Small patch discriminator for the paired engineering baseline, not BigGAN."""
import torch
from torch import nn
from torch.nn.utils import spectral_norm


class PatchDiscriminator(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        channels = [1, hidden, hidden*2, hidden*4, hidden*4]
        self.blocks = nn.ModuleList([nn.Sequential(spectral_norm(nn.Conv2d(a, b, 3, stride=2, padding=1)),
                                                   nn.LeakyReLU(.2)) for a, b in zip(channels, channels[1:])])
        self.score = spectral_norm(nn.Conv2d(channels[-1], 1, 1))

    def forward(self, image, features=False):
        values = []
        for block in self.blocks:
            image = block(image); values.append(image)
        score = self.score(image).mean((1, 2, 3))
        return (score, values) if features else score


def feature_loss(fake_features, real_features, widths, image_width):
    loss = []
    for fake, real in zip(fake_features[-3:], real_features[-3:]):
        pixel = (torch.arange(image_width, device=fake.device)[None, None, None, :]
                 < torch.as_tensor(widths, device=fake.device)[:, None, None, None]).float()
        mask = nn.functional.interpolate(pixel, size=fake.shape[-2:], mode='nearest')
        denominator = (mask.sum()*fake.shape[1]).clamp_min(1)
        scale = (real.detach().abs()*mask).sum().div(denominator).clamp_min(.1)
        loss.append(((fake-real.detach()).abs()*mask).sum()/denominator/scale)
    return torch.stack(loss).mean()
