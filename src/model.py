"""Small residual U-Net for single-channel image restoration (~1.9M params)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
    )


class UNet(nn.Module):
    def __init__(self, base: int = 32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.e1, self.e2, self.e3 = block(1, c[0]), block(c[0], c[1]), block(c[1], c[2])
        self.mid = block(c[2], c[3])
        self.u3, self.d3 = nn.ConvTranspose2d(c[3], c[2], 2, 2), block(c[3], c[2])
        self.u2, self.d2 = nn.ConvTranspose2d(c[2], c[1], 2, 2), block(c[2], c[1])
        self.u1, self.d1 = nn.ConvTranspose2d(c[1], c[0], 2, 2), block(c[1], c[0])
        self.out = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        # Pad to a multiple of 8 so any image size works at inference time.
        h, w = x.shape[-2:]
        ph, pw = (-h) % 8, (-w) % 8
        xp = F.pad(x, (0, pw, 0, ph), mode="reflect")
        e1 = self.e1(xp)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        m = self.mid(F.max_pool2d(e3, 2))
        d3 = self.d3(torch.cat([self.u3(m), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return (xp + self.out(d1))[..., :h, :w]  # residual: predict a correction
