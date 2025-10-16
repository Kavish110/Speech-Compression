import torch
import torch.nn as nn
from einops import rearrange

def downsample_avg(x, factor):
    # average pooling with stride=factor as in MSD
    pool = nn.AvgPool1d(kernel_size=factor, stride=factor, ceil_mode=False, count_include_pad=False)
    return pool(x)

class DBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, k, s):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, k, stride=s, padding=(k-1)//2)
        self.act = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        return self.act(self.conv(x))

class MSD(nn.Module):
    """Multi-Scale Discriminator: 3 scales (x, /2, /4), each 4 strided convs."""
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(DBlock1D(1, 16, 15, 1), DBlock1D(16, 64, 41, 4),
                          DBlock1D(64, 256, 41, 4), DBlock1D(256, 1024, 41, 4),
                          nn.Conv1d(1024, 1, 5, 1)),
            nn.Sequential(DBlock1D(1, 16, 15, 1), DBlock1D(16, 64, 41, 4),
                          DBlock1D(64, 256, 41, 4), DBlock1D(256, 1024, 41, 4),
                          nn.Conv1d(1024, 1, 5, 1)),
            nn.Sequential(DBlock1D(1, 16, 15, 1), DBlock1D(16, 64, 41, 4),
                          DBlock1D(64, 256, 41, 4), DBlock1D(256, 1024, 41, 4),
                          nn.Conv1d(1024, 1, 5, 1)),
        ])

    def forward(self, x):
        xs = [x, downsample_avg(x, 2), downsample_avg(x, 4)]
        logits = []
        feats   = []
        for s, b in zip(xs, self.blocks):
            f = s
            layer_feats = []
            for layer in b[:-1]:
                f = layer(f); layer_feats.append(f)
            out = b[-1](f)
            logits.append(out)
            feats.append(layer_feats)
        return logits, feats

class MPDPeriod(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 32, (5,1), (3,1), padding=(2,0)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 128, (5,1), (3,1), padding=(2,0)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 512, (5,1), (3,1), padding=(2,0)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1024, (5,1), (3,1), padding=(2,0)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(1024, 1, (3,1), 1, padding=(1,0)),
        ])

    def forward(self, x):
        # x: [B, 1, T] -> periodic 2D [B, 1, p, T//p]
        t = (x.size(-1) // self.p) * self.p
        x = x[..., -t:]
        x2d = rearrange(x, 'b c (t p) -> b c p t', p=self.p)
        feats = []
        f = x2d
        for layer in self.convs[:-1]:
            f = layer(f)
            feats.append(f)
        out = self.convs[-1](f)
        return out, feats

class MPD(nn.Module):
    def __init__(self, periods=(2,3,5,7,11)):
        super().__init__()
        self.ps = nn.ModuleList([MPDPeriod(p) for p in periods])

    def forward(self, x):
        logits, feats = [], []
        for p in self.ps:
            o, fs = p(x)
            logits.append(o)
            feats.append(fs)
        return logits, feats
