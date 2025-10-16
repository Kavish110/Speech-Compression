import torch
import torch.nn as nn

class XiHead(nn.Module):
    """Small projection head to re-embed waveforms back into CC representation spaces."""
    def __init__(self, sr=16000, kind='short', hidden=512, out_dim=64):
        super().__init__()
        # light conv stack + linear to 64-D
        if kind == 'short':
            downs = [5,4,2,2,2]
            kernels = [10,8,4,4,4]
        else:
            downs = [2,2,2]
            kernels = [4,4,4]
        ch = 1
        layers = []
        for k,s in zip(kernels, downs):
            layers += [nn.Conv1d(ch, hidden, k, stride=s, padding=(k-1)), nn.ReLU()]
            ch = hidden
        self.net = nn.Sequential(*layers)
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, wav):
        h = self.net(wav)        # [B, H, T']
        h = h.transpose(1, 2)    # [B, T', H]
        return self.proj(h)      # [B, T', 64]
