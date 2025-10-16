import torch
import torch.nn as nn
import torch.nn.functional as F
from .delta_quant import DeltaModulator1Bit

def causal_conv1d(in_ch, out_ch, k, stride=1, dilation=1)
    pad = (k - 1)  dilation  # left-pad only - causal
    return nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=stride,
                     dilation=dilation, padding=pad)

class LinearGRUCell(nn.Module)
    GRU-like cell with identity activation for candidate (linear instead of tanh).
    def __init__(self, input_size, hidden_size)
        super().__init__()
        self.hidden_size = hidden_size
        self.W_zr = nn.Linear(input_size, 2hidden_size)
        self.U_zr = nn.Linear(hidden_size, 2hidden_size, bias=False)
        self.W_n  = nn.Linear(input_size, hidden_size)
        self.U_n  = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x_t, h)
        zr = self.W_zr(x_t) + self.U_zr(h)
        z, r = zr.chunk(2, dim=-1)
        z = torch.sigmoid(z); r = torch.sigmoid(r)
        n = self.W_n(x_t) + r  self.U_n(h)   # no tanh, linear activation
        h_new = (1 - z)  h + z  n
        return h_new

class LinearGRU(nn.Module)
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False)
        super().__init__()
        assert not bidirectional
        self.layers = nn.ModuleList([
            LinearGRUCell(input_size if i==0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])

    def forward(self, x)
        # x [B, T, C]
        B, T, C = x.shape
        h = [x.new_zeros(B, self.layers[0].hidden_size) for _ in self.layers]
        outs = []
        for t in range(T)
            xt = x[, t]
            for i, cell in enumerate(self.layers)
                h[i] = cell(xt, h[i])
                xt = h[i]
            outs.append(xt)
        y = torch.stack(outs, dim=1)  # [B, T, H]
        return y

class CCLevel(nn.Module)
    def __init__(self, in_ch, conv_k, downsample, gru_hidden, hop)
        super().__init__()
        ch = in_ch
        layers = []
        for k, s in zip(conv_k, downsample)
            layers += [causal_conv1d(ch, 512, k, stride=s), nn.ReLU()]
            ch = 512
        self.conv = nn.Sequential(layers)   # [B, 512, T']
        self.hop = hop
        self.gru = LinearGRU(512, gru_hidden, num_layers=1)
        self.proj = nn.Linear(gru_hidden, 64)  # 64-D contextual representation

    def forward(self, x_wav)
        
        x_wav [B, 1, T_samples]
        returns Cs or Cl [B, T_frames, 64]
        
        h = self.conv(x_wav)                  # [B, 512, T']
        h = h.transpose(1, 2)                 # [B, T', 512]
        h = self.gru(h)                       # [B, T', H]
        z = self.proj(h)                      # [B, T', 64]
        return z

class CCEncoder(nn.Module)
    def __init__(self, sr=16000,
                 short_cfg=None, long_cfg=None)
        super().__init__()
        self.sr = sr
        self.short = CCLevel(1, short_cfg['conv_kernel'], short_cfg['downsample'],
                             short_cfg['gru_hidden'], short_cfg['hop_ms'])
        self.long  = CCLevel(1, long_cfg['conv_kernel'],  long_cfg['downsample'],
                             long_cfg['gru_hidden'],  long_cfg['hop_ms'])
        self.q_short = DeltaModulator1Bit(64, short_cfg['delta_step'])
        self.q_long  = DeltaModulator1Bit(64, long_cfg['delta_step'])

    @torch.no_grad()
    def quantize(self, x_wav)
        Cs = self.short(x_wav)
        Cl = self.long(x_wav)
        bs, _ = self.q_short(Cs)
        bl, _ = self.q_long(Cl)
        return bs, bl, Cs, Cl  # bits and pre-quant reps (for CC losses)
