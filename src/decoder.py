import torch
import torch.nn as nn
from einops import rearrange

class MRFFBlock(nn.Module):
    def __init__(self, ch, kernels=(3,7,11)):
        super().__init__()
        self.branches = nn.ModuleList([nn.Conv1d(ch, ch, k, padding=0) for k in kernels])
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        outs = []
        for conv in self.branches:
            k = conv.kernel_size[0]
            # no padding to avoid algorithmic delay (trim left to keep causality)
            x_pad = nn.functional.pad(x, (k-1, 0))
            outs.append(conv(x_pad))
        y = sum(outs) / len(outs)
        return self.act(y)

class UpStage(nn.Module):
    def __init__(self, in_ch, k_list, up_factors, start_ch, mrff_k=(3,7,11), mrff_repeat=3):
        super().__init__()
        ch = start_ch
        self.first = nn.Conv1d(in_ch, ch, 3, padding=0)
        blocks = []
        for k, u in zip(k_list, up_factors):
            # transposed conv without padding
            blocks += [nn.ConvTranspose1d(ch, ch//2, k, stride=u, padding=0), nn.LeakyReLU(0.2, True)]
            ch = ch//2
            for _ in range(mrff_repeat):
                blocks += [MRFFBlock(ch, mrff_k)]
        self.net = nn.Sequential(*blocks)
        self.final = nn.Conv1d(ch, 1, 7, padding=0)

    def forward(self, x):
        x = self.first(x)
        x = self.net(x)
        # final conv with left-only pad to keep causality
        k = self.final.kernel_size[0]
        x = nn.functional.pad(x, (k-1, 0))
        x = self.final(x)
        return x

class CognitiveDecoder(nn.Module):
    """
    Stage 1: upsample long-term Cl to short-term grid
    Stage 2: fuse with short-term Cs, then upsample to waveform
    """
    def __init__(self, top_cfg, low_cfg):
        super().__init__()
        self.up_long = UpStage(in_ch=64, k_list=top_cfg['deconv_kernel'],
                               up_factors=top_cfg['upsample'],
                               start_ch=top_cfg['start_channels'],
                               mrff_k=tuple(top_cfg['mrff_k']),
                               mrff_repeat=top_cfg['mrff_repeat'])
        # After top upsampling, concat with Cs along channel dim
        self.up_fuse = UpStage(in_ch=64+1,  # Cs as 64, but inject as 1x64 by pointwise conv below
                               k_list=low_cfg['deconv_kernel'],
                               up_factors=low_cfg['upsample'],
                               start_ch=low_cfg['start_channels'],
                               mrff_k=tuple(low_cfg['mrff_k']),
                               mrff_repeat=low_cfg['mrff_repeat'])
        self.cs_proj = nn.Conv1d(64, 1, 1)

    def forward(self, Cs, Cl):
        # Cs, Cl: [B, T*, 64] -> [B, 64, T*]
        Cl = Cl.transpose(1, 2)
        Cs = Cs.transpose(1, 2)
        up_l = self.up_long(Cl)                   # [B, 1, T_short]
        cs_1 = self.cs_proj(Cs)                   # [B, 1, T_short]
        fused = torch.cat([up_l, cs_1], dim=1)    # [B, 2, T_short]
        wav = self.up_fuse(fused)                 # [B, 1, T_wav]
        return wav
