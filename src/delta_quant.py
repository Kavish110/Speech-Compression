import torch
import torch.nn as nn

class DeltaModulator1Bit(nn.Module):
    """Per-feature 1-bit Δ-mod with learnable or fixed step."""
    def __init__(self, n_features: int, init_step: float = 0.05, learnable: bool = False):
        super().__init__()
        step = torch.full((1, n_features), init_step)
        self.step = nn.Parameter(step, requires_grad=learnable)

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """
        x: [B, T, C] float features
        returns:
          bits: [B, T, C] in {0,1}
          y: reconstructed running signal from Δ bits (same shape as x)
        """
        B, T, C = x.shape
        y = torch.zeros_like(x)
        bits = torch.zeros_like(x)
        prev = torch.zeros(B, C, device=x.device)
        for t in range(T):
            diff = x[:, t] - prev
            sgn = torch.sign(diff)
            sgn[sgn == 0] = 1.0
            delta = self.step * sgn
            curr = prev + delta
            y[:, t] = curr
            bits[:, t] = (sgn < 0).float()  # 1 for negative, 0 for positive
            prev = curr
        return bits, y
