import torch
from torch import nn
from cs336_basics.linear import Linear
from jaxtyping import Float

class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        """
        FFN(x) = SwiGLU(x,W1,W2,W3) = W2(SiLU(W1x)⊙W3x)
        SiLU(x) = x·σ(x)
        GLU(x,W1,W2) = σ(W1x)⊙W2x
        """
        super().__init__()
        self.w3 = Linear(d_model, d_ff, device, dtype)
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        
    def forward(
        self,
        x: Float[torch.Tensor, "... d_model"]
    ) -> Float[torch.Tensor, "... d_model"]:
        x1 = self.w1(x)
        x3 = self.w3(x)
        x1 = x1 * torch.sigmoid(x1)
        x2 = self.w2(x1 * x3)
        return x2