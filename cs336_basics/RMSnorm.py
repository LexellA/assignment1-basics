import torch
from torch import nn
from jaxtyping import Float
from einops import reduce


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int, 
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps
        
        
    def forward(
        self,
        x: Float[torch.Tensor, "batch seq d_model"]
    ) -> Float[torch.Tensor, "batch seq d_model"]:
        in_dtype = x.dtype
        x.to(torch.float32)
        
        rms = reduce(x ** 2, "batch seq d_model -> batch seq 1", "mean")
        rms = (rms + self.eps).sqrt()
        
        result = (x / rms) * self.gain
        
        return result.to(in_dtype)
        