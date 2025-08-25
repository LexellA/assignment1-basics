import torch
from torch import nn
from einops import einsum
from jaxtyping import Float

class Linear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None 
        ):
        """ Linear transformation module.
        
        Args:
            in_features (int): final dimension of the input
            out_features (int): final dimension of the output
            device (torch.device, optional): Device to store the parameters on
            dtype (torch.dtype, optional): Data type of the parameter
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        sigma = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, 0, sigma, -3 * sigma, 3 * sigma)
    
    def forward(self, x: Float[torch.Tensor, "... d_in"]) -> Float[torch.Tensor, "d_out"]:
        y = einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
        return y