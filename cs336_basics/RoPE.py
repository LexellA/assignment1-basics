import torch
from torch import nn
from jaxtyping import Int, Float
from einops import einsum, rearrange

class RotaryPositionalEmbedding(nn.Module):
    cos_thetas: torch.Tensor
    sin_thetas: torch.Tensor
    
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None
    ) -> None:
        """RoPE

        Args:
            theta (float): theta value for RoPE
            d_k (int): dimension of query and key vectors
            max_seq_len (int): Maximum sequence length that will be inputted
            device (torch.device): Device to store the buffer on 
        """
        super().__init__()
        inv_freq: Float[torch.Tensor, "d//2"] = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device) / d_k))
        positions: Int[torch.Tensor, "seq"] = torch.arange(0, max_seq_len, device=device)
        thetas: Float[torch.Tensor, "seq d//2"] = einsum(positions, inv_freq, "seq, d -> seq d")
        
        cos_thetas: Float[torch.Tensor, "seq d//2"] = thetas.cos()
        sin_thetas: Float[torch.Tensor, "seq d//2"] = thetas.sin()
        
        self.register_buffer("cos_thetas", cos_thetas, persistent=False)
        self.register_buffer("sin_thetas", sin_thetas, persistent=False)
        
        
    
    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len d_k"],
        token_positions: Int[torch.Tensor, "... seq_len"]
    ) -> Float[torch.Tensor, "... seq_len d_k"]:
        cos = self.cos_thetas[token_positions]
        sin = self.sin_thetas[token_positions]
        
        x = rearrange(x, "... seq (d two) -> two ... seq d", two = 2)
        x_even = x[0]
        x_odd = x[1]
        
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_odd * cos + x_even * sin
        
        x_rot = rearrange([x_rot_even, x_rot_odd], "two ... seq d -> ... seq (d two)")
        
        return x_rot
