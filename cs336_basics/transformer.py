import torch
from torch import nn
from cs336_basics.RMSnorm import RMSNorm
from cs336_basics.attention import MultiheadSelfAttention
from cs336_basics.SwiGLU_FFN import SwiGLUFFN
from cs336_basics.embedding import Embedding
from cs336_basics.softmax import softmax
from cs336_basics.linear import Linear
from jaxtyping import Float, Int

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_head: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        rms_eps: float = 1e-5,
        use_RoPE: bool = True,
        RoPE_theta: float = 10000.0,
        RoPE_max_seq_len: int = 2048,
        
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model, rms_eps, device, dtype)
        
        self.attention = MultiheadSelfAttention(
            d_model, num_head, use_RoPE, 
            RoPE_theta, RoPE_max_seq_len, device, dtype
        )
        
        self.norm2 = RMSNorm(d_model, rms_eps, device, dtype)
        
        self.ffn = SwiGLUFFN(d_model, d_ff, device, dtype)
        
    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len d_model"]
    ) -> Float[torch.Tensor, "... seq_len d_model"]:
        x = x + self.attention(self.norm1(x))
        y = x + self.ffn(self.norm2(x))
        return y
        


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_head: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        rms_eps: float = 1e-5,
        use_RoPE: bool = True,
        RoPE_theta: float = 10000.0,
    ) -> None:
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, device, dtype)
        
        self.blocks = [
            TransformerBlock(
                d_model, num_head, d_ff, device, dtype, 
                rms_eps, use_RoPE, RoPE_theta, context_length
            ) for _ in range(num_layers)
        ]
        
        self.norm = RMSNorm(d_model, rms_eps, device, dtype)
        
        self.linear = Linear(d_model, vocab_size, device, dtype)
        
    def forward(
        self,
        x: Int[torch.Tensor, "... seq_len"]
    ) -> Float[torch.Tensor, "... seq_len vocab_size"]:
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        y = self.linear(x)
        # y = softmax(y, -1) #vocab_size
        return y
        