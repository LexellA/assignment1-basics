from cs336_basics.softmax import softmax
from cs336_basics.linear import Linear
import torch
from torch import nn
from jaxtyping import Float, Bool
from einops import einsum, rearrange
from cs336_basics.RoPE import RotaryPositionalEmbedding


def ScaledDotProductAttention(
    Q: Float[torch.Tensor, "batch ... queries d_k"],
    K: Float[torch.Tensor, "batch ... keys d_k"],
    V: Float[torch.Tensor, "batch ... values d_v"],
    mask: Bool[torch.Tensor, "... queries keys"] | None = None,
) -> Float[torch.Tensor, "batch ... d_v"]:
    QK = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    d_k = Q.size(-1)
    
    pre: torch.Tensor = QK / (d_k ** 0.5)
    if mask is not None:
        pre = pre.masked_fill(~mask, float("-inf"))
        
    softmaxed = softmax(pre, -1)
    
    return einsum(V, softmaxed, "... keys d_v, ... queries keys -> ... queries d_v")
    
    
    
class MultiheadSelfAttention(nn.Module):
    """
    MultiHeadSelfAttention(x) = WOMultiHead(WQx,WKx,WV x)
    MultiHead(Q,K,V ) = Concat(head1,...,headh)
    for headi = Attention(Qi,Ki,Vi)
    """
    def __init__(
        self,
        d_model: int,
        num_head: int,
        use_rope: bool = True,
        rope_theta: float = 10000.0,
        rope_max_seq_len: int = 2048,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.d_k = d_model // num_head
        self.use_rope = use_rope
        
        self.W_q = Linear(d_model, d_model, device, dtype)
        self.W_k = Linear(d_model, d_model, device, dtype)
        self.W_v = Linear(d_model, d_model, device, dtype)
        self.W_o = Linear(d_model, d_model, device, dtype)
        
        self.RoPE = RotaryPositionalEmbedding(rope_theta, self.d_k, rope_max_seq_len, device)
        
    
    def forward(
        self,
        x: Float[torch.Tensor, " ... seq d_model"],
    ):
        Q: torch.Tensor = self.W_q(x)
        K: torch.Tensor = self.W_k(x)
        V: torch.Tensor = self.W_v(x)
        
        Q = rearrange(Q, "... queries (num_heads d_k) -> ... num_heads queries d_k", d_k=self.d_k)
        K = rearrange(K, "... keys (num_heads d_k) -> ... num_heads keys d_k", d_k=self.d_k)
        V = rearrange(V, "... values (num_heads d_k) -> ... num_heads values d_k", d_k=self.d_k)
        
        seq_length = Q.shape[-2]
        if self.use_rope is True:
            token_positions = torch.arange(0, seq_length, device=Q.device)
            Q = self.RoPE.forward(Q, token_positions)
            K = self.RoPE.forward(K, token_positions)
        
        mask = torch.triu(torch.ones(seq_length, seq_length, dtype=torch.bool), diagonal=1)
        
        attn = ScaledDotProductAttention(Q, K, V, ~mask)
        attn = rearrange(attn, "... num_heads queries d_k -> ... queries (num_heads d_k)")
        
        return self.W_o(attn)
        
        