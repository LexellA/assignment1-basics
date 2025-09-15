import torch
from torch import nn
from cs336_basics.RMSnorm import RMSNorm
from cs336_basics.attention import MultiheadSelfAttention
from cs336_basics.SwiGLU_FFN import SwiGLUFFN
from cs336_basics.embedding import Embedding
from cs336_basics.funciton import softmax
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
        self.context_length = context_length
        self.embedding = Embedding(vocab_size, d_model, device, dtype)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, num_head, d_ff, device, dtype, 
                rms_eps, use_RoPE, RoPE_theta, context_length
            ) for _ in range(num_layers)
        ])
        
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
        return y




from cs336_basics.tokenizer import Tokenizer
def generate(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int, 
    temperature: float, 
    top_p: float,
    end_token: str,
    device: torch.device | None = None
) -> str:
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, device=device) #(seq_len)
    end_token_id = tokenizer.encode(end_token)[0]
    
    model.eval()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if tokens.size(-1) > model.context_length:
                tokens = tokens[... ,-model.context_length:]
            logits = model(tokens)[..., -1:, :] #(1, vocab_size)
            
            #temprature scaling
            logits = logits / temperature
            
            probs = softmax(logits, -1)
            
            #top-p
            sorted_probs, sorted_indices = torch.sort(probs, descending=True) #(1, vocab_size)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            
            Vpmask = cum_probs < top_p
            Vpmask[..., 0] = True
            
            masked_porbs = sorted_probs.clone()
            masked_porbs[~Vpmask] = 0
            
            masked_porbs = masked_porbs / torch.sum(masked_porbs, keepdim=True, dim=-1)
            
            sample_indices = torch.multinomial(masked_porbs.squeeze(), num_samples=1) #(1)
            next_token = sorted_indices.squeeze().gather(-1, sample_indices)
            
            tokens = torch.cat([tokens, next_token], dim=-1) #(seq_len)
            
            if end_token_id is not None and next_token.item() == end_token_id:
                break
    
    generate_tokens = tokens.tolist()
    return tokenizer.decode(generate_tokens)