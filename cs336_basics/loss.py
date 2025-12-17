import torch
from cs336_basics.funciton import log_softmax
from jaxtyping import Float, Int

def cross_entropy(
    o: Float[torch.Tensor, " ... seq_len vocab_size"],
    x: Int[torch.Tensor, " ... seq_len"],
) -> Float[torch.Tensor, " ..."]:

    s_o = log_softmax(o, -1) # vocab_size
    
    p = s_o.gather(-1, x.unsqueeze(-1)).squeeze(-1)
    return -p.mean()