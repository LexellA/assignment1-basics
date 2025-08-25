import torch
from jaxtyping import Float
from einops import reduce, einsum


def softmax(
    in_features: Float[torch.Tensor, " ..."],
    dimension: int
) -> Float[torch.Tensor, " ..."]:
    max_values = in_features.max(dim=dimension, keepdim=True).values
    in_features = in_features - max_values + 1
    
    e_x = in_features.exp()
    sum_e_x = e_x.sum(dim=dimension, keepdim=True)
    
    return e_x / sum_e_x
    
    