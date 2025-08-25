import torch
from torch import nn
from jaxtyping import Int, Float
from einops import einsum

class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """embedding vector for each token ID by indexing into an embedding matrix

        Args:
            num_embeddings (int): Size of vocab
            embedding_dim (int): Dimension of the embedding vectors, d_model
            device (torch.device | None, optional): device to store the parameters on. Defaults to None.
            dtype (torch.dtype | None, optional): Data type of parameters. Defaults to None.
        """
        
        super().__init__()
        
        self.embedding_matrix = nn.Parameter(
                torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
            )
        sigma = (2 / (num_embeddings + embedding_dim)) ** 0.5
        nn.init.trunc_normal_(self.embedding_matrix, 0, sigma, -3 * sigma, 3 * sigma)
        
    
    def forward(
        self,
        token_ids: Int[torch.Tensor, " ..."],
    ) -> Float[torch.Tensor, " ... d_model"]:
        return self.embedding_matrix[token_ids]