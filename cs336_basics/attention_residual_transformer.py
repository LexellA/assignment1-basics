from __future__ import annotations

import torch
from jaxtyping import Float, Int
from torch import nn

from cs336_basics.RMSnorm import RMSNorm
from cs336_basics.SwiGLU_FFN import SwiGLUFFN
from cs336_basics.attention import MultiheadSelfAttention
from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear


def block_attention_residual(
    blocks: list[Float[torch.Tensor, "... seq_len d_model"]],
    partial_block: Float[torch.Tensor, "... seq_len d_model"],
    proj: Linear,
    norm: RMSNorm,
    return_weights: bool = False,
) -> Float[torch.Tensor, "... seq_len d_model"] | tuple[
    Float[torch.Tensor, "... seq_len d_model"],
    Float[torch.Tensor, "depth ... seq_len"],
]:
    """
    README pseudocode version of Block AttnRes.

    blocks:
        Completed block representations, already including token embeddings.
    partial_block:
        Current intra-block partial sum.
    """

    values = torch.stack(blocks + [partial_block], dim=0)
    keys = norm(values)
    logits = proj(keys).squeeze(-1)
    weights = torch.softmax(logits, dim=0)
    mixed = torch.sum(weights.unsqueeze(-1) * values, dim=0)

    if return_weights:
        return mixed, weights
    return mixed


class BlockAttentionResidualTransformerBlock(nn.Module):
    """
    Block AttnRes block, written to mirror the official README pseudocode.
    """

    def __init__(
        self,
        d_model: int,
        num_head: int,
        d_ff: int,
        layer_number: int,
        block_size: int = 8,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        rms_eps: float = 1e-5,
        use_RoPE: bool = True,
        RoPE_theta: float = 10000.0,
        RoPE_max_seq_len: int = 2048,
    ) -> None:
        super().__init__()
        if block_size <= 0 or block_size % 2 != 0:
            raise ValueError("block_size must be a positive even number because each transformer layer has attention + MLP.")

        self.layer_number = layer_number
        self.block_size = block_size

        self.attn_res_proj = Linear(d_model, 1, device, dtype)
        self.attn_res_norm = RMSNorm(d_model, rms_eps, device, dtype)
        self.attn_norm = RMSNorm(d_model, rms_eps, device, dtype)
        self.attn = MultiheadSelfAttention(
            d_model,
            num_head,
            use_RoPE,
            RoPE_theta,
            RoPE_max_seq_len,
            device,
            dtype,
        )

        self.mlp_res_proj = Linear(d_model, 1, device, dtype)
        self.mlp_res_norm = RMSNorm(d_model, rms_eps, device, dtype)
        self.mlp_norm = RMSNorm(d_model, rms_eps, device, dtype)
        self.mlp = SwiGLUFFN(d_model, d_ff, device, dtype)

    def forward(
        self,
        blocks: list[Float[torch.Tensor, "... seq_len d_model"]],
        hidden_states: Float[torch.Tensor, "... seq_len d_model"],
    ) -> tuple[
        list[Float[torch.Tensor, "... seq_len d_model"]],
        Float[torch.Tensor, "... seq_len d_model"],
    ]:
        partial_block = hidden_states

        h = block_attention_residual(
            blocks,
            partial_block,
            self.attn_res_proj,
            self.attn_res_norm,
        )

        if self.layer_number > 0 and self.layer_number % (self.block_size // 2) == 0:
            blocks = [*blocks, partial_block]
            partial_block = None

        attn_out = self.attn(self.attn_norm(h))
        partial_block = partial_block + attn_out if partial_block is not None else attn_out

        h = block_attention_residual(
            blocks,
            partial_block,
            self.mlp_res_proj,
            self.mlp_res_norm,
        )

        mlp_out = self.mlp(self.mlp_norm(h))
        partial_block = partial_block + mlp_out

        return blocks, partial_block


class AttentionResidualTransformerLM(nn.Module):
    """
    Transformer LM using the paper's practical Block AttnRes implementation.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_head: int,
        d_ff: int,
        block_size: int = 8,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        rms_eps: float = 1e-5,
        use_RoPE: bool = True,
        RoPE_theta: float = 10000.0,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.embedding = Embedding(vocab_size, d_model, device, dtype)
        self.layers = nn.ModuleList(
            [
                BlockAttentionResidualTransformerBlock(
                    d_model=d_model,
                    num_head=num_head,
                    d_ff=d_ff,
                    layer_number=layer_idx,
                    block_size=block_size,
                    device=device,
                    dtype=dtype,
                    rms_eps=rms_eps,
                    use_RoPE=use_RoPE,
                    RoPE_theta=RoPE_theta,
                    RoPE_max_seq_len=context_length,
                )
                for layer_idx in range(num_layers)
            ]
        )
        self.norm = RMSNorm(d_model, rms_eps, device, dtype)
        self.linear = Linear(d_model, vocab_size, device, dtype)

    def forward(
        self,
        x: Int[torch.Tensor, "... seq_len"],
    ) -> Float[torch.Tensor, "... seq_len vocab_size"]:
        hidden_states = self.embedding(x)
        blocks = [hidden_states]

        for layer in self.layers:
            blocks, hidden_states = layer(blocks, hidden_states)

        hidden_states = self.norm(hidden_states)
        return self.linear(hidden_states)
