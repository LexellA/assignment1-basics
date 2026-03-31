import torch

from cs336_basics.attention_residual_transformer import (
    AttentionResidualTransformerLM,
    BlockAttentionResidualTransformerBlock,
    block_attention_residual,
)
from cs336_basics.RMSnorm import RMSNorm
from cs336_basics.linear import Linear


def test_block_attention_residual_normalizes_over_depth():
    proj = Linear(8, 1)
    norm = RMSNorm(8)
    blocks = [torch.randn(2, 4, 8), torch.randn(2, 4, 8)]
    partial_block = torch.randn(2, 4, 8)

    mixed, weights = block_attention_residual(blocks, partial_block, proj, norm, return_weights=True)

    assert mixed.shape == partial_block.shape
    torch.testing.assert_close(weights.sum(dim=0), torch.ones_like(weights[0]), atol=1e-6, rtol=1e-6)


def test_block_attention_residual_transformer_block_shape():
    block = BlockAttentionResidualTransformerBlock(
        d_model=16,
        num_head=4,
        d_ff=32,
        layer_number=2,
        block_size=4,
        use_RoPE=False,
    )
    blocks = [torch.randn(2, 5, 16)]
    hidden_states = torch.randn(2, 5, 16)

    updated_blocks, partial_block = block(blocks, hidden_states)

    assert len(updated_blocks) == 2
    assert partial_block.shape == (2, 5, 16)


def test_attention_residual_transformer_lm_forward_shape():
    model = AttentionResidualTransformerLM(
        vocab_size=32,
        context_length=8,
        num_layers=2,
        d_model=16,
        num_head=4,
        d_ff=32,
        block_size=4,
        use_RoPE=False,
    )
    token_ids = torch.randint(0, 32, (2, 8))

    logits = model(token_ids)

    assert logits.shape == (2, 8, 32)
