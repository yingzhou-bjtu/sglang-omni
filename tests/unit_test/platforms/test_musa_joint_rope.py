# SPDX-License-Identifier: Apache-2.0
"""The MUSA joint-RoPE provider must match the reference rotation."""

from __future__ import annotations

import pytest
import torch
from x_transformers.x_transformers import apply_rotary_pos_emb

from sglang_omni.platforms import musa


def build_inputs(
    seq_len: int, heads: int, dim: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    phase = torch.rand(seq_len, dim // 2, dtype=torch.float32) * 6.0
    # note (yingzhou): x_transformers duplicates every half-frequency, and the
    # cache keeps the half table as cosines followed by sines.
    freqs = phase.repeat_interleave(2, dim=-1)
    cache = torch.cat((phase.cos(), phase.sin()), dim=-1).to(dtype)
    query = torch.randn(seq_len, heads, dim, dtype=dtype)
    key = torch.randn(seq_len, heads, dim, dtype=dtype)
    return query, key, cache, freqs, torch.arange(seq_len)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_musa_interleaved_rotation_matches_the_reference(dtype: torch.dtype) -> None:
    query, key, cache, freqs, positions = build_inputs(5, 3, 8, dtype)

    # note (yingzhou): apply_rotary_pos_emb reads the sequence from the
    # second-to-last axis, so [T, H, D] maps onto [batch, seq, heads * dim].
    def reference(tensor: torch.Tensor) -> torch.Tensor:
        flat = tensor.reshape(1, tensor.shape[0], -1)
        repeated = freqs.reshape(1, freqs.shape[0], -1).repeat(1, 1, tensor.shape[1])
        return apply_rotary_pos_emb(flat, repeated, 1.0).reshape(tensor.shape)

    expected_query = reference(query.clone())
    expected_key = reference(key.clone())

    musa.apply_rope_inplace(query, key, cache, positions, is_neox=False)

    # note (yingzhou): the provider accumulates in float32 and casts back, so
    # bfloat16 needs a cast-sized tolerance against the input-dtype reference.
    if dtype is torch.bfloat16:
        assert torch.allclose(query, expected_query, atol=2e-2, rtol=2e-2)
        assert torch.allclose(key, expected_key, atol=2e-2, rtol=2e-2)
    else:
        assert torch.allclose(query, expected_query, atol=1e-6)
        assert torch.allclose(key, expected_key, atol=1e-6)


def test_musa_neox_rotation_splits_the_head_in_half() -> None:
    dim = 8
    query, key, cache, _, positions = build_inputs(4, 2, dim, torch.float32)
    half = dim // 2
    cos = cache[:, :half].unsqueeze(1)
    sin = cache[:, half:].unsqueeze(1)
    leading, trailing = query[:, :, :half].clone(), query[:, :, half:].clone()
    expected = torch.cat(
        (leading * cos - trailing * sin, leading * sin + trailing * cos), dim=-1
    )

    musa.apply_rope_inplace(query, key, cache, positions, is_neox=True)

    assert torch.allclose(query, expected, atol=1e-6)
