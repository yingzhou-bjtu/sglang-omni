# SPDX-License-Identifier: Apache-2.0
"""The MUSA joint-RoPE provider reuses SGLang's native rotary embedding."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest
import torch

from sglang_omni.platforms import musa


def native_apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, is_neox: bool
) -> torch.Tensor:
    # Mirror sglang.srt.layers.rotary_embedding.utils.apply_rotary_emb.
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox:
        return torch.cat((o1, o2), dim=-1)
    return torch.stack((o1, o2), dim=-1).flatten(-2)


def build_inputs(
    seq_len: int, heads: int, dim: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    phase = torch.rand(seq_len, dim // 2, dtype=torch.float32) * 6.0
    cache = torch.cat((phase.cos(), phase.sin()), dim=-1).to(dtype)
    query = torch.randn(seq_len, heads, dim, dtype=dtype)
    key = torch.randn(seq_len, heads, dim, dtype=dtype)
    return query, key, cache, torch.arange(seq_len)


@pytest.mark.parametrize("is_neox", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_musa_inplace_rotation_matches_native_rotary(
    monkeypatch: pytest.MonkeyPatch, is_neox: bool, dtype: torch.dtype
) -> None:
    utils = ModuleType("sglang.srt.layers.rotary_embedding.utils")
    utils.apply_rotary_emb = native_apply_rotary_emb
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    query, key, cache, positions = build_inputs(5, 3, 8, dtype)
    rows = cache.index_select(0, positions)
    half = rows.shape[-1] // 2
    expected_query = native_apply_rotary_emb(
        query.clone(), rows[:, :half], rows[:, half:], is_neox
    )
    expected_key = native_apply_rotary_emb(
        key.clone(), rows[:, :half], rows[:, half:], is_neox
    )

    musa.apply_rope_inplace(query, key, cache, positions, is_neox=is_neox)

    torch.testing.assert_close(query, expected_query, rtol=0, atol=0)
    torch.testing.assert_close(key, expected_key, rtol=0, atol=0)
