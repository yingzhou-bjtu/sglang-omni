# SPDX-License-Identifier: Apache-2.0
"""MUSA platform adaptation tests for FishAudio S2-Pro."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.fishaudio_s2_pro import engine_builder
from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.text2semantic import (
    audio_decoder as fish_audio_decoder,
)


def test_fast_ar_attention_backend_uses_the_native_path_on_musa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        engine_builder,
        "current_platform",
        SimpleNamespace(is_npu=lambda: False, is_musa=lambda: True),
    )

    backend = engine_builder.resolve_fast_ar_attention_backend(gpu_id=0)

    assert backend == "torch_native"


def test_musa_kvcache_attention_matches_a_reference_reduction() -> None:
    torch.manual_seed(0)
    batch, heads, kv_heads, head_dim, cache_position = 2, 4, 2, 8, 3
    q = torch.randn(batch, 1, heads, head_dim, dtype=torch.float64)
    k = torch.randn(batch, 1, kv_heads, head_dim, dtype=torch.float64)
    v = torch.randn(batch, 1, kv_heads, head_dim, dtype=torch.float64)
    k_cache = torch.zeros(batch, 11, kv_heads, head_dim, dtype=torch.float64)
    v_cache = torch.zeros(batch, 11, kv_heads, head_dim, dtype=torch.float64)

    out = fish_audio_decoder.musa_kvcache_attention(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        k=k,
        v=v,
        cache_position=cache_position,
    )

    assert torch.equal(k_cache[:, cache_position : cache_position + 1], k)
    assert torch.equal(v_cache[:, cache_position : cache_position + 1], v)
    assert out.shape == q.shape

    query = q.transpose(1, 2)
    key = k_cache[:, : cache_position + 1].transpose(1, 2)
    value = v_cache[:, : cache_position + 1].transpose(1, 2)
    repeat = heads // kv_heads
    key = key.repeat_interleave(repeat, dim=1)
    value = value.repeat_interleave(repeat, dim=1)
    attn = torch.softmax((query / math.sqrt(head_dim)) @ key.transpose(-1, -2), dim=-1)
    expected = (attn @ value).transpose(1, 2)

    assert torch.allclose(out, expected, atol=1e-12)


def test_musa_kvcache_attention_requires_key_and_value() -> None:
    q = torch.randn(1, 1, 2, 4)
    cache = torch.zeros(1, 4, 2, 4)

    with pytest.raises(ValueError, match="requires k and v"):
        fish_audio_decoder.musa_kvcache_attention(
            q=q,
            k_cache=cache,
            v_cache=cache,
            k=None,
            v=None,
            cache_position=0,
        )
