# SPDX-License-Identifier: Apache-2.0
"""MUSA platform adaptation tests for FishAudio S2-Pro."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.fishaudio_s2_pro import engine_builder
from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.text2semantic import (
    audio_decoder as fish_audio_decoder,
)


def test_fast_ar_attention_backend_uses_fa3_on_musa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        engine_builder,
        "current_platform",
        SimpleNamespace(is_npu=lambda: False, is_musa=lambda: True),
    )

    backend = engine_builder.resolve_fast_ar_attention_backend(gpu_id=0)

    assert backend == "fa3"


def test_musa_fast_ar_uses_fa3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fish_audio_decoder,
        "current_platform",
        SimpleNamespace(is_musa=lambda: True),
    )
    fish_audio_decoder.fast_ar_uses_fa3.cache_clear()
    try:
        assert fish_audio_decoder.fast_ar_uses_fa3(0)
    finally:
        fish_audio_decoder.fast_ar_uses_fa3.cache_clear()


def test_musa_fast_ar_kvcache_reuses_sglang_flash_attn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def flash_attn_with_kvcache(**kwargs: object) -> torch.Tensor:
        calls.append(kwargs)
        return kwargs["q"]

    fake_flash_attn = SimpleNamespace(flash_attn_with_kvcache=flash_attn_with_kvcache)
    fake_sgl_kernel = types.ModuleType("sgl_kernel")
    fake_sgl_kernel.__path__ = []
    fake_sgl_kernel.flash_attn = fake_flash_attn
    monkeypatch.setitem(sys.modules, "sgl_kernel", fake_sgl_kernel)
    monkeypatch.setitem(sys.modules, "sgl_kernel.flash_attn", fake_flash_attn)
    monkeypatch.setattr(fish_audio_decoder, "fast_ar_uses_fa3", lambda _: True)

    q = SimpleNamespace(device=SimpleNamespace(index=0))
    k = torch.randn(2, 1, 2, 8)
    v = torch.randn(2, 1, 2, 8)
    k_cache = torch.zeros(2, 11, 2, 8)
    v_cache = torch.zeros(2, 11, 2, 8)
    cache_seqlens = torch.tensor([4, 4], dtype=torch.int32)

    out = fish_audio_decoder.fast_ar_kvcache_attention(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        k=k,
        v=v,
        cache_seqlens=cache_seqlens,
        causal=True,
        num_splits=1,
        cache_position=-1,
    )

    assert out is q
    assert len(calls) == 1
    assert calls[0] == {
        "q": q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "k": k,
        "v": v,
        "cache_seqlens": cache_seqlens,
        "causal": True,
        "num_splits": 1,
    }
