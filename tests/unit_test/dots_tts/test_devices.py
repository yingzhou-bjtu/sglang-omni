# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.dots_tts import stages
from sglang_omni.utils import device as device_utils


def _resolve_to(monkeypatch: pytest.MonkeyPatch, device_type: str) -> None:
    """Stage factories import resolve_concrete_device lazily, so patch the source."""
    monkeypatch.setattr(
        device_utils,
        "resolve_concrete_device",
        lambda device, gpu_id: torch.device(device_type, 0),
    )


def test_latent_engine_skips_the_cuda_guard_for_non_cuda_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _resolve_to(monkeypatch, "musa")

    class _Builder:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def build(self, model_path: str, **kwargs: object) -> str:
            assert model_path == "/model"
            return "scheduler"

    monkeypatch.setattr(
        "sglang_omni.models.dots_tts.engine_builder.DotsTTSEngineBuilder", _Builder
    )

    scheduler = stages.create_sglang_latent_engine_executor(
        "/model", device="musa", gpu_id=0
    )

    assert scheduler == "scheduler"


def test_vocoder_skips_the_cuda_guard_and_reads_the_real_capacity_attribute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _resolve_to(monkeypatch, "musa")
    monkeypatch.setattr(
        stages,
        "load_dots_audio_codec",
        lambda model_path, device: SimpleNamespace(device=device),
    )

    class _Vocoder:
        def __init__(self, codec: object, **kwargs: object) -> None:
            self.merge_steps = kwargs["merge_steps"]
            self.stream_slots = kwargs["stream_slots"]
            self.stream_chunk_batch_max = kwargs["max_batch_size"]
            self.pool_ready = False

        def ensure_slot_pool(self) -> None:
            self.pool_ready = True

    monkeypatch.setattr(stages, "DotsTTSStreamingVocoder", _Vocoder)

    vocoder = stages.create_vocoder_executor(
        "/model", device="musa", gpu_id=0, max_batch_size=3
    )

    assert vocoder.pool_ready
    assert vocoder.stream_chunk_batch_max == 3
    assert not hasattr(vocoder, "_stream_chunk_batch_max")
