# SPDX-License-Identifier: Apache-2.0
"""Tests for selecting the ZONOS2 speaker-embedding checkpoint."""

import pytest

pytest.importorskip("torchaudio")
pytest.importorskip("transformers")

import sglang_omni.models.zonos2.components.speaker_encoder as speaker_encoder  # noqa: E402
from sglang_omni.config.runtime import resolve_stage_typed_kwargs  # noqa: E402
from sglang_omni.config.schema import StageConfig  # noqa: E402
from sglang_omni.models.zonos2.components.speaker_encoder import (  # noqa: E402
    Qwen3SpeakerEmbedding,
)
from sglang_omni.models.zonos2.stages import (  # noqa: E402
    create_speaker_encode_executor,
)


class FakeAutoModel:
    def __init__(self) -> None:
        self.sources: list[str] = []

    def from_pretrained(self, source: str, **kwargs: object) -> "FakeModel":
        self.sources.append(source)
        assert kwargs == {"trust_remote_code": True}
        return FakeModel()


class FakeModel:
    def to(self, device: str) -> "FakeModel":
        return self

    def eval(self) -> "FakeModel":
        return self


def test_embedder_uses_the_configured_model(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_auto_model = FakeAutoModel()
    monkeypatch.setattr(speaker_encoder, "AutoModel", fake_auto_model)

    embedder = Qwen3SpeakerEmbedding(
        device="cpu",
        speaker_embedding_model="/models/qwen3_voice_embedding",
    )

    assert fake_auto_model.sources == ["/models/qwen3_voice_embedding"]
    assert embedder.speaker_embedding_model == "/models/qwen3_voice_embedding"


def test_factory_config_reaches_the_speaker_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    class Recorder:
        def __init__(self, **kwargs: object) -> None:
            seen.update(kwargs)

    monkeypatch.setattr(speaker_encoder, "SpeakerEncoder", Recorder)
    stage = StageConfig(
        name="speaker_encode",
        factory_path="sglang_omni.models.zonos2.stages.create_speaker_encode_executor",
        factory={"speaker_embedding_model": "/models/qwen3_voice_embedding"},
    )

    create_speaker_encode_executor(
        "/models/zonos2",
        device="cpu",
        **resolve_stage_typed_kwargs(stage),
    )

    assert seen["speaker_embedding_model"] == "/models/qwen3_voice_embedding"
