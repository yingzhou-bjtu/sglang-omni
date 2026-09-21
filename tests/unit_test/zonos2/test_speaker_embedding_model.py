# SPDX-License-Identifier: Apache-2.0
"""GPU-free tests for the ZONOS2 speaker-embedding weights source.

The Qwen3 voice-embedding checkpoint lives in its own Hugging Face repository, so
a node without network access has to be able to point the speaker encoder at a
staged snapshot instead of the hub id. These cover the resolution order and the
plumbing from the stage factory down to ``AutoModel.from_pretrained``; neither the
real checkpoint nor a GPU is required (the module still imports
torchaudio/transformers).
"""

from pathlib import Path

import pytest

pytest.importorskip("torchaudio")
pytest.importorskip("transformers")

import sglang_omni.models.zonos2.components.speaker_encoder as speaker_encoder  # noqa: E402
from sglang_omni.models.zonos2.components.speaker_encoder import (  # noqa: E402
    SPEAKER_EMBEDDING_PATH_ENV,
    Qwen3SpeakerEmbedding,
    SpeakerEncoder,
    resolve_speaker_embedding_source,
)
from sglang_omni.models.zonos2.stages import (  # noqa: E402
    create_speaker_encode_executor,
)


def test_configured_directory_wins_over_the_hub_id(tmp_path: Path) -> None:
    staged = tmp_path / "qwen3_voice_embedding"
    staged.mkdir()
    assert resolve_speaker_embedding_source(str(staged)) == str(staged)


def test_env_override_is_used_when_the_config_omits_the_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    staged = tmp_path / "qwen3_voice_embedding"
    staged.mkdir()
    monkeypatch.setenv(SPEAKER_EMBEDDING_PATH_ENV, str(staged))
    assert resolve_speaker_embedding_source() == str(staged)


def test_hub_id_is_kept_when_nothing_is_staged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(SPEAKER_EMBEDDING_PATH_ENV, raising=False)
    assert resolve_speaker_embedding_source() == Qwen3SpeakerEmbedding.MODEL_NAME
    # A configured source that is not a directory stays a hub id as well.
    missing = str(tmp_path / "not-staged")
    assert resolve_speaker_embedding_source(missing) == missing


class _FakeAutoModel:
    def __init__(self) -> None:
        self.sources: list[str] = []

    def from_pretrained(self, source: str, **kwargs: object) -> "_FakeModel":
        self.sources.append(source)
        assert kwargs == {"trust_remote_code": True}
        return _FakeModel()


class _FakeModel:
    def to(self, device: str) -> "_FakeModel":
        return self

    def eval(self) -> "_FakeModel":
        return self


def test_embedder_loads_the_configured_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(SPEAKER_EMBEDDING_PATH_ENV, raising=False)
    staged = tmp_path / "qwen3_voice_embedding"
    staged.mkdir()
    fake_auto = _FakeAutoModel()
    monkeypatch.setattr(speaker_encoder, "AutoModel", fake_auto)

    embedder = Qwen3SpeakerEmbedding(device="cpu", model_path=str(staged))

    assert fake_auto.sources == [str(staged)]
    assert embedder.model_path == str(staged)


def test_env_override_reaches_the_embedder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    staged = tmp_path / "qwen3_voice_embedding"
    staged.mkdir()
    monkeypatch.setenv(SPEAKER_EMBEDDING_PATH_ENV, str(staged))
    fake_auto = _FakeAutoModel()
    monkeypatch.setattr(speaker_encoder, "AutoModel", fake_auto)

    Qwen3SpeakerEmbedding(device="cpu")

    assert fake_auto.sources == [str(staged)]


def test_speaker_encoder_forwards_the_embedding_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    staged = tmp_path / "qwen3_voice_embedding"
    staged.mkdir()
    seen: dict[str, object] = {}

    class _Recorder:
        def __init__(self, **kwargs: object) -> None:
            seen.update(kwargs)

    monkeypatch.setattr(speaker_encoder, "Qwen3SpeakerEmbedding", _Recorder)

    SpeakerEncoder(device="cpu", embedding_model=str(staged)).get_embedder()

    assert seen["model_path"] == str(staged)
    assert seen["device"] == "cpu"


def test_speaker_encode_stage_forwards_the_embedding_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    staged = tmp_path / "qwen3_voice_embedding"
    staged.mkdir()
    seen: dict[str, object] = {}

    class _Recorder:
        def __init__(self, **kwargs: object) -> None:
            seen.update(kwargs)

    monkeypatch.setattr(speaker_encoder, "SpeakerEncoder", _Recorder)

    create_speaker_encode_executor(
        "/models/zonos2", device="cpu", speaker_embedding_model=str(staged)
    )

    assert seen["embedding_model"] == str(staged)
