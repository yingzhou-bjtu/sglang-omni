# SPDX-License-Identifier: Apache-2.0
"""The backbone config must be patched in a shadow, never in the checkpoint.

MiniMax Music 3 ships a backbone whose ``config.json`` is not a Qwen3 config, so
the loader needs ``model_type: qwen3``. Rewriting that file inside the
checkpoint breaks read-only weights mounts (and edits shared checkpoints); the
builder therefore mirrors the directory with symlinks and patches only the copy.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

from sglang_omni.models.minimax_music3.engine_builder import MiniMaxMusic3EngineBuilder


def _write_backbone(root, model_type: str):
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps({"model_type": model_type, "hidden_size": 8}), encoding="utf-8"
    )
    (root / "model.safetensors").write_bytes(b"weights")
    return root


def test_backbone_config_is_patched_outside_the_checkpoint(tmp_path) -> None:
    backbone = _write_backbone(tmp_path / "qwen_7B" / "qwen_7B", "qwen3_moe")

    shadow = MiniMaxMusic3EngineBuilder.normalize_backbone_config(
        backbone / "config.json"
    )

    assert shadow is not None
    assert shadow != backbone
    assert json.loads((shadow / "config.json").read_text())["model_type"] == "qwen3"
    # the checkpoint keeps its own config, and no .bak is dropped next to it
    assert (
        json.loads((backbone / "config.json").read_text())["model_type"] == "qwen3_moe"
    )
    assert not (backbone / "config.json.bak").exists()
    assert not list(backbone.glob("*.bak"))
    # weights are reachable through the shadow without copying them
    shadow_weights = shadow / "model.safetensors"
    assert shadow_weights.is_symlink()
    assert shadow_weights.read_bytes() == b"weights"


def test_qwen3_backbone_needs_no_shadow(tmp_path) -> None:
    backbone = _write_backbone(tmp_path / "already-qwen3", "qwen3")

    assert (
        MiniMaxMusic3EngineBuilder.normalize_backbone_config(backbone / "config.json")
        is None
    )


def test_shadow_directory_is_removed_with_the_builder(tmp_path) -> None:
    backbone = _write_backbone(tmp_path / "qwen_7B" / "qwen_7B", "qwen3_moe")
    (tmp_path / "flowmatching_vae.pth").write_bytes(b"dit")
    builder = MiniMaxMusic3EngineBuilder()
    builder.filter_audio_weights = lambda: None

    shadow_dir = Path(builder.resolve_checkpoint(str(tmp_path)))
    assert shadow_dir != backbone
    assert shadow_dir.is_dir()

    del builder
    gc.collect()

    assert not shadow_dir.exists()
