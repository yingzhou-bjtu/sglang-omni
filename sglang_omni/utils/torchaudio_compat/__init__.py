# SPDX-License-Identifier: Apache-2.0

"""Scoped torchaudio compatibility layer for SGLang-Omni audio paths.

Prefer the installed torchaudio package. The local implementations are used
only when that package cannot be imported in a MUSA runtime image.
"""

from __future__ import annotations

import io
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import torchaudio as _native_torchaudio
except (ImportError, OSError, RuntimeError):
    _native_torchaudio = None

if _native_torchaudio is None:
    from . import compliance, functional
else:
    from torchaudio import compliance as _native_compliance

    compliance = _native_compliance
    functional = _native_torchaudio.functional

__version__ = (
    getattr(_native_torchaudio, "__version__", "unknown")
    if _native_torchaudio is not None
    else "2.9.0+musa-compat"
)


def _read_audio_source(uri: Any) -> bytes:
    if isinstance(uri, (bytes, bytearray, memoryview)):
        return bytes(uri)
    if hasattr(uri, "read"):
        data = uri.read()
        return bytes(data)
    with open(Path(uri).expanduser(), "rb") as f:
        return f.read()


def _load_wav_bytes(data: bytes) -> tuple[torch.Tensor, int]:
    with wave.open(io.BytesIO(data), "rb") as wav_file:
        sample_rate = int(wav_file.getframerate())
        num_channels = int(wav_file.getnchannels())
        sample_width = int(wav_file.getsampwidth())
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width == 1:
        audio = torch.from_numpy(np.frombuffer(frames, dtype=np.uint8).copy())
        audio = audio.to(torch.float32)
        audio = (audio - 128.0) / 128.0
    elif sample_width == 2:
        audio = torch.from_numpy(np.frombuffer(frames, dtype="<i2").copy())
        audio = audio.to(torch.float32)
        audio = audio / 32768.0
    else:
        raise ValueError("unsupported WAV sample width")

    if num_channels > 1:
        audio = audio.reshape(-1, num_channels).transpose(0, 1).contiguous()
    else:
        audio = audio.reshape(1, -1)
    return audio, sample_rate


def load(uri, *_, **__) -> tuple[torch.Tensor, int]:
    if _native_torchaudio is not None:
        return _native_torchaudio.load(uri, *_, **__)
    data = _read_audio_source(uri)
    try:
        audio, sample_rate = _load_wav_bytes(data)
    except Exception:
        try:
            import soundfile as sf
        except Exception as exc:
            raise ValueError(
                "torchaudio shim only supports WAV input without soundfile"
            ) from exc
        audio_np, sample_rate = sf.read(
            io.BytesIO(data), always_2d=True, dtype="float32"
        )
        audio = torch.from_numpy(audio_np).transpose(0, 1).contiguous()
    return audio.to(dtype=torch.float32), int(sample_rate)


__all__ = ["compliance", "functional", "load"]
